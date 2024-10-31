"""
Functions for generating visualizations of training progress.
"""

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb

from einops import reduce
from functools import partial
from tqdm import tqdm
from typing import List

from .checkpoint import TrainState
from .transformer_model import ImageModel


@partial(jax.jit, static_argnames=["mdl"])
def _compute_losses_and_entropies(
    mdl: ImageModel,
    params: jax.Array,
    images: jax.Array,
    clip_embeddings: jax.Array,
    max_cos_distances: jax.Array,
):
    """Compute the per-token losses and entropies for a set of images"""
    logits = mdl.apply(
        params,
        images=images,
        clip_embeddings=clip_embeddings,
        max_cos_distances=max_cos_distances,
    ).astype(jnp.float32)
    losses = optax.softmax_cross_entropy(logits, jax.nn.one_hot(images, 8192))
    assert losses.shape == (images.shape[0], mdl.image_tokens)
    probs = jax.nn.softmax(logits, axis=-1)
    entropies = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
    assert entropies.shape == losses.shape
    return losses, entropies


def log_token_loss_visualization(
    train_state: TrainState,
    mdl: ImageModel,
    images: jax.Array,
    image_names: List[str],
    clip_embeddings: jax.Array,
    max_cos_distances: jax.Array,
    global_step: int,
):
    """Log charts of the per token loss and entropy for a set of test images"""
    img_cnt = images.shape[0]
    assert images.shape == (img_cnt, mdl.image_tokens)
    if mdl.clip_conditioning and mdl.clip_caps:
        assert clip_embeddings.shape == (img_cnt, mdl.clip_cap_count, 768)
        assert max_cos_distances.shape == (img_cnt, mdl.clip_cap_count)
    elif mdl.clip_conditioning and not mdl.clip_caps:
        assert clip_embeddings.shape == (img_cnt, 768)
        max_cos_distances = jnp.zeros((img_cnt, 0), dtype=jnp.float32)
    else:
        assert clip_embeddings.shape == (img_cnt, 0)
        max_cos_distances = jnp.zeros((img_cnt, 0), dtype=jnp.float32)
    params = train_state.get_eval_params()

    test_mdl = mdl.clone(dropout=None, image_dropout=None)

    losses, entropies = jax.device_get(
        _compute_losses_and_entropies(
            test_mdl,
            params,
            images,
            clip_embeddings,
            max_cos_distances,
        )
    )

    # Generate scatter plot for token losses
    fig_loss, ax_loss = plt.subplots(figsize=(12, 8))

    for i in range(img_cnt):
        ax_loss.scatter(
            range(mdl.image_tokens),
            losses[i],
            label=image_names[i],
            alpha=0.5,
            s=10,
        )

    ax_loss.set_xlabel("Token #")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Per-Token Loss for Test Images (step {global_step})")
    tick_positions = [x * mdl.image_tokens // 8 for x in range(9)]
    ax_loss.set_xticks(tick_positions)
    ax_loss.set_xticklabels(tick_positions)
    ax_loss.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    wandb.log(
        {"global_step": global_step, "token_loss_visualization": wandb.Image(fig_loss)}
    )
    plt.close(fig_loss)

    # Generate scatter plot for token entropies
    fig_entropy, ax_entropy = plt.subplots(figsize=(12, 8))

    for i in range(img_cnt):
        ax_entropy.scatter(
            range(mdl.image_tokens),
            entropies[i],
            label=image_names[i],
            alpha=0.5,
            s=10,
        )

    ax_entropy.set_xlabel("Token #")
    ax_entropy.set_xticks(tick_positions)
    ax_entropy.set_xticklabels(tick_positions)
    ax_entropy.set_ylabel("Entropy")
    ax_entropy.set_title(f"Per-Token Entropy for Test Images (step {global_step})")
    ax_entropy.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    wandb.log(
        {
            "global_step": global_step,
            "token_entropy_visualization": wandb.Image(fig_entropy),
        }
    )
    plt.close(fig_entropy)


@partial(jax.jit, static_argnames=["mdl"])
def _compute_attention_weights(
    mdl: ImageModel,
    train_state: TrainState,
    image: jax.Array,
    clip_embedding: jax.Array,
    max_cos_distance: jax.Array,
):
    """Compute the attention weights for an image"""
    mdl_record = mdl.copy(
        record_attention_weights=True, dropout=None, image_dropout=None
    )
    params = train_state.get_eval_params()

    logits, intermediates = mdl_record.apply(
        params,
        mutable=["intermediates"],
        images=image[None, ...],
        clip_embeddings=clip_embedding[None, ...],
        max_cos_distances=max_cos_distance[None, ...],
    )
    weights = intermediates["intermediates"]["transformer_layers"]["mha"][
        "attention_weights"
    ][0].astype(jnp.float32)
    assert weights.shape == (
        mdl.n_layers,
        1,
        mdl.num_heads,
        mdl.image_tokens,
        mdl.image_tokens,
    )

    weights_avgd = reduce(
        weights, "layers 1 head tok_q tok_k -> layers tok_q tok_k", "mean"
    )
    weights_head0 = weights[:, 0, 0]
    assert (
        weights_avgd.shape
        == weights_head0.shape
        == (mdl.n_layers, mdl.image_tokens, mdl.image_tokens)
    )

    return weights_avgd, weights_head0


def log_attention_maps(
    train_state: TrainState,
    mdl: ImageModel,
    image: jax.Array,
    image_name: str,
    clip_embedding: jax.Array,
    max_cos_distance: jax.Array,
    global_step: int,
):
    """Log attention maps for an image"""
    weights_avgd, weights_head0 = jax.device_get(
        _compute_attention_weights(
            mdl, train_state, image, clip_embedding, max_cos_distance
        )
    )

    to_log = {"global_step": global_step}

    for i in range(mdl.n_layers):
        layer_attn_weights_avgd = weights_avgd[i]
        layer_attn_weights_head0 = weights_head0[i]

        for weights, title, wandb_name in [
            (
                layer_attn_weights_avgd,
                f"Attention weights for {image_name}, all heads, layer {i} (step {global_step})",
                f"attention_maps/avgd_layer_{i:03d}",
            ),
            (
                layer_attn_weights_head0,
                f"Attention weights for {image_name}, head 0, layer {i} (step {global_step})",
                f"attention_maps/head0_layer_{i:03d}",
            ),
        ]:
            is_causal = np.allclose(np.triu(weights, k=1), 0)
            if not is_causal:
                tqdm.write(f"WARNING: attention weights at layer {i} are not causal")

            # The plots are super hard to read with a linear color scale, since tok 0 always puts
            # 100% of its attention weight on tok 0, meaning the max value is always 1. The min
            # value is always nearly 0, so the maps for relatively flat layers/heads are basically
            # a solid color. So we log transform and use a norm scheme that puts the median in
            # the middle.
            eps = 1e-10
            weights_log = np.log(weights + eps)

            # Mask the upper triangle when calculating the minimum and median
            mask = np.triu(np.ones_like(weights_log), k=1)
            masked_weights_log = np.ma.array(weights_log, mask=mask)

            # ensure vmin < vmedian < vmax. it's possible for one of the endpoints to be the median
            # and the scaling will error out in that case.
            vmin = np.min(masked_weights_log) - eps
            vmedian = np.ma.median(masked_weights_log)
            vmax = np.max(masked_weights_log) + eps

            fig, ax = plt.subplots()
            im = ax.imshow(
                weights_log,
                cmap="RdBu",
                aspect="auto",
                norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vmedian, vmax=vmax),
                # align so the center of each pixel corresponds exactly to a token index
                extent=(-0.5, weights.shape[1] - 0.5, weights.shape[0] - 0.5, -0.5),
            )

            # Add margins to make sure we can clearly see token 0
            margin = 5.5
            ax.set_xlim(-margin, weights.shape[1] - 1 + margin)
            ax.set_ylim(weights.shape[0] - 1 + margin, -margin)

            ax.set_title(title)
            ax.set_xlabel("Key token")
            ax.set_ylabel("Query token")

            # Set tick marks at 0, 1/4, 1/2, 3/4, and all of the total token count. it bothers my
            # programmer brain when they show up as 0, 250, 500, 750, 1000 instead of 0, 256, 512,
            # 768, 1024.
            token_count = weights.shape[0]
            tick_positions = [x * token_count // 4 for x in range(5)]
            ax.set_xticks(tick_positions)
            ax.set_yticks(tick_positions)
            ax.set_xticklabels(tick_positions)
            ax.set_yticklabels(tick_positions)

            # colorbar with original scale labels
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Attention weight")
            log_tick_locations = np.sort(
                np.concatenate([np.linspace(vmin, vmax, 4), np.array([vmedian])])
            )
            original_tick_locations = np.exp(log_tick_locations) - eps
            cbar.set_ticks(log_tick_locations)
            cbar.set_ticklabels([f"{loc:.2e}" for loc in original_tick_locations])

            fig.tight_layout()
            to_log[wandb_name] = wandb.Image(fig)
            plt.close(fig)

    wandb.log(to_log)
