import flash_attention_jax  # Pure JAX flash attention implementation by lucidrains
import flash_attn_jax as flash_attention_cpp  # C++ flash attention by Tri Dao et al, JAX bindings by nshepperd
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax  # type: ignore[import]
import pytest
from copy import copy
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from einops import rearrange, reduce, repeat
from flax import struct
from functools import partial
from infinidata import TableView
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from tqdm import tqdm, trange

from .cap_sampling import LogitsTable, random_pt_with_cosine_similarity, sample_cap
from .config import ModelConfig
from .gen_training_caps import gen_training_examples_from_tree
from .gpu_check import gpu_is_ampere_or_newer
from .load_pq_dir import load_pq_to_infinidata
from .spherical_space_partitioning import CapTree
from .triangle_schedule import triangle_schedule


AttnMethod = Enum("AttnMethod", ["STANDARD", "FLASH_JAX", "FLASH_CPP", "CUDNN"])


class ImageModel(nn.Module):
    """A transformer model for images encoded to a discrete representation."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    image_dropout: Optional[float]
    clip_dropout: Optional[float]
    n_layers: int
    image_tokens: int
    clip_conditioning: bool
    clip_caps: bool
    clip_cap_count: Optional[int]
    corrected_cap_projections: bool
    do_clip_feedforward: bool
    norm_clip_embeddings: bool
    use_biases: bool
    activations_dtype: jnp.dtype
    activation_function: Callable[[jax.Array], jax.Array]
    weights_dtype: jnp.dtype
    pre_norm: bool
    decode: bool = False
    attn_method: Optional[AttnMethod] = None
    record_attention_weights: bool = (
        False  # whether to record attention weights for visualization
    )

    def setup(self) -> None:
        # Follows PaLM: "PaLM: Scaling Language Modeling with Pathways"
        # https://arxiv.org/abs/2204.02311
        default_kernel_init = nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="normal"
        )
        self.in_embed = nn.Embed(
            num_embeddings=8192,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )

        if self.image_dropout is not None:
            self.image_dropout_layer = nn.Dropout(
                # This drops the features within the tokens, not entire tokens. Which makes less
                # sense as a strategy for reducing the model's ability to focus on learning local
                # structure, but empirically works substantially better.
                rate=self.image_dropout,
                deterministic=False,
            )
        else:
            self.image_dropout_layer = nn.Dropout(rate=0, deterministic=True)

        # A note on how CLIP conditioning works:
        # There are three modes:
        # 1) No conditioning. We prepend a zero token to the input sequence. This is partly
        # bitrotted at this point (2024-09-29)

        # 2) Conditioning on one CLIP embedding. We project the CLIP embedding into d_model
        # and prepend it to the input sequence. This works, and the model learns to produce images
        # that look like an image with that CLIP embedding. This works OK when prompting with an
        # embedding computed from another image, though well fit models will produce images that are
        # very similar to the input, which is probably not actually what you want. It works less
        # well when prompting with an embedding computed from text. Text embeddings and image
        # embeddings don't necessarily overlap, so a perfectly reasonable text prompt may produce
        # images that don't look anything like the prompt.

        # 3) Conditioning on one or more spherical caps. CLIP embeddings are unit vectors, and the
        # image's embedding must fall inside all the caps. This solves the problems with mode 2.
        # Text prompts work because now we're generating images that are within a certain distance
        # of the text embedding, and the distance being configurable means we can decide how similar
        # we want the output to be to the prompt at inference time. We project the cap's center and
        # the maximum cosine distance the embedding can have from the center into d_model, and then
        # sum them, producing one conditioning token per cap, which we then prepend to the input. So
        # the image's embedding must be in the intersection of all the caps. I've trained models
        # conditioned on one cap and they work well enough to prove the concept. Better models
        # coming soon. I haven't trained models conditioned on more than one cap with very much
        # compute yet. The results of such models may allow you to do things with prompting that are
        # impossible with a single prompt. Other models do allow multiple prompts but AFAICT they're
        # not doing *intersections* they're just taking an average.

        # When training, we generate caps for each image in ImageModel.gen_training_caps.

        assert (
            self.clip_conditioning or not self.clip_caps
        ), "Can't use clip_caps without clip_conditioning"

        if self.clip_caps:
            assert self.clip_cap_count is not None, "clip_cap_count must be set"
            assert self.clip_cap_count > 0, "clip_cap_count must be positive"

        if self.clip_dropout is not None:
            self.clip_dropout_layer = nn.Dropout(
                rate=self.clip_dropout,
                deterministic=False,
            )
        else:
            self.clip_dropout_layer = nn.Dropout(rate=0, deterministic=True)

        # The initializers for CLIP conditioning are chosen such that the conditioning tokens have
        # the same distribution as the token embeddings, assuming the clip embeddings are uniformly
        # distributed on the unit sphere and the max distances are drawn from U[0, 2]. See
        # https://chatgpt.com/share/66f1ee03-0874-800c-a230-9f5784f6d898 for the derivation.

        # We add the projections for the cap centers and the projections for the max cosine
        # distances so we need to scale the standard deviations of the projection weights by
        # sqrt(2) to ensure the variance is the same as it would be with a single projection.
        clip_proj_stddev = 1.0 if not self.clip_caps else 1.0 / np.sqrt(2)
        self.clip_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            # variance_scaling scale is in variance and normal stddev is in standard deviation, so
            # we need to square the stddev.
            kernel_init=nn.initializers.variance_scaling(
                scale=clip_proj_stddev**2.0, mode="fan_in", distribution="normal"
            )
            if self.norm_clip_embeddings
            else nn.initializers.normal(stddev=clip_proj_stddev),
        )
        self.max_cos_distance_proj = nn.Dense(
            features=self.d_model,
            use_bias=True,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=clip_proj_stddev),
        )

        if self.do_clip_feedforward:
            self.clip_ff_up = nn.Dense(
                features=self.ff_dim,
                kernel_init=default_kernel_init,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
            )
            self.clip_ff_down = nn.Dense(
                features=self.d_model,
                kernel_init=default_kernel_init,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
            )
            self.cond_tokens_layernorm = nn.LayerNorm(
                dtype=self.activations_dtype,
                param_dtype=jnp.float32,
            )

        if self.norm_clip_embeddings:
            self.clip_embeddings_norm = nn.LayerNorm(
                dtype=self.activations_dtype,
                param_dtype=jnp.float32,
            )

        self.positional_encoding = nn.Embed(
            num_embeddings=self.seq_len(),
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=1.0),
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )

        # Detect best available attention method
        if self.attn_method is None:
            if gpu_is_ampere_or_newer():
                if self.activations_dtype in [jnp.float16, jnp.bfloat16]:
                    if (
                        jax.local_device_count("gpu") > 1
                        and "H100" in jax.devices("gpu")[0].device_kind
                    ):
                        print(
                            "Using CUDNN flash attention to work around H100 deadlock bug"
                        )
                        attn_method = AttnMethod.CUDNN
                    else:
                        attn_method = AttnMethod.FLASH_CPP
                else:
                    print(
                        f"Warning: GPU is eligible for faster attention but activations_dtype "
                        f"is {self.activations_dtype}. Must be half precision for CUDNN or Tri Dao "
                        "flash attention to work, using pure JAX flash attention instead."
                    )
                    attn_method = AttnMethod.FLASH_JAX
            else:
                print(
                    "Warning: falling back to pure JAX flash attention because no >= Ampere "
                    "architecture GPU was detected."
                )
                attn_method = AttnMethod.FLASH_JAX
        else:
            attn_method = self.attn_method

        # it'd potentially be better to use nn.remat_scan here, but it makes inference massively
        # slower for some reason. Even though checkpointing should only affect gradient computation.
        # Might have to do with the fact that remat_scan creates a scan-of-scans? Could cause bad
        # optimization in JAX or XLA.
        self.transformer_layers = nn.scan(
            nn.remat(TransformerLayer),
            variable_axes={"params": 0, "cache": 0}
            | ({"intermediates": 0} if self.record_attention_weights else {}),
            variable_broadcast=False,
            split_rngs={"params": True, "dropout": True},
            length=self.n_layers,
        )(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            use_biases=self.use_biases,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            activation_function=self.activation_function,
            pre_norm=self.pre_norm,
            kernel_init=default_kernel_init,
            out_proj_kernel_init=default_kernel_init,
            decode=self.decode,
            attn_method=attn_method,
            record_attention_weights=self.record_attention_weights,
        )

        if self.pre_norm:
            self.final_layer_norm = nn.LayerNorm(
                # LLaMA uses float32 for layer norms and bf16 for the rest of the weights.
                # Presumably they know what they're doing.
                dtype=self.activations_dtype,
                param_dtype=jnp.float32,
            )

        self.logits_decoder = nn.Dense(
            features=8192,
            kernel_init=default_kernel_init,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )

        tokens_res = int(self.image_tokens**0.5)
        assert tokens_res * tokens_res == self.image_tokens

    def seq_len(self) -> int:
        """How many tokens are in the sequence being modeled."""
        return self.image_tokens + self.prepended_tokens() - 1

    def prepended_tokens(self) -> int:
        """How many tokens are prepended to the image tokens."""
        # We always prepend at least one, becuase the first image token must be conditioned on
        # *something* even when that's a constant.
        if self.clip_conditioning:
            if self.clip_caps:
                return self.clip_cap_count
            else:
                return 1
        else:
            return 1

    def gen_conditioning_tokens(
        self,
        clip_embeddings: jax.Array,
        max_cos_distances: jax.Array,
    ) -> jax.Array:
        """Generate the conditioning tokens that should be prepended to the image tokens. Returns a
        (batch_size, self.prepended_tokens(), self.d_model) shaped array."""
        batch_size = clip_embeddings.shape[0]
        if not self.clip_conditioning:
            assert clip_embeddings.shape == max_cos_distances.shape == (batch_size, 0)
            res = jnp.zeros((batch_size, 1, self.d_model), dtype=self.activations_dtype)
        else:
            if self.norm_clip_embeddings:
                clip_embeddings = self.clip_embeddings_norm(clip_embeddings)
            if not self.clip_caps:
                assert clip_embeddings.shape == (batch_size, 768)
                assert max_cos_distances.shape == (batch_size, 0)
                res = self.clip_proj(clip_embeddings)[:, None, :]
            else:
                assert clip_embeddings.shape == (batch_size, self.clip_cap_count, 768)
                assert max_cos_distances.shape == (batch_size, self.clip_cap_count)

                # Without this rearrange when we apply the dense layer it interprets the max cos
                # distances as a single vector of length n rather than n vectors of length 1, and
                # produces one embedding for the whole sequence of distances rather than one for
                # each cap. Shapes, man.
                max_cos_distances = rearrange(max_cos_distances, "b caps -> b caps 1")

                res_cap_centers = self.clip_proj(clip_embeddings)

                if self.corrected_cap_projections:
                    # We assume the cosine distances are drawn from U[0, 2]. This isn't true in the
                    # training data (I bias strongly towards max distances <= 1 because they're much
                    # more informative about the relationship between the CLIP embeddings and the
                    # images), but hopefully this will work well enough without adding additional
                    # complexity.
                    # The variance of U[0, 2] is 2^2/12 = 1/3 so if we want it to have unit variance
                    # and mean 0 we do this:
                    res_max_cos_distances = self.max_cos_distance_proj(
                        jnp.sqrt(3) * (max_cos_distances - 1)
                    )
                else:
                    # old behavior, kept around so we can still do inference with old checkpoints
                    res_max_cos_distances = self.max_cos_distance_proj(
                        1 - max_cos_distances
                    )
                assert res_cap_centers.shape == (
                    batch_size,
                    self.clip_cap_count,
                    self.d_model,
                )
                assert res_max_cos_distances.shape == (
                    batch_size,
                    self.clip_cap_count,
                    self.d_model,
                )
                if self.corrected_cap_projections:
                    res = res_max_cos_distances + res_cap_centers
                else:
                    res = (res_cap_centers + res_max_cos_distances) / 2
        assert res.shape == (batch_size, self.prepended_tokens(), self.d_model)
        if self.do_clip_feedforward:
            ff_out = self.activation_function(res)
            ff_out = self.clip_ff_up(ff_out)
            ff_out = self.activation_function(ff_out)
            ff_out = self.clip_ff_down(ff_out)
            res = self.cond_tokens_layernorm(res + ff_out)
            assert res.shape == (batch_size, self.prepended_tokens(), self.d_model)
        return res

    def output_shape_tokens(self) -> int:
        """What (2-D) shape of tokens is output by the model."""
        res = int(self.image_tokens**0.5)
        return (res, res)

    def __call__(
        self,
        images: jax.Array,
        clip_embeddings: jax.Array,
        max_cos_distances: jax.Array,
    ) -> jax.Array:
        """Run the model, returning log probabilities of the image tokens. No probabilities are computed
        for any CLIP conditioning tokens."""
        assert_msg = f"Expected images array with shape (N, {self.image_tokens}), got {images.shape}"
        assert len(images.shape) == 2, assert_msg
        assert images.shape[1] == self.image_tokens, assert_msg
        assert images.dtype == jnp.int32 or images.dtype == jnp.int64

        batch_size = images.shape[0]

        if self.clip_conditioning:
            if self.clip_caps:
                assert clip_embeddings.shape == (batch_size, self.clip_cap_count, 768)
                assert max_cos_distances.shape == (batch_size, self.clip_cap_count)
            else:
                assert clip_embeddings.shape == (batch_size, 768)
                assert max_cos_distances.shape == (batch_size, 0)
        else:
            assert clip_embeddings.shape == max_cos_distances.shape == (batch_size, 0)

        # Drop the last token from the images, it's not used to predict anything.
        images = images[:, :-1]

        # Separate dropout rates for the conditioning tokens and the image tokens. The noise from
        # dropout is useful for kicking us off saddle points, not just for regularization. Also,
        # hopefully dropout on the conditioning tokens will prevent exploding gradients from
        # allocating ~all attention weight to them.
        cond_embeds = self.gen_conditioning_tokens(clip_embeddings, max_cos_distances)
        assert cond_embeds.shape == (batch_size, self.prepended_tokens(), self.d_model)
        cond_pos_embeds = self.positional_encoding(jnp.arange(self.prepended_tokens()))
        assert cond_pos_embeds.shape == (self.prepended_tokens(), self.d_model)
        cond_embeds = self.clip_dropout_layer(cond_embeds + cond_pos_embeds)

        img_embeds = self.in_embed(images)
        assert img_embeds.shape == (batch_size, self.image_tokens - 1, self.d_model)
        img_pos_embeds = self.positional_encoding(
            jnp.arange(self.image_tokens - 1) + self.prepended_tokens()
        )
        assert img_pos_embeds.shape == (self.image_tokens - 1, self.d_model)
        img_embeds = img_embeds + img_pos_embeds
        img_embeds = self.image_dropout_layer(img_embeds)
        assert img_embeds.shape == (batch_size, self.image_tokens - 1, self.d_model)

        h = jnp.concatenate([cond_embeds, img_embeds], axis=1)
        assert h.shape == (batch_size, self.seq_len(), self.d_model)

        h, _ = self.transformer_layers(h, None)
        assert h.shape == (batch_size, self.seq_len(), self.d_model)
        if self.pre_norm:
            h = self.final_layer_norm(h)
        h = h[:, self.prepended_tokens() - 1 :]
        logits = self.logits_decoder(h)
        assert logits.shape == (batch_size, self.image_tokens, 8192)
        assert logits.dtype == self.activations_dtype

        return logits

    def decode_init(
        self,
        clip_embeddings: jax.Array,
        max_cos_distances: jax.Array,
    ):
        """Initialize the cache for decoding by computing and feeding the conditioning tokens. Returns
        the logits for the first image token. The cache should be ready for use with decode_step
        when this is done."""
        assert self.decode
        # TODO test CPP flash attention, maybe it works.
        assert (
            self.attn_method == AttnMethod.STANDARD
        ), "Only standard attention works with decoding."

        batch_size = clip_embeddings.shape[0]

        if self.clip_conditioning:
            if self.clip_caps:
                assert clip_embeddings.shape == (batch_size, self.clip_cap_count, 768)
                assert max_cos_distances.shape == (batch_size, self.clip_cap_count)
            else:
                assert clip_embeddings.shape == (batch_size, 768)
                assert max_cos_distances.shape == (batch_size, 0)
        else:
            assert (
                clip_embeddings.shape == max_cos_distances.shape == (batch_size, 0)
            ), f"Expected empty shapes, got {clip_embeddings.shape} and {max_cos_distances.shape}"

        cond_tokens = self.gen_conditioning_tokens(clip_embeddings, max_cos_distances)
        assert cond_tokens.shape == (batch_size, self.prepended_tokens(), self.d_model)

        h = cond_tokens + self.positional_encoding(jnp.arange(self.prepended_tokens()))
        assert h.shape == (batch_size, self.prepended_tokens(), self.d_model)

        for i in range(self.prepended_tokens()):
            # Feed the prepended tokens one by one to initialize kv cache
            tf_out, _ = self.transformer_layers(h[:, i : i + 1, :], None)
            assert tf_out.shape == (batch_size, 1, self.d_model)

        last_toks = tf_out[:, 0, :]
        assert last_toks.shape == (batch_size, self.d_model)

        if self.pre_norm:
            last_toks = self.final_layer_norm(last_toks)
        logits_out = self.logits_decoder(last_toks)
        assert logits_out.shape == (batch_size, 8192)
        return logits_out

    def decode_step(self, toks: jax.Array, idx: jax.Array) -> jax.Array:
        """Do a step of iterative decoding from the model. Returns the logits for the next set of
        tokens. See below tests for usage examples.
        """
        assert (
            self.decode
        ), "Can't call decode_step on a model that wasn't set up for decoding."
        assert (
            self.attn_method == AttnMethod.STANDARD
        ), "Only standard attention works with decoding."
        assert len(toks.shape) == 1
        batch_size = toks.shape[0]
        assert toks.dtype == jnp.int32 or toks.dtype == jnp.int64
        assert idx.shape == ()

        embed = self.in_embed(toks)
        assert embed.shape == (batch_size, self.d_model)

        h = embed + self.positional_encoding(idx + self.prepended_tokens())
        h = self.image_dropout_layer(h)
        assert h.shape == (batch_size, self.d_model)

        h, _ = self.transformer_layers(h[:, None, :], None)
        assert h.shape == (batch_size, 1, self.d_model)
        if self.pre_norm:
            h = self.final_layer_norm(h)
        return self.logits_decoder(h[:, 0, :])  # type: ignore[no-any-return]

    def dummy_inputs(self):
        images_dummy = jnp.zeros((1, self.image_tokens), dtype=jnp.int32)
        if self.clip_conditioning and self.clip_caps:
            max_cos_distance_dummy = jnp.zeros(
                (1, self.clip_cap_count), dtype=jnp.float32
            )
            clip_embeddings_dummy = jnp.zeros(
                (1, self.clip_cap_count, 768), dtype=jnp.float32
            )
        elif self.clip_conditioning and not self.clip_caps:
            clip_embeddings_dummy = jnp.zeros((1, 768), dtype=jnp.float32)
            max_cos_distance_dummy = jnp.zeros((1, 0), dtype=jnp.float32)
        else:
            clip_embeddings_dummy = jnp.zeros((1, 0), dtype=jnp.float32)
            max_cos_distance_dummy = jnp.zeros((1, 0), dtype=jnp.float32)
        return images_dummy, clip_embeddings_dummy, max_cos_distance_dummy

    def gen_training_caps(self, tbl, rng, clip_embedding):
        """Generate random caps to use for training with a single example. 'self' MUST be a static
        argument when JITting this."""
        assert self.clip_caps
        assert clip_embedding.shape == (768,)

        n_caps_rng, caps_rng, fill_rng = jax.random.split(rng, 3)

        # We draw the number of caps to use from a power law distribution, constrained such that
        # P(X = 1) = 0.1. The PMF is strictly monotonically increasing so long as clip_cap_count
        # < 10, constant if it's 10, and decreasing if it's larger. The principle here is that we
        # want a decent fraction of examples to have a single cap, since that's the majority
        # situation at inference time. But we also want most of them to have multiple caps, since
        # it's useful to be able to condition on multiple concepts at once. And we want as many
        # examples as possible to have lots, because an example with lots of caps is much more
        # informative to the model as to the CLIP embedding of the image - we should learn more per
        # example with more caps.

        # I don't know that this distribution is ideal. As of 2024-11-08 I haven't done very many
        # multi-cap experiments.

        # Find an alpha value for a power law distribution that gives P(X = 1) = 0.1. This search
        # runs once, at tracing time, and the result is reused when the JIT version runs on the GPU.
        if self.clip_cap_count > 1:
            lower = 0.0
            upper = 10.0
            for _ in range(20):
                alpha = (lower + upper) / 2
                probs = calculate_discrete_power_law_pmf(self.clip_cap_count, alpha)
                if probs[0] < 0.1:
                    upper = alpha
                else:
                    lower = alpha
            np.testing.assert_allclose(probs[0], 0.1, atol=3e-6, rtol=0)
        else:
            probs = np.array([1.0])

        # Draw a number of caps to use from the power law distribution.
        n_caps = jax.random.choice(
            n_caps_rng, jnp.arange(1, self.clip_cap_count + 1), p=probs
        )

        # Draw a complete set of caps. Because of JAX tracing constraints we can't only do n_caps
        # caps, we have to generate a static number of caps and select afterward.
        cap_centers, cap_d_maxes = jax.vmap(
            lambda rng: sample_cap(tbl, rng, clip_embedding, bias_d_max=True)
        )(jax.random.split(caps_rng, self.clip_cap_count))
        assert cap_centers.shape == (self.clip_cap_count, 768)
        assert cap_d_maxes.shape == (self.clip_cap_count,)

        # Fill in the rest of the caps with ones that cover the entire sphere and have uniformly
        # distributed centers
        fill_centers = jax.random.normal(fill_rng, (self.clip_cap_count, 768))
        fill_centers = fill_centers / jnp.linalg.norm(
            fill_centers, axis=-1, keepdims=True
        )
        fill_d_maxes = jnp.full((self.clip_cap_count,), 2.0)

        cap_mask = jnp.arange(self.clip_cap_count) < n_caps
        cap_centers = jnp.where(cap_mask[:, None], cap_centers, fill_centers)
        cap_d_maxes = jnp.where(cap_mask, cap_d_maxes, fill_d_maxes)
        assert cap_centers.shape == (self.clip_cap_count, 768)
        assert cap_d_maxes.shape == (self.clip_cap_count,)

        # Order smallest to largest. The smallest cap is most informative, so we want the later caps
        # to be able to attend to it when the model builds a representation of the intersection.
        sorted_idxs = jnp.argsort(cap_d_maxes)
        cap_centers = cap_centers[sorted_idxs]
        cap_d_maxes = cap_d_maxes[sorted_idxs]

        return cap_centers, cap_d_maxes


def calculate_discrete_power_law_pmf(n_max, alpha):
    """Calculate the probability mass function (as an array) for a discrete power law distribution
    over the integers 1 through n_max inclusive, with exponent alpha."""
    probs = np.arange(1, n_max + 1) ** alpha
    return probs / np.sum(probs)


@pytest.mark.parametrize("n_caps", [1, 2, 3, 4, 10])
@pytest.mark.parametrize("alpha", [0.5, 1.0, 4.0])
def test_calculate_discrete_power_law_pmf_increasing(n_caps: int, alpha: float):
    probs = calculate_discrete_power_law_pmf(n_caps, alpha)
    assert jnp.all(
        jnp.diff(probs) >= 0
    ), "Probabilities must be monotonically increasing"


@pytest.mark.parametrize("n_caps", [1, 2, 3, 4, 10])
def test_gen_training_caps(n_caps: int):
    tbl = LogitsTable(767, 8192)
    n_samples = 1000
    rng = jax.random.PRNGKey(0)
    config = copy(gpt_1_config)
    config.clip_caps = True
    config.clip_cap_count = n_caps
    mdl = ImageModel(**config.__dict__)

    def gen_caps_and_example_embedding(rng, mdl):
        ex_rng, cap_rng = jax.random.split(rng, 2)
        embedding = jax.random.normal(ex_rng, (768,))
        embedding = embedding / jnp.linalg.norm(embedding)
        return embedding, mdl.gen_training_caps(tbl, cap_rng, embedding)

    gen_caps_and_example_embedding_jv = jax.jit(
        lambda rng, mdl: jax.vmap(
            lambda rng_inner: gen_caps_and_example_embedding(rng_inner, mdl)
        )(jax.random.split(rng, n_samples)),
        static_argnames=["mdl"],
    )

    # Generate a bunch of caps and example embeddings
    embeddings, (centers, d_maxes) = gen_caps_and_example_embedding_jv(rng, mdl)
    assert embeddings.shape == (n_samples, 768)
    assert centers.shape == (n_samples, n_caps, 768)
    np.testing.assert_allclose(
        jnp.linalg.norm(embeddings, axis=-1), 1.0, atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        jnp.linalg.norm(centers, axis=-1), 1.0, atol=1e-6, rtol=0
    )
    assert d_maxes.shape == (n_samples, n_caps)

    # Check the example embeddings are inside the caps
    sims = jnp.sum(embeddings[:, None, :] * centers, axis=-1)
    dists = 1 - sims
    assert dists.shape == (n_samples, n_caps)
    embeddings_in_caps = dists <= d_maxes
    assert embeddings_in_caps.shape == (n_samples, n_caps)
    assert embeddings_in_caps.dtype == jnp.bool_
    assert jnp.mean(embeddings_in_caps) > 0.9999

    # Check that 10% of the time 1 cap is chosen
    n_caps_chosen = reduce(d_maxes != 2.0, "ex cap -> ex", "sum")
    assert n_caps_chosen.shape == (n_samples,)
    if n_caps > 1:
        np.testing.assert_allclose(jnp.mean(n_caps_chosen == 1), 0.1, atol=0.01, rtol=0)
    else:
        assert jnp.all(n_caps_chosen == 1)

    # Check that cap sizes are monotonically increasing
    diffs = jnp.diff(d_maxes, axis=1)
    assert diffs.shape == (n_samples, n_caps - 1)
    assert jnp.all(diffs >= 0)


@pytest.mark.parametrize(
    "config_modifications",
    [
        {},
        {"clip_conditioning": True},
        {"clip_conditioning": True, "clip_caps": True, "clip_cap_count": 4},
        {
            "clip_conditioning": True,
            "clip_caps": True,
            "clip_cap_count": 1,
            "do_clip_feedforward": True,
        },
    ],
)
def test_model_initialization_with_various_configs(config_modifications):
    """Test that the model can be initialized with different configurations."""
    config = replace(gpt_1_config, **config_modifications)
    model = ImageModel(**config.__dict__)

    rng = jax.random.PRNGKey(0)
    model.init(rng, *model.dummy_inputs())


def _assert_dicts_equal(d1, d2, name) -> None:
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert d1.keys() == d2.keys()
    for k in d1.keys():
        if isinstance(d1[k], dict):
            _assert_dicts_equal(d1[k], d2[k], f"{name}.{k}")
        elif isinstance(d1[k], jax.Array):
            np.testing.assert_allclose(
                np.array(d1[k]), np.array(d2[k]), atol=1e-8, rtol=0
            )
        else:
            assert False, f"unknown type {type(d1[k])} for {name}.{k}"


@pytest.mark.parametrize(
    "do_clip_feedforward",
    [
        pytest.param(True, id="with_clip_feedforward"),
        pytest.param(False, id="no_clip_feedforward"),
    ],
)
@pytest.mark.parametrize(
    "norm_clip_embeddings",
    [
        pytest.param(True, id="with_norm_embeddings"),
        pytest.param(False, id="no_norm_embeddings"),
    ],
)
def test_cap_cond_tokens_and_vqgan_embeds_are_same_distribution(
    do_clip_feedforward, norm_clip_embeddings
):
    """Test that, at initialization, the embeddings for the caps generated with
    gen_conditioning_tokens and the embeddings for the VQGAN tokens have the same mean, same mean
    magnitude, same standard deviation, and same mean standard deviation. If this is true the model
    should learn more gooder."""

    cfg = copy(gpt_1_config)
    cfg.dropout = None
    cfg.clip_conditioning = True
    cfg.clip_caps = True
    cfg.clip_cap_count = 1
    cfg.do_clip_feedforward = do_clip_feedforward
    cfg.norm_clip_embeddings = norm_clip_embeddings
    mdl = ImageModel(**cfg.__dict__)

    n_samples = 1_000
    n_models = 10

    # Generate dummy parameters and test inputs outside the loop
    dummy_img = jnp.zeros((1, 256), dtype=jnp.int32)
    dummy_clip_embeddings = jnp.zeros((1, 1, 768), dtype=jnp.float32)
    dummy_max_cos_distances = jnp.zeros((1, 1), dtype=jnp.float32)

    clips_rng, dists_rng, embeds_rng = jax.random.split(jax.random.PRNGKey(0), 3)

    # Generate random CLIP embeddings and max cosine distances
    clip_embeddings = jax.random.normal(clips_rng, (n_samples, 1, 768))
    clip_embeddings = clip_embeddings / jnp.linalg.norm(
        clip_embeddings, axis=-1, keepdims=True
    )
    max_cos_distances = jax.random.uniform(
        dists_rng, (n_samples, 1), minval=0, maxval=2
    )

    # Generate random VQGAN tokens
    vqgan_tokens = jax.random.randint(embeds_rng, (n_samples, 256), 0, 8192)

    # Initialize lists to store statistics for each model
    cap_cond_means = []
    vqgan_means = []
    cap_cond_mean_magnitudes = []
    vqgan_mean_magnitudes = []
    cap_cond_stds = []
    vqgan_stds = []
    cap_cond_mean_stds = []
    vqgan_mean_stds = []

    gen_params = jax.jit(
        lambda rng: mdl.init(
            rng, dummy_img, dummy_clip_embeddings, dummy_max_cos_distances
        )
    )

    for rng in jax.random.split(jax.random.PRNGKey(1), n_models):
        # Initialize model

        params = gen_params(rng)
        mdl_bound = mdl.bind(params)

        # Get cap conditioning tokens
        cap_cond_tokens = mdl_bound.gen_conditioning_tokens(
            clip_embeddings=clip_embeddings, max_cos_distances=max_cos_distances
        )[:, 0, :]
        assert cap_cond_tokens.shape == (n_samples, cfg.d_model)

        # Get VQGAN token embeddings
        vqgan_embeds = mdl_bound.in_embed(vqgan_tokens)
        assert vqgan_embeds.shape == (n_samples, 256, cfg.d_model)

        # Calculate and store statistics
        cap_cond_means.append(np.mean(cap_cond_tokens))
        vqgan_means.append(np.mean(vqgan_embeds))
        cap_cond_mean_magnitudes.append(
            np.mean(np.linalg.norm(cap_cond_tokens, axis=-1))
        )
        vqgan_mean_magnitudes.append(np.mean(np.linalg.norm(vqgan_embeds, axis=-1)))
        cap_cond_stds.append(np.std(cap_cond_tokens))
        vqgan_stds.append(np.std(vqgan_embeds))
        cap_cond_mean_stds.append(np.mean(np.std(cap_cond_tokens, axis=1)))
        vqgan_mean_stds.append(np.mean(np.std(vqgan_embeds, axis=2)))

    # Calculate average statistics across all models
    avg_cap_cond_mean = np.mean(cap_cond_means)
    avg_vqgan_mean = np.mean(vqgan_means)
    avg_cap_cond_mean_magnitude = np.mean(cap_cond_mean_magnitudes)
    avg_vqgan_mean_magnitude = np.mean(vqgan_mean_magnitudes)
    avg_cap_cond_std = np.mean(cap_cond_stds)
    avg_vqgan_std = np.mean(vqgan_stds)
    avg_cap_cond_mean_std = np.mean(cap_cond_mean_stds)
    avg_vqgan_mean_std = np.mean(vqgan_mean_stds)

    print(
        f"Average cap_cond_mean: {avg_cap_cond_mean}, Average vqgan_mean: {avg_vqgan_mean}"
    )
    print(
        f"Average cap_cond_mean_magnitude: {avg_cap_cond_mean_magnitude}, Average vqgan_mean_magnitude: {avg_vqgan_mean_magnitude}"
    )
    print(
        f"Average cap_cond_std: {avg_cap_cond_std}, Average vqgan_std: {avg_vqgan_std}"
    )
    print(
        f"Average cap_cond_mean_std: {avg_cap_cond_mean_std}, Average vqgan_mean_std: {avg_vqgan_mean_std}"
    )

    np.testing.assert_allclose(avg_cap_cond_mean, avg_vqgan_mean, atol=1e-3, rtol=0)
    np.testing.assert_allclose(avg_cap_cond_mean, 0, atol=1e-3, rtol=0)
    np.testing.assert_allclose(
        avg_cap_cond_mean_magnitude, avg_vqgan_mean_magnitude, atol=0, rtol=0.03
    )
    np.testing.assert_allclose(avg_cap_cond_std, avg_vqgan_std, atol=0.01, rtol=0)
    np.testing.assert_allclose(avg_cap_cond_std, 1, atol=0.01, rtol=0)
    np.testing.assert_allclose(
        avg_cap_cond_mean_std, avg_vqgan_mean_std, atol=0.03, rtol=0
    )
    np.testing.assert_allclose(avg_cap_cond_mean_std, 1, atol=0.03, rtol=0)


def _setup_test_sample(
    clip_conditioning: bool = False,
    clip_caps: bool = False,
    clip_cap_count: Optional[int] = None,
    pre_norm: bool = False,
    image_tokens: int = 256,
) -> Tuple[ImageModel, ImageModel, dict, jax.Array, jax.Array]:
    """Shared setup code for iterative sampling tests."""
    cfg_nodec = copy(gpt_1_config)
    cfg_nodec.dropout = None
    cfg_nodec.image_tokens = image_tokens
    # smaller model makes debug output easier to read
    cfg_nodec.n_layers = 2
    cfg_nodec.d_model = 64
    cfg_nodec.num_heads = 4
    if clip_conditioning:
        cfg_nodec.clip_conditioning = True
    if clip_caps:
        cfg_nodec.clip_caps = True
        if clip_cap_count is None:
            clip_cap_count = 2
        cfg_nodec.clip_cap_count = clip_cap_count
    cfg_nodec.pre_norm = pre_norm
    mdl_nodec = ImageModel(**cfg_nodec.__dict__)
    mdl_dec = mdl_nodec.clone(decode=True, attn_method=AttnMethod.STANDARD)

    img_toks = jax.random.randint(jax.random.PRNGKey(420), (image_tokens,), 0, 8192)
    if clip_conditioning:
        if clip_caps:
            clip_embedding = jax.random.normal(
                jax.random.PRNGKey(1337), (clip_cap_count, 768)
            )
            clip_embedding = clip_embedding / jnp.linalg.norm(
                clip_embedding, axis=-1, keepdims=True
            )
            max_cos_distance = jnp.full(clip_cap_count, 0.5)
        else:
            clip_embedding = jax.random.normal(jax.random.PRNGKey(1337), (768,))
            clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)
            max_cos_distance = jnp.array([])
    else:
        clip_embedding = max_cos_distance = jnp.array([])

    params = jax.jit(mdl_nodec.init)(
        jax.random.PRNGKey(69),
        images=img_toks[None, :],
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )
    # IMPORTANT: use regular __call__ here, not decode_step. The cache needs to be initialized to
    # the full seq_len size.
    params_dec = jax.jit(mdl_dec.init)(
        jax.random.PRNGKey(69),
        images=img_toks[None, :],
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )

    _assert_dicts_equal(params["params"], params_dec["params"], "params")

    logits_all = mdl_nodec.apply(
        params,
        images=img_toks[None, :],
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )[0]

    return (
        mdl_nodec,
        mdl_dec,
        params,
        params_dec["cache"],
        img_toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    )


@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize(
    "clip_conditioning,clip_caps,clip_cap_count",
    [
        (False, False, None),
        (True, False, None),
        (True, True, 1),
        (True, True, 2),
    ],
)
def test_sample_tok_0(
    clip_conditioning: bool,
    clip_caps: bool,
    clip_cap_count: Optional[int],
    pre_norm: bool,
) -> None:
    """Test that step-by-step decoding is equivalent to all at once for image token 0."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_caps, clip_cap_count, pre_norm)

    params = flax.core.copy(params, {"cache": cache})
    logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )
    assert logits_0.shape == (1, 8192)

    np.testing.assert_allclose(logits_all[0], logits_0[0], rtol=0, atol=3e-3)


@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize(
    "clip_conditioning,clip_caps",
    [
        (False, False),  # No CLIP
        (True, False),  # CLIP without caps
        (True, True),  # CLIP with caps
    ],
)
def test_sample_tok_1(clip_conditioning: bool, clip_caps: bool, pre_norm: bool) -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 1."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_caps, 1, pre_norm)

    params = flax.core.copy(params, {"cache": cache})

    _logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )
    params = flax.core.copy(params, cache)
    logits_1, _cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        toks=toks[None, 0],
        idx=jnp.array(0),
    )

    np.testing.assert_allclose(logits_all[1], logits_1[0], rtol=0, atol=1e-3)


@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize(
    "clip_conditioning,clip_caps,image_tokens",
    [
        (False, False, 256),  # No CLIP
        (True, False, 256),  # CLIP without caps
        (True, True, 256),  # CLIP with caps
        (True, True, 1024),  # Test longer sequence length
    ],
)
def test_sample_tok_all(
    clip_conditioning: bool, clip_caps: bool, image_tokens: int, pre_norm: bool
) -> None:
    """Test that step-by-step decoding is equivalent to all at once for all tokens."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_caps, None, pre_norm, image_tokens)

    decoded_logits = []
    params = flax.core.copy(params, {"cache": cache})

    # compute logits for image tok 0
    logits, new_cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=max_cos_distance[None, :],
    )
    logits = logits[0]
    assert logits.shape == (8192,)
    decoded_logits.append(logits)
    params = flax.core.copy(params, new_cache)

    step_j = jax.jit(
        lambda params, i: mdl_dec.apply(
            params,
            mutable=["cache"],
            method=mdl_dec.decode_step,
            toks=toks[None, i],
            idx=jnp.array(i),
        )
    )

    # compute logits for image toks 1-255 (inputting toks 0-254)
    for i in range(image_tokens - 1):
        logits, new_cache = step_j(params, i)
        logits = logits[0]
        assert logits.shape == (8192,)
        decoded_logits.append(logits)
        params = flax.core.copy(params, new_cache)

    decoded_logits = jnp.stack(decoded_logits, axis=0)
    assert decoded_logits.shape == (image_tokens, 8192)
    np.testing.assert_allclose(logits_all, decoded_logits, rtol=0, atol=0.003)


def test_batched_decode_consistency() -> None:
    """Test that decode_init and decode_step produce consistent results for different batch sizes."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache_1,
        _toks,
        _clip_embedding,
        _max_cos_distance,
        _logits_all,
    ) = _setup_test_sample(True, True)

    imgs_to_test = 3
    clips_rng, max_cos_rng, toks_rng = jax.random.split(jax.random.PRNGKey(1425), 3)
    clip_embeddings = jax.random.normal(
        clips_rng, (imgs_to_test, mdl_nodec.clip_cap_count, 768)
    )
    clip_embeddings = clip_embeddings / jnp.linalg.norm(
        clip_embeddings, axis=-1, keepdims=True
    )
    max_cos_distances = jax.random.uniform(
        max_cos_rng,
        shape=(imgs_to_test, mdl_nodec.clip_cap_count),
        minval=0.0,
        maxval=2.0,
    )
    toks = jax.random.randint(toks_rng, (imgs_to_test, mdl_nodec.image_tokens), 0, 8192)
    logits_1 = np.zeros((imgs_to_test, mdl_nodec.image_tokens, 8192))
    logits_3 = np.zeros((imgs_to_test, mdl_nodec.image_tokens, 8192))

    decode_init_j = jax.jit(
        lambda params, clip_embeddings, max_cos_distances: mdl_dec.apply(
            params,
            mutable=["cache"],
            method=mdl_dec.decode_init,
            clip_embeddings=clip_embeddings,
            max_cos_distances=max_cos_distances,
        )
    )
    decode_step_j = jax.jit(
        lambda params, toks, idx: mdl_dec.apply(
            params,
            mutable=["cache"],
            method=mdl_dec.decode_step,
            toks=toks,
            idx=idx,
        )
    )

    # Test batch size 1
    for i in trange(imgs_to_test, desc="Testing batch size 1"):
        params_1 = flax.core.copy(params, {"cache": cache_1})
        logits_1[i, 0], new_cache = decode_init_j(
            params_1,
            clip_embeddings=clip_embeddings[i : i + 1],
            max_cos_distances=max_cos_distances[i : i + 1],
        )
        params_1 = flax.core.copy(params_1, new_cache)
        for j in trange(
            mdl_nodec.image_tokens - 1,
            desc=f"Processing tokens for image {i+1}",
            leave=False,
        ):
            logits_1[i, j + 1], new_cache = decode_step_j(
                params_1,
                toks=toks[i : i + 1, j],
                idx=jnp.array(j),
            )
            params_1 = flax.core.copy(params_1, new_cache)

    # Test batch size 3
    params_3 = mdl_dec.init(
        jax.random.PRNGKey(0),
        images=toks,
        clip_embeddings=clip_embeddings,
        max_cos_distances=max_cos_distances,
    )
    params_3 = flax.core.copy(params, {"cache": params_3["cache"]})
    logits_3[:, 0, :], new_cache = decode_init_j(
        params_3,
        clip_embeddings=clip_embeddings,
        max_cos_distances=max_cos_distances,
    )
    params_3 = flax.core.copy(params_3, new_cache)
    for i in trange(mdl_nodec.image_tokens - 1, desc="Testing batch size 3"):
        logits_3[:, i + 1, :], new_cache = decode_step_j(
            params_3,
            toks=toks[:, i],
            idx=jnp.array(i),
        )
        params_3 = flax.core.copy(params_3, new_cache)

    np.testing.assert_allclose(logits_1, logits_3, rtol=0, atol=2e-3)


def test_clip_does_anything() -> None:
    """Test that changing the CLIP embedding changes the logits."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    ) = _setup_test_sample(True, False)

    clip_embedding = jnp.zeros_like(clip_embedding)
    logits_all_zero = mdl_nodec.apply(
        params,
        images=toks[None, :],
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=jnp.array([[]]),
    )

    assert not jnp.allclose(logits_all, logits_all_zero, rtol=0, atol=1e-3)


def test_clip_caps_do_anything() -> None:
    """Test that changing the CLIP cap size changes the logits."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        max_cos_distance,
        logits_all,
    ) = _setup_test_sample(True, True)

    logits_full_range = mdl_nodec.apply(
        params,
        images=toks[None, :],
        clip_embeddings=clip_embedding[None, :],
        max_cos_distances=jnp.array([[2.0, 0.85]]),
    )

    assert not jnp.allclose(logits_all, logits_full_range, rtol=0, atol=1e-3)


LogitFilterMethod = Enum("LogitFilterMethod", ["TOP_P", "MIN_P"])


# _init_decode and _step_decode are morally part of sample() but they need to be defined at the top
# level so the results of jitting them get cached.
@partial(jax.jit, static_argnames=("mdl", "filter_method"))
def _init_decode(
    mdl: ImageModel,
    params,
    clip_embeddings: jax.Array,
    max_cos_distances: jax.Array,
    rngs: jax.Array,
    filter_method: LogitFilterMethod,
    filter_threshold: float,
    temperature: float,
):
    """Do the first step of decoding, choosing the 0th token for every image. This is part of
    sample() below."""
    batch_size = clip_embeddings.shape[0]
    if mdl.clip_conditioning and mdl.clip_caps:
        assert clip_embeddings.shape == (batch_size, mdl.clip_cap_count, 768)
        assert max_cos_distances.shape == (batch_size, mdl.clip_cap_count)
    elif mdl.clip_conditioning and not mdl.clip_caps:
        assert clip_embeddings.shape == (batch_size, 768)
        assert max_cos_distances.shape == (batch_size, 0)
    else:
        assert clip_embeddings.shape == max_cos_distances.shape == (batch_size, 0)
    assert rngs.shape == (batch_size, 2)

    _, cache = mdl.apply(
        params,
        mutable=["cache"],
        images=jnp.zeros((batch_size, mdl.image_tokens), dtype=jnp.int32),
        clip_embeddings=clip_embeddings,
        max_cos_distances=max_cos_distances,
    )
    params = flax.core.copy(params, cache)

    logits_0, cache = mdl.apply(
        params,
        mutable=["cache"],
        method=mdl.decode_init,
        clip_embeddings=clip_embeddings,
        max_cos_distances=max_cos_distances,
    )

    filtered_logits_0 = jax.vmap(_filter_logits, in_axes=(0, None, None, None))(
        logits_0, filter_method, filter_threshold, temperature
    )
    rngs_split = jax.vmap(jax.random.split, in_axes=(0, None))(rngs, 2)
    rngs, rngs_sample = rngs_split[:, 0], rngs_split[:, 1]
    toks_0 = jax.vmap(jax.random.categorical, in_axes=(0, 0))(
        rngs_sample, filtered_logits_0
    )
    assert toks_0.shape == (batch_size,)

    return toks_0, cache, rngs


@partial(
    jax.jit,
    static_argnames=("mdl", "filter_method"),
    donate_argnames=("cache", "image_toks", "rngs"),
)
def _step_decode(
    mdl: ImageModel,
    params,
    cache,
    image_toks,
    idx,
    rngs,
    filter_method: LogitFilterMethod,
    filter_threshold: float,
    temperature: float,
):
    """Do a single step of decoding for all images. This is part of sample() below."""
    batch_size = image_toks.shape[0]

    params = flax.core.copy(params, cache)

    logits, new_cache = mdl.apply(
        params,
        mutable=["cache"],
        method=mdl.decode_step,
        toks=image_toks[:, idx],
        idx=idx,
    )
    assert logits.shape == (batch_size, 8192)
    filtered_logits = jax.vmap(_filter_logits, in_axes=(0, None, None, None))(
        logits, filter_method, filter_threshold, temperature
    )
    rngs_split = jax.vmap(jax.random.split, in_axes=(0, None))(rngs, 2)
    rngs, rngs_sample = rngs_split[:, 0], rngs_split[:, 1]
    toks = jax.vmap(jax.random.categorical, in_axes=(0, 0))(
        rngs_sample, filtered_logits
    )
    assert toks.shape == (batch_size,)
    image_toks = image_toks.at[:, idx + 1].set(toks)
    return new_cache, image_toks, rngs


def sample(
    mdl: ImageModel,
    params: dict[str, Any],
    clip_embeddings: jax.Array,
    max_cos_distances: jax.Array,
    rngs: jax.Array,
    filter_method: LogitFilterMethod = LogitFilterMethod.TOP_P,
    filter_threshold: float = 0.95,
    temperature: float = 1.0,
) -> jax.Array:
    """Sample a single image from the model. Returns an array of codes to be passed to the
    LDM decoder."""
    batch_size = clip_embeddings.shape[0]
    if mdl.clip_conditioning and mdl.clip_caps:
        assert clip_embeddings.shape == (batch_size, mdl.clip_cap_count, 768)
        assert max_cos_distances.shape == (batch_size, mdl.clip_cap_count)
    elif mdl.clip_conditioning and not mdl.clip_caps:
        assert clip_embeddings.shape == (batch_size, 768)
        assert max_cos_distances.shape == (batch_size, 0)
    else:
        assert clip_embeddings.shape == max_cos_distances.shape == (batch_size, 0)
    assert rngs.shape == (batch_size, 2)

    # Flash attention doesn't work with Flax's fast decoding. Something to do with how masks are
    # handled. Would be nice to fix it, but for now we just use the slower attention when sampling.
    mdl_decode = mdl.clone(
        decode=True,
        attn_method=AttnMethod.STANDARD,
        dropout=None,
        image_dropout=None,
        clip_dropout=None,
    )

    with tqdm(total=mdl.image_tokens * batch_size, unit="token", leave=False) as pbar:
        toks_0, cache, rngs = _init_decode(
            mdl_decode,
            params,
            clip_embeddings,
            max_cos_distances,
            rngs,
            filter_method,
            filter_threshold,
            temperature,
        )
        pbar.update(batch_size)
        image_toks = (
            jnp.zeros((batch_size, mdl.image_tokens), dtype=jnp.int32)
            .at[:, 0]
            .set(toks_0)
        )

        for i in range(mdl.image_tokens - 1):
            cache, image_toks, rngs = _step_decode(
                mdl_decode,
                params,
                cache,
                image_toks,
                i,
                rngs,
                filter_method,
                filter_threshold,
                temperature,
            )
            pbar.update(batch_size)

    return image_toks


def _filter_logits(
    logits: jax.Array, method: LogitFilterMethod, p: float, temperature: float = 1.0
) -> jax.Array:
    """Filter an array of logits using the specified method. Returns the filtered array."""
    if method == LogitFilterMethod.TOP_P:
        logits = _filter_top_p(logits, p)
    elif method == LogitFilterMethod.MIN_P:
        logits = _filter_min_p(logits, p)
    else:
        raise ValueError(f"Invalid logit filter method: {method}")
    return logits / temperature


def _filter_top_p(logits: jax.Array, top_p: float) -> jax.Array:
    """Filter an array of logits to include the smallest subset of possibilities that has
    proability mass at least p i.e. top p/nucleus sampling. Returns the filtered array.
    """
    assert jnp.issubdtype(logits.dtype, jnp.floating), "logits must be floating point"

    # Under certain circumstances, softmax can produce NaN values. I don't really understand it, but
    # it only happens with extreme logits and only with bf16, so it should be fine to do the softmax
    # in float32 and check for NaNs afteward just in case. If there are NaNs in the output, we just
    # return the input logits. Not much difference between picking a token 99.99999999% of the time
    # and 100% of the time.

    probs = jax.nn.softmax(logits.astype(jnp.float32))

    def do_filter(probs):
        sorted_indices = jnp.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = jnp.cumsum(sorted_probs)

        # collect the minimal set of possibilites with probability <= top_p
        mask = cumulative_probs <= top_p
        # Find the index of the first possibility that has cumulative probability >= top_p
        # this might be the last element we found above or might be the one after it.
        # we could do only the argmax and set the mask with a range but that's not JIT-able so we do
        # this.
        last_idx = jnp.argmax(cumulative_probs >= top_p)
        mask = mask.at[last_idx].set(True)

        # permute the mask back to the original order
        mask = mask[sorted_indices.argsort()]

        return jnp.where(mask, logits, -np.inf)

    return jax.lax.cond(jnp.any(jnp.isnan(probs)), lambda _: logits, do_filter, probs)


def test_filter_top_p_10() -> None:
    """Test that filter_top_p is the identity function when top_p = 1.0."""
    logits = jnp.arange(10, dtype=jnp.float32)
    filtered_logits = _filter_top_p(logits, 1.0)
    assert jnp.allclose(
        filtered_logits, logits
    ), "filter_top_p doesn't match the identity function when top_p = 1.0"


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0, -0.25, 500.0])
def test_filter_top_p_05(offset) -> None:
    """Test that filter_top_p removes low-probability elements when top_p = 0.5."""
    probabilities = jnp.array([0.35, 0.35, 0.1, 0.1, 0.1])
    assert jnp.isclose(jnp.sum(probabilities), 1.0)
    logits = jnp.log(probabilities) + offset
    filtered_logits = _filter_top_p(logits, 0.5)
    np.testing.assert_allclose(
        np.array(jax.nn.softmax(filtered_logits)), np.array([0.5, 0.5, 0, 0, 0])
    )


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0, -0.25, 500.0])
def test_filter_top_p_out_of_order(offset) -> None:
    """Test that filter_top_p removes low-probability elements when inputs do not start sorted."""
    probabilities = np.repeat(1000.0, 7)
    big_indices = np.array([3, 5])
    medium_indices = np.array([2, 4])
    small_indices = np.array([0, 1, 6])
    probabilities[big_indices] = 0.25
    probabilities[medium_indices] = 0.2
    probabilities[small_indices] = 0.1 / 3.0
    np.testing.assert_allclose(np.sum(probabilities), 1.0)

    logits = jnp.log(probabilities) + offset
    filtered_logits = _filter_top_p(logits, 0.75)
    filtered_probabilities = np.array(jax.nn.softmax(filtered_logits))

    np.testing.assert_allclose(
        filtered_probabilities[small_indices], 0.0, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        filtered_probabilities[medium_indices], 0.2 / 0.9, rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(
        filtered_probabilities[big_indices], 0.25 / 0.9, rtol=0, atol=1e-6
    )


def _filter_min_p(logits: jax.Array, min_p: float) -> jax.Array:
    """
    Filter an array of logits to include only those possibilities that have probability >= min_p *
    the probability of the most probable token. Returns the filtered array.
    """
    probs = jax.nn.softmax(logits)
    min_prob = min_p * jnp.max(probs)
    return jnp.where(probs >= min_prob, logits, -np.inf)


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0, -0.25, 500.0])
def test_filter_min_p_identity(offset):
    """Test that filter_min_p is the identity function when min_p = 0."""
    logits = jnp.array([1.0, 2.0, 3.0, 4.0]) + offset
    filtered_logits = _filter_min_p(logits, 0.0)
    np.testing.assert_allclose(filtered_logits, logits, rtol=0, atol=1e-5)


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0, -0.25, 500.0])
def test_filter_min_p_threshold(offset):
    """Test that filter_min_p correctly applies the threshold."""
    probs = jnp.array([0.1, 0.3, 0.5, 0.1])
    logits = jnp.log(probs) + offset
    filtered_logits = _filter_min_p(logits, 0.5)
    filtered_probs = jax.nn.softmax(filtered_logits)
    np.testing.assert_allclose(
        filtered_probs, np.array([0.0, 0.3 / 0.8, 0.5 / 0.8, 0.0]), rtol=0, atol=3e-6
    )


@pytest.mark.parametrize("offset", [0.0, 1.0, -1.0, -0.25, 500.0])
def test_filter_min_p_all_filtered(offset):
    """Test that filter_min_p filters out all tokens when min_p = 1."""
    probs = jnp.array([0.4, 0.1, 0.3, 0.2])
    logits = jnp.log(probs) + offset
    filtered_logits = _filter_min_p(logits, 1.0)
    filtered_probs = jax.nn.softmax(filtered_logits)
    np.testing.assert_allclose(filtered_probs, np.array([1.0, 0, 0, 0]))


class TransformerLayer(nn.Module):
    """A single transformer layer."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    use_biases: bool
    activations_dtype: jnp.dtype
    activation_function: Callable[[jax.Array], jax.Array]
    weights_dtype: jnp.dtype
    pre_norm: bool
    kernel_init: Callable[..., jnp.ndarray]
    out_proj_kernel_init: Callable[..., jnp.ndarray]
    decode: bool
    attn_method: AttnMethod
    record_attention_weights: bool = False

    def setup(self) -> None:
        if self.record_attention_weights:
            # Recording attention weights requires vanilla attention
            attn_function = nn.attention.dot_product_attention
        elif self.attn_method == AttnMethod.FLASH_JAX:
            # Use fast flash attention implementation
            def attn_function(
                q,
                k,
                v,
                bias=None,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=False,
                dtype=None,
                precision=None,
            ):
                assert (
                    len(q.shape) == len(k.shape) == len(v.shape) == 4
                ), f"q k v shapes: {q.shape} {k.shape} {v.shape}, expected: (batch, seq_len, heads, head_dim)"
                assert (
                    q.shape[0] == k.shape[0] == v.shape[0]
                ), "batch dimensions must match"
                batch_size = q.shape[0]
                assert q.shape[1] == k.shape[1] == v.shape[1], "seq_len must match"
                seq_len = q.shape[1]
                assert q.shape[2] == k.shape[2] == v.shape[2], "num_heads must match"
                num_heads = q.shape[2]
                assert q.shape[3] == k.shape[3], "q & k head_dim must match"
                qk_head_dim, v_head_dim = q.shape[3], v.shape[3]

                rearrange_qkv = lambda x: rearrange(
                    x, "batch seq_len heads head_dim -> batch heads seq_len head_dim"
                )
                q, k, v = map(rearrange_qkv, (q, k, v))

                assert bias == None, "attention bias not implemented"
                assert (
                    mask == None
                ), "attention mask is redundant with causal_flash_attention"
                assert dropout_rate == 0.0, "attention dropout not implemented"

                try:
                    res = flash_attention_jax.causal_flash_attention(q, k, v)
                except TypeError as e:
                    if "cannot reshape array of shape" in str(e):
                        raise ValueError(
                            (
                                "Got an exception from causal_flash_attention: {}. You may have "
                                "run into its bug with sequence lengths that are not a multiple of "
                                "the chunk size."
                            ).format(e)
                        )
                    else:
                        raise e
                if dtype != None:
                    assert res.dtype == dtype
                assert res.shape == (batch_size, num_heads, seq_len, v_head_dim)
                res = rearrange(
                    res, "batch heads seq_len head_dim -> batch seq_len heads head_dim"
                )
                assert res.shape == (batch_size, seq_len, num_heads, v_head_dim)
                return res

        elif self.attn_method == AttnMethod.FLASH_CPP:

            def attn_function(
                q,
                k,
                v,
                bias=None,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=False,
                dtype=None,
                precision=None,
            ):
                assert bias == None, "attention bias not implemented"
                assert (
                    mask == None
                ), "attention mask is redundant with causal_flash_attention"
                assert dropout_rate == 0.0, "attention dropout not implemented"
                assert dtype in [
                    jnp.bfloat16,
                    jnp.float16,
                ], "CPP flash attention only supports bfloat16 & float16"

                res = flash_attention_cpp.flash_mha(q, k, v, is_causal=True)
                assert res.shape == v.shape
                return res

        elif self.attn_method == AttnMethod.CUDNN:

            def attn_function(
                q,
                k,
                v,
                bias=None,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=False,
                dtype=None,
                precision=None,
            ):
                assert mask is None, "attention mask should be None for cudnn attention"
                assert (
                    dropout_rate == 0.0
                ), "attention dropout not implemented for cudnn attention"
                assert dtype in [
                    jnp.bfloat16,
                    jnp.float16,
                ], "cudnn attention only supports half precision"
                res = jax.nn.dot_product_attention(
                    q,
                    k,
                    v,
                    bias=bias,
                    mask=None,
                    is_causal=True,
                    implementation="cudnn",
                )
                assert res.shape == v.shape
                return res

        elif self.attn_method == AttnMethod.STANDARD:
            attn_function = nn.attention.dot_product_attention
        else:
            raise ValueError(f"Invalid attention method: {self.attn_method}")

        self.mha = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            # dropout in the attention matrix was introduced in
            # https://arxiv.org/abs/1907.11065, it's *not* the normal thing
            # from Attention is All You Need.
            dropout_rate=0,
            deterministic=False,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=self.kernel_init,
            out_kernel_init=self.out_proj_kernel_init,
            decode=self.decode,
            attention_fn=attn_function,
        )
        self.layer_norm_1 = nn.LayerNorm(
            dtype=self.activations_dtype, param_dtype=jnp.float32
        )
        self.linear_1 = nn.Dense(
            features=self.ff_dim,
            use_bias=self.use_biases,
            kernel_init=self.kernel_init,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )
        self.linear_2 = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            kernel_init=self.out_proj_kernel_init,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )
        self.layer_norm_2 = nn.LayerNorm(
            dtype=self.activations_dtype, param_dtype=jnp.float32
        )
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout, deterministic=False)
        else:
            self.dropout_layer = nn.Dropout(rate=0, deterministic=True)

    def __call__(self, embeds: jax.Array, _) -> jax.Array:
        assert_msg = (
            f"embeds.shape: {embeds.shape}, expected: (batch, seq_len, {self.d_model})"
        )
        assert len(embeds.shape) == 3, assert_msg
        assert embeds.shape[2] == self.d_model, assert_msg
        batch_size = embeds.shape[0]
        seq_len = embeds.shape[1]

        if (
            self.attn_method == AttnMethod.FLASH_JAX
            or self.attn_method == AttnMethod.FLASH_CPP
            or self.attn_method == AttnMethod.CUDNN
        ) and not self.record_attention_weights:
            mask = None
        else:
            mask = jnp.tril(
                jnp.ones((batch_size, self.num_heads, embeds.shape[1], embeds.shape[1]))
            )

        attn_in = self.layer_norm_1(embeds) if self.pre_norm else embeds
        attn_output = self.mha(
            attn_in, mask=mask, sow_weights=self.record_attention_weights
        )
        if not self.pre_norm:
            attn_output = self.layer_norm_1(attn_output)
        embeds = embeds + self.dropout_layer(attn_output)

        ff_in = self.layer_norm_2(embeds) if self.pre_norm else embeds
        ff_output = self.linear_2(self.activation_function(self.linear_1(ff_in)))
        if not self.pre_norm:
            ff_output = self.layer_norm_2(ff_output)
        embeds = embeds + self.dropout_layer(ff_output)

        assert embeds.shape == (batch_size, seq_len, self.d_model)
        return embeds, None


@pytest.mark.parametrize(
    "flash_method",
    [
        pytest.param(AttnMethod.FLASH_JAX),
        pytest.param(AttnMethod.FLASH_CPP, marks=pytest.mark.requires_ampere_or_newer),
        pytest.param(AttnMethod.CUDNN, marks=pytest.mark.requires_ampere_or_newer),
    ],
)
def test_flash_attention_equals_standard(flash_method: AttnMethod) -> None:
    """Test that flash attention gives the same results as Flax's standard attention."""
    activations_dtype = (
        jnp.float32 if flash_method == AttnMethod.FLASH_JAX else jnp.bfloat16
    )
    mdl_std = TransformerLayer(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        dropout=None,
        use_biases=False,
        activations_dtype=activations_dtype,
        activation_function=jax.nn.relu,
        weights_dtype=jnp.float32,
        pre_norm=False,
        kernel_init=nn.initializers.normal(stddev=0.02),
        out_proj_kernel_init=nn.initializers.normal(stddev=0.02 / jnp.sqrt(2 * 12)),
        decode=False,
        attn_method=AttnMethod.STANDARD,
    )

    input_shape = (4, 64, 768)
    input_vals = jax.random.normal(jax.random.PRNGKey(0), input_shape)

    params = mdl_std.init(
        jax.random.PRNGKey(1), jnp.ones(input_shape, dtype=jnp.float32), None
    )

    out_std, _ = mdl_std.apply(params, input_vals, None)

    mdl_flash = mdl_std.clone(attn_method=flash_method)
    out_flash, _ = mdl_flash.apply(params, input_vals, None)

    # Numerical differences are obscenely large, but only on certain hardware????
    np.testing.assert_allclose(out_std, out_flash, atol=0.05, rtol=0)


def loss_batch_tokens(
    model: ImageModel,
    params: dict[str, Any],
    dropout_rng: jax.Array,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for each token in a batch of examples."""
    batch_size = batch_imgs.shape[0]
    assert batch_imgs.shape == (
        batch_size,
        model.image_tokens,
    ), f"batch_img.shape: {batch_imgs.shape}, expected: {(batch_size, model.image_tokens)}"
    if model.clip_conditioning and not model.clip_caps:
        assert batch_clips.shape == (batch_size, 768)
        assert batch_max_cos_distances.shape == (batch_size, 0)
    elif model.clip_conditioning and model.clip_caps:
        assert batch_clips.shape == (batch_size, model.clip_cap_count, 768)
        assert batch_max_cos_distances.shape == (batch_size, model.clip_cap_count)
    else:
        assert batch_clips.shape == batch_max_cos_distances.shape == (batch_size, 0)
    logits: jax.Array = model.apply(
        params,
        rngs={"dropout": dropout_rng},
        images=batch_imgs,
        clip_embeddings=batch_clips,
        max_cos_distances=batch_max_cos_distances,
    )
    per_token_loss = optax.softmax_cross_entropy(
        logits, jax.nn.one_hot(batch_imgs, 8192)
    )
    assert per_token_loss.shape == (
        batch_size,
        model.image_tokens,
    ), f"per_token_loss.shape: {per_token_loss.shape}"
    return per_token_loss


def loss_batch(
    model: ImageModel,
    params: dict[str, Any],
    dropout_rng: jax.Array,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
) -> jax.Array:
    """Compute the average cross-entropy loss for a batch of examples."""
    per_token_loss = loss_batch_tokens(
        model, params, dropout_rng, batch_imgs, batch_clips, batch_max_cos_distances
    )
    return jnp.mean(per_token_loss)


# Parameters taken from GPT-1, except seq_len is 256 instead of 1024
gpt_1_config = ModelConfig(
    d_model=768,
    num_heads=12,
    ff_dim=3072,
    dropout=0.1,
    n_layers=12,
    image_tokens=256,
    use_biases=True,
    activation_function=jax.nn.relu,
    clip_conditioning=False,
    corrected_cap_projections=True,
)


def test_cap_train() -> None:
    """Test the model can memorize some image/clip pairs."""
    mdl_cfg = copy(gpt_1_config)
    mdl_cfg.clip_conditioning = True
    mdl_cfg.clip_caps = True
    mdl_cfg.clip_cap_count = 9
    mdl_cfg.dropout = None
    mdl_cfg.pre_norm = True

    n_imgs = 8

    mdl = ImageModel(**mdl_cfg.__dict__)

    (
        img_rng,
        clip_rng,
        max_cos_distance_rng,
        params_rng,
        train_rng,
        test_rng,
    ) = jax.random.split(jax.random.PRNGKey(0), 6)

    imgs = jax.random.randint(img_rng, (n_imgs, mdl.image_tokens), 0, 8192)
    clips = jax.random.normal(clip_rng, (n_imgs, mdl.clip_cap_count, 768))
    clips = clips / jnp.linalg.norm(clips, axis=-1, keepdims=True)
    max_cos_distances = jax.random.uniform(
        max_cos_distance_rng, shape=(n_imgs, mdl.clip_cap_count), minval=0.0, maxval=2.0
    )

    params = mdl.init(
        {"params": params_rng, "dropout": jax.random.PRNGKey(0)}, *mdl.dummy_inputs()
    )

    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    steps = 100
    opt = optax.contrib.schedule_free_adamw(
        learning_rate=1e-3, b1=0.98, warmup_steps=30
    )
    opt_state = opt.init(params)

    def opt_step(params, opt_state, rng):
        dropout_rng, rng2 = jax.random.split(rng, 2)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            batch_imgs=imgs,
            batch_clips=clips,
            batch_max_cos_distances=max_cos_distances,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        norm = optax.global_norm(grads)
        return new_params, opt_state, rng2, loss, norm

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    for i in trange(steps):
        params, opt_state, train_rng, loss, norm = opt_step(
            params, opt_state, train_rng
        )
        loss, norm = jax.device_get((loss, norm))
        tqdm.write(f"iter {i:04d} loss: {loss:0.4f} grad norm: {norm:7.2f}")

    params = optax.contrib.schedule_free_eval_params(opt_state, params)

    # This only tests the ability to memorize training examples, and not even the ability to
    # generalize to different sets of caps for the same image.

    sample_rng = jax.random.split(test_rng, n_imgs)
    sampled_imgs = sample(
        mdl,
        params,
        clips,
        max_cos_distances,
        sample_rng,
        filter_method=LogitFilterMethod.TOP_P,
        filter_threshold=0.25,
    )

    assert sampled_imgs.shape == (n_imgs, mdl.image_tokens)

    # Test that sampled images are the training images
    correct_imgs = [0] * n_imgs
    for img in sampled_imgs:
        for i, train_img in enumerate(imgs):
            if jnp.all(img == train_img):
                correct_imgs[i] += 1
                break
    correct_img_cnt = sum(correct_imgs)
    print(f"correct_imgs: {correct_imgs}, count {correct_img_cnt}")
    assert correct_img_cnt == n_imgs

    # Test that sampled images match the CLIP conditioning
    correct_conds = [False] * n_imgs
    for i in range(n_imgs):
        correct_conds[i] = jnp.array_equal(imgs[i], sampled_imgs[i])
    correct_conds = np.array(correct_conds)
    print(f"correct_conds: {correct_conds}")
    assert all(correct_conds)


def train_loop_simple(
    data: jax.Array,
    mdl: ImageModel,
    iters: int,
    learning_rate: float = 3e-5,
    warmup_steps: Optional[int] = None,
) -> Tuple[jax.Array, dict[str, Any]]:
    """Train the model repeatedly on a single batch for testing."""
    assert mdl.clip_conditioning is False
    assert len(data.shape) == 2
    batch_size = data.shape[0]
    assert data.shape == (batch_size, mdl.image_tokens)

    params = mdl.init(
        {"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
        *mdl.dummy_inputs(),
    )
    if warmup_steps is None:
        warmup_steps = iters // 10
    opt = optax.contrib.schedule_free_adamw(
        learning_rate=learning_rate, b1=0.98, warmup_steps=warmup_steps
    )
    opt_state = opt.init(params)
    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    def opt_step(
        params: dict[str, Any],
        opt_state: Any,
        rng: jax.Array,
        batch_imgs: jax.Array,
    ) -> Tuple[dict[str, Any], Any, jax.Array, jax.Array]:
        dropout_rng, rng2 = jax.random.split(rng, 2)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            batch_imgs=batch_imgs,
            batch_clips=jnp.zeros((batch_imgs.shape[0], 0), dtype=jnp.float32),
            batch_max_cos_distances=jnp.zeros(
                (batch_imgs.shape[0], 0), dtype=jnp.float32
            ),
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params: dict[str, Any] = optax.apply_updates(params, updates)
        return new_params, opt_state, rng2, loss

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    train_rng = jax.random.PRNGKey(0)
    for i in trange(iters):
        params, opt_state, train_rng, loss = opt_step(  # type:ignore[misc]
            params, opt_state, train_rng, data
        )
        tqdm.write(f"iter {i} loss: {loss}")
    params = optax.contrib.schedule_free_eval_params(opt_state, params)
    return loss, params  # type:ignore[return-value]


@pytest.mark.parametrize("weights_dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("pre_norm", [True, False])
def test_learn_zeros(pre_norm: bool, weights_dtype: jnp.dtype) -> None:
    """Test whether the model can learn to predict all zeros."""
    mdl_cfg = copy(gpt_1_config)
    mdl_cfg.pre_norm = pre_norm
    mdl_cfg.activations_dtype = weights_dtype
    mdl_cfg.weights_dtype = weights_dtype
    mdl = ImageModel(**mdl_cfg.__dict__)
    data = jnp.zeros((16, gpt_1_config.image_tokens), dtype=jnp.int32)
    loss, params = train_loop_simple(
        data, mdl, iters=7, learning_rate=1e-1, warmup_steps=1
    )
    assert loss < 1e-10

    sampled_arr = sample(
        mdl,
        params,
        jnp.zeros((1, 0), dtype=jnp.float32),
        jnp.zeros((1, 0), dtype=jnp.float32),
        jax.random.PRNGKey(0)[None, :],
    )
    assert jnp.all(sampled_arr == 0)


@pytest.mark.parametrize(
    "weights_dtype, activations_dtype",
    [
        pytest.param(jnp.float32, jnp.float32, id="wf32-af32"),
        pytest.param(jnp.float32, jnp.bfloat16, id="wf32-abf16"),
        pytest.param(jnp.bfloat16, jnp.bfloat16, id="wbf16-abf16"),
    ],
)
def test_learn_sequential(
    weights_dtype: jnp.dtype, activations_dtype: jnp.dtype
) -> None:
    """Test whether the model can learn to predict consecutive numbers."""
    mdl_cfg = copy(gpt_1_config)
    mdl_cfg.pre_norm = True
    mdl_cfg.activations_dtype = activations_dtype
    mdl_cfg.weights_dtype = weights_dtype
    mdl = ImageModel(**mdl_cfg.__dict__)

    data_rng, params_rng, train_rng, sample_rng = jax.random.split(
        jax.random.PRNGKey(777), 4
    )

    # Generate a dataset of sequential numbers
    n_imgs = 8192
    start_vals = jax.random.randint(data_rng, (n_imgs,), 0, 8192)
    data = (start_vals[:, None] + jnp.arange(gpt_1_config.image_tokens)) % 8192
    assert data.shape == (n_imgs, gpt_1_config.image_tokens)
    data = jax.device_get(data)
    dset_all = TableView({"encoded_img": data}).shuffle(seed=0)
    test_set_size = n_imgs // 10
    dset_train = dset_all.new_view(slice(None, -test_set_size))
    dset_test = dset_all.new_view(slice(-test_set_size, None))

    # Train a model
    params = mdl.init(
        {"params": params_rng, "dropout": jax.random.PRNGKey(0)}, *mdl.dummy_inputs()
    )

    steps = 300
    batch_size = 64

    opt = optax.contrib.schedule_free_adamw(
        learning_rate=1e-3, b1=0.98, warmup_steps=10
    )
    opt_state = opt.init(params)
    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    def opt_step(params, opt_state, rng, batch_imgs):
        dropout_rng, rng2 = jax.random.split(rng, 2)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            batch_imgs=batch_imgs,
            batch_clips=jnp.zeros((batch_size, 0), dtype=jnp.float32),
            batch_max_cos_distances=jnp.zeros((batch_size, 0), dtype=jnp.float32),
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, rng2, loss

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    with tqdm(total=steps) as pbar:
        while pbar.n < steps:
            for batch in dset_train.shuffle(seed=pbar.n).batch_iter(
                batch_size, drop_last_batch=True
            ):
                params, opt_state, train_rng, loss = opt_step(
                    params, opt_state, train_rng, batch["encoded_img"]
                )
                loss = jax.device_get(loss)
                if pbar.n % 10 == 0:
                    tqdm.write(f"iter {pbar.n:04d} loss: {loss:0.4f}")
                pbar.update(1)
                pbar.set_postfix({"loss": loss})
                if pbar.n >= steps:
                    break
    print(f"Final train loss: {loss}")
    params = optax.contrib.schedule_free_eval_params(opt_state, params)

    # Compute test loss
    test_mdl = mdl.clone(dropout=None, image_dropout=None, clip_dropout=None)
    calc_loss = jax.jit(
        lambda imgs: loss_batch(
            test_mdl,
            params,
            jax.random.PRNGKey(0),
            imgs,
            jnp.zeros((imgs.shape[0], 0), dtype=jnp.float32),
            jnp.zeros((imgs.shape[0], 0), dtype=jnp.float32),
        )
    )
    test_losses = []
    test_batches = len(dset_test) // batch_size + (
        1 if len(dset_test) % batch_size > 0 else 0
    )
    for batch in tqdm(
        dset_test.batch_iter(batch_size, drop_last_batch=False), total=test_batches
    ):
        test_losses.append(calc_loss(batch["encoded_img"]))
    test_loss = jnp.mean(jnp.array(test_losses))
    print(f"Final test loss: {test_loss}")
    assert test_loss < 0.04

    # Sample from the model
    sample_batch_size = 64
    sample_batches = 4

    samples = []
    for _ in range(sample_batches):
        split_rngs = jax.random.split(sample_rng, sample_batch_size + 1)
        sample_rng = split_rngs[-1]
        samples.append(
            sample(
                mdl,
                params,
                jnp.zeros((sample_batch_size, 0), dtype=jnp.float32),
                jnp.zeros((sample_batch_size, 0), dtype=jnp.float32),
                split_rngs[:-1],
                filter_threshold=0.8,
            )
        )
    sampled_arr = jnp.concatenate(samples, axis=0)

    assert sampled_arr.shape == (
        sample_batch_size * sample_batches,
        gpt_1_config.image_tokens,
    )

    # Test that each row in sampled_arr is an increasing sequence
    expected = (sampled_arr[:, :1] + np.arange(gpt_1_config.image_tokens)) % 8192
    np.testing.assert_array_equal(np.array(sampled_arr), expected)


@pytest.mark.skip("extremely slow and flaky")
@pytest.mark.parametrize(
    "weights_dtype, activations_dtype",
    [
        pytest.param(jnp.bfloat16, jnp.bfloat16, id="wbf16-abf16"),
        pytest.param(jnp.float32, jnp.bfloat16, id="wf32-abf16"),
        pytest.param(jnp.float32, jnp.float32, id="wf32-af32"),
    ],
)
def test_clip_caps_overfit(weights_dtype: jnp.dtype, activations_dtype: jnp.dtype):
    """Using a collection of ~100k, cap-image pair training examples, train a model and check test
    loss, then sample and check the generated samples. The training data is 100k pairs generated
    from 100 images (drawn with replacement). This tests the model's ability to memorize a small
    set of image encodings and learn to sample based on their CLIP encodings being within caps.
    The trained model should be overfit on the images, but not overfit on the caps. We test against
    a holdout set of cap-image pairs from the same 100 images."""

    params_rng, sample_rng = jax.random.split(jax.random.PRNGKey(42), 2)
    # run get-test-data.sh to download this
    dset_all = load_pq_to_infinidata(
        Path(__file__).parent.parent / "test-images/examples-100.pq"
    ).shuffle(seed=420_69)
    dset_train = dset_all.new_view(slice(None, -16))
    dset_test = dset_all.new_view(slice(-16, None))

    mdl_cfg = copy(gpt_1_config)
    mdl_cfg.clip_conditioning = True
    mdl_cfg.clip_caps = True
    mdl_cfg.clip_cap_count = 1
    mdl_cfg.dropout = None
    mdl_cfg.activation_function = jax.nn.gelu
    mdl_cfg.activations_dtype = activations_dtype
    mdl_cfg.weights_dtype = weights_dtype
    mdl_cfg.pre_norm = True

    mdl = ImageModel(**mdl_cfg.__dict__)

    params = _train_clip_caps_overfit(dset_train, mdl, params_rng)
    _test_clip_caps_overfit_test_loss(dset_test, mdl, params)
    _test_clip_caps_overfit_samples(dset_test, mdl, params, sample_rng)


def _train_clip_caps_overfit(dset_train, mdl, rng):
    """Train a test model"""
    params_rng, dropout_rng = jax.random.split(rng, 2)

    params = mdl.init({"params": params_rng}, *mdl.dummy_inputs())

    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    steps = 1200
    batch_size = 64

    opt = optax.contrib.schedule_free_adamw(
        learning_rate=1e-3, b1=0.98, warmup_steps=100
    )
    opt_state = opt.init(params)

    def opt_step(params, opt_state, rng, images, clips, max_cos_distances):
        dropout_rng, rng2 = jax.random.split(rng, 2)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            batch_imgs=images,
            batch_clips=clips,
            batch_max_cos_distances=max_cos_distances,
        )
        updates, opt_state = opt.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        norm = optax.global_norm(grads)
        return new_params, opt_state, rng2, loss, norm

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    with tqdm(total=steps) as pbar:
        while pbar.n < steps:
            dset_shuf = dset_train.shuffle(seed=pbar.n)
            for batch in dset_shuf.batch_iter(batch_size, drop_last_batch=True):
                images, cap_centers, max_cos_distances = (
                    batch["encoded_img"],
                    rearrange(batch["cap_center"], "b c -> b 1 c"),
                    rearrange(batch["cap_max_cos_distance"], "b -> b 1"),
                )
                assert (
                    images.shape[0]
                    == cap_centers.shape[0]
                    == max_cos_distances.shape[0]
                )
                params, opt_state, dropout_rng, loss, norm = opt_step(
                    params,
                    opt_state,
                    dropout_rng,
                    images,
                    cap_centers,
                    max_cos_distances,
                )
                pbar.update(1)
                loss, norm = jax.device_get((loss, norm))
                pbar.set_postfix({"loss": loss, "grad norm": norm})
                if pbar.n % 10 == 0:
                    tqdm.write(
                        f"iter {pbar.n:04d} loss: {loss:0.4f} grad norm: {norm:05.2f}"
                    )
                if pbar.n >= steps:
                    break
    return optax.contrib.schedule_free_eval_params(opt_state, params)


def _test_clip_caps_overfit_test_loss(dset_test, mdl, params):
    """Compute test loss and assert it's low enough"""
    examples = dset_test[:]
    cap_centers = rearrange(examples["cap_center"], "b c -> b 1 c")
    max_cos_distances = rearrange(examples["cap_max_cos_distance"], "b -> b 1")
    imgs = examples["encoded_img"]

    computed_loss = loss_batch(
        mdl, params, jax.random.PRNGKey(0), imgs, cap_centers, max_cos_distances
    )
    print(f"test loss: {computed_loss}")
    assert computed_loss < 0.05


def _test_clip_caps_overfit_samples(dset_test, mdl, params, rng):
    """Sample from the model and check it returns images with embeddings inside the test caps."""
    pt_rng, sample_rng = jax.random.split(rng, 2)

    examples = dset_test[:]
    n_samples = len(examples["encoded_img"])
    embeddings = examples["clip_embedding"]

    # Generate random query points with cosine similarity 0.8 (cosine distance 0.2) to the
    # embeddings.
    query_pts = jax.vmap(random_pt_with_cosine_similarity, in_axes=(0, 0, None))(
        jax.random.split(pt_rng, n_samples), embeddings, 0.8
    )
    query_pts = query_pts.reshape((n_samples, 1, 768))
    # Sample from caps centered on those points with max cosine distance 0.25.
    query_max_cos_distances = jnp.full((n_samples, 1), 0.25)

    sample_rngs = jax.random.split(sample_rng, n_samples)
    toks = sample(
        mdl,
        params,
        query_pts,
        query_max_cos_distances,
        sample_rngs,
        filter_method=LogitFilterMethod.TOP_P,
        filter_threshold=0.05,
    )
    assert toks.shape == (n_samples, gpt_1_config.image_tokens)
    matches = jnp.all(toks == examples["encoded_img"], axis=1)

    print(f"Found {matches.sum()} matches out of {n_samples} samples")
    assert matches.sum() >= round(n_samples * 0.9)
