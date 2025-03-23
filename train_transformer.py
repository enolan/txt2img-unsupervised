"""Train the image model."""

import os

# necessary to use more of the GPU's memory. Default is 0.75. It's supposed to be able to
# dynamically allocate more, but there are fragmentation issues since we allocate ginormous arrays.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import datetime
import gc
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type:ignore[import]
import orbax.checkpoint as ocp
import torch
import transformers
import wandb
from copy import copy
from distutils.util import strtobool
from einops import rearrange, repeat
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm, trange
from typing import Optional, Tuple

from txt2img_unsupervised.checkpoint import TransformerTrainState
from txt2img_unsupervised.config import (
    LearningRateSchedule,
    TransformerModelConfig,
    TrainingConfig,
)
from txt2img_unsupervised.ldm_autoencoder import LDMAutoencoder
from txt2img_unsupervised.train_data_loading import get_batch
from txt2img_unsupervised.training_infra import (
    init_common_train_state,
    init_wandb_training,
    leading_dims_to_subtrees,
    load_dataset,
    plan_steps,
    save_checkpoint,
    setup_common_arguments,
    setup_jax_for_training,
    setup_profiling_server,
    setup_sharding,
    SignalHandler,
)
from txt2img_unsupervised.training_visualizations import (
    log_attention_maps,
    log_token_loss_visualization,
)
import txt2img_unsupervised.cap_sampling as cap_sampling
import txt2img_unsupervised.sample as sample
import txt2img_unsupervised.transformer_model as transformer_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    # Add common arguments for all model types
    parser = setup_common_arguments(parser)

    # Add transformer-specific arguments
    parser.add_argument("--sample-batch-size", type=int, default=None)
    parser.add_argument("--ae-cfg", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--image-dropout", type=float, default=None)
    parser.add_argument("--clip-conditioning", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--clip-caps", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--clip-cap-count", type=int)
    parser.add_argument(
        "--skip-sampling", action="store_true", help="Skip sampling during training"
    )

    args, _unknown = parser.parse_known_args()
    return args


def init_train_state(
    model_cfg: TransformerModelConfig,
    training_cfg: TrainingConfig,
    total_steps: int,
    resume_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_path: Optional[Path] = None,
    sample_batch_size: Optional[int] = None,
    start_where_finetune_source_left_off: bool = False,
):
    """Set up our initial TransformerTrainState using the provided configs.

    Args:
        model_cfg: The model configuration
        training_cfg: The training configuration
        total_steps: Total number of training steps
        resume_checkpoint_path: Path to checkpoint to resume from, if any
        finetune_checkpoint_path: Path to checkpoint to finetune from, if any
        sample_batch_size: Batch size for sampling, if different from training
        start_where_finetune_source_left_off: Whether to start training from where the finetune source left off
    """
    (
        global_step,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    ) = init_common_train_state(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        total_steps=total_steps,
        train_state_class=TransformerTrainState,
        resume_checkpoint_path=resume_checkpoint_path,
        finetune_checkpoint_path=finetune_checkpoint_path,
        start_where_finetune_source_left_off=start_where_finetune_source_left_off,
        create_model_fn=transformer_model.ImageModel,
    )

    if sample_batch_size is None:
        sample_batch_size = training_cfg.batch_size

    print(mdl.tabulate(jax.random.PRNGKey(0), *mdl.dummy_inputs()))

    return (
        global_step,
        sample_batch_size,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    )


def get_clip_mdl():
    """Get the CLIP model and processor."""
    # The model must match the one used in the preprocessor
    clip_mdl_name = "openai/clip-vit-large-patch14"
    clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name)
    clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)
    return clip_mdl, clip_processor


def find_sfw_indices(clip_mdl, clip_processor, dset, n: int):
    """Find indices of n hopefully SFW images based on CLIP embeddings."""
    text_tokens = clip_processor.tokenizer("nsfw", return_tensors="jax", padding=True)
    nsfw_text_features = clip_mdl.get_text_features(**text_tokens)
    nsfw_text_embeds = nsfw_text_features / jnp.linalg.norm(nsfw_text_features)

    clip_embeds = dset["clip_embedding"]
    sims = jnp.dot(clip_embeds, nsfw_text_embeds.T)
    ok_sims = sims < 0.2
    print(f"Found {ok_sims.sum()} images with similarity < 0.2")
    ok_indices = jnp.where(ok_sims)[0]
    assert (
        len(ok_indices) >= n
    ), f"Found {len(ok_indices)} images with similarity < 0.2, expected at least {n}"
    return jax.device_get(ok_indices[:n])


def get_clip_text_embeddings_to_sample(n: int, clip_mdl, clip_processor) -> jax.Array:
    """Generate some text embeddings to test the model with."""
    prompts = [
        "A woman’s face",
        "A screenshot of Grand Theft Auto",
        "Barack Obama riding a bicycle",
        "a painting of a cat",
        "The Golden Gate Bridge at sunset",
        "Samoyed puppies!",
        "Taylor Swift in concert",
        "my favorite art car from Burning Man 2018",
    ]
    assert n <= len(prompts), "write more prompts?"
    text_tokens = clip_processor(prompts, return_tensors="jax", padding=True)
    text_features = clip_mdl.get_text_features(**text_tokens)
    text_embeds = text_features / jnp.linalg.norm(text_features, axis=1, keepdims=True)
    assert text_embeds.shape == (len(prompts), 768)
    return {
        "clip_embedding": text_embeds[:n],
        "name": np.array(prompts)[:n],
    }


def mk_image_prompt_conditions(image_prompt_clips, grid_size):
    """Make conditioning data for the image prompts."""
    assert image_prompt_clips.shape[1] == 768
    # Use a larger and a smaller cap. The smaller cap should generate images that are very
    # similar to the image prompt, the larger one should generate images that are semantically
    # similar but not necessarily visually similar.
    if mdl.clip_caps:
        max_cos_distance_choices = jnp.array([0.75, 0.4], dtype=jnp.float32)

        samples_count = (
            len(image_prompt_clips) * len(max_cos_distance_choices) * grid_size
        )

        # With more than one cap slot, we fill in the unused ones with caps that cover the entire
        # space.
        cap_centers_fill, cap_max_cos_distances_fill = sample.mk_filler_caps(
            mdl, samples_count, 1, jax.random.PRNGKey(20240214)
        )
        cap_centers_fill = rearrange(
            cap_centers_fill,
            "(n ds g) cap clip -> n ds g cap clip",
            n=len(image_prompt_clips),
            ds=len(max_cos_distance_choices),
            g=grid_size,
        )
        cap_max_cos_distances_fill = rearrange(
            cap_max_cos_distances_fill,
            "(n ds g) cap -> n ds g cap",
            n=len(image_prompt_clips),
            ds=len(max_cos_distance_choices),
            g=grid_size,
        )
        cap_centers_cond = repeat(
            image_prompt_clips,
            "n clip -> n ds g 1 clip",
            ds=len(max_cos_distance_choices),
            g=grid_size,
        )
        cap_max_cos_distances_cond = repeat(
            max_cos_distance_choices,
            "ds -> n ds g 1",
            n=len(image_prompt_clips),
            g=grid_size,
        )

        cap_centers = jnp.concatenate([cap_centers_cond, cap_centers_fill], axis=3)
        cap_max_cos_distances = jnp.concatenate(
            [cap_max_cos_distances_cond, cap_max_cos_distances_fill], axis=3
        )
        assert cap_centers.shape == (
            len(image_prompt_clips),
            len(max_cos_distance_choices),
            grid_size,
            mdl.clip_cap_count,
            768,
        )
        assert cap_max_cos_distances.shape == (
            len(image_prompt_clips),
            len(max_cos_distance_choices),
            grid_size,
            mdl.clip_cap_count,
        )
        assert np.prod(cap_centers.shape[:3]) == samples_count
        return cap_centers, cap_max_cos_distances, max_cos_distance_choices
    else:
        samples_count = len(image_prompt_clips) * grid_size
        clip_embeddings = repeat(
            image_prompt_clips,
            "n clip -> n g clip",
            g=grid_size,
        )
        assert np.prod(clip_embeddings.shape[:2]) == samples_count
        return clip_embeddings, None


def mk_txt_prompt_conditions(text_prompt_clips, grid_size):
    """Make conditioning data for the text prompts."""
    assert text_prompt_clips.shape[1] == 768
    samples_count = len(text_prompt_clips) * grid_size
    if mdl.clip_caps:
        cap_centers_fill, cap_max_cos_distances_fill = sample.mk_filler_caps(
            mdl, samples_count, 1, jax.random.PRNGKey(20240214)
        )
        cap_centers_fill = rearrange(
            cap_centers_fill,
            "(prompt g) cap clip -> prompt g cap clip",
            prompt=len(text_prompt_clips),
            g=grid_size,
        )
        cap_max_cos_distances_fill = rearrange(
            cap_max_cos_distances_fill,
            "(prompt g) cap -> prompt g cap",
            prompt=len(text_prompt_clips),
            g=grid_size,
        )

        cap_centers_cond = repeat(
            text_prompt_clips, "prompt clip -> prompt g 1 clip", g=grid_size
        )
        cap_max_cos_distances_cond = np.full(
            (len(text_prompt_clips), grid_size, 1), 0.70
        )

        cap_centers = np.concatenate([cap_centers_cond, cap_centers_fill], axis=2)
        cap_max_cos_distances = np.concatenate(
            [cap_max_cos_distances_cond, cap_max_cos_distances_fill], axis=2
        )

        assert cap_centers.shape == (
            len(text_prompt_clips),
            grid_size,
            mdl.clip_cap_count,
            768,
        )
        assert cap_max_cos_distances.shape == (
            len(text_prompt_clips),
            grid_size,
            mdl.clip_cap_count,
        )
        return cap_centers, cap_max_cos_distances
    else:
        clip_embeddings = repeat(
            text_prompt_clips, "prompt clip -> prompt g clip", g=grid_size
        )
        assert clip_embeddings.shape == (len(text_prompt_clips), grid_size, 768)
        return clip_embeddings, None


def load_autoencoder(ae_cfg_path: Path, ae_ckpt_path: Path):
    """Load the autoencoder model and parameters."""
    ae_cfg = OmegaConf.load(ae_cfg_path)["model"]["params"]
    ae_mdl = LDMAutoencoder(ae_cfg)
    # don't keep these on the GPU when we're not using them
    ae_params_torch = torch.load(ae_ckpt_path, map_location="cpu")
    return ae_cfg, ae_mdl, ae_params_torch


def sample_and_log(
    ts: TransformerTrainState, sample_batch_size: int, global_step: int
) -> None:
    """Sample from the model and log to wandb."""

    ae_params = LDMAutoencoder.params_from_torch(ae_params_torch, ae_cfg)
    eval_params = ts.get_eval_params()

    if mdl.clip_conditioning:
        # Create a grid of samples for each set of conditions.
        grid_size = 9

        image_prompt_names = visualization_dset["name"]
        text_prompt_texts = text_prompt_clips["name"]

        if mdl.clip_caps:
            (
                img_cap_centers,
                img_max_cos_distances,
                img_max_cos_distance_choices,
            ) = mk_image_prompt_conditions(
                visualization_dset["clip_embedding"], grid_size
            )

            img_centers_for_sampling = rearrange(
                img_cap_centers,
                "prompts ds grid cap clip -> (prompts ds grid) cap clip",
            )
            img_max_cos_distances_for_sampling = rearrange(
                img_max_cos_distances,
                "prompts ds grid cap -> (prompts ds grid) cap",
            )

            text_cap_centers, text_max_cos_distances = mk_txt_prompt_conditions(
                text_prompt_clips["clip_embedding"], grid_size
            )

            text_centers_for_sampling = rearrange(
                text_cap_centers,
                "prompts grid cap clip -> (prompts grid) cap clip",
            )
            text_max_cos_distances_for_sampling = rearrange(
                text_max_cos_distances,
                "prompts grid cap -> (prompts grid) cap",
            )

            all_centers_for_sampling = np.concatenate(
                [img_centers_for_sampling, text_centers_for_sampling], axis=0
            )
            all_max_cos_distances_for_sampling = np.concatenate(
                [
                    img_max_cos_distances_for_sampling,
                    text_max_cos_distances_for_sampling,
                ],
                axis=0,
            )

            imgs_list = sample.sample_loop(
                mdl,
                eval_params,
                ae_mdl,
                ae_params,
                sample_batch_size,
                all_centers_for_sampling,
                all_max_cos_distances_for_sampling,
                ts.rng,
                logit_filter_method=transformer_model.LogitFilterMethod.TOP_P,
                logit_filter_threshold=0.95,
            )
            # sample_loop returns a list of PIL image objects, but we want to rearrange them back
            # so we make a numpy array. Numpy is *very* overeager to convert the PIL objects into
            # pixel arrays so we use this dumb hack.
            imgs = np.empty((len(imgs_list),), dtype=object)
            imgs[:] = imgs_list
            assert imgs.shape == (
                len(all_centers_for_sampling),
            ), f"imgs.shape {imgs.shape}"

            img_imgs = imgs[: len(img_centers_for_sampling)]
            text_imgs = imgs[len(img_centers_for_sampling) :]
            assert len(img_imgs) == len(img_centers_for_sampling)
            assert len(text_imgs) == len(text_centers_for_sampling)
            img_imgs = rearrange(
                img_imgs,
                "(prompts ds grid)-> prompts ds grid",
                prompts=len(image_prompt_names),
                ds=len(img_max_cos_distance_choices),
                grid=grid_size,
            )
            text_imgs = rearrange(
                text_imgs,
                "(prompts grid) -> prompts grid",
                prompts=len(text_prompt_texts),
                grid=grid_size,
            )

            tqdm.write(
                f"Sampled grids of shapes {img_imgs.shape} and {text_imgs.shape}"
            )

            to_log = {"global_step": global_step}

            for i, name in enumerate(image_prompt_names):
                for j, dist in enumerate(img_max_cos_distance_choices):
                    grid_pil = sample.make_grid(
                        [img for img in img_imgs[i, j]],
                    )
                    to_log[
                        f"samples/imgprompts/{name}/max_dist{dist:.2f}"
                    ] = wandb.Image(grid_pil)

            for i, name in enumerate(text_prompt_clips["name"]):
                grid_pil = sample.make_grid(
                    [img for img in text_imgs[i]],
                )
                to_log[f"samples/txtprompts/{name}"] = wandb.Image(grid_pil)
            wandb.log(to_log)
        else:
            img_clip_embeddings, _ = mk_image_prompt_conditions(
                visualization_dset["clip_embedding"], grid_size
            )
            img_clip_embeddings_for_sampling = rearrange(
                img_clip_embeddings,
                "prompts grid clip -> (prompts grid) clip",
            )
            text_clip_embeddings, _ = mk_txt_prompt_conditions(
                text_prompt_clips["clip_embedding"], grid_size
            )
            text_clip_embeddings_for_sampling = rearrange(
                text_clip_embeddings,
                "prompts grid clip -> (prompts grid) clip",
            )

            all_clip_embeddings_for_sampling = np.concatenate(
                [img_clip_embeddings_for_sampling, text_clip_embeddings_for_sampling],
                axis=0,
            )

            imgs_list = sample.sample_loop(
                mdl,
                eval_params,
                ae_mdl,
                ae_params,
                sample_batch_size,
                all_clip_embeddings_for_sampling,
                None,
                ts.rng,
                logit_filter_method=transformer_model.LogitFilterMethod.TOP_P,
                logit_filter_threshold=0.95,
            )

            imgs = np.empty((len(imgs_list),), dtype=object)
            imgs[:] = imgs_list
            assert imgs.shape == (
                len(all_clip_embeddings_for_sampling),
            ), f"imgs.shape {imgs.shape}"

            img_imgs = imgs[: len(img_clip_embeddings_for_sampling)]
            text_imgs = imgs[len(img_clip_embeddings_for_sampling) :]
            assert len(img_imgs) == len(img_clip_embeddings_for_sampling)
            assert len(text_imgs) == len(text_clip_embeddings_for_sampling)
            img_imgs = rearrange(
                img_imgs,
                "(prompts grid)-> prompts grid",
                prompts=len(image_prompt_names),
                grid=grid_size,
            )
            text_imgs = rearrange(
                text_imgs,
                "(prompts grid) -> prompts grid",
                prompts=len(text_prompt_texts),
                grid=grid_size,
            )

            tqdm.write(
                f"Sampled grids of shape: {img_imgs.shape} and {text_imgs.shape}"
            )

            to_log = {"global_step": global_step}

            for i, name in enumerate(image_prompt_names):
                grid_pil = sample.make_grid(
                    [img for img in img_imgs[i]],
                )
                to_log[f"samples/imgprompts/{name}"] = wandb.Image(grid_pil)

            for i, name in enumerate(text_prompt_clips["name"]):
                grid_pil = sample.make_grid(
                    [img for img in text_imgs[i]],
                )
                to_log[f"samples/txtprompts/{name}"] = wandb.Image(grid_pil)

            wandb.log(to_log)
    else:
        # Sample 100 unconditioned images for a 10x10 grid
        clip_embeddings = np.zeros((100, 0), dtype=jnp.float32)

        imgs = sample.sample_loop(
            mdl,
            eval_params,
            ae_mdl,
            ae_params,
            sample_batch_size,
            clip_embeddings,
            None,
            ts.rng,
            logit_filter_method=transformer_model.LogitFilterMethod.TOP_P,
            logit_filter_threshold=0.95,
        )
        assert len(imgs) == 100
        grid = sample.make_grid(imgs)
        wandb.log(
            {
                "samples/unconditioned": wandb.Image(grid),
                "global_step": global_step,
            }
        )


@partial(
    jax.jit, donate_argnames=["state"], static_argnames=["return_weights_and_grads"]
)
def train_step(
    state: TransformerTrainState,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
    return_weights_and_grads: bool = False,
) -> Tuple[TransformerTrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    dropout_rng, rng2 = jax.random.split(state.rng, 2)
    loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
    loss, grads = loss_grad_fn(
        mdl,
        state.params,
        dropout_rng,
        batch_imgs,
        batch_clips,
        batch_max_cos_distances,
    )
    new_state = state.apply_gradients(
        grads=grads, rng=rng2
    )  # type:ignore[no-untyped-call]
    # If we're using adaptive gradient clip, we've already computed the global norm. Reusing it
    # saves, surprisingly, a substantial amount of memory. IDK why the memory used when adaptive
    # gradient clip computes the norm isn't freed when it's done, but whatever.
    if training_cfg.adaptive_gradient_clip:
        norm = new_state.get_last_norm()
        assert norm is not None
    else:
        assert new_state.get_last_norm() is None
        norm = optax.global_norm(grads)

    if return_weights_and_grads:
        # TODO this uses a lot of VRAM and reduces max model size/max batch size. Instead of
        # returning the gradients from train_step, when we want to log weights and grads we should
        # copy the train state to host RAM, free everything but the weights on the GPU, then compute
        # gradients for logging, then copy the train state back to the GPU and resume training.
        return (
            new_state,
            loss,
            norm,
            (
                state.params["params"],
                grads["params"],
            ),
        )
    else:
        return new_state, loss, norm, None


last_checkpoint_time = None


def save_checkpoint_and_log_images(
    my_train_state,
    sample_batch_size: int,
    global_step: int,
    skip_sampling: bool,
    skip_saving: bool,
) -> None:
    save_checkpoint(my_train_state, checkpoint_manager, global_step, skip_saving)

    if not skip_sampling:
        tqdm.write("Sampling")
        sample_and_log(my_train_state, sample_batch_size, global_step)
        tqdm.write("Done sampling")
        visualization_img_names = visualization_dset["name"]
        fields = ["encoded_img"]
        if mdl.clip_conditioning:
            fields.append("clip_embedding")

        visualization_batch = get_batch(
            visualization_dset,
            len(visualization_dset),
            0,
            fields=fields,
            sharding=examples_sharding,
        )

        visualization_img_encodings = visualization_batch["encoded_img"]
        visualization_embeddings = visualization_batch.get(
            "clip_embedding", jnp.zeros((len(visualization_dset), 0))
        )
        if mdl.clip_caps:
            visualization_embeddings, visualization_max_cos_distances = gen_caps(
                jax.random.PRNGKey(0), visualization_embeddings, mdl.clip_cap_count, mdl
            )
        else:
            visualization_max_cos_distances = jnp.zeros(
                (visualization_embeddings.shape[0], 0), dtype=jnp.float32
            )
        tqdm.write("Logging attention maps")
        log_attention_maps(
            my_train_state,
            mdl,
            visualization_img_encodings[0],
            visualization_img_names[0],
            visualization_embeddings[0],
            visualization_max_cos_distances[0],
            global_step,
        )
        tqdm.write("Done logging attention maps")
        # The attention maps are very large (~3G for a single set with gpt-2-m) so we need to make
        # sure the VRAM is freed before we try training again. My kingdom for predictable memory
        # deallocation...
        gc.collect()
        tqdm.write("Logging token loss visualization")
        log_token_loss_visualization(
            my_train_state,
            mdl,
            visualization_img_encodings,
            visualization_img_names,
            visualization_embeddings,
            visualization_max_cos_distances,
            global_step,
        )
        tqdm.write("Done logging token loss visualization")
    else:
        tqdm.write("Skipping sampling")


def train_loop(
    global_step,
    train_state,
    train_imgs,
    test_imgs,
    mdl,
    training_cfg,
    total_steps,
    total_epochs,
    steps_per_epoch,
    steps_in_partial_epoch,
    sample_batch_size,
    data_offset,
    args,
    examples_sharding,
):
    """Run the training loop across epochs."""
    assert global_step < total_steps, "training run is over my dude"

    start_epoch = global_step // steps_per_epoch
    if data_offset > 0:
        print(f"Using data offset for finetuning: {data_offset} batches")
    start_step = global_step % steps_per_epoch
    tqdm.write(f"Starting at epoch {start_epoch}, step {start_step}")

    eval_loss = None
    last_checkpoint_time = None
    next_train_step_outputs = None
    signal_handler = SignalHandler()

    # In order to ensure the GPU doesn't wait on CPU stuff, we ensure that there's always at least one
    # batch in flight. So at the beginning of each inner loop we enqueue the next step before doing
    # anything that would wait for the current step to finish.
    def prefetch_and_train(
        current_state, current_step, log_weight_and_grad_interval: int = 0
    ):
        """Prefetch next batch and enqueue next train step."""

        fields = ["encoded_img"]
        if mdl.clip_conditioning:
            fields.append("clip_embedding")

        batch = get_batch(
            train_imgs,
            training_cfg.batch_size,
            # In order to support --start-where-finetune-source-left-off, we offset where we're
            # reading the data from by the amount of examples seen in the source run (expressed in
            # units of this run's batch size).
            current_step + data_offset,
            fields=fields,
            sharding=examples_sharding,
        )

        batch_imgs = batch["encoded_img"]
        batch_clips = (
            batch["clip_embedding"]
            if "clip_embedding" in batch
            else jnp.zeros((training_cfg.batch_size, 0))
        )
        if mdl.clip_conditioning:
            assert batch_clips.shape == (training_cfg.batch_size, 768)
        else:
            assert batch_clips.shape == (training_cfg.batch_size, 0)
        if mdl.clip_caps:
            caps_rng, rng = jax.random.split(current_state.rng, 2)
            current_state = current_state.replace(rng=rng)
            batch_cap_centers, batch_max_cos_distances = gen_caps(
                caps_rng, batch_clips, mdl.clip_cap_count, mdl
            )
            assert batch_cap_centers.shape == (
                batch_clips.shape[0],
                mdl.clip_cap_count,
                768,
            )
            assert batch_max_cos_distances.shape == (
                batch_clips.shape[0],
                mdl.clip_cap_count,
            )
            batch_clips = batch_cap_centers
        else:
            batch_max_cos_distances = jnp.zeros(
                (batch_clips.shape[0], 0), dtype=jnp.float32
            )
        train_state, loss, norm, weights_and_grads = train_step(
            current_state,
            batch_imgs,
            batch_clips,
            batch_max_cos_distances,
            return_weights_and_grads=log_weight_and_grad_interval > 0
            and global_step % log_weight_and_grad_interval == 0,
        )
        opt_state = train_state.opt_state

        to_log = {
            "train/loss": loss,
            "grad_global_norm": norm,
        }

        # The train state, including the optimizer state and the rng, is donated to train_step, so we
        # need to copy any values within it that we want to use after beginning the next step. Note
        # these values are *JAX arrays*, and get copied back to host RAM when needed, after this
        # function returns.
        to_log["debug/notfinite_count"] = opt_state.notfinite_count.copy()
        if training_cfg.adaptive_gradient_clip:
            to_log["debug/clipped_updates"] = opt_state.inner_state.clip_count.copy()
            clipped_last = opt_state.inner_state.clipped_last.copy()
        else:
            to_log["debug/clipped_updates"] = 0
            clipped_last = False

        # The example data is used for auxillary loss logging (eval loss, unweighted loss) so we return
        # it, even though it's not used for training outside this function.
        return (
            train_state,
            train_state.rng.copy(),
            batch_imgs,
            batch_clips,
            batch_max_cos_distances,
            to_log,
            weights_and_grads,
            clipped_last,
        )

    for epoch in trange(
        start_epoch,
        total_epochs,
        initial=start_epoch,
        total=total_epochs,
        desc="epochs",
    ):
        # If we're doing a partial epoch, set the number of batches to do
        if epoch == total_epochs - 1:
            # The number of batches for the epoch, assuming we're not resuming
            batches_for_this_epoch = (
                steps_in_partial_epoch
                if steps_in_partial_epoch is not None
                else steps_per_epoch
            )
        else:
            batches_for_this_epoch = steps_per_epoch

        this_start_step = start_step if epoch == start_epoch else 0
        actual_batches = batches_for_this_epoch - this_start_step
        this_end_step = this_start_step + actual_batches

        tqdm.write(
            f"Epoch {epoch} starting at step {this_start_step}, doing {actual_batches} steps, ending at step {this_end_step}"
        )
        with tqdm(
            total=batches_for_this_epoch,
            leave=False,
            desc="train batches",
            initial=this_start_step,
        ) as pbar:
            for batch_idx in range(actual_batches):
                if next_train_step_outputs is not None:
                    # The step we enqueued last time is now the current step
                    (
                        train_state,
                        train_rng,
                        batch_imgs,
                        batch_clips,
                        batch_max_cos_distances,
                        train_step_to_log,
                        weights_and_grads,
                        clipped_last,
                    ) = next_train_step_outputs
                    next_train_step_outputs = None
                else:
                    (
                        train_state,
                        train_rng,
                        batch_imgs,
                        batch_clips,
                        batch_max_cos_distances,
                        train_step_to_log,
                        weights_and_grads,
                        clipped_last,
                    ) = prefetch_and_train(
                        train_state, global_step, args.log_weight_and_grad_interval
                    )

                train_step_to_log["global_step"] = global_step
                if global_step % 20 == 0:
                    extra_to_log = {"global_step": global_step}
                    if (
                        training_cfg.learning_rate_schedule
                        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
                    ):
                        # Since the params used for gradient computation with a schedule-free optimizer
                        # are not the same as the params used for inference, we want to test with the
                        # inference params occasionally for charting.
                        eval_params = train_state.get_eval_params()
                        eval_loss = loss_fn(
                            eval_params,
                            train_rng,
                            batch_imgs,
                            batch_clips,
                            batch_max_cos_distances,
                        )
                        extra_to_log["eval/loss"] = eval_loss
                        del eval_params

                else:
                    extra_to_log = None
                global_step += 1
                if signal_handler.exit_requested:
                    tqdm.write("Saving checkpoint and exiting early")
                    save_checkpoint_and_log_images(
                        train_state,
                        sample_batch_size,
                        global_step,
                        skip_sampling=True,
                        skip_saving=args.skip_saving,
                    )
                    exit(0)
                # Save checkpoint every 30 minutes. This does one after step 0 too, which is nice so we
                # don't have to wait half an hour to find out if it crashes.
                if (
                    last_checkpoint_time is None
                    or (datetime.datetime.now() - last_checkpoint_time)
                    > datetime.timedelta(minutes=30)
                    or signal_handler.early_checkpoint_requested
                ):
                    save_checkpoint_and_log_images(
                        train_state,
                        sample_batch_size,
                        global_step,
                        args.skip_sampling,
                        args.skip_saving,
                    )
                    last_checkpoint_time = datetime.datetime.now()
                    signal_handler.reset_checkpoint_flag()
                if batch_idx + 1 < actual_batches:
                    next_train_step_outputs = prefetch_and_train(
                        train_state, global_step, args.log_weight_and_grad_interval
                    )
                # At this point we've enqueued all the work for this step and queued the next step, so
                # we can block on the current step and be sure the GPU will have stuff to do. There's a
                # small but real perf improvement to be had by fetching only train_step_to_log at this
                # point and waiting a bit before fetching extra_to_log.
                clipped_last, train_step_to_log, weights_and_grads = jax.device_get(
                    (clipped_last, train_step_to_log, weights_and_grads)
                )

                if not np.isfinite(train_step_to_log["train/loss"]):
                    tqdm.write(f"Loss nonfinite 😢 ({train_step_to_log['train/loss']})")
                if clipped_last:
                    tqdm.write(
                        f"Clipped update due to large gradient norm: {train_step_to_log['grad_global_norm']}"
                    )
                if train_step_to_log["debug/notfinite_count"] > 50:
                    tqdm.write(f"Too many nonfinite values in gradients, giving up")
                    exit(1)
                if weights_and_grads is not None:
                    for name, vals_tree in [
                        ("weights", weights_and_grads[0]),
                        ("grads", weights_and_grads[1]),
                    ]:
                        transformed_vals = copy(vals_tree)
                        transformed_vals[
                            "transformer_layers"
                        ] = leading_dims_to_subtrees(vals_tree["transformer_layers"])
                        log_tree = {
                            f"{name}/{k}": v
                            for k, v in jax.tree.map(
                                # NumPy histograms don't know what bf16 is
                                lambda arr: wandb.Histogram(arr.astype(np.float32)),
                                transformed_vals,
                            ).items()
                        }
                        train_step_to_log |= log_tree
                wandb.log(train_step_to_log)
                if extra_to_log is not None:
                    # If we actually got any extra stuff to log, fetch it now and do the logging.
                    extra_to_log, eval_loss = jax.device_get((extra_to_log, eval_loss))
                    wandb.log(extra_to_log)
                if (
                    training_cfg.learning_rate_schedule
                    == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
                    and eval_loss is not None  # can be None if resuming from checkpoint
                ):
                    pbar.set_postfix(
                        train_loss=f"{train_step_to_log['train/loss']:.4f}",
                        eval_loss=f"{eval_loss:.4f}",
                    )
                else:
                    pbar.set_postfix(
                        train_loss=f"{train_step_to_log['train/loss']:.4f}"
                    )
                pbar.update()

        # Save checkpoint at end of epoch
        save_checkpoint_and_log_images(
            train_state,
            sample_batch_size,
            global_step,
            skip_sampling=False,
            skip_saving=args.skip_saving,
        )
        last_checkpoint_time = datetime.datetime.now()

        # Evaluate on test set
        losses = []
        eval_params = train_state.get_eval_params()
        test_rng = jax.random.PRNGKey(7357)
        for batch_idx in trange(
            len(test_imgs) // training_cfg.batch_size,
            desc="test batches",
        ):
            dropout_rng, cap_rng, test_rng = jax.random.split(test_rng, 3)
            fields = ["encoded_img"]
            if mdl.clip_conditioning:
                fields.append("clip_embedding")

            batch = get_batch(
                test_imgs,
                training_cfg.batch_size,
                batch_idx,
                fields=fields,
                sharding=examples_sharding,
            )

            batch_imgs = batch["encoded_img"]
            batch_clips = batch.get(
                "clip_embedding", jnp.zeros((training_cfg.batch_size, 0))
            )
            if mdl.clip_caps:
                batch_clips, batch_max_cos_distances = gen_caps(
                    cap_rng, batch_clips, mdl.clip_cap_count, mdl
                )
            else:
                batch_max_cos_distances = jnp.zeros(
                    (batch_clips.shape[0], 0), dtype=jnp.float32
                )
            losses.append(
                loss_fn(
                    eval_params,
                    dropout_rng,
                    batch_imgs,
                    batch_clips,
                    batch_max_cos_distances,
                )
            )
        del test_rng
        del eval_params
        test_loss = jnp.mean(jnp.stack(losses))
        wandb.log({"global_step": global_step, "test/loss": test_loss})
        tqdm.write(
            f"Epoch {epoch} done, train loss: {train_step_to_log['train/loss']:.4f}, test loss {test_loss:.4f}"
        )

    return global_step, train_state


cap_logits_table = cap_sampling.LogitsTable(767, 16384)


@partial(jax.jit, static_argnames=["n_caps", "model"])
def gen_caps(rng, batch_clips, n_caps, model):
    """Generate containing spherical caps for a batch of examples."""
    ex_rngs = jax.random.split(rng, batch_clips.shape[0])

    cap_centers, cap_max_cos_distances = jax.vmap(
        lambda rng, embedding: model.gen_training_caps(cap_logits_table, rng, embedding)
    )(ex_rngs, batch_clips)
    assert cap_centers.shape == (batch_clips.shape[0], n_caps, 768)
    assert cap_max_cos_distances.shape == (batch_clips.shape[0], n_caps)
    return cap_centers, cap_max_cos_distances


if __name__ == "__main__":
    args = parse_arguments()

    setup_jax_for_training()
    setup_profiling_server(args.profiling_server)

    wandb_settings = wandb.Settings(code_dir="txt2img_unsupervised")

    model_cfg, training_cfg, _ = init_wandb_training(
        args.resume, args.model_config, args.training_config, args, wandb_settings
    )

    train_imgs, test_imgs = load_dataset(args.pq_dir)

    (
        steps_per_epoch,
        total_steps,
        complete_epochs,
        total_epochs,
        steps_in_partial_epoch,
    ) = plan_steps(
        train_set_size=train_imgs.shape[0],
        batch_size=training_cfg.batch_size,
        epochs=training_cfg.epochs,
        examples=training_cfg.training_images,
        steps=0,  # We don't use the steps parameter directly
    )

    print(
        f"Training for {total_steps * training_cfg.batch_size} images in {total_steps} steps over {complete_epochs} full epochs plus {steps_in_partial_epoch if steps_in_partial_epoch is not None else 0} extra batches"
    )

    (
        global_step,
        sample_batch_size,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    ) = init_train_state(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        total_steps=total_steps,
        resume_checkpoint_path=args.resume,
        finetune_checkpoint_path=args.finetune,
        sample_batch_size=args.sample_batch_size,
        start_where_finetune_source_left_off=args.start_where_finetune_source_left_off,
    )

    mesh = setup_sharding(training_cfg.batch_size)
    train_state = train_state.replicate_for_multi_gpu(mesh)
    examples_sharding = NamedSharding(mesh, PartitionSpec("dev"))

    loss_fn = jax.jit(partial(transformer_model.loss_batch, mdl))

    image_prompts_to_sample = 8
    clip_mdl, clip_processor = get_clip_mdl()

    sfw_indices = find_sfw_indices(
        clip_mdl, clip_processor, test_imgs[:2048], image_prompts_to_sample
    )
    visualization_dset = test_imgs.select(sfw_indices[:8])
    print(
        f"Using {[visualization_dset['name'][i] for i in range(8)]} for visualization & sampling"
    )
    text_prompt_clips = get_clip_text_embeddings_to_sample(8, clip_mdl, clip_processor)

    del clip_mdl, clip_processor

    ae_cfg, ae_mdl, ae_params_torch = load_autoencoder(args.ae_cfg, args.ae_ckpt)

    global_step, train_state = train_loop(
        global_step,
        train_state,
        train_imgs,
        test_imgs,
        mdl,
        training_cfg,
        total_steps,
        total_epochs,
        steps_per_epoch,
        steps_in_partial_epoch,
        sample_batch_size,
        data_offset,
        args,
        examples_sharding,
    )
