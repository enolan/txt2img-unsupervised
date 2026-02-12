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
    IntervalTimer,
    leading_dims_to_subtrees,
    load_dataset,
    plan_steps,
    save_checkpoint,
    setup_common_arguments,
    setup_jax_for_training,
    setup_profiling_server,
    setup_sharding,
    SignalHandler,
    train_loop,
)
from txt2img_unsupervised.training_visualizations import (
    log_attention_maps,
    log_token_loss_visualization,
)
import txt2img_unsupervised.cap_sampling as cap_sampling
import txt2img_unsupervised.sample as sample
import txt2img_unsupervised.transformer_model as transformer_model
import txt2img_unsupervised.training_infra as training_infra


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
    text_tokens = clip_processor(text=prompts, return_tensors="jax", padding=True)
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


# Loss function that processes batches and handles cap generation
def loss_fn(params, batch, rng, mdl=None):
    dropout_rng, caps_rng = jax.random.split(rng)

    batch_imgs = batch["batch_imgs"]
    batch_clips = batch["batch_clips"]

    # Generate caps if needed
    if mdl.clip_caps:
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

    # Call the model's loss function
    return transformer_model.loss_batch(
        mdl,
        params,
        dropout_rng,
        batch_imgs,
        batch_clips,
        batch_max_cos_distances,
    )


# Fast path hook that runs operations that don't need the full train state
def fast_post_step_hook(loss, metrics, global_step, norm):
    to_log = {
        "train/loss": loss,
        "grad_global_norm": norm,
        "global_step": global_step,
    }

    # Add metrics that were copied before donation
    if "notfinite_count" in metrics:
        to_log["debug/notfinite_count"] = metrics["notfinite_count"]
    if "clip_count" in metrics:
        to_log["debug/clipped_updates"] = metrics["clip_count"]
    else:
        to_log["debug/clipped_updates"] = 0

    # Log warnings based on metrics
    if not np.isfinite(loss):
        tqdm.write(f"Loss nonfinite 😢 ({loss})")

    if metrics.get("notfinite_count", 0) > 50:
        tqdm.write(f"Too many nonfinite values in gradients, giving up")
        exit(1)

    if metrics.get("clipped_last", False):
        tqdm.write(f"Clipped update due to large gradient norm: {norm}")

    # Log to wandb - this will materialize the JAX arrays, but that's OK
    # because we've already enqueued the next step
    wandb.log(to_log)


# Slow path hook that runs operations that need the full train state
def slow_post_step_hook(loss, state, global_step, norm):
    if signal_handler.exit_requested:
        tqdm.write("Saving checkpoint and exiting early")
        save_checkpoint_and_log_images(
            state,
            sample_batch_size,
            global_step,
            skip_sampling=True,
            skip_saving=args.skip_saving,
        )
        exit(0)

    # If we got a signal to save a checkpoint, do so. If we didn't, save a checkpoint only if it's
    # time.
    if signal_handler.early_checkpoint_requested:
        checkpoint_timer.run_and_reset(
            state,
            global_step,
            skip_sampling=args.skip_sampling,
            skip_saving=args.skip_saving,
        )
        signal_handler.reset_checkpoint_flag()
    else:
        checkpoint_timer.check_and_run(state, global_step)

    # Schedule-free specific logging, stubbed out. I'm going to drop support for schedule free
    # optimizers soon, so I'm not bothering to make this work.
    if (
        global_step % 20 == 0
        and training_cfg.learning_rate_schedule
        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
    ):
        eval_params = state.get_eval_params()
        # TODO: Add evaluation metrics logging for schedule-free optimizer
        del eval_params

    # Always continue training unless explicitly exited above
    return False


# Function to determine if we need to run the slow path
def slow_path_condition(global_step):
    if signal_handler.exit_requested or signal_handler.early_checkpoint_requested:
        return True
    if checkpoint_timer.check_if_should_run():
        return True
    if (
        global_step % 20 == 0
        and training_cfg.learning_rate_schedule
        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
    ):
        return True
    return False


def post_epoch_hook(state, epoch_idx, global_step):
    checkpoint_timer.run_and_reset(
        state, global_step, skip_sampling=False, skip_saving=args.skip_saving
    )
    losses = []
    eval_params = state.get_eval_params()
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
            eval_loss_fn(
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
    tqdm.write(f"Epoch {epoch_idx} done, test loss {test_loss:.4f}")


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

    # Create checkpoint timer and signal handler
    checkpoint_timer = IntervalTimer(
        datetime.timedelta(minutes=30),
        lambda state, step, **kwargs: save_checkpoint_and_log_images(
            state,
            sample_batch_size,
            step,
            skip_sampling=kwargs.get("skip_sampling", args.skip_sampling),
            skip_saving=kwargs.get("skip_saving", args.skip_saving),
        ),
    )
    signal_handler = SignalHandler()

    # Create loss function for training
    jitted_loss_fn = jax.jit(partial(loss_fn, mdl=mdl))

    # Define eval loss function for post-epoch hook
    eval_loss_fn = jax.jit(partial(transformer_model.loss_batch, mdl))

    # Use the generic train loop with async execution
    train_state, global_step = train_loop(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        complete_epochs=complete_epochs,
        total_epochs=total_epochs,
        steps_in_partial_epoch=steps_in_partial_epoch,
        initial_step=global_step,
        initial_train_state=train_state,
        get_batch_fn=lambda step: (
            lambda b: {
                "batch_imgs": b["encoded_img"],
                "batch_clips": b.get(
                    "clip_embedding", jnp.zeros((training_cfg.batch_size, 0))
                ),
            }
        )(
            get_batch(
                train_imgs,
                training_cfg.batch_size,
                step + data_offset,
                fields=["encoded_img"]
                + (["clip_embedding"] if mdl.clip_conditioning else []),
                sharding=examples_sharding,
            )
        ),
        loss_fn=jitted_loss_fn,
        post_step_hook_fn=None,  # Not used with async implementation
        post_epoch_hook_fn=post_epoch_hook,
        fast_post_step_hook_fn=fast_post_step_hook,
        slow_post_step_hook_fn=slow_post_step_hook,
        slow_path_condition_fn=slow_path_condition,
    )

    # Only save a final checkpoint if not at an epoch boundary
    if global_step % steps_per_epoch != 0:
        save_checkpoint_and_log_images(
            train_state,
            sample_batch_size,
            global_step,
            skip_sampling=False,
            skip_saving=args.skip_saving,
        )
