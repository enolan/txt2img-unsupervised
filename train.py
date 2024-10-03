"""Train the image model."""

import os

# neccessary to use more of the GPU's memory. Default is 0.75. It's supposed to be able to
# dynamically allocate more, but there are fragmentation issues since we allocate ginormous arrays.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import datasets
import datetime
import flax.core
import gc
import importlib.util
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optax  # type:ignore[import]
import optax.contrib
import orbax.checkpoint as ocp
import PIL.Image
import signal
import torch
import transformers
import wandb
from copy import copy
from datasets import Dataset
from distutils.util import strtobool
from einops import rearrange, reduce, repeat
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import OmegaConf
from pathlib import Path
from sys import exit
from tqdm import tqdm, trange
from typing import Any, Callable, Tuple

from txt2img_unsupervised.checkpoint import (
    mk_checkpoint_manager,
    setup_checkpoint_manager_and_initial_state,
    setup_optimizer,
    TrainState,
)
from txt2img_unsupervised.config import (
    LearningRateSchedule,
    ModelConfig,
    TrainingConfig,
    str_to_activation,
    str_to_dtype,
    str_to_learning_rate_schedule,
)
from txt2img_unsupervised.ldm_autoencoder import LDMAutoencoder
from txt2img_unsupervised.load_pq_dir import load_pq_dir
from txt2img_unsupervised.training_visualizations import (
    log_attention_maps,
    log_token_loss_visualization,
)
import txt2img_unsupervised.config as config
import txt2img_unsupervised.sample as sample
import txt2img_unsupervised.transformer_model as transformer_model


parser = argparse.ArgumentParser()


def argparse_from_dict(d: dict[str, Any]) -> Callable[[str], Any]:
    """Create an argparse argument type from a dictionary."""

    def f(x: str) -> Any:
        if x in d:
            return d[x]
        else:
            raise argparse.ArgumentTypeError(f"Unknown value {x}")

    return f


parser.add_argument("--pq-dir", type=Path, required=True)
parser.add_argument("--model-config", type=Path, required=True)
parser.add_argument("--training-config", type=Path, required=True)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--sample-batch-size", type=int, default=None)
parser.add_argument(
    "--profiling-server", action="store_true", help="Enable JAX profiling server"
)
parser.add_argument("--epochs", type=int)
parser.add_argument("--training-images", type=int)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument(
    "--learning-rate-schedule", type=argparse_from_dict(str_to_learning_rate_schedule)
)
parser.add_argument("--warmup-steps", type=int, default=None)
parser.add_argument("--schedule-free-beta1", type=float, default=None)
parser.add_argument("--gradient-accumulation-steps", type=int)
parser.add_argument("--use-biases", type=lambda x: bool(strtobool(x)))
parser.add_argument("--gradient-clipping", type=float, default=None)
parser.add_argument("--loss-decay-constant", type=float, default=1.0)
parser.add_argument("--image-dropout", type=float, default=None)
parser.add_argument("--ae-cfg", type=Path, required=True)
parser.add_argument("--ae-ckpt", type=Path, required=True)
parser.add_argument("--activations-dtype", type=argparse_from_dict(str_to_dtype))
parser.add_argument("--activation-function", type=argparse_from_dict(str_to_activation))
parser.add_argument("--clip-conditioning", type=lambda x: bool(strtobool(x)))
parser.add_argument("--clip-caps", type=lambda x: bool(strtobool(x)))
parser.add_argument("--clip-cap-count", type=int)
parser.add_argument("--resume", type=Path)
parser.add_argument("--finetune", type=Path)
parser.add_argument(
    "--start-where-finetune-source-left-off",
    type=lambda x: bool(strtobool(x)),
    help="start the training data from where the finetune source run left off",
    default=False,
)
parser.add_argument(
    "--skip-sampling", action="store_true", help="Skip sampling during training"
)
args, _unknown = parser.parse_known_args()


def setup_profiling_server():
    if args.profiling_server:
        if importlib.util.find_spec("tensorflow") is None:
            print("You gotta install tensorflow for profiling bro")
            exit(1)

        jax.profiler.start_server(6969)
        print("JAX profiling server started on port 6969")


setup_profiling_server()

# Turn on JAX compilation cache
jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")


def json_pretty(dict):
    """Print a dictionary as pretty JSON."""
    return json.dumps(dict, indent=2)


def init_train_state():
    """Set up our ModelConfig and TrainingConfig, initialize wandb, and create our initial
    TrainState."""
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        keep_time_interval=datetime.timedelta(hours=6),
        # Async checkpointing can hide out of disk errors, so we disable it.
        enable_async_checkpointing=False,
    )
    wandb_settings = wandb.Settings(code_dir="txt2img_unsupervised")
    assert ((args.resume is None) != (args.finetune is None)) or (
        args.resume is None and args.finetune is None
    ), "Must specify one of --resume or --finetune or neither, not both"
    if args.resume is not None:
        print(f"Resuming from checkpoint {args.resume}...")
        checkpoint_dir = args.resume.absolute()
        checkpoint_manager = mk_checkpoint_manager(checkpoint_dir)
        # Checkpoint saved after step n, so we start at step n+1
        global_step = checkpoint_manager.latest_step() + 1
        metadata = checkpoint_manager.metadata()
        model_cfg = ModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
        run_id = metadata["run_id"]
        data_step_offset = metadata.get("data_step_offset", 0)
        print(f"Resuming run {run_id}")
        print(
            "ALL TRAINING AND MODEL PARAMETERS PASSED ON THE COMMAND LINE WILL BE IGNORED."
        )
        print(f"ModelConfig {json_pretty(model_cfg.to_json_dict())}")
        print(f"TrainingConfig {json_pretty(training_cfg.to_json_dict())}")
        wandb.init(id=run_id, resume="must", settings=wandb_settings)

        train_state, mdl = TrainState.load_from_checkpoint(
            checkpoint_manager,
            global_step - 1,
            1,  # batches total will be update after we load the dataset
        )
    else:
        print("Starting new run...")
        wandb.init(settings=wandb_settings)

        global_step = 0
        # Load model configuration
        with open(args.model_config) as f:
            model_cfg = ModelConfig.from_json_dict(json.load(f))
        config.merge_attrs(model_cfg, args)
        # Load training configuration
        with open(args.training_config) as f:
            training_cfg = TrainingConfig.from_json_dict(json.load(f))
        config.merge_attrs(training_cfg, args)

        # Send config to wandb
        wandb.config.update(model_cfg.to_json_dict())
        wandb.config.update(training_cfg.to_json_dict())

        # Read potentially sweep-controlled parameters from wandb
        model_cfg = ModelConfig.from_json_dict(wandb.config.as_dict())
        training_cfg = TrainingConfig.from_json_dict(wandb.config.as_dict())
        print(f"Model config post-wandb: {json_pretty(model_cfg.to_json_dict())}")
        print(f"Training config post-wandb: {json_pretty(training_cfg.to_json_dict())}")

        assert (
            0 < training_cfg.loss_decay_constant <= 1
        ), "loss_decay_constant must be in (0, 1]"

        if args.finetune is not None:
            finetune_src_checkpoint_dir = args.finetune.absolute()
            finetune_src_checkpoint_manager = mk_checkpoint_manager(
                finetune_src_checkpoint_dir
            )
            if args.start_where_finetune_source_left_off:
                # data_step_offset is offset (in batches) we use into the training data, equal to
                # the number of batches the source run trained on. We track this number so we don't
                # repeat any training data the source run saw before we have to.
                data_step_offset = finetune_src_checkpoint_manager.latest_step() + 1
            else:
                data_step_offset = 0
            extra_metadata = (
                "finetune_src_config",
                finetune_src_checkpoint_manager.metadata(),
            )
        else:
            data_step_offset = 0
            extra_metadata = None
        # Set up checkpoint manager and initial state

        checkpoint_dir = Path(f"checkpoints/{wandb.run.id}").absolute()
        checkpoint_manager, train_state = setup_checkpoint_manager_and_initial_state(
            checkpoint_options,
            checkpoint_dir,
            wandb.run.id,
            model_cfg,
            training_cfg,
            jax.random.PRNGKey(1337),
            1,  # We don't know the total number of batches until we load the dataset
            data_step_offset=data_step_offset,
            extra_metadata=extra_metadata,
        )

        if args.finetune is not None:
            train_state = train_state.replace(params=None)
            gc.collect()
            print(f"Loading params from {args.finetune} for finetuning...")

            finetune_src_ts, finetune_src_mdl = TrainState.load_from_checkpoint(
                finetune_src_checkpoint_manager,
                finetune_src_checkpoint_manager.latest_step(),
                1,
            )
            finetune_src_params = finetune_src_ts.get_eval_params()
            del finetune_src_ts
            gc.collect()

            train_state = train_state.replace(params=finetune_src_params)
            print(
                "Finetuning parameters loaded successfully. Overwriting initial checkpoint."
            )
            train_state.save_checkpoint(checkpoint_manager, 0)
            print("Done.")
            wandb.config.update(
                {"finetune_src_metadata": finetune_src_checkpoint_manager.metadata()}
            )

        mdl = transformer_model.ImageModel(**model_cfg.__dict__)

    train_state = train_state.replicate_for_multi_gpu()
    if args.sample_batch_size is None:
        sample_batch_size = training_cfg.batch_size
    else:
        sample_batch_size = args.sample_batch_size

    print(mdl.tabulate(jax.random.PRNGKey(0), *mdl.dummy_inputs()))

    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("test/loss", summary="last")
    wandb.define_metric("train/loss", summary="last")

    return (
        global_step,
        model_cfg,
        training_cfg,
        sample_batch_size,
        checkpoint_manager,
        train_state,
        mdl,
        data_step_offset,
    )


(
    global_step,
    model_cfg,
    training_cfg,
    sample_batch_size,
    checkpoint_manager,
    train_state,
    mdl,
    data_step_offset,
) = init_train_state()


def load_dataset(dir: Path) -> Tuple[Dataset, Dataset]:
    # The paths don't necessarily come out in the same order on every machine, so we sort to make the
    # example order consistent.
    dset_all = load_pq_dir(dir)
    dset_split = dset_all.train_test_split(test_size=0.01, seed=19900515)
    train_imgs = dset_split["train"]
    test_imgs = dset_split["test"]
    print(f"Train set {train_imgs.shape}, test set {test_imgs.shape}")
    wandb.config.update(
        {"train_set_size": len(train_imgs), "test_set_size": len(test_imgs)}
    )
    return train_imgs, test_imgs


train_imgs, test_imgs = load_dataset(args.pq_dir)
# Make the RNG partitionable across devices
jax.config.update("jax_threefry_partitionable", True)


def calculate_steps(train_set_size, training_cfg, train_state):
    """Calculate the number of steps and epochs to do."""

    batches_for_image_count = training_cfg.training_images // training_cfg.batch_size
    batches_per_epoch = train_set_size // training_cfg.batch_size
    image_count_epochs = batches_for_image_count // batches_per_epoch
    extra_batches = batches_for_image_count % batches_per_epoch
    epochs_total = (
        image_count_epochs + training_cfg.epochs + (1 if extra_batches > 0 else 0)
    )
    batches_total = (
        training_cfg.epochs + image_count_epochs
    ) * batches_per_epoch + extra_batches
    assert epochs_total > 0, "Can't train for 0 steps"

    print(
        f"Training for {batches_total * training_cfg.batch_size} images in {batches_total} steps over {image_count_epochs + training_cfg.epochs} full epochs plus {extra_batches} extra batches"
    )
    # Reconfigure optimizer now that we know the total number of batches
    opt = setup_optimizer(training_cfg, batches_total)
    opt_state = opt.init(train_state.params)
    train_state = train_state.replace(opt_state=opt_state, tx=opt)
    return batches_total, epochs_total, batches_per_epoch, extra_batches, train_state


(
    batches_total,
    epochs_total,
    batches_per_epoch,
    extra_batches,
    train_state,
) = calculate_steps(train_imgs.shape[0], training_cfg, train_state)


def setup_sharding():
    print(
        f"Sharding batches of {training_cfg.batch_size} across {jax.device_count()} devices, {training_cfg.batch_size / jax.device_count()} per device"
    )
    assert training_cfg.batch_size % jax.device_count() == 0
    # NamedSharding is overkill for the simple batch parallelism we do, but it's necessary to get
    # orbax to save checkpoints correctly.
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    return mesh


mesh = setup_sharding()


loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
loss_fn = jax.jit(
    partial(transformer_model.loss_batch, mdl), static_argnames=["loss_decay_constant"]
)


# TODO delete this, unnecessary now
# Set up for sampling:
sample_cfg = copy(model_cfg)
sample_cfg.dropout = None
sample_mdl = transformer_model.ImageModel(**sample_cfg.__dict__, decode=True)
sample_params = sample_mdl.init(jax.random.PRNGKey(0), *sample_mdl.dummy_inputs())
sample_cache = sample_params["cache"]
del sample_params

sample_jv = jax.jit(
    jax.vmap(
        lambda params, clip_embedding, max_cos_distance, rng, top_p: transformer_model.sample(
            sample_mdl,
            flax.core.copy(params, {"cache": sample_cache}),
            clip_embedding,
            max_cos_distance,
            rng,
            top_p,
        ),
        in_axes=(None, 0, 0, 0, 0),
    )
)

# CLIP embeddings to use for diagnostic sampling
image_prompts_to_sample = 8


def get_clip_mdl():
    """Get the CLIP model and processor."""
    # The model must match the one used in the preprocessor
    clip_mdl_name = "openai/clip-vit-large-patch14"
    clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name)
    clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)
    return clip_mdl, clip_processor


clip_mdl, clip_processor = get_clip_mdl()


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


def get_image_prompt_clip_embeddings(indices: np.ndarray, dset: Dataset) -> jax.Array:
    """Find n hopefully SFW CLIP embeddings to use for diagnostic sampling."""

    print(f"Using {[dset['name'][i] for i in indices]} for diagnostic sampling")
    return {
        "clip_embedding": dset["clip_embedding"][indices],
        "name": dset["name"][indices],
    }


def get_clip_text_embeddings_to_sample(n: int) -> jax.Array:
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


sfw_indices = find_sfw_indices(
    clip_mdl, clip_processor, test_imgs[:2048], image_prompts_to_sample
)
image_prompt_clips = get_image_prompt_clip_embeddings(sfw_indices, test_imgs[:2048])
text_prompt_clips = get_clip_text_embeddings_to_sample(8)

del clip_mdl, clip_processor


def mk_image_prompt_conditions(image_prompt_clips, grid_size):
    """Make conditioning data for the image prompts."""
    assert image_prompt_clips.shape[1] == 768
    # Use a larger and a smaller cap. The smaller cap should generate images that are very
    # similar to the image prompt, the larger one should generate images that are semantically
    # similar but not necessarily visually similar.
    if model_cfg.clip_caps:
        max_cos_distance_choices = jnp.array([0.75, 0.4], dtype=jnp.float32)

        samples_count = (
            len(image_prompt_clips) * len(max_cos_distance_choices) * grid_size
        )

        # With more than one cap slot, we fill in the unused ones with caps that cover the entire
        # space.
        cap_centers_fill, cap_max_cos_distances_fill = sample.mk_filler_caps(
            model_cfg, samples_count, 1, jax.random.PRNGKey(20240214)
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
            model_cfg.clip_cap_count,
            768,
        )
        assert cap_max_cos_distances.shape == (
            len(image_prompt_clips),
            len(max_cos_distance_choices),
            grid_size,
            model_cfg.clip_cap_count,
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
    if model_cfg.clip_caps:
        cap_centers_fill, cap_max_cos_distances_fill = sample.mk_filler_caps(
            model_cfg, samples_count, 1, jax.random.PRNGKey(20240214)
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
            model_cfg.clip_cap_count,
            768,
        )
        assert cap_max_cos_distances.shape == (
            len(text_prompt_clips),
            grid_size,
            model_cfg.clip_cap_count,
        )
        return cap_centers, cap_max_cos_distances
    else:
        clip_embeddings = repeat(
            text_prompt_clips, "prompt clip -> prompt g clip", g=grid_size
        )
        assert clip_embeddings.shape == (len(text_prompt_clips), grid_size, 768)
        return clip_embeddings, None


ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)
# don't keep these on the GPU when we're not using them
ae_params_torch = torch.load(args.ae_ckpt, map_location="cpu")


def sample_and_log(ts: TrainState, sample_batch_size: int, global_step: int) -> None:
    """Sample from the model and log to wandb."""

    ae_params = LDMAutoencoder.params_from_torch(ae_params_torch, ae_cfg)
    eval_params = ts.get_eval_params()

    if model_cfg.clip_conditioning:
        # Create a grid of samples for each set of conditions.
        grid_size = 9

        image_prompt_names = image_prompt_clips["name"]
        text_prompt_texts = text_prompt_clips["name"]

        if model_cfg.clip_caps:
            (
                img_cap_centers,
                img_max_cos_distances,
                img_max_cos_distance_choices,
            ) = mk_image_prompt_conditions(
                image_prompt_clips["clip_embedding"], grid_size
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
                image_prompt_clips["clip_embedding"], grid_size
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
                model_cfg,
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
        # Sample with different top_p values
        top_ps = [0.2, 0.6, 0.8, 0.9, 0.95]

        def round_down_to_multiples(x, a, b):
            # Round something down so it's a multiple of a and b
            less = min(a, b)
            more = max(a, b)
            while True:
                if x % less != 0:
                    x -= x % less
                if x % more != 0:
                    x -= x % more
                if x % less == 0 and x % more == 0:
                    break
                assert x > 0
            return x

        imgs_to_sample = min(80, training_cfg.batch_size)
        imgs_to_sample = round_down_to_multiples(
            imgs_to_sample, len(top_ps), jax.device_count()
        )
        img_top_ps = jnp.concatenate(
            [jnp.repeat(p, imgs_to_sample // len(top_ps)) for p in top_ps]
        )
        assert len(img_top_ps) % jax.device_count() == 0
        img_top_ps = jax.device_put(img_top_ps, sharding_1d)

        rngs = jax.device_put(jax.random.split(ts.rng, imgs_to_sample), sharding_2d)

        clip_embeddings = jnp.zeros((imgs_to_sample, 0), dtype=jnp.float32)

        samples = sample_jv(eval_params, clip_embeddings, rngs, img_top_ps)

        decoded = sample.decode_jv(ae_mdl, ae_params, samples)
        decoded = np.array(decoded)

        decoded_grouped = rearrange(decoded, "(p n) h w c -> p n h w c", p=len(top_ps))
        to_log = {
            f"samples/top-p-{p:.02f}": [
                wandb.Image(PIL.Image.fromarray(np.array(img))) for img in imgs
            ]
            for p, imgs in zip(top_ps, decoded_grouped)
        }
        to_log["global_step"] = global_step
        wandb.log(to_log)


last_cap_set = None  # We track the last one so we can print when it changes


def rearrange_batch_caps(
    centers, max_cos_distances, model_cap_count, epoch, is_test=False
):
    """Rearrange the cap data from parquet into the shape expected by the model and select the
    appropriate subset of the caps for the current epoch. For reasons, parquet does not support
    multidimensional arrays. Prints a message when the cap set changes, unless is_test is True, in
    which case it doesn't print anything."""
    assert len(centers.shape) == 2, f"centers shape {centers.shape}"
    batch_size = centers.shape[0]
    assert centers.shape[1] % 768 == 0
    dset_cap_count = centers.shape[1] // 768

    # normalize max_cos_distances shape, one of the dimensions should be dset_cap_count
    if dset_cap_count == 1:
        assert max_cos_distances.shape == (
            batch_size,
        ), f"max_cos_distances batch shape {max_cos_distances.shape}, should be unidimensional if the dataset contains only a sinple cap"
        max_cos_distances = rearrange(max_cos_distances, "b -> b 1")

    assert max_cos_distances.shape[0] == batch_size
    assert max_cos_distances.shape[1] == dset_cap_count

    centers = rearrange(centers, "b (n c) -> b n c", n=dset_cap_count, c=768)

    assert centers.shape[1] == max_cos_distances.shape[1]
    dset_cap_count = centers.shape[1]
    assert (
        dset_cap_count % model_cap_count == 0
    ), f"Dataset cap count {dset_cap_count} not divisible by model cap count {model_cap_count}"

    # This just feeds all the non-overlapping contiguous subsequences of the dataset caps to the
    # model sequentially - one subsequence per epoch. Ideally we'd go through all the permutations
    # in a smart order - non-overlapping sets first, but that's more complicated.
    distinct_cap_sets = dset_cap_count // model_cap_count
    this_cap_set = epoch % distinct_cap_sets
    this_cap_set_start = this_cap_set * model_cap_count
    this_cap_set_end = this_cap_set_start + model_cap_count

    if not is_test:
        global last_cap_set
        if last_cap_set != this_cap_set:
            tqdm.write(
                f"Switching to cap set #{this_cap_set} of {distinct_cap_sets} total - {this_cap_set_start}:{this_cap_set_end}"
            )
            last_cap_set = this_cap_set

    centers = centers[:, this_cap_set_start:this_cap_set_end, :]
    max_cos_distances = max_cos_distances[:, this_cap_set_start:this_cap_set_end]

    return centers, max_cos_distances


@partial(jax.jit, donate_argnames=["state"], static_argnames=["loss_decay_constant"])
def train_step(
    state: TrainState,
    loss_decay_constant: float,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    dropout_rng, rng2 = jax.random.split(state.rng, 2)
    loss, grads = loss_grad_fn(
        mdl,
        state.params,
        loss_decay_constant,
        dropout_rng,
        batch_imgs,
        batch_clips,
        batch_max_cos_distances,
    )
    new_state = state.apply_gradients(
        grads=grads, rng=rng2
    )  # type:ignore[no-untyped-call]
    norm = optax.global_norm(grads)
    return new_state, loss, norm


last_checkpoint_time = None


def save_checkpoint_and_log_images(
    my_train_state, sample_batch_size, global_step, skip_sampling
) -> None:
    # TPU VMs run out of disk space a lot. Retrying in a loop lets me manually clean up the disk
    my_train_state.save_checkpoint(checkpoint_manager, global_step)
    tqdm.write("Saved checkpoint")
    if not skip_sampling:
        tqdm.write("Sampling")
        sample_and_log(my_train_state, sample_batch_size, global_step)
        tqdm.write("Done sampling")
        visualization_imgs = test_imgs[sfw_indices[:8]]
        visualization_img_names = visualization_imgs["name"]
        visualization_img_encodings = visualization_imgs["encoded_img"]
        if mdl.clip_conditioning and mdl.clip_caps:
            (
                visualization_embeddings,
                visualization_max_cos_distances,
            ) = rearrange_batch_caps(
                visualization_imgs["cap_center"],
                visualization_imgs["cap_max_cos_distance"],
                mdl.clip_cap_count,
                epoch=0,
                is_test=True,
            )
        elif mdl.clip_conditioning and not mdl.clip_caps:
            visualization_embeddings = visualization_imgs["clip_embedding"]
            visualization_max_cos_distances = jnp.zeros((len(visualization_imgs), 0))
        else:
            visualization_embeddings = jnp.zeros((len(visualization_imgs), 0))
            visualization_max_cos_distances = jnp.zeros((len(visualization_imgs), 0))
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


assert global_step < batches_total, "training run is over my dude"
# to support --start-where-finetune-source-left-off, we need to track the progress in the *run*
# separately from the progress through the dataset, which will be offset.
start_epoch = global_step // batches_per_epoch
start_data_epoch = (global_step + data_step_offset) // batches_per_epoch
if data_step_offset > 0:
    print(
        f"Using data offset for finetuning: epoch {start_data_epoch}, step {data_step_offset % batches_per_epoch}"
    )
start_step = global_step % batches_per_epoch
tqdm.write(f"Starting at epoch {start_epoch}, step {start_step}")

examples_sharding = NamedSharding(mesh, PartitionSpec("dev"))

eval_loss = None

exit_requested = False


def exit_early_signal_handler(signum, frame):
    # It's important to checkpoint and exit at well-defined times (not in the middle of a train
    # step), so we use a flag and check for it rather than exiting immediately upon recieving the
    # signal.
    global exit_requested
    exit_requested = True


signal.signal(signal.SIGTERM, exit_early_signal_handler)
signal.signal(signal.SIGINT, exit_early_signal_handler)

for epoch in trange(
    start_epoch,
    epochs_total,
    initial=start_epoch,
    total=epochs_total,
    desc="epochs",
):
    # We use a consistent shuffle so resuming gets the same ordering.

    # Separately, to support --start-where-finetune-source-left-off, we concatenate the portion of
    # the epoch that the source run didn't see with a shuffled version of the portion of the epoch
    # it did see. So if the source run did 200 steps on a dataset with 300 batches per epoch, our
    # first epoch is the 100 batches the source would've seen if it continued plus the 200 batches
    # it did see.
    intra_epoch_offset = data_step_offset % batches_per_epoch
    shuffled_train_imgs = train_imgs.shuffle(seed=start_data_epoch)
    epoch_part_1 = shuffled_train_imgs.select(
        range(intra_epoch_offset, len(train_imgs))
    )
    epoch_part_2 = shuffled_train_imgs.select(
        range(len(train_imgs) - intra_epoch_offset)
    ).shuffle(seed=start_data_epoch + 1)
    shuffled_train_imgs = datasets.concatenate_datasets([epoch_part_1, epoch_part_2])

    # If we're doing a partial epoch, set the number of batches to do
    if epoch == epochs_total - 1:
        # The number of batches for the epoch, assuming we're not resuming
        batches_for_this_epoch = (
            extra_batches if extra_batches > 0 else batches_per_epoch
        )
    else:
        batches_for_this_epoch = batches_per_epoch

    this_start_step = start_step if epoch == start_epoch else 0
    actual_batches = batches_for_this_epoch - this_start_step
    this_end_step = this_start_step + actual_batches

    # Much faster to skip leading examples this way than with islice
    shuffled_train_imgs = shuffled_train_imgs.select(
        range(
            this_start_step * training_cfg.batch_size,
            this_end_step * training_cfg.batch_size,
        )
    )
    iter = shuffled_train_imgs.iter(
        batch_size=training_cfg.batch_size, drop_last_batch=True
    )
    tqdm.write(
        f"Epoch {epoch} starting at step {this_start_step}, doing {actual_batches} steps, ending at step {this_end_step}"
    )
    with tqdm(
        total=batches_for_this_epoch,
        leave=False,
        desc="train batches",
        initial=this_start_step,
    ) as pbar:
        for batch in iter:
            batch_imgs = jax.device_put(batch["encoded_img"], examples_sharding)
            if model_cfg.clip_conditioning and not model_cfg.clip_caps:
                batch_clips = jax.device_put(batch["clip_embedding"], examples_sharding)
                batch_max_cos_distances = jax.device_put(
                    jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
                )
            elif model_cfg.clip_conditioning and model_cfg.clip_caps:
                batch_centers, batch_max_cos_distances = rearrange_batch_caps(
                    batch["cap_center"],
                    batch["cap_max_cos_distance"],
                    model_cfg.clip_cap_count,
                    (global_step + data_step_offset) // batches_per_epoch,
                )
                batch_clips = jax.device_put(batch_centers, examples_sharding)
                batch_max_cos_distances = jax.device_put(
                    batch_max_cos_distances, examples_sharding
                )
            else:
                batch_clips = jax.device_put(
                    jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
                )
                batch_max_cos_distances = jax.device_put(
                    jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
                )
            train_state, train_loss, norm = train_step(
                train_state,
                training_cfg.loss_decay_constant,
                batch_imgs,
                batch_clips,
                batch_max_cos_distances,
            )
            opt_state = train_state.opt_state
            train_loss, notfinite_count, norm = jax.device_get(
                (train_loss, opt_state.notfinite_count, norm)
            )
            if not jnp.isfinite(train_loss):
                tqdm.write(f"Loss nonfinite 😢 ({train_loss})")
            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": train_loss,
                    "grad_global_norm": norm,
                    "debug/notfinite_count": notfinite_count,
                }
            )
            # Save checkpoint every 30 minutes. This does one at step 0 too, which is nice so we
            # don't have to wait half an hour to find out if it crashes.
            if last_checkpoint_time is None or (
                datetime.datetime.now() - last_checkpoint_time
            ) > datetime.timedelta(minutes=30):
                save_checkpoint_and_log_images(
                    train_state,
                    sample_batch_size,
                    global_step,
                    args.skip_sampling,
                )
                last_checkpoint_time = datetime.datetime.now()

            if notfinite_count > 50:
                tqdm.write(f"Too many nonfinite values in gradients, giving up")
                exit(1)
            if (
                global_step % 20 == 0
                and training_cfg.learning_rate_schedule
                == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
            ):
                # Since the params used for gradient computation with a schedule-free optimizer are
                # not the same as the params used for inference, we want to test with the inference
                # params occasionally for charting.
                eval_params = train_state.get_eval_params()
                to_log = {"global_step": global_step}
                eval_loss_weighted = loss_fn(
                    eval_params,
                    training_cfg.loss_decay_constant,
                    train_state.rng,
                    batch_imgs,
                    batch_clips,
                    batch_max_cos_distances,
                )
                if training_cfg.loss_decay_constant != 1.0:
                    eval_loss_unweighted = loss_fn(
                        eval_params,
                        1.0,
                        train_state.rng,
                        batch_imgs,
                        batch_clips,
                        batch_max_cos_distances,
                    )
                    eval_loss_weighted, eval_loss_unweighted = jax.device_get(
                        (eval_loss_weighted, eval_loss_unweighted)
                    )
                    to_log["eval/loss"] = eval_loss_weighted
                    to_log["eval/loss_unweighted"] = eval_loss_unweighted
                else:
                    to_log["eval/loss"] = jax.device_get(eval_loss_weighted)
                eval_loss = eval_loss_weighted
                del eval_params
                wandb.log(to_log)
            if global_step % 20 == 0 and training_cfg.loss_decay_constant != 1.0:
                train_loss_unweighted = jax.device_get(
                    loss_fn(
                        train_state.params,
                        1.0,
                        train_state.rng,
                        batch_imgs,
                        batch_clips,
                        batch_max_cos_distances,
                    )
                )
                wandb.log(
                    {
                        "global_step": global_step,
                        "train/loss_unweighted": train_loss_unweighted,
                    }
                )
            if (
                training_cfg.learning_rate_schedule
                == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
                and eval_loss is not None  # can be None if resuming from checkpoint
            ):
                pbar.set_postfix(
                    train_loss=f"{train_loss:.4f}", eval_loss=f"{eval_loss:.4f}"
                )
            else:
                pbar.set_postfix(train_loss=f"{train_loss:.4f}")
            pbar.update()
            if exit_requested:
                tqdm.write("Saving checkpoint and exiting early")
                save_checkpoint_and_log_images(
                    train_state, sample_batch_size, global_step, skip_sampling=True
                )
                exit(0)
            global_step += 1
    # Evaluate on test set
    losses = []
    eval_params = train_state.get_eval_params()
    for batch in tqdm(
        # The last batch needs to be a multiple of the number of devices, and it isn't guaranteed
        # to be, so we drop it. Shouldn't matter much when even 1% of the dataset is thousands of
        # images.
        test_imgs.iter(batch_size=training_cfg.batch_size, drop_last_batch=True),
        total=len(test_imgs) // training_cfg.batch_size,
        desc="test batches",
    ):
        dropout_rng, rng = jax.random.split(train_state.rng, 2)
        train_state = train_state.replace(rng=rng)
        batch_imgs = jax.device_put(batch["encoded_img"], examples_sharding)
        if model_cfg.clip_conditioning:
            if model_cfg.clip_caps:
                batch_cap_centers, batch_max_cos_distances = rearrange_batch_caps(
                    batch["cap_center"],
                    batch["cap_max_cos_distance"],
                    model_cfg.clip_cap_count,
                    epoch=0,  # could iterate and do every cap set but this is faster and fine with lots of test images
                    is_test=True,
                )
                batch_clips = jax.device_put(batch_cap_centers, examples_sharding)
                batch_max_cos_distances = jax.device_put(
                    batch_max_cos_distances, examples_sharding
                )
            else:
                batch_clips = jax.device_put(batch["clip_embedding"], examples_sharding)
                batch_max_cos_distances = jax.device_put(
                    jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
                )
        else:
            batch_clips = jax.device_put(
                jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
            )
            batch_max_cos_distances = jax.device_put(
                jnp.zeros((batch_imgs.shape[0], 0)), examples_sharding
            )
        losses.append(
            loss_fn(
                eval_params,
                training_cfg.loss_decay_constant,
                dropout_rng,
                batch_imgs,
                batch_clips,
                batch_max_cos_distances,
            )
        )
    del eval_params
    test_loss = jnp.mean(jnp.stack(losses))
    wandb.log({"global_step": global_step, "test/loss": test_loss})
    tqdm.write(
        f"Epoch {epoch} done, train loss: {train_loss:.4f}, test loss {test_loss:.4f}",
        end="",
    )
    save_checkpoint_and_log_images(
        train_state, sample_batch_size, global_step, skip_sampling=False
    )
    last_checkpoint_time = datetime.datetime.now()
