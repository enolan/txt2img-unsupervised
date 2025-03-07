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
from math import ceil
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
from txt2img_unsupervised.train_data_loading import get_batch
from txt2img_unsupervised.training_visualizations import (
    log_attention_maps,
    log_token_loss_visualization,
)
import txt2img_unsupervised.cap_sampling as cap_sampling
import txt2img_unsupervised.config as config
import txt2img_unsupervised.sample as sample
import txt2img_unsupervised.transformer_model as transformer_model


def argparse_from_dict(d: dict[str, Any]) -> Callable[[str], Any]:
    """Create an argparse argument type from a dictionary."""

    def f(x: str) -> Any:
        if x in d:
            return d[x]
        else:
            raise argparse.ArgumentTypeError(f"Unknown value {x}")

    return f


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
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
        "--learning-rate-schedule",
        type=argparse_from_dict(str_to_learning_rate_schedule),
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--schedule-free-beta1", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--use-biases", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--gradient-clipping", type=float, default=None)
    parser.add_argument("--adaptive-gradient-clip", type=lambda x: bool(strtobool(x)))
    parser.add_argument("--adaptive-gradient-clip-history-len", type=int, default=None)
    parser.add_argument(
        "--adaptive-gradient-clip-threshold-factor", type=float, default=None
    )
    parser.add_argument("--adaptive-gradient-clip-quantile", type=float, default=None)
    parser.add_argument("--image-dropout", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ae-cfg", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--activations-dtype", type=argparse_from_dict(str_to_dtype))
    parser.add_argument("--weights-dtype", type=argparse_from_dict(str_to_dtype))
    parser.add_argument(
        "--activation-function", type=argparse_from_dict(str_to_activation)
    )
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
    parser.add_argument(
        "--skip-saving",
        action="store_true",
        help="Skip saving checkpoints during training",
    )
    parser.add_argument(
        "--log-weight-and-grad-interval",
        type=int,
        default=0,
        help="Log attention weights and gradients every N steps",
    )
    args, _unknown = parser.parse_known_args()
    return args


args = parse_arguments()


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
        checkpoint_manager = mk_checkpoint_manager(checkpoint_dir, checkpoint_options)
        # The step recorded in a checkpoint is the number of steps completed, so our starting step
        # index is that number. If the checkpoint we're resuming from completed 0 steps, we start
        # at step 0, if it completed 5 steps, that means it did 0-4 inclusive, so we start at step
        # 5.
        global_step = checkpoint_manager.latest_step()
        metadata = checkpoint_manager.metadata()
        model_cfg = ModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
        run_id = metadata["run_id"]
        data_offset = metadata.get("data_offset", 0)
        print(f"Resuming run {run_id}")
        print(
            "ALL TRAINING AND MODEL PARAMETERS PASSED ON THE COMMAND LINE WILL BE IGNORED."
        )
        print(f"ModelConfig {json_pretty(model_cfg.to_json_dict())}")
        print(f"TrainingConfig {json_pretty(training_cfg.to_json_dict())}")
        wandb.init(id=run_id, resume="must", settings=wandb_settings)

        train_state, mdl = TrainState.load_from_checkpoint(
            checkpoint_manager,
            global_step,
            # we don't know the total number of batches until calculate_steps
            training_cfg.warmup_steps + 1
            if training_cfg.warmup_steps is not None
            else 1,
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

        if args.finetune is not None:
            finetune_src_checkpoint_dir = args.finetune.absolute()
            finetune_src_checkpoint_manager = mk_checkpoint_manager(
                finetune_src_checkpoint_dir
            )
            if args.start_where_finetune_source_left_off:
                # data_offset is the offset (in batches) we use into the training data, equal to the
                # number of examples the source run trained on divided by our batch size, rounded
                # up. We track this number so we don't repeat any training data the source run saw
                # before we have to.
                data_offset_examples = (
                    finetune_src_checkpoint_manager.latest_step()
                    * finetune_src_checkpoint_manager.metadata()["training_cfg"][
                        "batch_size"
                    ]
                )
                data_offset = ceil(data_offset_examples / training_cfg.batch_size)
            else:
                data_offset = 0
            extra_metadata = (
                "finetune_src_config",
                finetune_src_checkpoint_manager.metadata(),
            )
        else:
            data_offset = 0
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
            # we don't know the total number of batches until calculate_steps
            training_cfg.warmup_steps + 1
            if training_cfg.warmup_steps is not None
            else 1,
            data_offset=data_offset,
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
        data_offset,
    )


(
    global_step,
    model_cfg,
    training_cfg,
    sample_batch_size,
    checkpoint_manager,
    train_state,
    mdl,
    data_offset,
) = init_train_state()


def load_dataset(dir: Path) -> Tuple[Dataset, Dataset]:
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
    # TODO this is a super dubious thing to be doing. We should ideally always set the optimizer
    # state and the optimizer itself at the same time. It's fine so long as the shape of the
    # optimizer state doesn't depend on batches_total, but we should really get all the info needed
    # to set up the optimizer before we set up the optimizer.
    opt = setup_optimizer(training_cfg, batches_total)
    train_state = train_state.replace(tx=opt)
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
train_state = train_state.replicate_for_multi_gpu(mesh)

loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
loss_fn = jax.jit(partial(transformer_model.loss_batch, mdl))


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
visualization_dset = test_imgs.select(sfw_indices[:8])
print(
    f"Using {[visualization_dset['name'][i] for i in range(8)]} for visualization & sampling"
)
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

        image_prompt_names = visualization_dset["name"]
        text_prompt_texts = text_prompt_clips["name"]

        if model_cfg.clip_caps:
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


def leading_dims_to_subtrees(tree):
    """Given a dict pytree, return a new dict pytree with each array split into a dict of arrays
    indexed by leading dimension.
    """
    if not isinstance(tree, dict):
        raise ValueError(f"Expected dict, got {type(tree)}")
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = leading_dims_to_subtrees(v)
        elif isinstance(v, np.ndarray):
            if v.ndim < 2:
                raise ValueError(f"Expected array with at least 2 dims, got {v.ndim}")
            else:
                out[k] = {f"{idx:03d}": v[idx] for idx in range(v.shape[0])}
        else:
            raise ValueError(f"Unknown type: {type(v)} for key {k}")
    return out


@partial(
    jax.jit, donate_argnames=["state"], static_argnames=["return_weights_and_grads"]
)
def train_step(
    state: TrainState,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
    return_weights_and_grads: bool = False,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    dropout_rng, rng2 = jax.random.split(state.rng, 2)
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
    my_train_state, sample_batch_size, global_step, skip_sampling, skip_saving
) -> None:
    if not skip_saving:
        my_train_state.save_checkpoint(checkpoint_manager, global_step)
        tqdm.write("Saved checkpoint")
    else:
        tqdm.write("Skipping checkpoint save")

    if not skip_sampling:
        tqdm.write("Sampling")
        sample_and_log(my_train_state, sample_batch_size, global_step)
        tqdm.write("Done sampling")
        visualization_imgs = test_imgs[sfw_indices[:8]]
        visualization_img_names = visualization_imgs["name"]
        (
            visualization_img_encodings,
            visualization_embeddings,
        ) = get_batch(
            visualization_dset,
            len(visualization_dset),
            0,
            clip_conditioning=mdl.clip_conditioning,
            sharding=examples_sharding,
        )
        if mdl.clip_caps:
            visualization_embeddings, visualization_max_cos_distances = gen_caps(
                jax.random.PRNGKey(0), visualization_embeddings, mdl.clip_cap_count
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


assert global_step < batches_total, "training run is over my dude"

start_epoch = global_step // batches_per_epoch
if data_offset > 0:
    print(f"Using data offset for finetuning: {data_offset} batches")
start_step = global_step % batches_per_epoch
tqdm.write(f"Starting at epoch {start_epoch}, step {start_step}")

examples_sharding = NamedSharding(mesh, PartitionSpec("dev"))

eval_loss = None


class SignalHandler:
    """
    Class to handle signals for clean exit and checkpointing. It's important to checkpoint and exit
    at well-defined times (not in the middle of a train step), so we use a flag and check for it
    rather than exiting immediately upon receiving the signal.

    Attributes:
        exit_requested: Flag indicating whether exit has been requested.
        early_checkpoint_requested: Flag indicating whether early checkpointing has been requested.
    """

    def __init__(self):
        """Initialize signal handler."""
        self.exit_requested = False
        self.early_checkpoint_requested = False

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._exit_early_signal_handler)
        signal.signal(signal.SIGINT, self._exit_early_signal_handler)
        signal.signal(signal.SIGUSR1, self._early_checkpoint_signal_handler)

    def _exit_early_signal_handler(self, signum, frame):
        if self.exit_requested:
            tqdm.write(
                "CTRL-C pressed twice, exiting immediately without checkpointing"
            )
            exit(1)
        else:
            tqdm.write("CTRL-C pressed, doing clean exit after checkpointing")
            self.exit_requested = True

    def _early_checkpoint_signal_handler(self, signum, frame):
        # Same deal as exit_early_signal_handler, but we don't exit, just checkpoint.
        self.early_checkpoint_requested = True

    def reset_checkpoint_flag(self):
        """Reset the early checkpoint flag after it's been handled."""
        self.early_checkpoint_requested = False


signal_handler = SignalHandler()


cap_logits_table = cap_sampling.LogitsTable(767, 16384)


@partial(jax.jit, static_argnames=["n_caps"])
def gen_caps(rng, batch_clips, n_caps):
    """Generate containing spherical caps for a batch of examples."""
    ex_rngs = jax.random.split(rng, batch_clips.shape[0])

    cap_centers, cap_max_cos_distances = jax.vmap(
        lambda rng, embedding: mdl.gen_training_caps(cap_logits_table, rng, embedding)
    )(ex_rngs, batch_clips)
    assert cap_centers.shape == (batch_clips.shape[0], n_caps, 768)
    assert cap_max_cos_distances.shape == (batch_clips.shape[0], n_caps)
    return cap_centers, cap_max_cos_distances


# In order to ensure the GPU doesn't wait on CPU stuff, we ensure that there's always at least one
# batch in flight. So at the beginning of each inner loop we enqueue the next step before doing
# anything that would wait for the current step to finish.
def prefetch_and_train(current_state, current_step):
    """Prefetch next batch and enqueue next train step."""

    batch_imgs, batch_clips = get_batch(
        train_imgs,
        training_cfg.batch_size,
        # In order to support --start-where-finetune-source-left-off, we offset where we're
        # reading the data from by the amount of examples seen in the source run (expressed in
        # units of this run's batch size).
        current_step + data_offset,
        clip_conditioning=mdl.clip_conditioning,
        sharding=examples_sharding,
    )
    if mdl.clip_conditioning:
        assert batch_clips.shape == (training_cfg.batch_size, 768)
    else:
        assert batch_clips.shape == (training_cfg.batch_size, 0)
    if mdl.clip_caps:
        caps_rng, rng = jax.random.split(current_state.rng, 2)
        current_state = current_state.replace(rng=rng)
        batch_cap_centers, batch_max_cos_distances = gen_caps(
            caps_rng, batch_clips, mdl.clip_cap_count
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
        return_weights_and_grads=args.log_weight_and_grad_interval > 0
        and global_step % args.log_weight_and_grad_interval == 0,
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


next_train_step_outputs = None

for epoch in trange(
    start_epoch,
    epochs_total,
    initial=start_epoch,
    total=epochs_total,
    desc="epochs",
):
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
                ) = prefetch_and_train(train_state, global_step)

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
                next_train_step_outputs = prefetch_and_train(train_state, global_step)
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
                    transformed_vals["transformer_layers"] = leading_dims_to_subtrees(
                        vals_tree["transformer_layers"]
                    )
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
                pbar.set_postfix(train_loss=f"{train_step_to_log['train/loss']:.4f}")
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
        batch_imgs, batch_clips = get_batch(
            test_imgs,
            training_cfg.batch_size,
            batch_idx,
            clip_conditioning=mdl.clip_conditioning,
            sharding=examples_sharding,
        )
        if mdl.clip_caps:
            batch_clips, batch_max_cos_distances = gen_caps(
                cap_rng, batch_clips, mdl.clip_cap_count
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
