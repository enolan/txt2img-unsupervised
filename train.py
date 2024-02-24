"""Train the image model."""
import argparse
import datetime
import flax.core
import jax
import jax.numpy as jnp
import json
import numpy as np
import optax  # type:ignore[import]
import orbax.checkpoint as ocp
import PIL.Image
import time
import torch
import transformers
import wandb
from copy import copy
from datasets import Dataset
from distutils.util import strtobool
from einops import rearrange, repeat
from flax.training import train_state
from functools import partial
from itertools import islice
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import OmegaConf
from pathlib import Path
from sys import exit
from tqdm import tqdm, trange
from typing import Any, Callable, Tuple

from txt2img_unsupervised.config import (
    ModelConfig,
    TrainingConfig,
    str_to_activation,
    str_to_dtype,
)
from txt2img_unsupervised.ldm_autoencoder import LDMAutoencoder
from txt2img_unsupervised.load_pq_dir import load_pq_dir
from txt2img_unsupervised.triangle_schedule import triangle_schedule
import txt2img_unsupervised.config as config
import txt2img_unsupervised.ldm_autoencoder as ldm_autoencoder
import txt2img_unsupervised.sample as sample
import txt2img_unsupervised.transformer_model as transformer_model


# TODO next sweep:
# - learning rate
# - gradient accumulation
# - gradient clipping
# - biases
# - alt activation functions

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
parser.add_argument("--epochs", type=int)
parser.add_argument("--training-images", type=int)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument(
    "--triangle-schedule", type=lambda x: bool(strtobool(x)), default=True
)
parser.add_argument("--gradient-accumulation-steps", type=int)
parser.add_argument("--use-biases", type=lambda x: bool(strtobool(x)))
parser.add_argument("--gradient-clipping", type=float, default=None)
parser.add_argument("--ae-cfg", type=Path, required=True)
parser.add_argument("--ae-ckpt", type=Path, required=True)
parser.add_argument("--activations-dtype", type=argparse_from_dict(str_to_dtype))
parser.add_argument("--activation-function", type=argparse_from_dict(str_to_activation))
parser.add_argument("--clip-conditioning", type=lambda x: bool(strtobool(x)))
parser.add_argument("--clip-caps", type=lambda x: bool(strtobool(x)))
parser.add_argument("--clip-cap-count", type=int)
parser.add_argument("--resume", type=Path)
args, _unknown = parser.parse_known_args()


def json_pretty(dict):
    """Print a dictionary as pretty JSON."""
    return json.dumps(dict, indent=2)


def setup_cfg_and_wandb():
    """Set up our ModelConfig and TrainingConfig and initialize wandb."""
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=3, keep_time_interval=datetime.timedelta(hours=6)
    )
    checkpoint_manager_items = ("params", "opt_state", "rng")
    if args.resume is not None:
        print(f"Resuming from checkpoint {args.resume}...")
        checkpoint_dir = args.resume.absolute()
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=checkpoint_options,
            item_names=checkpoint_manager_items,
        )
        global_step = checkpoint_manager.latest_step()
        restoring = True
        metadata = checkpoint_manager.metadata()
        model_cfg = ModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
        run_id = metadata["run_id"]
        print(f"Resuming run {run_id}")
        print(
            "ALL TRAINING AND MODEL PARAMETERS PASSED ON THE COMMAND LINE WILL BE IGNORNED."
        )
        print(f"ModelConfig {json_pretty(model_cfg.to_json_dict())}")
        print(f"TrainingConfig {json_pretty(training_cfg.to_json_dict())}")
        wandb.init(id=run_id, resume="must")
    else:
        print("Starting new run...")
        wandb.init()

        global_step = 0
        restoring = False
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

        checkpoint_dir = Path(f"checkpoints/{wandb.run.id}").absolute()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=checkpoint_options,
            item_names=checkpoint_manager_items,
            metadata={
                "model_cfg": model_cfg.to_json_dict(),
                "training_cfg": training_cfg.to_json_dict(),
                "run_id": wandb.run.id,
            },
        )

    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("test/loss", summary="last")
    wandb.define_metric("train/loss", summary="last")

    return global_step, model_cfg, training_cfg, checkpoint_manager, restoring


(
    global_step,
    model_cfg,
    training_cfg,
    checkpoint_manager,
    restoring,
) = setup_cfg_and_wandb()


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


class TrainState(train_state.TrainState):  # type:ignore[no-untyped-call]
    rng: jax.Array  # type:ignore[misc]


def calculate_steps(train_set_size, training_cfg):
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
    return batches_total, epochs_total, batches_per_epoch, extra_batches


batches_total, epochs_total, batches_per_epoch, extra_batches = calculate_steps(
    train_imgs.shape[0], training_cfg
)


def setup_optimizer(
    model_cfg, training_cfg, batches_total, checkpoint_manager, global_step, restoring
):
    """Set up a model and create a - possibly restored from disk - TrainState to begin training
    with."""

    # Set up model
    mdl = transformer_model.ImageModel(**model_cfg.__dict__)

    # Set up optimizer
    if training_cfg.triangle_schedule:
        opt = optax.adam(
            learning_rate=triangle_schedule(
                training_cfg.learning_rate,
                batches_total,
            )
        )
    else:
        opt = optax.adam(learning_rate=training_cfg.learning_rate)

    assert training_cfg.gradient_accumulation_steps > 0
    if training_cfg.gradient_accumulation_steps > 1:
        opt = optax.MultiSteps(
            opt, every_k_schedule=training_cfg.gradient_accumulation_steps
        )
    if training_cfg.gradient_clipping is not None:
        clip = optax.clip_by_global_norm(training_cfg.gradient_clipping)
    else:
        clip = optax.identity()
    opt = optax.apply_if_finite(optax.chain(clip, opt), 20)

    # We need template inputs to get initial parameters, regardless of if we're loading or starting
    # anew.
    max_cos_distance_dummy = (
        jnp.zeros((model_cfg.clip_cap_count,), dtype=jnp.float32)
        if model_cfg.clip_caps
        else jnp.array([], dtype=jnp.float32)
    )
    if model_cfg.clip_conditioning and not model_cfg.clip_caps:
        clip_embedding_dummy = jnp.zeros((768,), dtype=jnp.float32)
    elif model_cfg.clip_conditioning and model_cfg.clip_caps:
        clip_embedding_dummy = jnp.zeros(
            (model_cfg.clip_cap_count, 768), dtype=jnp.float32
        )
    else:
        clip_embedding_dummy = jnp.zeros((0,), dtype=jnp.float32)

    rngs_dummy = {"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)}
    image_dummy = jnp.zeros((model_cfg.image_tokens,), dtype=jnp.int32)
    params_empty = mdl.init(
        rngs=rngs_dummy,
        image=image_dummy,
        clip_embedding=clip_embedding_dummy,
        max_cos_distance=max_cos_distance_dummy,
    )
    table_str = mdl.tabulate(
        rngs=rngs_dummy,
        image=image_dummy,
        clip_embedding=clip_embedding_dummy,
        max_cos_distance=max_cos_distance_dummy,
    )
    print(table_str)

    # Set up parameters
    if restoring:
        opt_state_skeleton = opt.init(params_empty)
        restored = checkpoint_manager.restore(
            global_step,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(),
                opt_state=ocp.args.StandardRestore(opt_state_skeleton),
                rng=ocp.args.JaxRandomKeyRestore(),
            ),
        )
        ts = TrainState(
            step=global_step,
            apply_fn=mdl.apply,
            params=restored["params"],
            tx=opt,
            rng=restored["rng"],
            opt_state=restored["opt_state"],
        )
    else:
        ts = TrainState.create(
            apply_fn=mdl.apply,
            params=params_empty,
            tx=opt,
            rng=jax.random.PRNGKey(1337),
        )

    return mdl, ts


mdl, my_train_state = setup_optimizer(
    model_cfg, training_cfg, batches_total, checkpoint_manager, global_step, restoring
)


loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
loss_fn = jax.jit(partial(transformer_model.loss_batch, mdl))


# TODO delete this, unnecessary now
# Set up for sampling:
sample_cfg = copy(model_cfg)
sample_cfg.dropout = None
sample_mdl = transformer_model.ImageModel(**sample_cfg.__dict__, decode=True)
max_cos_distance_dummy = (
    jnp.zeros((model_cfg.clip_cap_count,), dtype=jnp.float32)
    if model_cfg.clip_caps
    else jnp.array([], dtype=jnp.float32)
)
if model_cfg.clip_conditioning and not model_cfg.clip_caps:
    clip_embedding_dummy = jnp.zeros((768,), dtype=jnp.float32)
    max_cos_distance_dummy = jnp.zeros((0,), dtype=jnp.float32)
elif model_cfg.clip_conditioning and model_cfg.clip_caps:
    clip_embedding_dummy = jnp.zeros((model_cfg.clip_cap_count, 768), dtype=jnp.float32)
    max_cos_distance_dummy = jnp.zeros((model_cfg.clip_cap_count,), dtype=jnp.float32)
else:
    clip_embedding_dummy = jnp.zeros((0,), dtype=jnp.float32)
    max_cos_distance_dummy = jnp.zeros((0,), dtype=jnp.float32)
sample_params = sample_mdl.init(
    jax.random.PRNGKey(0),
    image=jnp.arange(model_cfg.image_tokens, dtype=jnp.int32),
    clip_embedding=clip_embedding_dummy,
    max_cos_distance=max_cos_distance_dummy,
)
sample_cache = sample_params["cache"]
del sample_params
del clip_embedding_dummy
del max_cos_distance_dummy

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


def get_image_prompt_clip_embeddings(
    clip_mdl, clip_processor, n: int, dset
) -> jax.Array:
    """Find n hopefully SFW CLIP embeddings to use for diagnostic sampling."""

    text_tokens = clip_processor.tokenizer("nsfw", return_tensors="jax", padding=True)
    nsfw_text_features = clip_mdl.get_text_features(**text_tokens)
    nsfw_text_embeds = nsfw_text_features / jnp.linalg.norm(nsfw_text_features)

    # compute cosine similarity between images in dset and nsfw_text_embeds and
    # take the first n images with similarity < 0.2
    clip_embeds = dset["clip_embedding"]
    sims = jnp.dot(clip_embeds, nsfw_text_embeds.T)
    ok_sims = sims < 0.2
    print(f"Found {ok_sims.sum()} images with similarity < 0.2")
    ok_indices = jnp.where(ok_sims)[0]
    assert (
        len(ok_indices) >= n
    ), f"Found {len(ok_indices)} images with similarity < 0.2, expected at least {n}"
    print(f"Using {[dset['name'][i] for i in ok_indices[:n]]} for diagnostic sampling")
    return {
        "clip_embedding": clip_embeds[ok_indices[:n]],
        "name": dset["name"][ok_indices[:n]],
    }


def get_clip_text_embeddings_to_sample(n: int) -> jax.Array:
    """Generate some text embeddings to test the model with."""
    prompts = [
        "Barack Obama riding a bicycle",
        "a painting of a cat",
        "The Golden Gate Bridge at sunset",
        "Samoyed puppies!",
        "Taylor Swift in concert",
        "Manhattan from above #dronephotography",
        "my favorite art car from Burning Man 2018",
        "engineering diagram of an internal combustion engine",
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


image_prompt_clips = get_image_prompt_clip_embeddings(
    clip_mdl, clip_processor, image_prompts_to_sample, test_imgs[:2048]
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
            (len(text_prompt_clips), grid_size, 1), 0.75
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


def sample_and_log(ts: TrainState, global_step: int) -> None:
    """Sample from the model and log to wandb."""

    ae_params = LDMAutoencoder.params_from_torch(ae_params_torch, ae_cfg)

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
                model_cfg,
                ts.params,
                ae_mdl,
                ae_params,
                training_cfg.batch_size,
                all_centers_for_sampling,
                all_max_cos_distances_for_sampling,
                ts.rng,
                0.95,
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
                ts.params,
                ae_mdl,
                ae_params,
                training_cfg.batch_size,
                all_clip_embeddings_for_sampling,
                None,
                ts.rng,
                0.95,
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

        samples = sample_jv(ts.params, clip_embeddings, rngs, img_top_ps)

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


def rearrange_batch_caps(centers, max_cos_distances, cap_count):
    """Rearrange the data from parquet into the shape expected by the model. For reasons, parquet
    does not support multidimensional arrays."""
    centers = rearrange(centers, "b (n c) -> b n c", n=cap_count, c=768)
    max_cos_distances = rearrange(max_cos_distances, "(b n) -> b n", n=cap_count)
    return centers, max_cos_distances


@partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: TrainState,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
    batch_max_cos_distances: jax.Array,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    dropout_rng, rng2 = jax.random.split(state.rng, 2)
    loss, grads = loss_grad_fn(
        mdl, state.params, dropout_rng, batch_imgs, batch_clips, batch_max_cos_distances
    )
    new_state = state.apply_gradients(
        grads=grads, rng=rng2
    )  # type:ignore[no-untyped-call]
    norm = optax.global_norm(grads)
    return new_state, loss, norm


# Set up sharding
print(
    f"Sharding batches of {training_cfg.batch_size} across {jax.device_count()} devices, {training_cfg.batch_size / jax.device_count()} per device"
)
assert training_cfg.batch_size % jax.device_count() == 0
# NamedSharding is overkill for the simple batch parallelism we do, but it's necessary to get orbax
# to save checkpoints correctly.
devices = mesh_utils.create_device_mesh((jax.device_count(),))
mesh = Mesh(devices, axis_names=("dev",))
my_train_state = my_train_state.replace(
    params=jax.device_put(
        my_train_state.params, NamedSharding(mesh, PartitionSpec(None))
    )
)


last_checkpoint_time = None


def save_checkpoint_and_sample(my_train_state, global_step) -> None:
    # TPU VMs run out of disk space a lot. Retrying in a loop lets me manually clean up the disk
    tqdm.write("Attempting to save checkpoint")
    while True:
        try:
            checkpoint_manager.save(
                global_step,
                args=ocp.args.Composite(
                    params=ocp.args.StandardSave(my_train_state.params),
                    opt_state=ocp.args.StandardSave(my_train_state.opt_state),
                    rng=ocp.args.JaxRandomKeySave(my_train_state.rng),
                ),
            )
            break
        except (OSError, ValueError) as e:
            tqdm.write(f"Error saving checkpoint: {e}")
            tqdm.write("Retrying in 60 seconds")
            time.sleep(60)
    tqdm.write("Saved checkpoint")
    tqdm.write("Sampling")
    sample_and_log(my_train_state, global_step)
    tqdm.write("Done sampling")


assert global_step < batches_total, "training run is over my dude"
start_epoch = global_step // batches_per_epoch
start_step = global_step % batches_per_epoch
tqdm.write(f"Starting at epoch {start_epoch}, step {start_step}")

examples_sharding = NamedSharding(mesh, PartitionSpec("dev"))
for epoch in trange(
    start_epoch,
    epochs_total,
    initial=start_epoch,
    total=epochs_total,
    desc="epochs",
):
    # Consistent shuffle so resuming gets the same ordering
    shuffled_train_imgs = train_imgs.shuffle(seed=epoch)

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
        iter = shuffled_train_imgs.iter(
            batch_size=training_cfg.batch_size, drop_last_batch=True
        )
        for batch in islice(iter, this_start_step, this_end_step):
            # Save checkpoint every 30 minutes. This does one at step 0 too, which is nice so we
            # don't have to wait half an hour to find out if it crashes.
            if last_checkpoint_time is None or (
                datetime.datetime.now() - last_checkpoint_time
            ) > datetime.timedelta(minutes=30):
                save_checkpoint_and_sample(my_train_state, global_step)
                last_checkpoint_time = datetime.datetime.now()

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
            my_train_state, loss, norm = train_step(
                my_train_state, batch_imgs, batch_clips, batch_max_cos_distances
            )
            # TODO check if moving this check inside an if opt_state.notfinite_count > 0 is faster
            if not jnp.isfinite(loss):
                tqdm.write(f"Loss nonfinite 😢 ({loss})")
            opt_state = my_train_state.opt_state
            global_step += 1
            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": loss,
                    "grad_global_norm": norm,
                    "debug/notfinite_count": opt_state.notfinite_count,
                }
            )
            if opt_state.notfinite_count > 50:
                tqdm.write(f"Too many nonfinite values in gradients, giving up")
                exit(1)
            pbar.update()
            pbar.set_postfix(loss=f"{loss:.4f}")
    # Evaluate on test set
    losses = []
    for batch in tqdm(
        # The last batch needs to be a multiple of the number of devices, and it isn't guaranteed
        # to be, so we drop it. Shouldn't matter much when even 1% of the dataset is thousands of
        # images.
        test_imgs.iter(batch_size=training_cfg.batch_size, drop_last_batch=True),
        total=len(test_imgs) // training_cfg.batch_size,
        desc="test batches",
    ):
        dropout_rng, rng = jax.random.split(my_train_state.rng, 2)
        my_train_state = my_train_state.replace(rng=rng)
        batch_imgs = jax.device_put(batch["encoded_img"], examples_sharding)
        if model_cfg.clip_conditioning:
            if model_cfg.clip_caps:
                batch_cap_centers, batch_max_cos_distances = rearrange_batch_caps(
                    batch["cap_center"],
                    batch["cap_max_cos_distance"],
                    model_cfg.clip_cap_count,
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
                my_train_state.params,
                dropout_rng,
                batch_imgs,
                batch_clips,
                batch_max_cos_distances,
            )
        )
    test_loss = jnp.mean(jnp.stack(losses))
    wandb.log({"global_step": global_step, "test/loss": test_loss})
    tqdm.write(
        f"Epoch {epoch} done, train loss: {loss:.4f}, test loss {test_loss:.4f}",
        end="",
    )
    save_checkpoint_and_sample(my_train_state, global_step)
    last_checkpoint_time = datetime.datetime.now()
