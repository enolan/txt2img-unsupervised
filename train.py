"""Train the image model."""
import argparse
import config
import datetime
import jax
import jax.numpy as jnp
import json
import numpy as np
import optax  # type:ignore[import]
import orbax.checkpoint  # type:ignore[import]
import PIL.Image
import sample
import time
import torch
import transformers
import transformer_model
import wandb
from config import ModelConfig, TrainingConfig, str_to_activation, str_to_dtype
from copy import copy
from datasets import Dataset
from distutils.util import strtobool
from einops import rearrange, repeat
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from functools import partial
from jax.experimental import mesh_utils
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from sys import exit
from tqdm import tqdm, trange
from typing import Any, Callable, Tuple

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
args, _unknown = parser.parse_known_args()


wandb.init()
global_step: int = 0  # gradients computed so far
wandb.define_metric("*", step_metric="global_step")


# Load the dataset
dset_paths = list(args.pq_dir.glob("**/*.parquet"))
dset_all = Dataset.from_parquet([str(path) for path in dset_paths])
dset_all.set_format("numpy")
dset_split = dset_all.train_test_split(test_size=0.01, seed=19900515)
train_imgs = dset_split["train"]
test_imgs = dset_split["test"]
print(f"Train set {train_imgs.shape}, test set {test_imgs.shape}")

# Make the RNG partitionable across devices
jax.config.update("jax_threefry_partitionable", True)

# Load the model configuration
with open(args.model_config) as f:
    model_cfg = ModelConfig.from_json_dict(json.load(f))
config.merge_attrs(model_cfg, args)

with open(args.training_config) as f:
    training_cfg = TrainingConfig.from_json_dict(json.load(f))
config.merge_attrs(training_cfg, args)

wandb.config.update(model_cfg.to_json_dict())
wandb.config.update(training_cfg.to_json_dict())

# Get possibly sweep-controlled parameters from wandb
model_cfg = ModelConfig.from_json_dict(wandb.config.as_dict())
training_cfg = TrainingConfig.from_json_dict(wandb.config.as_dict())
print(f"Model config post-wandb: {json.dumps(model_cfg.to_json_dict(), indent=2)}")
print(
    f"Training config post-wandb: {json.dumps(training_cfg.to_json_dict(), indent=2)}"
)

wandb.define_metric("test/loss", summary="last")
wandb.define_metric("train/loss", summary="last")

mdl = transformer_model.ImageModel(**model_cfg.__dict__)

params = mdl.init(
    rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
    image=jnp.zeros((model_cfg.image_tokens,), dtype=jnp.int32),
    clip_embedding=jnp.zeros((768,), dtype=jnp.float32)
    if model_cfg.clip_conditioning
    else jnp.zeros((0,), dtype=jnp.float32),
)


def triangle_schedule(max_lr: float, total_steps: int) -> optax.Schedule:
    """Simple linear trianguar learning rate schedule. Best schedule found in Cramming paper.
    https://arxiv.org/abs/2212.14034"""
    sched_up = optax.linear_schedule(
        init_value=0, end_value=max_lr, transition_steps=total_steps // 2
    )
    sched_down = optax.linear_schedule(
        init_value=max_lr, end_value=0, transition_steps=total_steps // 2
    )
    return optax.join_schedules([sched_up, sched_down], [total_steps // 2])


if training_cfg.triangle_schedule:
    opt = optax.adam(
        learning_rate=triangle_schedule(
            training_cfg.learning_rate, training_cfg.epochs * len(train_imgs)
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
loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
loss_fn = jax.jit(
    lambda params, rng, batch_imgs, batch_clips: transformer_model.loss_batch(
        mdl, params, rng, batch_imgs, batch_clips
    )
)


class TrainState(train_state.TrainState):  # type:ignore[no-untyped-call]
    rng: jax.random.KeyArray  # type:ignore[misc]


# Set up for sampling:
sample_cfg = copy(model_cfg)
sample_cfg.dropout = None
sample_mdl = transformer_model.ImageModel(**sample_cfg.__dict__, decode=True)
sample_params = sample_mdl.init(
    jax.random.PRNGKey(0),
    image=jnp.arange(256),
    clip_embedding=jnp.zeros((768,), dtype=jnp.float32)
    if model_cfg.clip_conditioning
    else jnp.zeros((0,), dtype=jnp.float32),
)
sample_cache = sample_params["cache"]
del sample_params

sample_jv = jax.jit(
    jax.vmap(
        lambda params, clip_embedding, rng, top_p: transformer_model.sample(
            sample_mdl, params.copy({"cache": sample_cache}), clip_embedding, rng, top_p
        ),
        in_axes=(None, 0, 0, 0),
    )
)

# CLIP embeddings to use for diagnostic sampling
embeddings_to_sample = 8

def get_clip_embeddings_to_sample(n: int, dset) -> jax.Array:
    """Find n hopefully SFW CLIP embeddings to use for diagnostic sampling."""

    # Must match model used in preprocessor
    clip_mdl_name = "openai/clip-vit-large-patch14"
    clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name)
    clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

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
    return {
        "clip_embedding": clip_embeds[ok_indices[:n]],
        "name": dset["name"][ok_indices[:n]],
    }


sampling_clips = get_clip_embeddings_to_sample(embeddings_to_sample, test_imgs[:2048])

ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)
# don't keep these on the GPU when we're not using them
ae_params_torch = torch.load(args.ae_ckpt, map_location="cpu")


def sample_and_log(ts: TrainState, global_step: int) -> None:
    """Sample from the model and log to wandb."""

    # We need one sharding per rank if we're just always doing data parallelism.
    sharding_1d = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((jax.device_count(),))
    )

    sharding_2d = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((jax.device_count(), 1))
    )

    ae_params = LDMAutoencoder.params_from_torch(ae_params_torch, ae_cfg)
    ae_params = jax.device_put(ae_params, sharding_2d.replicate())

    if model_cfg.clip_conditioning:
        # Create a grid of samples for each testing CLIP embedding/top-p pair.
        top_ps = jnp.array([0.6, 0.8, 0.9, 0.95], dtype=jnp.float32)
        grid_size = 16

        samples_count = embeddings_to_sample * len(top_ps) * grid_size
        assert (samples_count % training_cfg.batch_size) % jax.device_count() == 0

        conditioning_embeds = sampling_clips["clip_embedding"]
        conditioning_names = sampling_clips["name"]

        conditioning_embeds_rep = repeat(
            conditioning_embeds,
            "clips clip_dim -> (clips t g) clip_dim",
            t=len(top_ps),
            g=grid_size,
            clip_dim=768,
        )
        top_ps_rep = repeat(
            top_ps,
            "t -> (clips t g)",
            t=len(top_ps),
            g=grid_size,
            clips=embeddings_to_sample,
        )
        rngs = jax.random.split(ts.rng, samples_count)

        assert (
            len(conditioning_embeds_rep)
            == len(top_ps_rep)
            == len(rngs)
            == samples_count
        )

        conditioning_embeds_rep = jax.device_put(conditioning_embeds_rep, sharding_2d)
        top_ps_rep = jax.device_put(top_ps_rep, sharding_1d)
        rngs = jax.device_put(rngs, sharding_2d)

        batches = sample.batches_split(
            training_cfg.batch_size, len(conditioning_embeds_rep)
        )

        sampled_codes = []

        with tqdm(total=samples_count, desc="sampling", unit="img") as pbar:
            for batch_size in batches:
                sampled_codes.append(
                    sample_jv(
                        ts.params,
                        conditioning_embeds_rep[:batch_size],
                        rngs[:batch_size],
                        top_ps_rep[:batch_size],
                    )
                )
                conditioning_embeds_rep = conditioning_embeds_rep[batch_size:]
                rngs = rngs[batch_size:]
                top_ps_rep = top_ps_rep[batch_size:]

                pbar.update(batch_size)

        sampled_imgs = []
        with tqdm(total=samples_count, desc="decoding", unit="img") as pbar:
            for codes_batch in sampled_codes:
                sampled_imgs.append(sample.decode_jv(ae_mdl, ae_params, codes_batch))
                pbar.update(len(codes_batch))
        imgs_np = np.concatenate(sampled_imgs)

        assert imgs_np.shape[0] == samples_count

        grid_imgs = rearrange(
            imgs_np,
            "(clips t g) w h c -> clips t g w h c",
            clips=embeddings_to_sample,
            t=len(top_ps),
            g=grid_size,
        )
        tqdm.write(f"Made grids of shape: {grid_imgs.shape}")
        assert grid_imgs.shape[0] == embeddings_to_sample

        to_log = {"global_step": global_step}

        for i, name in enumerate(conditioning_names):
            for j, top_p in enumerate(top_ps):
                grid_pil = sample.make_grid(
                    [PIL.Image.fromarray(np.array(img)) for img in grid_imgs[i, j]]
                )
                to_log[f"samples/{name}/top_p_{top_p:.2f}"] = wandb.Image(grid_pil)

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


@partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: TrainState,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    rng, rng2 = jax.random.split(state.rng)
    loss, grads = loss_grad_fn(mdl, state.params, rng, batch_imgs, batch_clips)
    new_state = state.apply_gradients(
        grads=grads, rng=rng2
    )  # type:ignore[no-untyped-call]
    norm = optax.global_norm(grads)
    return new_state, loss, norm


# Set up sharding
devices = jax.device_count()
print(
    f"Sharding batches of {training_cfg.batch_size} across {devices} devices, {training_cfg.batch_size / devices} per device"
)
assert training_cfg.batch_size % devices == 0
sharding = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((devices, 1)))
params = jax.device_put(params, sharding.replicate())

my_train_state: TrainState = TrainState.create(  # type:ignore[no-untyped-call]
    apply_fn=mdl.apply, params=params, tx=opt, rng=jax.random.PRNGKey(1337)
)

run_id = wandb.run.id  # type:ignore[union-attr]

checkpoint_dir = Path(f"checkpoints/{run_id}")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=3, keep_time_interval=datetime.timedelta(hours=6)
)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir,
    orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler()),
    options=checkpoint_options,
    metadata={
        "model_cfg": model_cfg.to_json_dict(),
        "training_cfg": training_cfg.to_json_dict(),
        "run_id": run_id,
    },
)

last_checkpoint_time = datetime.datetime.now()


def save_checkpoint() -> None:
    # TPU VMs run out of disk space a lot. Retrying in a loop lets me manually clean up the disk
    while True:
        try:
            checkpoint_manager.save(global_step, my_train_state)
            break
        except ValueError as e:
            tqdm.write(f"Error saving checkpoint: {e}")
            tqdm.write("Retrying in 60 seconds")
            time.sleep(60)


rng = jax.random.PRNGKey(1337)
rng_np = np.random.default_rng(1337)
for epoch in trange(training_cfg.epochs):
    train_imgs = train_imgs.shuffle(generator=rng_np)
    batches = train_imgs.shape[0] // training_cfg.batch_size
    with tqdm(total=batches, leave=False, desc="train batches") as pbar:
        for batch in train_imgs.iter(
            batch_size=training_cfg.batch_size, drop_last_batch=True
        ):
            batch_imgs = jax.device_put(batch["encoded_img"], sharding)
            if model_cfg.clip_conditioning:
                batch_clips = jax.device_put(batch["clip_embedding"], sharding)
            else:
                batch_clips = jax.device_put(
                    jnp.zeros((batch_imgs.shape[0], 0)), sharding
                )
            my_train_state, loss, norm = train_step(
                my_train_state, batch_imgs, batch_clips
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
            # Save checkpoint every 10 minutes
            if (datetime.datetime.now() - last_checkpoint_time) > datetime.timedelta(
                minutes=30
            ):
                tqdm.write("Saving checkpoint...", end="")
                save_checkpoint()
                last_checkpoint_time = datetime.datetime.now()
                tqdm.write(" DONE")
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
        rng, rng2 = jax.random.split(rng)
        batch_imgs = jax.device_put(batch["encoded_img"], sharding)
        if model_cfg.clip_conditioning:
            batch_clips = jax.device_put(batch["clip_embedding"], sharding)
        else:
            batch_clips = jax.device_put(jnp.zeros((batch_imgs.shape[0], 0)), sharding)
        losses.append(loss_fn(my_train_state.params, rng2, batch_imgs, batch_clips))
    test_loss = jnp.mean(jnp.stack(losses))
    wandb.log({"global_step": global_step, "test/loss": test_loss})
    tqdm.write(
        f"Epoch {epoch} done, train loss: {loss:.4f}, test loss {test_loss:.4f} saving...",
        end="",
    )
    save_checkpoint()
    tqdm.write(" DONE")
    tqdm.write("Sampling...", end="")
    sample_and_log(my_train_state, global_step)
    tqdm.write(" DONE")
