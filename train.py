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
import time
import torch
import transformer_model
import wandb
from config import ModelConfig, TrainingConfig, str_to_activation, str_to_dtype
from copy import copy
from datasets import Dataset
from distutils.util import strtobool
from einops import rearrange
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
    image=jnp.zeros((model_cfg.seq_len,), dtype=jnp.int32),
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
    lambda params, rng, batch: transformer_model.loss_batch(mdl, params, rng, batch)
)


class TrainState(train_state.TrainState):  # type:ignore[no-untyped-call]
    rng: jax.random.KeyArray  # type:ignore[misc]


# Set up for sampling:
sample_mdl = copy(mdl)
sample_mdl.dropout = 0.0

sample_jv = jax.jit(
    jax.vmap(
        lambda params, rng, top_p: transformer_model.sample(
            sample_mdl, params, rng, top_p
        ),
        in_axes=(None, 0, 0),
    )
)


ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)
# don't keep these on the GPU when we're not using them
ae_params_torch = torch.load(args.ae_ckpt, map_location="cpu")

decode_jv = jax.jit(
    jax.vmap(
        (
            lambda ae_params, codes: ae_mdl.apply(
                ae_params, method=ae_mdl.decode, x=codes, shape=(16, 16)
            )
        ),
        in_axes=(None, 0),
    )
)


def sample_and_log(ts: TrainState, global_step: int) -> None:
    """Sample from the model and log to wandb."""
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
    # Can't use the sharding for examples here because it has two dimensions and we need one.
    p_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((jax.device_count(),))
    )
    img_top_ps = jax.device_put(img_top_ps, p_sharding)

    rng_sharding = jax.sharding.PositionalSharding(
        mesh_utils.create_device_mesh((jax.device_count(), 1))
    )
    rngs = jax.device_put(jax.random.split(ts.rng, imgs_to_sample), rng_sharding)

    samples = sample_jv(ts.params, rngs, img_top_ps)

    ae_params = LDMAutoencoder.params_from_torch(ae_params_torch, ae_cfg)

    decoded = decode_jv(ae_params, samples)
    decoded = jnp.clip(-1.0, decoded, 1.0)
    decoded = (decoded + 1.0) * 127.5
    decoded = np.array(decoded.astype(jnp.uint8))

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
    batch: jax.Array,
) -> Tuple[TrainState, jax.Array, jax.Array]:
    """Compute a single optimization step."""
    rng, rng2 = jax.random.split(state.rng)
    loss, grads = loss_grad_fn(mdl, state.params, rng, batch)
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
            print(f"Error saving checkpoint: {e}")
            print("Retrying in 60 seconds")
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
            batch = jax.device_put(batch["encoded_img"], sharding)
            my_train_state, loss, norm = train_step(my_train_state, batch)
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
        batch = jax.device_put(batch["encoded_img"], sharding)
        losses.append(loss_fn(my_train_state.params, rng2, batch))
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
