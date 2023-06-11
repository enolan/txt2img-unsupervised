"""Train the image model."""
import argparse
import datetime
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type:ignore[import]
import orbax.checkpoint  # type:ignore[import]
import PIL.Image
import random
import string
import torch
import transformer_model
import wandb
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
from tqdm import tqdm, trange
from typing import Any, Tuple

# TODO next sweep:
# - learning rate
# - gradient accumulation
# - gradient clipping
# - biases
# - alt activation functions

parser = argparse.ArgumentParser()
parser.add_argument("train_pq", type=Path)
parser.add_argument("test_pq", type=Path)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument(
    "--triangle_schedule", type=lambda x: bool(strtobool(x)), default=True
)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--use-biases", type=lambda x: bool(strtobool(x)), default=True)
parser.add_argument("--gradient-clipping", type=float, default=None)
parser.add_argument("--ae-cfg", type=Path, required=True)
parser.add_argument("--ae-ckpt", type=Path, required=True)
parser.add_argument("--activations-dtype", type=str, default="float32")
args, _unknown = parser.parse_known_args()


wandb.init()
global_step: int = 0  # gradients computed so far
wandb.define_metric("*", step_metric="global_step")


# Load the dataset
def load_dataset(p: Path) -> Dataset:
    ds = Dataset.from_parquet(str(p))
    ds.set_format("numpy")
    return ds


train_imgs = load_dataset(args.train_pq)
test_imgs = load_dataset(args.test_pq)
print(f"Train set {train_imgs.shape}, test set {test_imgs.shape}")

# Make the RNG partitionable across devices
jax.config.update("jax_threefry_partitionable", True)

cfg = transformer_model.gpt_1_config

wandb_config = {}
wandb_config.update(cfg.__dict__)
wandb_config.update(vars(args))
wandb.config.update(wandb_config)
wandb.define_metric("test/loss", summary="last")
wandb.define_metric("train/loss", summary="last")


# Setup the model
cfg.use_biases = wandb.config.use_biases
if wandb.config.activations_dtype == "float32":
    cfg.activations_dtype = jnp.float32
elif wandb.config.activations_dtype == "bfloat16":
    cfg.activations_dtype = jnp.bfloat16
else:
    raise ValueError("Invalid activations_dtype")
mdl = transformer_model.ImageModel(**cfg.__dict__)

params = mdl.init(
    rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
    image=jnp.zeros((cfg.seq_len,), dtype=jnp.int32),
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


if wandb.config.triangle_schedule:
    opt = optax.adam(
        learning_rate=triangle_schedule(
            wandb.config.lr, wandb.config.epochs * len(train_imgs)
        )
    )
else:
    opt = optax.adam(learning_rate=wandb.config.lr)

assert wandb.config.gradient_accumulation_steps > 0
if wandb.config.gradient_accumulation_steps > 1:
    opt = optax.MultiSteps(
        opt, every_k_schedule=wandb.config.gradient_accumulation_steps
    )
if wandb.config.gradient_clipping is not None:
    clip = optax.clip_by_global_norm(wandb.config.gradient_clipping)
else:
    clip = optax.identity()
opt = optax.chain(clip, opt)
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


def sample_and_log(ts: TrainState, global_step: int, sharding) -> None:
    """Sample from the model and log to wandb."""
    # Sample with different top_p values
    top_ps = [0.2, 0.6, 0.8, 0.9, 0.95]

    def round_down_to_multiples(x, a, b):
        # Round something down so it's a multiple of a and b
        less = min(a, b)
        more = max(a, b)
        while x % less != 0 or x % more != 0:
            assert x > 0
            x -= less
        return x

    imgs_to_sample = min(80, args.batch_size)
    imgs_to_sample = round_down_to_multiples(
        imgs_to_sample, len(top_ps), jax.device_count()
    )
    img_top_ps = jnp.concatenate(
        [jnp.repeat(p, imgs_to_sample // len(top_ps)) for p in top_ps]
    )
    assert len(img_top_ps) % jax.device_count() == 0
    img_top_ps = jax.device_put(img_top_ps, sharding)

    rngs = jax.device_put(jax.random.split(ts.rng, imgs_to_sample), sharding)

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
    f"Sharding batches of {args.batch_size} across {devices} devices, {args.batch_size / devices} per device"
)
assert args.batch_size % devices == 0
sharding = jax.sharding.PositionalSharding(mesh_utils.create_device_mesh((devices, 1)))
params = jax.device_put(params, sharding.replicate())

my_train_state: TrainState = TrainState.create(  # type:ignore[no-untyped-call]
    apply_fn=mdl.apply, params=params, tx=opt, rng=jax.random.PRNGKey(1337)
)

run_id = wandb.run.id  # type:ignore[union-attr]

checkpoint_dir = Path(f"checkpoints/{run_id}")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=3, keep_time_interval=datetime.timedelta(hours=1)
)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax.checkpoint.PyTreeCheckpointer(), options=checkpoint_options
)

last_checkpoint_time = datetime.datetime.now()


def save_checkpoint() -> None:
    checkpoint_manager.save(
        global_step, {"state": my_train_state, "cfg": cfg}
    )


rng = jax.random.PRNGKey(1337)
rng_np = np.random.default_rng(1337)
for epoch in trange(wandb.config.epochs):
    train_imgs = train_imgs.shuffle(generator=rng_np)
    batches = train_imgs.shape[0] // args.batch_size
    with tqdm(total=batches, leave=False, desc="train batches") as pbar:
        for batch in train_imgs.iter(batch_size=args.batch_size, drop_last_batch=True):
            batch = jax.device_put(batch["encoded_img"], sharding)
            my_train_state, loss, norm = train_step(my_train_state, batch)
            global_step += 1
            wandb.log(
                {
                    "global_step": global_step,
                    "train/loss": loss,
                    "grad_global_norm": norm,
                }
            )
            pbar.update()
            pbar.set_postfix(loss=f"{loss:.4f}")
            # Save checkpoint every 10 minutes
            if (datetime.datetime.now() - last_checkpoint_time) > datetime.timedelta(
                minutes=10
            ):
                tqdm.write("Saving checkpoint...", end="")
                save_checkpoint()
                last_checkpoint_time = datetime.datetime.now()
                tqdm.write(" DONE")
    # Evaluate on test set
    losses = []
    for batch in tqdm(
        test_imgs.iter(batch_size=args.batch_size),
        total=len(test_imgs) // args.batch_size,
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
    sample_and_log(my_train_state, global_step, sharding)
    tqdm.write(" DONE")
