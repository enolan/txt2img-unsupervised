"""Train the image model."""
import argparse
import datetime
import jax
import jax.numpy as jnp
import optax  # type:ignore[import]
import orbax.checkpoint  # type:ignore[import]
import random
import string
import transformer_model
import wandb
from distutils.util import strtobool
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from functools import partial
from pathlib import Path
from tqdm import tqdm, trange
from typing import Any, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("train_dir", type=Path)
parser.add_argument("test_dir", type=Path)
parser.add_argument("batch_size", type=int)
parser.add_argument("epochs", type=int)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--triangle_schedule", type=lambda x: bool(strtobool(x)), default=True)
args, _unknown = parser.parse_known_args()

wandb.init()
global_step: int = 0 # gradients computed so far
wandb.define_metric("*", step_metric="global_step")

# Load the dataset
def load_dir(path: Path) -> jax.Array:
    imgs = []
    for img_path in path.iterdir():
        imgs.append(jnp.load(img_path))
    return jnp.stack(imgs)


train_imgs = load_dir(args.train_dir)
test_imgs = load_dir(args.test_dir)
print(f"Train set {train_imgs.shape}, test set {test_imgs.shape}")

# Setup the model
cfg = transformer_model.gpt_1_config
mdl = transformer_model.ImageModel(**cfg.__dict__)

wandb_config = {}
wandb_config.update(vars(args))
wandb_config.update(cfg.__dict__)
wandb.config.update(wandb_config)

params = mdl.init(
    rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
    image=jnp.zeros((cfg.seq_len,), dtype=jnp.int32),
)


def triangle_schedule(max_lr: float, total_steps: int) -> optax.Schedule:
    """Simple linear trianguar learning rate schedule. Best schedule found in Cramming paper.
    https://arxiv.org/abs/2212.14034"""
    sched_up = optax.linear_schedule(init_value=0, end_value=max_lr, transition_steps=total_steps // 2)
    sched_down = optax.linear_schedule(init_value=max_lr, end_value=0, transition_steps=total_steps // 2)
    return optax.join_schedules([sched_up, sched_down], [total_steps // 2])

if wandb.config.triangle_schedule:
    opt = optax.adam(learning_rate=triangle_schedule(wandb.config.lr, wandb.config.epochs * len(train_imgs)))
else:
    opt = optax.adam(learning_rate=wandb.config.lr)

loss_grad_fn = jax.value_and_grad(transformer_model.loss_batch, argnums=1)
loss_fn = jax.jit(
    lambda params, rng, batch: transformer_model.loss_batch(mdl, params, rng, batch)
)


class TrainState(train_state.TrainState):  # type:ignore[no-untyped-call]
    rng: jax.random.KeyArray  # type:ignore[misc]


@partial(jax.jit, donate_argnums=(0,))
def train_step(
    state: TrainState,
    batch: jax.Array,
) -> Tuple[TrainState, jax.Array]:
    """Compute a single optimization step."""
    rng, rng2 = jax.random.split(state.rng)
    loss, grads = loss_grad_fn(mdl, state.params, rng, batch)
    new_state = state.apply_gradients(
        grads=grads, rng=rng2
    )  # type:ignore[no-untyped-call]
    return new_state, loss


my_train_state: TrainState = TrainState.create(  # type:ignore[no-untyped-call]
    apply_fn=mdl.apply, params=params, tx=opt, rng=jax.random.PRNGKey(1337)
)

run_id = wandb.run.id

checkpoint_dir = Path(f"checkpoints/{run_id}")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
checkpoint_options = orbax.checkpoint.CheckpointManagerOptions(
    max_to_keep=3, keep_time_interval=datetime.timedelta(hours=1)
)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    checkpoint_dir, orbax.checkpoint.PyTreeCheckpointer(), options=checkpoint_options
)

rng = jax.random.PRNGKey(1337)
for epoch in trange(wandb.config.epochs):
    rng, rng2 = jax.random.split(rng)
    train_imgs = jax.random.permutation(rng2, train_imgs, axis=0)
    batches = train_imgs.shape[0] // args.batch_size
    with tqdm(total=batches, leave=False) as pbar:
        for i_batch in range(train_imgs.shape[0] // args.batch_size):
            batch = train_imgs[
                i_batch * args.batch_size : (i_batch + 1) * args.batch_size
            ]
            my_train_state, loss = train_step(my_train_state, batch)
            global_step += 1
            wandb.log({"global_step": global_step, "train/loss": loss})
            pbar.update()
            pbar.set_postfix(loss=f"{loss:.4f}")
    # Evaluate on test set
    losses = []
    for i_batch in range(test_imgs.shape[0] // args.batch_size):
        batch = test_imgs[i_batch * args.batch_size : (i_batch + 1) * args.batch_size]
        losses.append(loss_fn(my_train_state.params, rng, batch))
    test_loss = jnp.mean(jnp.stack(losses))
    wandb.log({"global_step": global_step, "test/loss": test_loss})
    print(
        f"Epoch {epoch} done, train loss: {loss:.4f}, test loss {test_loss:.4f} saving...",
        end="",
        flush=True,
    )
    checkpoint_manager.save(epoch, my_train_state)
    print(" DONE")
