"""
Infrastructure code for training models.
"""

import argparse
import datetime
import importlib.util
import jax
import jax.numpy as jnp
import json
import optax
import orbax.checkpoint as ocp
import signal
import wandb
from copy import copy
from datasets import Dataset
from distutils.util import strtobool
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from math import ceil
from pathlib import Path
from sys import exit
from tqdm import tqdm, trange
from typing import Any, Callable, Dict, List, Optional, Tuple

from txt2img_unsupervised.checkpoint import (
    BaseTrainState,
    FlowMatchingTrainState,
    TransformerTrainState,
    get_model_from_checkpoint,
    mk_checkpoint_manager,
    setup_checkpoint_manager_and_initial_state,
)
from txt2img_unsupervised.config import (
    BaseModelConfig,
    LearningRateSchedule,
    TrainingConfig,
    str_to_activation,
    str_to_dtype,
    str_to_learning_rate_schedule,
)
from txt2img_unsupervised.load_pq_dir import load_pq_dir
from txt2img_unsupervised.train_data_loading import get_batch
import txt2img_unsupervised.config as config
import pytest


def plan_steps(
    train_set_size: int,
    batch_size: int,
    epochs: int = 0,
    examples: int = 0,
    steps: int = 0,
) -> Tuple[int, int, int, int, Optional[int]]:
    """
    Plan the number of epochs and steps to train for. Given a requested number of epochs, examples,
    and steps, this function calculates the steps and epochs needed to train for the sum of all
    three. So for example you could pass epochs=0, examples=20_000, and steps=0 and if the train set
    size was 15_000, and the batch size was 10, this function would return (1500, 2000, 1, 2, 500)
    meaning you'd train for 1 full epoch and a partial epoch of 500 steps for a total of 2000 steps.

    Args:
        train_set_size: The number of training examples.
        batch_size: The number of examples per batch.
        epochs: The number of epochs to train for.
        examples: The number of examples to train for.
        steps: The number of steps to train for.

    Returns:
        A tuple of (steps_per_epoch, total_steps, complete_epochs, total_epochs, steps_in_partial_epoch)
    """
    steps_per_epoch = train_set_size // batch_size

    extra_examples = examples + steps * batch_size
    extra_steps = extra_examples // batch_size
    extra_full_epochs = extra_steps // steps_per_epoch
    extra_steps_in_partial_epoch = extra_steps % steps_per_epoch

    complete_epochs = epochs + extra_full_epochs
    total_epochs = complete_epochs + (1 if extra_steps_in_partial_epoch > 0 else 0)

    return (
        steps_per_epoch,
        complete_epochs * steps_per_epoch + extra_steps_in_partial_epoch,
        complete_epochs,
        total_epochs,
        extra_steps_in_partial_epoch if extra_steps_in_partial_epoch > 0 else None,
    )


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


def argparse_from_dict(d: Dict[str, Any]) -> Callable[[str], Any]:
    """Create an argparse argument type from a dictionary."""

    def f(x: str) -> Any:
        if x in d:
            return d[x]
        else:
            raise argparse.ArgumentTypeError(f"Unknown value {x}")

    return f


def setup_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Add common arguments to an argument parser.

    Args:
        parser: The argument parser to add to

    Returns:
        The updated argument parser
    """
    parser.add_argument("--pq-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--activations-dtype", type=argparse_from_dict(str_to_dtype))
    parser.add_argument("--weights-dtype", type=argparse_from_dict(str_to_dtype))
    parser.add_argument(
        "--activation-function", type=argparse_from_dict(str_to_activation)
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--finetune", type=Path)
    parser.add_argument(
        "--start-where-finetune-source-left-off",
        type=lambda x: bool(strtobool(x)),
        help="start the training data from where the finetune source run left off",
        default=False,
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
        help="Log weights and gradients every N steps",
    )
    # Muon optimizer arguments
    parser.add_argument(
        "--use-muon",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use Muon optimizer for projection matrices",
    )
    parser.add_argument(
        "--muon-beta",
        type=float,
        default=0.95,
        help="Momentum parameter for Muon optimizer",
    )
    parser.add_argument(
        "--muon-learning-rate",
        type=float,
        help="Learning rate for Muon parameters (if None, uses learning_rate)",
    )
    parser.add_argument(
        "--adam-learning-rate",
        type=float,
        help="Learning rate for Adam parameters when using Muon (if None, uses learning_rate)",
    )
    return parser


def setup_profiling_server(profiling_server: bool = False):
    """
    Set up the JAX profiling server if requested.

    Args:
        profiling_server: Whether to start the profiling server
    """
    if profiling_server:
        if importlib.util.find_spec("tensorflow") is None:
            print("You gotta install tensorflow for profiling bro")
            exit(1)

        jax.profiler.start_server(6969)
        print("JAX profiling server started on port 6969")


def json_pretty(dict_obj):
    """
    Print a dictionary as pretty JSON.

    Args:
        dict_obj: Dictionary to format

    Returns:
        Formatted JSON string
    """
    return json.dumps(dict_obj, indent=2)


def init_common_train_state(
    model_cfg: BaseModelConfig,
    training_cfg: TrainingConfig,
    total_steps: int,
    train_state_class: type,
    resume_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_path: Optional[Path] = None,
    start_where_finetune_source_left_off: bool = False,
    create_model_fn: Callable = None,  # Kept for backward compatibility but no longer needed
):
    """
    Set up our initial TrainState using the provided configs.

    Args:
        model_cfg: The model configuration
        training_cfg: The training configuration
        total_steps: Total number of training steps
        train_state_class: The appropriate TrainState class to use (TransformerTrainState or FlowMatchingTrainState)
        resume_checkpoint_path: Path to checkpoint to resume from, if any
        finetune_checkpoint_path: Path to checkpoint to finetune from, if any
        start_where_finetune_source_left_off: Whether to start training from where the finetune source left off
        create_model_fn: Function to create a model instance from the config

    Returns:
        Tuple of (global_step, checkpoint_manager, train_state, model, data_offset)
    """
    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=3,
        keep_time_interval=datetime.timedelta(hours=6),
        # Async checkpointing can hide out of disk errors, so we disable it.
        enable_async_checkpointing=False,
    )

    if resume_checkpoint_path is not None:
        print(f"Resuming from checkpoint {resume_checkpoint_path}...")
        checkpoint_dir = resume_checkpoint_path.absolute()
        checkpoint_manager = mk_checkpoint_manager(checkpoint_dir, checkpoint_options)
        # The step recorded in a checkpoint is the index of the last completed step.
        # We need to start from the next step (index + 1).
        # For example: If checkpoint step is 0, we've completed step 0, so we start at step 1.
        # If checkpoint step is 5, we've completed steps 0-5, so we start at step 6.
        checkpoint_step = checkpoint_manager.latest_step()
        # Start from the next step after the one in the checkpoint
        global_step = checkpoint_step + 1
        data_offset = checkpoint_manager.metadata().get("data_offset", 0)

        # Use the provided train state class to load the checkpoint
        train_state, mdl = train_state_class.load_from_checkpoint(
            checkpoint_manager,
            checkpoint_step,  # Use the original checkpoint step for loading
            total_steps,
        )
    else:
        global_step = 0
        data_offset = 0
        extra_metadata = None

        if finetune_checkpoint_path is not None:
            finetune_src_checkpoint_dir = finetune_checkpoint_path.absolute()
            finetune_src_checkpoint_manager = mk_checkpoint_manager(
                finetune_src_checkpoint_dir
            )
            if start_where_finetune_source_left_off:
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

            extra_metadata = (
                "finetune_src_config",
                finetune_src_checkpoint_manager.metadata(),
            )

        # Set up checkpoint manager and initial state
        checkpoint_dir = Path(f"checkpoints/{wandb.run.id}").absolute()
        checkpoint_manager, train_state = setup_checkpoint_manager_and_initial_state(
            checkpoint_options,
            checkpoint_dir,
            wandb.run.id,
            model_cfg,
            training_cfg,
            jax.random.PRNGKey(1337),
            total_steps,
            data_offset=data_offset,
            extra_metadata=extra_metadata,
        )

        if finetune_checkpoint_path is not None:
            train_state = train_state.replace(params=None)
            import gc

            gc.collect()
            print(f"Loading params from {finetune_checkpoint_path} for finetuning...")

            # Get model from finetune source checkpoint
            _, finetune_src_mdl = get_model_from_checkpoint(finetune_checkpoint_path)

            # Use the provided train state class for loading the finetune source
            finetune_src_ts, finetune_src_mdl = train_state_class.load_from_checkpoint(
                finetune_src_checkpoint_manager,
                finetune_src_checkpoint_manager.latest_step(),
                total_steps,
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

        # Initialize the model using the train state class's model creation method
        mdl = train_state_class._create_model_from_config(model_cfg)

    return (
        global_step,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    )


def load_dataset(dir: Path) -> Tuple[Dataset, Dataset]:
    """
    Load dataset and split into train/test sets.

    Args:
        dir: Path to the dataset directory

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    dset_all = load_pq_dir(dir)
    dset_split = dset_all.train_test_split(test_size=10_000, seed=19900515)
    train_dset = dset_split["train"]
    test_dset = dset_split["test"]
    print(f"Train set {train_dset.shape}, test set {test_dset.shape}")
    wandb.config.update(
        {"train_set_size": len(train_dset), "test_set_size": len(test_dset)}
    )
    return train_dset, test_dset


def setup_sharding(batch_size):
    """
    Set up sharding across devices.

    Args:
        batch_size: Batch size to shard

    Returns:
        Mesh for sharding
    """
    print(
        f"Sharding batches of {batch_size} across {jax.device_count()} devices, {batch_size / jax.device_count()} per device"
    )
    assert batch_size % jax.device_count() == 0
    # NamedSharding is overkill for the simple batch parallelism we do, but it's necessary to get
    # orbax to save checkpoints correctly.
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    return mesh


def leading_dims_to_subtrees(tree):
    """
    Given a dict pytree, return a new dict pytree with each array split into a dict of arrays
    indexed by leading dimension.
    """
    if not isinstance(tree, dict):
        raise ValueError(f"Expected dict, got {type(tree)}")
    out = {}
    for k, v in tree.items():
        if isinstance(v, dict):
            out[k] = leading_dims_to_subtrees(v)
        elif (
            isinstance(v, jnp.ndarray)
            or isinstance(v, jax.Array)
            or isinstance(v, jnp.number)
        ):
            if v.ndim < 2:
                raise ValueError(f"Expected array with at least 2 dims, got {v.ndim}")
            else:
                out[k] = {f"{idx:03d}": v[idx] for idx in range(v.shape[0])}
        else:
            raise ValueError(f"Unknown type: {type(v)} for key {k}")
    return out


def save_checkpoint(
    train_state, checkpoint_manager, global_step, skip_saving: bool
) -> None:
    """
    Save a checkpoint.

    Args:
        train_state: The training state to save
        checkpoint_manager: The checkpoint manager
        global_step: The current global step
        skip_saving: Whether to skip saving
    """
    if not skip_saving:
        train_state.save_checkpoint(checkpoint_manager, global_step)
        tqdm.write("Saved checkpoint")
    else:
        tqdm.write("Skipping checkpoint save")


def init_wandb_training(
    resume_checkpoint_path,
    model_cfg_path,
    training_cfg_path,
    args,
    wandb_settings,
    extra_config=None,
    project=None,
):
    """
    Initialize Weights & Biases for training.

    Args:
        resume_checkpoint_path: Path to checkpoint to resume from, if any
        model_cfg_path: Path to model config file
        training_cfg_path: Path to training config file
        args: Parsed command-line arguments
        wandb_settings: W&B settings
        extra_config: Optional additional config to log to W&B
        project: Optional project name for wandb (defaults to "txt2img-unsupervised")

    Returns:
        Tuple of (model_cfg, training_cfg, extra config values)
    """
    # Load and set up configs
    extra_values = {}
    if extra_config:
        extra_values = extra_config

    if resume_checkpoint_path is not None:
        # Resuming from checkpoint - load configs from checkpoint
        print(f"Resuming from checkpoint {resume_checkpoint_path}...")
        checkpoint_dir = resume_checkpoint_path.absolute()
        checkpoint_options = ocp.CheckpointManagerOptions(
            max_to_keep=3,
            keep_time_interval=datetime.timedelta(hours=6),
            enable_async_checkpointing=False,
        )
        checkpoint_manager = mk_checkpoint_manager(checkpoint_dir, checkpoint_options)
        metadata = checkpoint_manager.metadata()
        model_cfg = BaseModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
        run_id = metadata["run_id"]

        # Extract any extra values from metadata
        if extra_config:
            for key in extra_config.keys():
                extra_values[key] = metadata.get(key, extra_config[key])

        print(f"Resuming run {run_id}")
        print(
            "ALL TRAINING AND MODEL PARAMETERS PASSED ON THE COMMAND LINE WILL BE IGNORED."
        )
        print(f"Model Config {json_pretty(model_cfg.to_json_dict())}")
        print(f"TrainingConfig {json_pretty(training_cfg.to_json_dict())}")

        # Use the provided project name or default
        wandb_init_kwargs = {"id": run_id, "resume": "must", "settings": wandb_settings}
        if project:
            wandb_init_kwargs["project"] = project
        wandb.init(**wandb_init_kwargs)
    else:
        print("Starting new run...")

        # Use the provided project name or default
        wandb_init_kwargs = {"settings": wandb_settings}
        if project:
            wandb_init_kwargs["project"] = project
        wandb.init(**wandb_init_kwargs)

        with open(model_cfg_path) as f:
            model_cfg = BaseModelConfig.from_json_dict(json.load(f))
        config.merge_attrs(model_cfg, args)

        with open(training_cfg_path) as f:
            training_cfg = TrainingConfig.from_json_dict(json.load(f))
        config.merge_attrs(training_cfg, args)

        # Send config to wandb
        wandb.config.update(model_cfg.to_json_dict())
        wandb.config.update(training_cfg.to_json_dict())
        if extra_config:
            wandb.config.update(extra_values)

        # Read potentially sweep-controlled parameters from wandb
        model_cfg = BaseModelConfig.from_json_dict(wandb.config.as_dict())
        training_cfg = TrainingConfig.from_json_dict(wandb.config.as_dict())
        if extra_config:
            for key in extra_config.keys():
                extra_values[key] = wandb.config.get(key, extra_config[key])

        print(f"Model config post-wandb: {json_pretty(model_cfg.to_json_dict())}")
        print(f"Training config post-wandb: {json_pretty(training_cfg.to_json_dict())}")

    # Set up wandb metrics
    wandb.define_metric("*", step_metric="global_step")
    wandb.define_metric("test/loss", summary="last")
    wandb.define_metric("train/loss", summary="last")

    return model_cfg, training_cfg, extra_values


def setup_jax_for_training():
    """
    Configure JAX for training.
    """
    # Enable caching of JIT-compiled functions
    jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")
    # Make the RNG partitionable across devices
    jax.config.update("jax_threefry_partitionable", True)


@pytest.mark.parametrize(
    "train_set_size, batch_size, epochs, examples, steps, expected_steps_per_epoch, expected_total_steps, expected_complete_epochs, expected_total_epochs, expected_steps_in_partial_epoch",
    [
        # Example from docstring
        (15_000, 10, 0, 20_000, 0, 1500, 2000, 1, 2, 500),
        # Training for exactly one epoch
        (10_000, 32, 1, 0, 0, 312, 312, 1, 1, None),
        # Training for a specific number of steps
        (5_000, 16, 0, 0, 500, 312, 500, 1, 2, 188),
        # Combination of epochs and examples
        (8_000, 64, 2, 3_200, 0, 125, 300, 2, 3, 50),
        # Multiple epochs with no partial epoch
        (6_000, 50, 3, 0, 0, 120, 360, 3, 3, None),
        # Using all three parameters
        (4_000, 25, 1, 1_000, 20, 160, 220, 1, 2, 60),
    ],
)
def test_plan_steps(
    train_set_size,
    batch_size,
    epochs,
    examples,
    steps,
    expected_steps_per_epoch,
    expected_total_steps,
    expected_complete_epochs,
    expected_total_epochs,
    expected_steps_in_partial_epoch,
):
    """Test that plan_steps works correctly with various inputs."""
    (
        steps_per_epoch,
        total_steps,
        complete_epochs,
        total_epochs,
        steps_in_partial_epoch,
    ) = plan_steps(
        train_set_size=train_set_size,
        batch_size=batch_size,
        epochs=epochs,
        examples=examples,
        steps=steps,
    )

    assert steps_per_epoch == expected_steps_per_epoch
    assert total_steps == expected_total_steps
    assert complete_epochs == expected_complete_epochs
    assert total_epochs == expected_total_epochs
    assert steps_in_partial_epoch == expected_steps_in_partial_epoch


def make_train_step_with_metrics(loss_fn):
    @partial(jax.jit, donate_argnames=["state"])
    def _train_step_with_metrics(
        state: BaseTrainState, batch: Any
    ) -> Tuple[BaseTrainState, float, float, Dict[str, Any]]:
        """
        Performs a single training step, extracting metrics before donation.

        Args:
            state: Current training state
            batch: Batch of data to train on

        Returns:
            Tuple of (new_state, loss, gradient_norm, metrics_dict)
        """
        step_rng, new_rng = jax.random.split(state.rng)

        # Extract any metrics we need to preserve before donation
        metrics = {}
        if hasattr(state, "opt_state"):
            if hasattr(state.opt_state, "notfinite_count"):
                metrics["notfinite_count"] = state.opt_state.notfinite_count.copy()
            if hasattr(state.opt_state, "inner_state") and hasattr(
                state.opt_state.inner_state, "clip_count"
            ):
                metrics["clip_count"] = state.opt_state.inner_state.clip_count.copy()
                if hasattr(state.opt_state.inner_state, "clipped_last"):
                    metrics[
                        "clipped_last"
                    ] = state.opt_state.inner_state.clipped_last.copy()

        grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        loss, grads = grad_fn(state.params, batch, step_rng)
        new_state = state.apply_gradients(grads=grads)

        # Use the gradient norm from state if available, otherwise compute it
        if new_state.get_last_norm() is not None:
            norm = new_state.get_last_norm()
        else:
            norm = optax.global_norm(grads)

        new_state = new_state.replace(rng=new_rng)
        return new_state, loss, norm, metrics

    return _train_step_with_metrics


def train_loop_async(
    steps_per_epoch: int,
    total_steps: int,
    complete_epochs: int,
    total_epochs: int,
    steps_in_partial_epoch: Optional[int],
    initial_step: int,
    initial_train_state: BaseTrainState,
    get_batch_fn: Callable[[int], Any],
    loss_fn: Callable[[Any, Any, jax.random.PRNGKey], float],
    fast_post_step_hook_fn: Callable[[float, Dict[str, Any], int, float], None],
    slow_post_step_hook_fn: Callable[[float, BaseTrainState, int, float], bool],
    slow_path_condition_fn: Callable[[int], bool],
    post_epoch_hook_fn: Optional[Callable[[BaseTrainState, int, int], None]] = None,
) -> Tuple[Any, int]:
    """
    Runs an asynchronous training loop that maximizes GPU utilization. In general, this keeps at
    least one batch in flight on the GPU at all times, enqueueing the next batch before using the
    results of the previous one.

    Args:
        steps_per_epoch: Number of steps per epoch.
        total_steps: Total number of steps to train for.
        complete_epochs: Number of full epochs to train for.
        total_epochs: Total number of epochs (including partial).
        steps_in_partial_epoch: Number of steps in the final partial epoch, if any.
        initial_step: The step number to start training from (0-indexed).
        initial_train_state: The initial state of the training process (must have apply_gradients method).
        get_batch_fn: Function to fetch a batch of data. Takes (step: int) -> batch.
        loss_fn: Function to calculate the loss. Takes (params: Any, batch: Any, rng: jax.random.PRNGKey) -> loss: float.
                 This function will be used with jax.value_and_grad.
        fast_post_step_hook_fn: Callback to run after every step. Cannot access the train state.
                                Takes (loss: float, metrics: Dict[str, Any], step: int, norm: float) -> None.
        slow_post_step_hook_fn: Callback to run after train steps for which slow_path_condition_fn
                                returns True. Can access the train state.
                                Takes (loss: float, state: BaseTrainState, step: int, norm: float) -> should_stop: bool.
        slow_path_condition_fn: Function that determines if slow path should run. As the name
                                implies, this slows down training substantially, since we can't have
                                two batches in flight if we need to access the state between steps.
                                Takes (step: int) -> should_run_slow_path: bool.
        post_epoch_hook_fn: Optional function called after each epoch.
                           Takes (state: BaseTrainState, epoch_idx: int, global_step: int).

    Returns:
        A tuple containing the final training state and the final step number (the number of steps completed).
    """
    train_state = initial_train_state
    global_step = initial_step
    start_epoch = initial_step // steps_per_epoch

    # Create our training step function
    _train_step_with_metrics = make_train_step_with_metrics(loss_fn)

    pbar_epoch = trange(
        start_epoch,
        total_epochs,
        initial=start_epoch,
        total=total_epochs,
        desc=f"Epoch",
        unit="epoch",
        leave=True,
        position=0,
    )

    # Start the first step before entering the main loop
    if global_step < total_steps:
        initial_batch = get_batch_fn(global_step)
        (
            current_state,
            current_loss,
            current_norm,
            current_metrics,
        ) = _train_step_with_metrics(train_state, initial_batch)
    else:
        # Already done
        return train_state, global_step

    for current_epoch_idx in pbar_epoch:
        is_partial_epoch = (current_epoch_idx == total_epochs - 1) and (
            steps_in_partial_epoch is not None
        )
        steps_in_this_epoch = (
            steps_in_partial_epoch if is_partial_epoch else steps_per_epoch
        )

        epoch_start_step = current_epoch_idx * steps_per_epoch
        epoch_end_step = epoch_start_step + steps_in_this_epoch

        start_step_in_epoch = 0
        if global_step > epoch_start_step:
            start_step_in_epoch = global_step - epoch_start_step

        pbar_step = trange(
            start_step_in_epoch,
            steps_in_this_epoch,
            initial=start_step_in_epoch,
            total=steps_in_this_epoch,
            desc=f"Step",
            unit="step",
            leave=False,
            position=1,
        )

        for step_in_epoch_idx in pbar_step:
            current_step = global_step
            global_step = current_step + 1  # Update for next iteration

            # Check if we need the slow path
            need_slow_path = slow_path_condition_fn(current_step)

            if need_slow_path:
                # Call fast path hook first (with metrics extracted before donation)
                fast_post_step_hook_fn(
                    current_loss, current_metrics, current_step, current_norm
                )

                # Run slow path hook with full train state (will materialize implicitly when used)
                should_stop = slow_post_step_hook_fn(
                    current_loss, current_state, current_step, current_norm
                )

                if should_stop:
                    tqdm.write(
                        f"Stopping training early after step {current_step} due to slow_post_step_hook."
                    )
                    pbar_step.close()
                    pbar_epoch.close()
                    return current_state, global_step

                # Get ready for next step after slow path is done
                if global_step < total_steps:
                    next_batch = get_batch_fn(global_step)
                    (
                        current_state,
                        current_loss,
                        current_norm,
                        current_metrics,
                    ) = _train_step_with_metrics(current_state, next_batch)
                else:
                    # Done with all steps
                    break
            else:
                # Get the next batch and enqueue right away (fast path)
                if global_step < total_steps:
                    next_batch = get_batch_fn(global_step)
                    (
                        next_state,
                        next_loss,
                        next_norm,
                        next_metrics,
                    ) = _train_step_with_metrics(current_state, next_batch)

                # Call fast path hook
                fast_post_step_hook_fn(
                    current_loss, current_metrics, current_step, current_norm
                )

                # Move to next step
                if global_step < total_steps:
                    current_state = next_state
                    current_loss = next_loss
                    current_norm = next_norm
                    current_metrics = next_metrics
                else:
                    # Done with all steps
                    break

            pbar_step.set_description(f"Loss: {current_loss:.4f}")

        pbar_step.close()

        # Run post-epoch hook if provided
        if post_epoch_hook_fn is not None:
            tqdm.write(
                f"Epoch {current_epoch_idx + 1} finished. Running post-epoch hook..."
            )
            post_epoch_hook_fn(current_state, current_epoch_idx, current_step)
            tqdm.write("Post-epoch hook complete.")

    pbar_epoch.close()
    return current_state, global_step


def train_loop(
    steps_per_epoch: int,
    total_steps: int,
    complete_epochs: int,
    total_epochs: int,
    steps_in_partial_epoch: Optional[int],
    initial_step: int,
    initial_train_state: BaseTrainState,
    get_batch_fn: Callable[[int], Any],
    loss_fn: Callable[[Any, Any, jax.random.PRNGKey], float],
    fast_post_step_hook_fn: Callable[[float, Dict[str, Any], int, float], None],
    slow_post_step_hook_fn: Callable[[float, BaseTrainState, int, float], bool],
    slow_path_condition_fn: Callable[[int], bool],
    post_epoch_hook_fn: Optional[Callable[[BaseTrainState, int, int], None]] = None,
    post_step_hook_fn: Optional[
        Callable[[float, Any, int, float], bool]
    ] = None,  # Kept for API compatibility but not used
) -> Tuple[Any, int]:
    """
    Entry point for asynchronous training loop that maximizes GPU utilization.

    Args:
        steps_per_epoch: Number of steps per epoch.
        total_steps: Total number of steps to train for.
        complete_epochs: Number of full epochs to train for.
        total_epochs: Total number of epochs (including partial).
        steps_in_partial_epoch: Number of steps in the final partial epoch, if any.
        initial_step: The step number to start training from (0-indexed).
        initial_train_state: The initial state of the training process (must have apply_gradients method).
        get_batch_fn: Function to fetch a batch of data. Takes (step: int) -> batch.
        loss_fn: Function to calculate the loss. Takes (params: Any, batch: Any, rng: jax.random.PRNGKey) -> loss: float.
                 This function will be used with jax.value_and_grad.
        fast_post_step_hook_fn: Function for fast operations that don't need realized train state.
                                Takes (loss: float, metrics: Dict[str, Any], step: int, norm: float) -> None.
        slow_post_step_hook_fn: Function for slow operations that need realized train state.
                                Takes (loss: float, state: BaseTrainState, step: int, norm: float) -> should_stop: bool.
        slow_path_condition_fn: Function that determines if slow path should run.
                                Takes (step: int) -> should_run_slow_path: bool.
        post_epoch_hook_fn: Optional function called after each epoch. Takes (state: BaseTrainState, epoch_idx: int, global_step: int).
        post_step_hook_fn: Legacy parameter, not used but kept for API compatibility.

    Returns:
        A tuple containing the final training state and the final step number (the number of steps completed).
    """
    return train_loop_async(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        complete_epochs=complete_epochs,
        total_epochs=total_epochs,
        steps_in_partial_epoch=steps_in_partial_epoch,
        initial_step=initial_step,
        initial_train_state=initial_train_state,
        get_batch_fn=get_batch_fn,
        loss_fn=loss_fn,
        fast_post_step_hook_fn=fast_post_step_hook_fn,
        slow_post_step_hook_fn=slow_post_step_hook_fn,
        slow_path_condition_fn=slow_path_condition_fn,
        post_epoch_hook_fn=post_epoch_hook_fn,
    )


class IntervalTimer:
    """
    Helper class to track time intervals and trigger actions periodically by calling a callback.
    """

    def __init__(self, interval: datetime.timedelta, callback_fn: Callable[..., Any]):
        """
        Initialize the timer with a specific interval and a callback function.

        Args:
            interval: The time duration that needs to pass before check_and_run() calls the callback.
            callback_fn: The function to call when the interval has passed. It can accept arguments.
        """
        if not isinstance(interval, datetime.timedelta):
            raise ValueError("Interval must be a datetime.timedelta object")
        if interval <= datetime.timedelta(0):
            raise ValueError("Interval must be positive")
        if not callable(callback_fn):
            raise ValueError("callback_fn must be callable")

        self.interval = interval
        self.callback_fn = callback_fn
        # Initialize last_trigger_time to the past so the first check triggers immediately
        self.last_trigger_time = datetime.datetime.min

    def check_if_should_run(self) -> bool:
        """
        Check if the specified interval has passed since the last time the callback was run.

        Returns:
            True if the interval has passed, False otherwise.
        """
        now = datetime.datetime.now()
        return now - self.last_trigger_time >= self.interval

    def check_and_run(self, *args, **kwargs) -> bool:
        """
        Check if the specified interval has passed since the last time the callback was run.

        If the interval has passed, runs the callback with the provided *args and **kwargs,
        updates the internal last trigger time, and returns True. Otherwise, returns False.

        Args:
            *args: Positional arguments to pass to the callback function.
            **kwargs: Keyword arguments to pass to the callback function.

        Returns:
            True if the callback was run, False otherwise.
        """
        now = datetime.datetime.now()
        if now - self.last_trigger_time >= self.interval:
            self.callback_fn(*args, **kwargs)
            self.last_trigger_time = now
            return True
        return False

    def run_and_reset(self, *args, **kwargs):
        """
        Unconditionally run the callback function with the provided *args and **kwargs,
        and reset the timer's last trigger time.

        Args:
            *args: Positional arguments to pass to the callback function.
            **kwargs: Keyword arguments to pass to the callback function.
        """
        now = datetime.datetime.now()
        self.callback_fn(*args, **kwargs)
        self.last_trigger_time = now
