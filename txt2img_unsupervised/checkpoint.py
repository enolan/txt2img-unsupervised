"""Create, load, and save TrainStates and checkpoints."""

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import time
import subprocess

from flax.training import train_state
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple

from .config import LearningRateSchedule, ModelConfig, TrainingConfig
from .transformer_model import ImageModel
from .triangle_schedule import triangle_schedule


def setup_optimizer(training_cfg: TrainingConfig, batches_total: int):
    """Set up an optimizer based on the TrainingConfig.
    Args:
        training_cfg: The TrainingConfig to set up the optimizer for.
        batches_total: The total number of batches to train for.
    Returns:
        An optax optimizer.
    """

    def get_schedule_error_message(schedule, warmup_required, beta1_required):
        warmup_state = "set" if warmup_required else "unset"
        beta1_state = "set" if beta1_required else "unset"
        return f"{schedule} schedule requires warmup_steps to be {warmup_state} and schedule_free_beta1 to be {beta1_state}"

    if training_cfg.learning_rate_schedule == LearningRateSchedule.CONSTANT:
        if (
            training_cfg.warmup_steps is not None
            or training_cfg.schedule_free_beta1 is not None
        ):
            raise ValueError(get_schedule_error_message("constant", False, False))
    elif training_cfg.learning_rate_schedule == LearningRateSchedule.TRIANGLE:
        if (
            training_cfg.warmup_steps is not None
            or training_cfg.schedule_free_beta1 is not None
        ):
            raise ValueError(get_schedule_error_message("triangle", False, False))
    elif training_cfg.learning_rate_schedule == LearningRateSchedule.WARMUP_PLUS_COSINE:
        if (
            training_cfg.warmup_steps is None
            or training_cfg.schedule_free_beta1 is not None
        ):
            raise ValueError(
                get_schedule_error_message("warmup plus cosine", True, False)
            )
    elif (
        training_cfg.learning_rate_schedule
        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
    ):
        if (
            training_cfg.warmup_steps is None
            or training_cfg.schedule_free_beta1 is None
        ):
            raise ValueError(
                get_schedule_error_message("warmup plus schedule-free", True, True)
            )
    else:
        raise ValueError(
            f"Unknown learning rate schedule {training_cfg.learning_rate_schedule}"
        )
    if training_cfg.learning_rate_schedule == LearningRateSchedule.CONSTANT:
        opt = optax.adam(learning_rate=training_cfg.learning_rate)
    elif training_cfg.learning_rate_schedule == LearningRateSchedule.TRIANGLE:
        opt = optax.adam(
            learning_rate=triangle_schedule(
                training_cfg.learning_rate,
                batches_total,
            )
        )
    elif training_cfg.learning_rate_schedule == LearningRateSchedule.WARMUP_PLUS_COSINE:
        opt = optax.adam(
            learning_rate=optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=training_cfg.learning_rate,
                warmup_steps=training_cfg.warmup_steps,
                decay_steps=batches_total,
                end_value=training_cfg.learning_rate * 0.05,
            )
        )
    elif (
        training_cfg.learning_rate_schedule
        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
    ):
        opt = optax.contrib.schedule_free_adamw(
            learning_rate=training_cfg.learning_rate,
            warmup_steps=training_cfg.warmup_steps,
            b1=training_cfg.schedule_free_beta1,
        )
    else:
        raise ValueError(
            f"Unknown learning rate schedule {training_cfg.learning_rate_schedule}"
        )

    # Apply gradient accumulation if needed
    if training_cfg.gradient_accumulation_steps > 1:
        opt = optax.MultiSteps(
            opt, every_k_schedule=training_cfg.gradient_accumulation_steps
        )

    # Apply gradient clipping if needed
    if training_cfg.gradient_clipping is not None:
        clip = optax.clip_by_global_norm(training_cfg.gradient_clipping)
    else:
        clip = optax.identity()
    opt = optax.chain(clip, opt)

    # Apply finite check
    opt = optax.apply_if_finite(opt, 20)

    return opt


class TrainState(train_state.TrainState):
    rng: jax.Array

    @staticmethod
    def new(rng, mdl, training_cfg, batches_total):
        """Create a new TrainState with random initial parameters."""

        # Set up parameters
        images_dummy, clip_embeddings_dummy, max_cos_distance_dummy = mdl.dummy_inputs()
        params = jax.jit(mdl.init)(
            rng, images_dummy, clip_embeddings_dummy, max_cos_distance_dummy
        )

        opt = setup_optimizer(training_cfg, batches_total)

        return TrainState.create(
            apply_fn=mdl.apply,
            params=params,
            tx=opt,
            rng=rng,
        )

    def replicate_for_multi_gpu(self):
        """Replicate parameters for multi-GPU training."""
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices, axis_names=("dev",))
        replicated_params = jax.device_put(
            self.params, NamedSharding(mesh, PartitionSpec(None))
        )
        return self.replace(params=replicated_params)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_manager: ocp.CheckpointManager,
        step: int,
        batches_total: Optional[int] = None,
    ) -> Tuple["TrainState", ImageModel]:
        """Load a TrainState from a checkpoint.
        Args:
            checkpoint_manager: The CheckpointManager to load from.
            step: The step to load from.
            batches_total: The total number of batches to train for. If None, training may be
                broken, since learning rate schedules depend on knowing the total number of batches.
        Returns:
            A tuple containing the TrainState and the ImageModel.
        """
        metadata = checkpoint_manager.metadata()
        model_cfg = ModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
        mdl = ImageModel(**model_cfg.__dict__)

        # Jitting this means we don't actually do any flops, just compute new random params.
        @jax.jit
        def mk_templates(rng):
            template_ts = TrainState.new(rng, mdl, training_cfg, batches_total=0)
            return (template_ts.params, template_ts.opt_state, template_ts.rng)

        params_template, opt_state_template, rng_template = mk_templates(
            jax.random.PRNGKey(0)
        )
        # Free the memory of the templates as soon as we know the shapes and dtypes
        params_template, opt_state_template, rng_template = map(
            lambda x: jax.tree.map(ocp.utils.to_shape_dtype_struct, x),
            (params_template, opt_state_template, rng_template),
        )

        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(params_template),
                opt_state=ocp.args.StandardRestore(opt_state_template),
                rng=ocp.args.ArrayRestore(rng_template),
            ),
        )

        # Normalize placement of the restored values. Orbax records the sharding in the checkpoint
        # and when you load a checkpoint the arrays are committed to whatever device they were on
        # when it was saved. Sometimes this breaks things.
        params, opt_state, rng = jax.device_get(
            (restored.params, restored.opt_state, restored.rng)
        )
        del restored
        params, opt_state, rng = map(
            lambda x: jax.device_put(x), (params, opt_state, rng)
        )

        opt = setup_optimizer(training_cfg, batches_total)

        train_state = cls(
            apply_fn=mdl.apply,
            params=params,
            tx=opt,
            opt_state=opt_state,
            rng=rng,
            step=step,
        )

        return train_state, mdl

    def get_eval_params(self):
        """Get the parameters for evaluation. With schedule-free optimizers this is different than
        the params used for training."""
        # Find the innermost optimizer state
        if isinstance(self.opt_state, optax.ApplyIfFiniteState):
            opt_state = self.opt_state.inner_state[
                1
            ]  # The main optimizer state is the second element

        # Handle chain (for gradient clipping)
        if isinstance(opt_state, list):
            opt_state = opt_state[-1]  # The main optimizer is the last in the chain

        # Handle gradient accumulation
        if isinstance(opt_state, optax.MultiStepsState):
            opt_state = opt_state.inner_opt_state

        # Check for schedule-free optimizer
        if isinstance(opt_state, optax.contrib.ScheduleFreeState):
            return optax.contrib.schedule_free_eval_params(opt_state, self.params)

        # For all other cases, return the regular params
        return self.params

    def save_checkpoint(
        self, checkpoint_manager: ocp.CheckpointManager, global_step: int
    ) -> None:
        """Save the TrainState to a checkpoint.

        Args:
            checkpoint_manager: The CheckpointManager to save with.
            global_step: The current global step.
        """
        tqdm.write("Attempting to save checkpoint")
        while True:
            try:
                save_args = ocp.args.Composite(
                    params=ocp.args.StandardSave(self.params),
                    opt_state=ocp.args.StandardSave(self.opt_state),
                    rng=ocp.args.ArraySave(self.rng),
                )

                checkpoint_manager.save(global_step, args=save_args)
                tqdm.write(f"Saved checkpoint at step {global_step}")
                break
            except (OSError, ValueError) as e:
                tqdm.write(f"Error saving checkpoint: {e}")
                tqdm.write("Retrying in 60 seconds")
                time.sleep(60)


def mk_checkpoint_manager(checkpoint_dir: Path) -> ocp.CheckpointManager:
    """Create a CheckpointManager for a directory that already has checkpoints in it."""
    item_names = ("params", "opt_state", "rng")
    return ocp.CheckpointManager(checkpoint_dir.absolute(), item_names=item_names)


def get_imagemodel_from_checkpoint(checkpoint_dir: Path) -> ImageModel:
    """Get the ImageModel from a checkpoint.
    Args:
        checkpoint_dir: The directory path containing the checkpoints.
    Returns:
        The ImageModel instance.
    """
    return ModelConfig.from_json_dict(
        mk_checkpoint_manager(checkpoint_dir).metadata()["model_cfg"]
    )


def load_eval_params(
    checkpoint_dir: Path, step: Optional[int] = None
) -> Tuple[dict, int, ImageModel]:
    """Load the evaluation parameters from a checkpoint.
    Args:
        checkpoint_dir: The directory path containing the checkpoints.
        step: The step to load from. If None, loads the latest checkpoint.
    Returns:
        A tuple containing:
        - A dictionary of parameters
        - The step number
        - The ImageModel instance
    """
    checkpoint_manager = mk_checkpoint_manager(checkpoint_dir)
    if step is None:
        step = checkpoint_manager.latest_step()
    ts, mdl = TrainState.load_from_checkpoint(checkpoint_manager, step)
    ts = ts.replicate_for_multi_gpu()
    return ts.get_eval_params(), step, mdl


def setup_checkpoint_manager_and_initial_state(
    checkpoint_options: ocp.CheckpointManagerOptions,
    checkpoint_dir: Path,
    run_id: str,
    model_cfg: ModelConfig,
    training_cfg: TrainingConfig,
    rng: jax.random.PRNGKey,
    batches_total: int,
) -> Tuple[ocp.CheckpointManager, TrainState]:
    """
    Set up a CheckpointManager and create an initial TrainState with a saved checkpoint.

    Args:
        checkpoint_options: Options for the CheckpointManager.
        checkpoint_dir: Directory to save checkpoints.
        run_id: Unique identifier for the run.
        model_cfg: Model configuration.
        training_cfg: Training configuration.
        rng: JAX random number generator key.
        batches_total: Total number of batches for the entire training run.

    Returns:
        A tuple containing the CheckpointManager and the initial TrainState.
    """
    # Ensure the checkpoint directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get the git commit hash
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    # Set up the CheckpointManager
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=checkpoint_options,
        item_names=("params", "opt_state", "rng"),
        metadata={
            "model_cfg": model_cfg.to_json_dict(),
            "training_cfg": training_cfg.to_json_dict(),
            "run_id": run_id,
            "commit_hash": commit_hash,
        },
    )

    # Create the initial TrainState
    mdl = ImageModel(**model_cfg.__dict__)
    initial_state = TrainState.new(rng, mdl, training_cfg, batches_total)

    # Save the initial checkpoint
    initial_state.save_checkpoint(checkpoint_manager, global_step=0)

    return checkpoint_manager, initial_state
