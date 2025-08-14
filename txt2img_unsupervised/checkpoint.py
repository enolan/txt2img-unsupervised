"""Create, load, and save TrainStates and checkpoints."""

import gc
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
import time
import subprocess

from contextlib import nullcontext
from copy import copy
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from pathlib import Path
from tqdm import tqdm
from typing import Any, Literal, Optional, Tuple, Union

from .adaptive_gradient_clip import AdaptiveGradientClipState, adaptive_gradient_clip
from .config import (
    BaseModelConfig,
    FlowMatchingModelConfig,
    LearningRateSchedule,
    TrainingConfig,
    TransformerModelConfig,
)
from .flow_matching import VectorField
from .muon import muon
from .transformer_model import ImageModel
from .triangle_schedule import triangle_schedule


def setup_optimizer(training_cfg: TrainingConfig, batches_total: int, mdl=None):
    """Set up an optimizer based on the TrainingConfig.
    Args:
        training_cfg: The TrainingConfig to set up the optimizer for.
        batches_total: The total number of batches to train for.
        mdl: The model instance, used for muP learning rate scaling if it has mk_partition_map and scale_lr properties.
    Returns:
        An optax optimizer.
    """
    if training_cfg.use_muon and mdl is None:
        raise ValueError("Muon optimizer requires a model instance for muP scaling")
    if training_cfg.use_muon and not (
        hasattr(mdl, "mk_partition_map") and hasattr(mdl, "scale_lr")
    ):
        raise ValueError(
            "Muon optimizer requires a model muP scaling - mk_partition_map and scale_lr methods must exist"
        )

    # Determine if we should use muP scaling
    use_mup_scaling = (
        mdl is not None
        and hasattr(mdl, "mk_partition_map")
        and hasattr(mdl, "scale_lr")
    )

    # Determine learning rates for Adam and Muon
    adam_lr = (
        training_cfg.adam_learning_rate
        if training_cfg.adam_learning_rate is not None
        else training_cfg.learning_rate
    )
    muon_lr = (
        training_cfg.muon_learning_rate
        if training_cfg.muon_learning_rate is not None
        else training_cfg.learning_rate
    )

    # Create learning rate schedules
    def create_lr_schedule(base_lr):
        if training_cfg.learning_rate_schedule == LearningRateSchedule.CONSTANT:
            return base_lr
        elif training_cfg.learning_rate_schedule == LearningRateSchedule.TRIANGLE:
            return triangle_schedule(base_lr, batches_total)
        elif (
            training_cfg.learning_rate_schedule
            == LearningRateSchedule.WARMUP_PLUS_COSINE
        ):
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=base_lr,
                warmup_steps=training_cfg.warmup_steps,
                decay_steps=batches_total,
                end_value=base_lr * 0.05,
            )
        elif (
            training_cfg.learning_rate_schedule
            == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
        ):
            # Schedule-free is handled separately
            return base_lr
        elif (
            training_cfg.learning_rate_schedule
            == LearningRateSchedule.CONSTANT_PLUS_LINEAR_DECAY
        ):
            # Constant learning rate for (batches_total - decay_steps), then linear decay over decay_steps
            constant_schedule = optax.constant_schedule(base_lr)
            decay_schedule = optax.linear_schedule(
                init_value=base_lr,
                end_value=0.0,
                transition_steps=training_cfg.decay_steps,
            )
            return optax.join_schedules(
                schedules=[constant_schedule, decay_schedule],
                boundaries=[batches_total - training_cfg.decay_steps],
            )
        else:
            raise ValueError(
                f"Unknown learning rate schedule {training_cfg.learning_rate_schedule}"
            )

    # Handle schedule-free Adam (not compatible with Muon)
    if (
        training_cfg.learning_rate_schedule
        == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
    ):
        if training_cfg.use_muon:
            raise ValueError("Schedule-free optimizers are not compatible with Muon")

        if use_mup_scaling:
            opt_fixed_lr = optax.contrib.schedule_free_adamw(
                learning_rate=adam_lr,
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
            opt_scaled_lr = optax.contrib.schedule_free_adamw(
                learning_rate=mdl.scale_lr(adam_lr),
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
            opt = optax.transforms.partition(
                {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr},
                mdl.mk_partition_map(use_muon=False),
            )
        else:
            opt = optax.contrib.schedule_free_adamw(
                learning_rate=adam_lr,
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
    else:
        # Handle regular optimizers (Adam, Muon, or mixed)
        adam_lr_schedule = create_lr_schedule(adam_lr)
        muon_lr_schedule = create_lr_schedule(muon_lr)

        # Common Adam parameters
        adam_params = {
            "weight_decay": training_cfg.weight_decay,
            "b2": training_cfg.adam_beta2,
        }

        if training_cfg.use_muon:
            # Mixed Muon/Adam optimization with muP scaling (always enabled for Muon)
            if callable(adam_lr_schedule):
                adam_scaled_lr_schedule = lambda step: mdl.scale_lr(
                    adam_lr_schedule(step)
                )
            else:
                adam_scaled_lr_schedule = mdl.scale_lr(adam_lr_schedule)

            if callable(muon_lr_schedule):
                muon_scaled_lr_schedule = lambda step: mdl.scale_lr(
                    muon_lr_schedule(step)
                )
            else:
                muon_scaled_lr_schedule = mdl.scale_lr(muon_lr_schedule)

            # Create optimizers for each group
            adam_fixed_opt = optax.adamw(learning_rate=adam_lr_schedule, **adam_params)
            adam_scaled_opt = optax.adamw(
                learning_rate=adam_scaled_lr_schedule, **adam_params
            )
            muon_params = {
                "beta": training_cfg.muon_beta,
                "weight_decay": training_cfg.weight_decay,
            }
            muon_fixed_opt = muon(muon_lr_schedule, **muon_params)
            muon_scaled_opt = muon(muon_scaled_lr_schedule, **muon_params)

            opt = optax.transforms.partition(
                {
                    "adam_fixed": adam_fixed_opt,
                    "adam_scaled": adam_scaled_opt,
                    "muon_fixed": muon_fixed_opt,
                    "muon_scaled": muon_scaled_opt,
                },
                mdl.mk_partition_map(use_muon=True),
            )
        else:
            # Pure Adam optimization
            if use_mup_scaling:
                # For function schedules, we need to create a wrapper function for scaling
                if callable(adam_lr_schedule):
                    scaled_lr_schedule = lambda step: mdl.scale_lr(
                        adam_lr_schedule(step)
                    )
                else:
                    scaled_lr_schedule = mdl.scale_lr(adam_lr_schedule)

                opt_fixed_lr = optax.adamw(
                    learning_rate=adam_lr_schedule, **adam_params
                )
                opt_scaled_lr = optax.adamw(
                    learning_rate=scaled_lr_schedule, **adam_params
                )

                opt = optax.transforms.partition(
                    {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr},
                    mdl.mk_partition_map(use_muon=False),
                )
            else:
                opt = optax.adamw(learning_rate=adam_lr_schedule, **adam_params)

    if training_cfg.gradient_accumulation_steps > 1:
        opt = optax.MultiSteps(
            opt, every_k_schedule=training_cfg.gradient_accumulation_steps
        )

    if training_cfg.gradient_clipping is not None:
        clip = optax.clip_by_global_norm(training_cfg.gradient_clipping)
    else:
        clip = optax.identity()
    opt = optax.chain(clip, opt)

    if training_cfg.adaptive_gradient_clip:
        opt = adaptive_gradient_clip(
            opt,
            training_cfg.adaptive_gradient_clip_history_len,
            training_cfg.adaptive_gradient_clip_threshold_factor,
        )

    opt = optax.apply_if_finite(opt, 20)

    return opt


class BaseTrainState(train_state.TrainState):
    """Base class for all train states."""

    rng: jax.Array

    @classmethod
    def new(cls, rng, mdl, training_cfg, batches_total):
        """
        Create a new train state with random initial parameters.

        Args:
            rng: JAX random key
            mdl: The model instance
            training_cfg: Training configuration
            batches_total: Total batches for the training run

        Returns:
            A new train state with initialized parameters
        """
        params = jax.jit(mdl.init)(rng, *mdl.dummy_inputs())
        opt = setup_optimizer(training_cfg, batches_total, mdl=mdl)

        beta2_in_dtype = jnp.astype(training_cfg.adam_beta2, mdl.weights_dtype)
        if beta2_in_dtype >= 1.0:
            raise ValueError(
                f"Adam beta2 {training_cfg.adam_beta2} is too large for {mdl.weights_dtype}"
            )

        return cls.create(
            apply_fn=mdl.apply,
            params=params,
            tx=opt,
            rng=rng,
        )

    def replicate_for_multi_gpu(self, mesh: Mesh):
        """Replicate parameters for multi-GPU training."""
        return jax.device_put(self, NamedSharding(mesh, PartitionSpec()))

    @classmethod
    def _create_model_from_config(cls, model_cfg):
        """Create a model instance from configuration. To be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_manager: ocp.CheckpointManager,
        step: int,
        batches_total: Optional[int] = None,
    ):
        """
        Load a train state from a checkpoint.

        Args:
            checkpoint_manager: The CheckpointManager to load from
            step: The step to load from
            batches_total: Total number of batches (needed for learning rate schedules)

        Returns:
            A tuple of (train_state, model)
        """
        metadata = checkpoint_manager.metadata()
        model_cfg = BaseModelConfig.from_json_dict(metadata["model_cfg"])
        training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])

        # Create model based on config - this is model-specific and implemented by subclasses
        mdl = cls._create_model_from_config(model_cfg)

        # Create a template train state. We need to know the shapes & dtypes to get orbax to load
        # our checkpoint.
        template_state = cls.new(
            jax.random.PRNGKey(0), mdl, training_cfg, batches_total
        )

        # Convert to shape/dtype structs
        params_template = jax.tree.map(
            ocp.utils.to_shape_dtype_struct, template_state.params
        )
        opt_state_template = jax.tree.map(
            ocp.utils.to_shape_dtype_struct, template_state.opt_state
        )
        rng_template = jax.tree.map(ocp.utils.to_shape_dtype_struct, template_state.rng)
        # Save optimizer (python object, no arrays)
        opt = template_state.tx

        # Delete the template to free VRAM.
        del template_state
        gc.collect()

        # Restore from checkpoint using the templates
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(params_template),
                opt_state=ocp.args.StandardRestore(opt_state_template),
                rng=ocp.args.ArrayRestore(rng_template),
            ),
        )

        # Create and return the train state
        train_state = cls(
            apply_fn=mdl.apply,
            params=restored.params,
            tx=opt,
            opt_state=restored.opt_state,
            rng=restored.rng,
            step=step,
        )

        return train_state, mdl

    @jax.jit
    def get_eval_params(self):
        """Get the parameters for evaluation. With schedule-free optimizers this is different than
        the params used for training."""
        # Find the innermost optimizer state
        if isinstance(self.opt_state, optax.ApplyIfFiniteState):
            opt_state = self.opt_state.inner_state
        else:
            opt_state = self.opt_state

        # adaptive gradient clip
        if isinstance(opt_state, AdaptiveGradientClipState):
            opt_state = opt_state.inner_state

        # chain for gradient clipping
        if isinstance(opt_state, tuple):
            opt_state = opt_state[-1]  # The main optimizer is the last in the chain

        # gradient accumulation
        if isinstance(opt_state, optax.MultiStepsState):
            opt_state = opt_state.inner_opt_state

        # Check for schedule-free optimizer
        if isinstance(opt_state, optax.contrib.ScheduleFreeState):
            return optax.contrib.schedule_free_eval_params(opt_state, self.params)

        # For scheduleful optimizers, return the regular params
        return self.params

    def get_last_norm(self):
        """If adaptive gradient clip is enabled, return the norm of the last update. Otherwise
        return None."""
        if isinstance(self.opt_state, optax.ApplyIfFiniteState):
            opt_state = self.opt_state.inner_state
        else:
            opt_state = self.opt_state
        if isinstance(opt_state, AdaptiveGradientClipState):
            return opt_state.last_norm
        return None

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
            # VMs sometimes have small disks. Retrying in a loop gives me an opportunity to go and
            # delete stuff.
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


class TransformerTrainState(BaseTrainState):
    """Train state specific to transformer models."""

    @classmethod
    def _create_model_from_config(cls, model_cfg):
        """Create a transformer model instance from configuration."""
        if not isinstance(model_cfg, TransformerModelConfig):
            raise ValueError(f"Expected TransformerModelConfig, got {type(model_cfg)}")
        return ImageModel(**model_cfg.__dict__)


class FlowMatchingTrainState(BaseTrainState):
    """Train state specific to flow matching models."""

    @classmethod
    def _create_model_from_config(cls, model_cfg):
        """Create a flow matching model instance from configuration."""
        if not isinstance(model_cfg, FlowMatchingModelConfig):
            raise ValueError(f"Expected FlowMatchingModelConfig, got {type(model_cfg)}")
        cfg_dict = copy(model_cfg.__dict__)
        cfg_dict["conditioning_dim"] = 0
        return VectorField(**cfg_dict)


def mk_checkpoint_manager(
    checkpoint_dir: Path,
    checkpoint_manager_options: Optional[ocp.CheckpointManagerOptions] = None,
    for_training: bool = True,
) -> ocp.CheckpointManager:
    """Create a CheckpointManager for a directory that already has checkpoints in it."""
    item_names = ("params", "rng", "opt_state") if for_training else ("params",)
    return ocp.CheckpointManager(
        checkpoint_dir.absolute(),
        item_names=item_names,
        options=checkpoint_manager_options,
    )


def get_model_from_checkpoint(checkpoint_dir: Path):
    """Get the model config and instance from a checkpoint.
    Args:
        checkpoint_dir: The directory path containing the checkpoints.
    Returns:
        A tuple of (model config, model instance)
    """
    metadata = mk_checkpoint_manager(checkpoint_dir).metadata()
    model_cfg = BaseModelConfig.from_json_dict(metadata["model_cfg"])

    # Use the appropriate train state class to create the model based on config type
    if isinstance(model_cfg, TransformerModelConfig):
        return model_cfg, TransformerTrainState._create_model_from_config(model_cfg)
    elif isinstance(model_cfg, FlowMatchingModelConfig):
        return model_cfg, FlowMatchingTrainState._create_model_from_config(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {type(model_cfg)}")


def _init_model_with_dummy_inputs(mdl, rng=None):
    """Initialize model parameters with appropriate dummy inputs based on model type.

    Args:
        mdl: The model instance to initialize
        rng: Optional JAX PRNGKey. If None, creates a new one.

    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    return jax.jit(mdl.init)(rng, *mdl.dummy_inputs())


def load_params(
    checkpoint_dir: Path,
    step: Optional[int] = None,
    device: Literal["gpu", "cpu"] = "gpu",
) -> Tuple[dict, int, Any]:
    """Load the evaluation parameters from a checkpoint.
    Args:
        checkpoint_dir: The directory path containing the checkpoints.
        step: The step to load from. If None, loads the latest checkpoint.
        device: Device to load parameters to. Either "gpu" or "cpu". Defaults to "gpu".
    Returns:
        A tuple containing:
        - A dictionary of parameters
        - The step number
        - The model instance
    """
    if device not in ("gpu", "cpu"):
        raise ValueError(f"Device must be 'gpu' or 'cpu', got '{device}'")

    checkpoint_manager = mk_checkpoint_manager(checkpoint_dir, for_training=False)
    if step is None:
        step = checkpoint_manager.latest_step()

    model_cfg, mdl = get_model_from_checkpoint(checkpoint_dir)

    # Create dummy inputs and initialize model
    params_template = _init_model_with_dummy_inputs(mdl)
    params_template = jax.tree.map(ocp.utils.to_shape_dtype_struct, params_template)
    gc.collect()

    # Load parameters on system memory
    with jax.default_device(jax.devices("cpu")[0]):
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.Composite(
                params=ocp.args.StandardRestore(params_template),
            ),
        )
        params = restored.params

    # Send the params to all GPUs if requested.
    if device == "gpu":
        devices = mesh_utils.create_device_mesh((jax.device_count(),))
        mesh = Mesh(devices, axis_names=("dev",))
        params = jax.device_put(params, NamedSharding(mesh, PartitionSpec(None)))

    return params, step, mdl


def setup_checkpoint_manager_and_initial_state(
    checkpoint_options: ocp.CheckpointManagerOptions,
    checkpoint_dir: Path,
    run_id: str,
    model_cfg: BaseModelConfig,
    training_cfg: TrainingConfig,
    rng: jax.random.PRNGKey,
    batches_total: int,
    data_offset: int = 0,
    extra_metadata: Optional[Tuple[str, Any]] = None,
) -> Tuple[ocp.CheckpointManager, Union[TransformerTrainState, FlowMatchingTrainState]]:
    """
    Set up a CheckpointManager and create an initial TrainState. Does NOT save an initial
    checkpoint.

    Args:
        checkpoint_options: Options for the CheckpointManager.
        checkpoint_dir: Directory to save checkpoints.
        run_id: Unique identifier for the run.
        model_cfg: Model configuration.
        training_cfg: Training configuration.
        rng: JAX random number generator key.
        batches_total: Total number of batches for the entire training run.
        data_offset: Offset in the dataset (for finetuning)
        extra_metadata: Optional extra metadata to include in the checkpoint.

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

    if extra_metadata is not None:
        extra_metadata = {extra_metadata[0]: extra_metadata[1]}
    else:
        extra_metadata = {}

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
            "data_offset": data_offset,
        }
        | extra_metadata,
    )

    # Create the appropriate model and initial train state based on model config type
    if isinstance(model_cfg, TransformerModelConfig):
        mdl = TransformerTrainState._create_model_from_config(model_cfg)
        initial_state = TransformerTrainState.new(rng, mdl, training_cfg, batches_total)
    elif isinstance(model_cfg, FlowMatchingModelConfig):
        mdl = FlowMatchingTrainState._create_model_from_config(model_cfg)
        initial_state = FlowMatchingTrainState.new(
            rng, mdl, training_cfg, batches_total
        )
    else:
        raise ValueError(f"Unsupported model config type: {type(model_cfg)}")

    return checkpoint_manager, initial_state


@pytest.mark.parametrize("schedule_free", [True, False])
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
@pytest.mark.parametrize("gradient_clipping", [None, 1.0])
@pytest.mark.parametrize("adaptive_gradient_clip", [True, False])
def test_get_eval_params(
    adaptive_gradient_clip,
    gradient_clipping,
    gradient_accumulation_steps,
    schedule_free,
):
    # Set up a TrainState with either schedule-free or non-schedule-free optimizer
    rng = jax.random.PRNGKey(0)

    if adaptive_gradient_clip:
        adaptive_gradient_clip_cfg = {
            "adaptive_gradient_clip": True,
            "adaptive_gradient_clip_history_len": 100,
            "adaptive_gradient_clip_threshold_factor": 1.1,
        }
    else:
        adaptive_gradient_clip_cfg = {}

    if schedule_free:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
            schedule_free_beta1=0.9,
            warmup_steps=100,
            learning_rate=1e-4,
            batch_size=128,
            epochs=1,
            gradient_clipping=gradient_clipping,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **adaptive_gradient_clip_cfg,
        )
    else:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.CONSTANT,
            learning_rate=1e-4,
            batch_size=128,
            epochs=1,
            gradient_clipping=gradient_clipping,
            gradient_accumulation_steps=gradient_accumulation_steps,
            **adaptive_gradient_clip_cfg,
        )

    optimizer = setup_optimizer(training_cfg, batches_total=1000, mdl=None)

    # Initialize parameters with random values
    params = jax.random.normal(rng, (10,))

    state = TransformerTrainState.create(
        apply_fn=lambda x: x,  # dummy apply function
        params=params,
        tx=optimizer,
        rng=rng,
    )

    # We have to do some steps for the average to potentially diverge from the original params
    step_grads = jax.jit(
        lambda state: state.apply_gradients(grads=jnp.ones_like(params))
    )
    for _ in range(10 * gradient_accumulation_steps):
        state = step_grads(state)

    # Get eval params
    eval_params = state.get_eval_params()

    if schedule_free:
        # schedule-free eval params should be different from the params for gradient computation
        assert not jnp.allclose(eval_params, state.params)
    else:
        # scheduleful eval params should be the same
        np.testing.assert_array_equal(eval_params, state.params)
