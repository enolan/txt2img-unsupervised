"""Create, load, and save TrainStates and checkpoints."""

import gc
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import pytest
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm import tqdm

from .adaptive_gradient_clip import AdaptiveGradientClipState, adaptive_gradient_clip
from .config import (
    BaseModelConfig,
    EuclideanVDMConfig,
    FlowMatchingModelConfig,
    LearningRateSchedule,
    TrainingConfig,
    TransformerModelConfig,
    migrate_transformer_config_json,
)
from .euclidean_vdm import EuclideanDiffusionModel
from .function_weighted_flow_model import FunctionWeightedFlowModel
from .muon import muon
from .transformer_model import ImageModel, gpt_1_config
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

    # Check if model has "schedule" params (e.g., EuclideanDiffusionModel)
    has_schedule_params = use_mup_scaling and "schedule" in mdl.mk_partition_map(
        use_muon=False
    ).get("params", {})

    # Create schedule optimizer if needed
    def _mk_schedule_opt(base_lr_schedule):
        schedule_lr = training_cfg.schedule_learning_rate or (
            base_lr_schedule
            if not callable(base_lr_schedule)
            else training_cfg.learning_rate
        )
        sched_lr = create_lr_schedule(schedule_lr)
        return optax.adam(
            learning_rate=sched_lr,
            b2=training_cfg.adam_beta2,
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
                learning_rate=training_cfg.learning_rate,
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
            opt_scaled_lr = optax.contrib.schedule_free_adamw(
                learning_rate=mdl.scale_lr(training_cfg.learning_rate),
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
            partition_dict = {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr}
            if has_schedule_params:
                partition_dict["schedule"] = _mk_schedule_opt(
                    training_cfg.learning_rate
                )
            opt = optax.transforms.partition(
                partition_dict,
                mdl.mk_partition_map(use_muon=False),
            )
        else:
            opt = optax.contrib.schedule_free_adamw(
                learning_rate=training_cfg.learning_rate,
                warmup_steps=training_cfg.warmup_steps,
                b1=training_cfg.schedule_free_beta1,
                b2=training_cfg.adam_beta2,
                weight_decay=training_cfg.weight_decay,
            )
    else:
        # Handle regular optimizers (Adam, Muon, or mixed)
        lr_schedule = create_lr_schedule(training_cfg.learning_rate)

        # Common Adam parameters
        adam_params = {
            "weight_decay": training_cfg.weight_decay,
            "b2": training_cfg.adam_beta2,
        }

        if training_cfg.use_muon:
            # Mixed Muon/Adam optimization with muP scaling (always enabled for Muon)
            if callable(lr_schedule):
                scaled_lr_schedule = lambda step: mdl.scale_lr(lr_schedule(step))
            else:
                scaled_lr_schedule = mdl.scale_lr(lr_schedule)

            # Create optimizers for each group
            adam_fixed_opt = optax.adamw(learning_rate=lr_schedule, **adam_params)
            adam_scaled_opt = optax.adamw(
                learning_rate=scaled_lr_schedule, **adam_params
            )
            muon_params = {
                "beta": training_cfg.muon_beta,
                "weight_decay": training_cfg.weight_decay,
            }
            muon_fixed_opt = muon(lr_schedule, **muon_params)
            muon_scaled_opt = muon(scaled_lr_schedule, **muon_params)

            partition_dict = {
                "adam_fixed": adam_fixed_opt,
                "adam_scaled": adam_scaled_opt,
                "muon_fixed": muon_fixed_opt,
                "muon_scaled": muon_scaled_opt,
            }
            if has_schedule_params:
                partition_dict["schedule"] = _mk_schedule_opt(lr_schedule)
            opt = optax.transforms.partition(
                partition_dict,
                mdl.mk_partition_map(use_muon=True),
            )
        else:
            # Pure Adam optimization
            if use_mup_scaling:
                if callable(lr_schedule):
                    scaled_lr_schedule = lambda step: mdl.scale_lr(lr_schedule(step))
                else:
                    scaled_lr_schedule = mdl.scale_lr(lr_schedule)

                opt_fixed_lr = optax.adamw(learning_rate=lr_schedule, **adam_params)
                opt_scaled_lr = optax.adamw(
                    learning_rate=scaled_lr_schedule, **adam_params
                )

                partition_dict = {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr}
                if has_schedule_params:
                    partition_dict["schedule"] = _mk_schedule_opt(lr_schedule)
                opt = optax.transforms.partition(
                    partition_dict,
                    mdl.mk_partition_map(use_muon=False),
                )
            else:
                opt = optax.adamw(learning_rate=lr_schedule, **adam_params)

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
        batches_total: int | None = None,
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
        metadata = checkpoint_metadata(checkpoint_manager)
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

        # muP models partition their params into learning rate groups, each with its own copy of
        # the optimizer
        if isinstance(opt_state, optax.transforms.PartitionState):
            partition_states = [
                masked.inner_state for masked in opt_state.inner_states.values()
            ]
            is_schedule_free = [
                isinstance(s, optax.contrib.ScheduleFreeState) for s in partition_states
            ]
            if all(is_schedule_free):
                return _schedule_free_eval_params_partitioned(
                    partition_states, self.params
                )
            elif any(is_schedule_free):
                raise ValueError(
                    "Some partitions use schedule-free optimizers and some don't"
                )
            else:
                return self.params

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
        return FunctionWeightedFlowModel(**model_cfg.__dict__)


class EuclideanVDMTrainState(BaseTrainState):
    """Train state specific to Euclidean VDM models."""

    @classmethod
    def _create_model_from_config(cls, model_cfg):
        """Create a EuclideanDiffusionModel instance from configuration."""
        if not isinstance(model_cfg, EuclideanVDMConfig):
            raise ValueError(f"Expected EuclideanVDMConfig, got {type(model_cfg)}")
        vf_kwargs = model_cfg.vector_field_kwargs()
        # conditioning_dim is a computed property, not a field
        vf_kwargs.pop("conditioning_dim")
        return EuclideanDiffusionModel(
            **vf_kwargs,
            init_log_snr_min=model_cfg.init_log_snr_min,
            init_log_snr_max=model_cfg.init_log_snr_max,
            schedule_hidden_dim=model_cfg.schedule_hidden_dim,
            schedule_n_quadrature_points=model_cfg.schedule_n_quadrature_points,
            cap_conditioning=model_cfg.cap_conditioning,
            d_max_dist=model_cfg.d_max_dist,
            vlb_variance_loss_weight=model_cfg.vlb_variance_loss_weight,
            sigma_radial=model_cfg.sigma_radial,
            sde_max_leak=model_cfg.sde_max_leak,
            sde_min_d_max=model_cfg.sde_min_d_max,
            classifier_loss_weight=model_cfg.classifier_loss_weight,
            cap_features=model_cfg.cap_features,
            residual_eps=model_cfg.residual_eps,
        )


def _schedule_free_eval_params_partitioned(partition_states, params):
    """
    Eval params for a schedule-free optimizer split up by optax.transforms.partition. Each
    partition's state holds z only for the params it owns, with MaskedNode everywhere else, so
    compute each partition's eval params on the params it owns and merge the results.
    """
    is_masked = lambda x: isinstance(x, optax.transforms.MaskedNode)
    eval_params = params
    for state in partition_states:
        # A MaskedNode in z stands in for a whole subtree of params the partition doesn't own.
        # Expand z to the full param structure, using the params themselves as placeholders
        # there, and mark which leaves the partition owns.
        owned = jax.tree.map(
            lambda z, x: jax.tree.map(lambda _: not is_masked(z), x),
            state.z,
            params,
            is_leaf=is_masked,
        )
        z_full = jax.tree.map(
            lambda z, x: x if is_masked(z) else z, state.z, params, is_leaf=is_masked
        )
        partition_eval_params = optax.contrib.schedule_free_eval_params(
            state._replace(z=z_full), params
        )
        eval_params = jax.tree.map(
            lambda current, new, is_owned: new if is_owned else current,
            eval_params,
            partition_eval_params,
            owned,
        )
    return eval_params


def train_state_class_for_config(model_cfg: BaseModelConfig) -> type[BaseTrainState]:
    """The train state class for a model config."""
    if isinstance(model_cfg, TransformerModelConfig):
        return TransformerTrainState
    elif isinstance(model_cfg, FlowMatchingModelConfig):
        return FlowMatchingTrainState
    elif isinstance(model_cfg, EuclideanVDMConfig):
        return EuclideanVDMTrainState
    else:
        raise ValueError(f"Unknown model config type: {type(model_cfg)}")


def partition_opt_state(opt_state, partitioned_template):
    """
    Reshape the state of an optimizer into the state of the same optimizer wrapped in
    optax.transforms.partition, so that a checkpoint saved before its model had a partition map
    can be resumed. Each partition's state has the structure of the whole optimizer's state with
    MaskedNode in place of the leaves that belong to other partitions, so every leaf of the
    partitioned state is the leaf at the same path in the unpartitioned state, once the
    `.inner_states[label].inner_state` segment is dropped from the path.

    Args:
        opt_state: The unpartitioned optimizer state.
        partitioned_template: The partitioned optimizer's state, or a tree of ShapeDtypeStructs
            with its structure.

    Returns:
        The partitioned optimizer state, with opt_state's arrays.
    """
    unpartitioned_leaves = {
        jax.tree_util.keystr(path): leaf
        for path, leaf in jax.tree_util.tree_leaves_with_path(opt_state)
    }

    def leaf_for(path, template_leaf):
        keys = list(path)
        partition_prefix_starts = [
            i
            for i, k in enumerate(keys)
            if isinstance(k, jax.tree_util.GetAttrKey) and k.name == "inner_states"
        ]
        if partition_prefix_starts:
            i = partition_prefix_starts[0]
            del keys[i : i + 3]
        leaf = unpartitioned_leaves[jax.tree_util.keystr(keys)]
        assert leaf.shape == template_leaf.shape, (
            f"{jax.tree_util.keystr(path)}: {leaf.shape} != {template_leaf.shape}"
        )
        return leaf

    return jax.tree_util.tree_map_with_path(leaf_for, partitioned_template)


def migrate_transformer_checkpoint_to_mup(
    src_dir: Path, dst_dir: Path, step: int | None, batches_total: int | None
) -> None:
    """
    Migrate a transformer checkpoint saved before ImageModel supported muP to the current format,
    so it can be resumed or finetuned from. The model config gets the muP fields with values that
    reproduce the model's existing behaviour (see migrate_transformer_config_json), and the
    optimizer state is reshaped into the per-learning-rate-group partitions muP models use. Params
    and the RNG are copied as they are.

    Args:
        src_dir: Checkpoint directory to migrate.
        dst_dir: Directory to write the migrated checkpoint to.
        step: Which step to migrate, or None for the latest.
        batches_total: Total batches in the training run, needed for learning rate schedules that
            depend on it, otherwise None.
    """
    src_manager = mk_checkpoint_manager(src_dir)
    step = src_manager.latest_step() if step is None else step
    metadata = checkpoint_metadata(src_manager)
    model_cfg_json = migrate_transformer_config_json(metadata["model_cfg"])
    model_cfg = TransformerModelConfig.from_json_dict(model_cfg_json)
    training_cfg = TrainingConfig.from_json_dict(metadata["training_cfg"])
    mdl = ImageModel(**model_cfg.__dict__)

    params_template = jax.eval_shape(
        lambda: mdl.init(jax.random.PRNGKey(0), *mdl.dummy_inputs())
    )
    rng_template = ocp.utils.to_shape_dtype_struct(jax.random.PRNGKey(0))
    # A model with no partition map gets an unpartitioned optimizer
    unpartitioned_opt = setup_optimizer(training_cfg, batches_total, mdl=None)
    partitioned_opt = setup_optimizer(training_cfg, batches_total, mdl=mdl)
    unpartitioned_template = jax.eval_shape(unpartitioned_opt.init, params_template)
    partitioned_template = jax.eval_shape(partitioned_opt.init, params_template)

    print(f"Loading step {step} from {src_dir}")
    restored = src_manager.restore(
        step,
        args=ocp.args.Composite(
            params=ocp.args.StandardRestore(params_template),
            opt_state=ocp.args.StandardRestore(unpartitioned_template),
            rng=ocp.args.ArrayRestore(rng_template),
        ),
    )
    opt_state = partition_opt_state(restored.opt_state, partitioned_template)

    print(f"Saving migrated checkpoint to {dst_dir}")
    dst_manager = ocp.CheckpointManager(
        dst_dir.absolute(),
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
        item_names=("params", "opt_state", "rng"),
        metadata=metadata | {"model_cfg": model_cfg_json},
    )
    dst_manager.save(
        step,
        args=ocp.args.Composite(
            params=ocp.args.StandardSave(restored.params),
            opt_state=ocp.args.StandardSave(opt_state),
            rng=ocp.args.ArraySave(restored.rng),
        ),
    )
    dst_manager.close()


def checkpoint_metadata(checkpoint_manager: ocp.CheckpointManager) -> dict[str, Any]:
    """The metadata dict a checkpoint directory was created with (model and training configs, run
    id, and so on)."""
    return checkpoint_manager.metadata().custom_metadata


def mk_checkpoint_manager(
    checkpoint_dir: Path,
    checkpoint_manager_options: ocp.CheckpointManagerOptions | None = None,
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
    train_state_class = train_state_class_for_config(model_cfg)
    return model_cfg, train_state_class._create_model_from_config(model_cfg)


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
    step: int | None = None,
    device: Literal["gpu", "cpu"] = "gpu",
) -> tuple[dict, int, Any]:
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
    extra_metadata: tuple[str, Any] | None = None,
) -> tuple[ocp.CheckpointManager, TransformerTrainState | FlowMatchingTrainState]:
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

    train_state_class = train_state_class_for_config(model_cfg)
    mdl = train_state_class._create_model_from_config(model_cfg)
    initial_state = train_state_class.new(rng, mdl, training_cfg, batches_total)

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


_tiny_mup_transformer_cfg = replace(
    gpt_1_config,
    n_layers=1,
    d_model=32,
    num_heads=2,
    ff_dim=64,
    image_tokens=16,
    dropout=None,
    pre_norm=True,
)


def _random_grads_like(params, key):
    """A tree of standard normal arrays with params' structure."""
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(
        [jax.random.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)]
    )


def _assert_trees_close(tree_a, tree_b, **kwargs):
    """Assert two pytrees have the same structure and numerically close leaves."""
    assert jax.tree_util.tree_structure(tree_a) == jax.tree_util.tree_structure(tree_b)
    for a, b in zip(
        jax.tree_util.tree_leaves(tree_a), jax.tree_util.tree_leaves(tree_b)
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), **kwargs)


@pytest.mark.parametrize("schedule_free", [True, False])
@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2])
def test_get_eval_params_partitioned(schedule_free, gradient_accumulation_steps):
    """Test eval params with a muP model's partitioned optimizer. At its base width every partition
    gets the same learning rate, so the partitioned optimizer must trace the unpartitioned one
    exactly, eval params included."""
    cfg = replace(_tiny_mup_transformer_cfg, d_model_base=32, head_dim_base=16)
    mdl = ImageModel(**cfg.__dict__)
    assert mdl.scale_lr(1.0) == 1.0
    rng = jax.random.PRNGKey(0)
    params = jax.jit(mdl.init)(rng, *mdl.dummy_inputs())
    if schedule_free:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
            schedule_free_beta1=0.9,
            warmup_steps=5,
            learning_rate=1e-2,
            batch_size=1,
            epochs=1,
            gradient_clipping=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
    else:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.CONSTANT,
            learning_rate=1e-2,
            batch_size=1,
            epochs=1,
            gradient_clipping=None,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    states = [
        TransformerTrainState.create(apply_fn=mdl.apply, params=params, tx=opt, rng=rng)
        for opt in [
            setup_optimizer(training_cfg, batches_total=100, mdl=mdl),
            setup_optimizer(training_cfg, batches_total=100, mdl=None),
        ]
    ]
    partitioned_inner_state = states[0].opt_state.inner_state[-1]
    if gradient_accumulation_steps > 1:
        partitioned_inner_state = partitioned_inner_state.inner_opt_state
    assert isinstance(partitioned_inner_state, optax.transforms.PartitionState)
    step = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))
    for grad_key in jax.random.split(
        jax.random.PRNGKey(1), 10 * gradient_accumulation_steps
    ):
        grads = _random_grads_like(params, grad_key)
        states = [step(state, grads) for state in states]

    partitioned, unpartitioned = states
    _assert_trees_close(partitioned.params, unpartitioned.params, rtol=1e-6, atol=0)
    partitioned_eval = partitioned.get_eval_params()
    _assert_trees_close(
        partitioned_eval, unpartitioned.get_eval_params(), rtol=1e-6, atol=0
    )
    if schedule_free:
        assert not all(
            jnp.allclose(a, b)
            for a, b in zip(
                jax.tree_util.tree_leaves(partitioned_eval),
                jax.tree_util.tree_leaves(partitioned.params),
            )
        )


def test_partition_opt_state():
    """Test that an unpartitioned optimizer state reshaped into partitions matches the state the
    partitioned optimizer would have reached itself, when the partitions share a learning rate."""
    params = {"a": {"kernel": jnp.ones(3), "bias": jnp.ones(2)}, "b": jnp.ones(4)}
    labels = {"a": {"kernel": "scaled", "bias": "fixed"}, "b": "scaled"}
    mk_adam = lambda: optax.contrib.schedule_free_adamw(1e-2, warmup_steps=2)
    unpartitioned = optax.apply_if_finite(optax.chain(optax.identity(), mk_adam()), 20)
    partitioned = optax.apply_if_finite(
        optax.chain(
            optax.identity(),
            optax.transforms.partition(
                {"fixed": mk_adam(), "scaled": mk_adam()}, labels
            ),
        ),
        20,
    )
    states = [unpartitioned.init(params), partitioned.init(params)]
    grads = jax.tree.map(lambda p: 0.5 * p, params)
    for opt, i in [(unpartitioned, 0), (partitioned, 1)]:
        for _ in range(3):
            _, states[i] = opt.update(grads, states[i], params)

    migrated = partition_opt_state(states[0], jax.eval_shape(partitioned.init, params))
    _assert_trees_close(migrated, states[1], rtol=1e-6, atol=0)


@pytest.mark.parametrize("schedule_free", [True, False])
def test_migrate_transformer_checkpoint_to_mup(tmp_path, schedule_free):
    """Test migrating a transformer checkpoint saved before muP: the result loads as a
    TransformerTrainState whose model behaves as the old one did, with the old params, eval params,
    and an optimizer state that continues training identically."""
    old_cfg_json = _tiny_mup_transformer_cfg.to_json_dict()
    for mup_field in [
        "d_model_base",
        "head_dim_base",
        "variance_base",
        "alpha_input",
        "alpha_output",
    ]:
        del old_cfg_json[mup_field]
    # Params have the same shapes whatever the muP settings, so this model stands in for the old
    # one when saving.
    mdl = ImageModel(**_tiny_mup_transformer_cfg.__dict__)
    if schedule_free:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
            schedule_free_beta1=0.9,
            warmup_steps=5,
            learning_rate=1e-2,
            batch_size=1,
            epochs=1,
            gradient_clipping=None,
            gradient_accumulation_steps=1,
        )
    else:
        training_cfg = TrainingConfig(
            learning_rate_schedule=LearningRateSchedule.TRIANGLE,
            learning_rate=1e-2,
            batch_size=1,
            epochs=1,
            gradient_clipping=None,
            gradient_accumulation_steps=1,
        )
    batches_total = 100
    rng = jax.random.PRNGKey(0)
    params = jax.jit(mdl.init)(rng, *mdl.dummy_inputs())
    old_state = TransformerTrainState.create(
        apply_fn=mdl.apply,
        params=params,
        tx=setup_optimizer(training_cfg, batches_total, mdl=None),
        rng=rng,
    )
    step = jax.jit(lambda state, grads: state.apply_gradients(grads=grads))
    grad_keys = jax.random.split(jax.random.PRNGKey(1), 4)
    for grad_key in grad_keys[:3]:
        old_state = step(old_state, _random_grads_like(params, grad_key))

    src_dir = tmp_path / "src"
    src_manager = ocp.CheckpointManager(
        src_dir,
        options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False),
        item_names=("params", "opt_state", "rng"),
        metadata={
            "model_cfg": old_cfg_json,
            "training_cfg": training_cfg.to_json_dict(),
            "run_id": "test",
            "commit_hash": "test",
            "data_offset": 0,
        },
    )
    old_state.save_checkpoint(src_manager, 3)
    src_manager.close()

    dst_dir = tmp_path / "dst"
    migrate_transformer_checkpoint_to_mup(src_dir, dst_dir, None, batches_total)

    dst_manager = mk_checkpoint_manager(dst_dir)
    assert checkpoint_metadata(dst_manager)["run_id"] == "test"
    new_state, new_mdl = TransformerTrainState.load_from_checkpoint(
        dst_manager, 3, batches_total
    )
    assert new_state.step == 3
    assert new_mdl.scale_lr(1.0) == 1.0
    assert new_mdl.hidden_variance_base * new_mdl.d_model_base == 1.0
    assert new_mdl.attention_logit_scale == 1 / np.sqrt(new_mdl.head_dim)
    _assert_trees_close(new_state.params, old_state.params, rtol=0, atol=0)
    _assert_trees_close(
        new_state.get_eval_params(), old_state.get_eval_params(), rtol=1e-6, atol=0
    )

    grads = _random_grads_like(params, grad_keys[3])
    _assert_trees_close(
        step(new_state, grads).params, step(old_state, grads).params, rtol=1e-6, atol=0
    )
