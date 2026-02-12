"""Train flow matching model."""

import os

# necessary to use more of the GPU's memory. Default is 0.75. It's supposed to be able to
# dynamically allocate more, but there are fragmentation issues since we allocate ginormous arrays.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import jax
import wandb
import matplotlib.pyplot as plt
from distutils.util import strtobool
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec
from pathlib import Path
from typing import Optional

from txt2img_unsupervised.checkpoint import FlowMatchingTrainState
from txt2img_unsupervised.config import (
    FlowMatchingModelConfig,
    TrainingConfig,
)
from txt2img_unsupervised.flow_matching import (
    compute_batch_loss,
    LogitsTable,
    create_mollweide_projection_figure,
)
from txt2img_unsupervised.function_weighted_flow_model import (
    BaseDistribution,
    generate_samples,
    compute_nll,
    compute_hemisphere_masses,
    WeightingFunction,
    CapIndicatorExtraParams,
    SmoothedCapIndicatorExtraParams,
    sample_loop,
    sample_full_sphere,
)
from txt2img_unsupervised.train_data_loading import get_batch
from txt2img_unsupervised.training_infra import (
    fast_post_step_hook,
    init_common_train_state,
    init_wandb_training,
    load_dataset,
    log_test_set_mean_cosine_similarity,
    make_checkpoint_hooks,
    mean_cosine_similarity,
    plan_steps,
    save_checkpoint_and_evaluate_vector_model,
    setup_common_arguments,
    setup_jax_for_training,
    setup_profiling_server,
    setup_sharding,
    train_loop,
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser = setup_common_arguments(parser)

    # Add flow matching-specific arguments
    parser.add_argument(
        "--vector-column",
        type=str,
        default="clip_embedding",
        help="Column name in the dataset that contains the vectors to train on",
    )
    parser.add_argument(
        "--viz-samples",
        type=int,
        default=100,
        help="Number of samples to generate for visualization (only used when domain_dim=3)",
    )
    parser.add_argument(
        "--viz-batch-size",
        type=int,
        default=2048,
        help="Number of samples to generate for visualization (only used when domain_dim=3)",
    )
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=16,
        help="Number of integration steps for both sampling and NLL calculation",
    )
    parser.add_argument(
        "--nll-n-projections",
        type=int,
        default=32,
        help="Number of projections for NLL calculation during evaluation",
    )
    parser.add_argument(
        "--nll-batch-size",
        type=int,
        default=None,
        help="Batch size to use for NLL evaluation (defaults to training batch size)",
    )
    parser.add_argument(
        "--max-nll-examples",
        type=int,
        default=1000,
        help="Maximum number of examples to evaluate NLL on",
    )

    # Add arguments for FlowMatchingModelConfig fields
    parser.add_argument("--n-layers", type=int, help="Number of layers in the model")
    parser.add_argument(
        "--domain-dim", type=int, help="Dimension of the domain (sphere dimension)"
    )
    parser.add_argument(
        "--reference-directions",
        type=int,
        help="Number of reference directions for spherical embeddings",
    )
    parser.add_argument("--time-dim", type=int, help="Dimension of time embedding")
    parser.add_argument(
        "--use-pre-mlp-projection",
        type=lambda x: bool(strtobool(x)),
        help="Whether to use pre-MLP projection",
    )
    parser.add_argument("--d-model", type=int, help="Hidden dimension size")
    parser.add_argument(
        "--mlp-expansion-factor", type=int, help="Expansion factor for MLP blocks"
    )
    parser.add_argument(
        "--mlp-dropout-rate", type=float, help="Dropout rate for MLP layers"
    )
    parser.add_argument(
        "--input-dropout-rate", type=float, help="Dropout rate for inputs"
    )
    parser.add_argument("--alpha-input", type=float, help="Alpha scaling for inputs")
    parser.add_argument("--alpha-output", type=float, help="Alpha scaling for outputs")

    # Add arguments for FunctionWeightedFlowModel weighting function configuration
    parser.add_argument(
        "--weighting-function",
        type=str,
        choices=["constant", "cap_indicator", "smoothed_cap_indicator"],
        help="Weighting function type for FunctionWeightedFlowModel",
    )
    args, _unknown = parser.parse_known_args()
    return args


def init_train_state(
    model_cfg: FlowMatchingModelConfig,
    training_cfg: TrainingConfig,
    total_steps: int,
    resume_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_path: Optional[Path] = None,
    start_where_finetune_source_left_off: bool = False,
):
    """Set up our initial FlowMatchingTrainState using the provided configs.

    Args:
        model_cfg: The model configuration
        training_cfg: The training configuration
        total_steps: Total number of training steps
        resume_checkpoint_path: Path to checkpoint to resume from, if any
        finetune_checkpoint_path: Path to checkpoint to finetune from, if any
        start_where_finetune_source_left_off: Whether to start training from where the finetune source left off
    """
    (
        global_step,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    ) = init_common_train_state(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        total_steps=total_steps,
        train_state_class=FlowMatchingTrainState,
        resume_checkpoint_path=resume_checkpoint_path,
        finetune_checkpoint_path=finetune_checkpoint_path,
        start_where_finetune_source_left_off=start_where_finetune_source_left_off,
    )

    print(mdl.tabulate(jax.random.PRNGKey(0), *mdl.dummy_inputs()))

    return (
        global_step,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    )


def visualize_model_samples(mdl, params, n_samples, batch_size, rng, step, n_steps=100):
    """
    Generate samples from the model and visualize them using a Mollweide projection.

    Args:
        mdl: The model instance
        params: Model parameters
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        rng: JAX random key
        step: Current training step (for logging)
        n_steps: Number of integration steps for sample generation

    Returns:
        None, but logs the visualization to wandb
    """
    # Only visualize for 3D models
    if mdl.domain_dim != 3:
        return

    # Generate appropriate weighting function parameters for visualization
    if mdl.weighting_function == WeightingFunction.CONSTANT:
        weighting_function_params = None
        samples = sample_loop(
            mdl,
            params,
            rng,
            weighting_function_params,
            n_samples,
            batch_size,
            n_steps,
        )
    elif mdl.weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        samples = sample_full_sphere(mdl, params, rng, n_samples, batch_size, n_steps)
    else:
        raise ValueError(
            f"Unsupported weighting function for visualization: {mdl.weighting_function}"
        )
    mean_sim = mean_cosine_similarity(samples)

    samples = jax.device_get(samples)
    assert samples.shape == (n_samples, 3), f"Samples shape: {samples.shape}"

    fig = create_mollweide_projection_figure(
        samples, title=f"Flow Matching Model Samples at Step {step}"
    )

    wandb.log(
        {
            "global_step": step,
            "model_samples": wandb.Image(fig),
            "sample_mean_cosine_similarity": mean_sim,
        }
    )

    plt.close(fig)


if __name__ == "__main__":
    args = parse_arguments()

    setup_jax_for_training()
    setup_profiling_server(args.profiling_server)

    wandb_settings = wandb.Settings(code_dir="txt2img_unsupervised")

    # Initialize wandb with the flow matching project
    model_cfg, training_cfg, _ = init_wandb_training(
        args.resume,
        args.model_config,
        args.training_config,
        args,
        wandb_settings,
        project="txt2img-unsupervised-flow",
    )

    train_dataset, test_dataset = load_dataset(args.pq_dir)

    log_test_set_mean_cosine_similarity(test_dataset, args.vector_column)

    (
        steps_per_epoch,
        total_steps,
        complete_epochs,
        total_epochs,
        steps_in_partial_epoch,
    ) = plan_steps(
        train_set_size=train_dataset.shape[0],
        batch_size=training_cfg.batch_size,
        epochs=training_cfg.epochs,
        examples=training_cfg.training_images,
        steps=0,  # We don't use the steps parameter directly
    )

    print(
        f"Training for {total_steps * training_cfg.batch_size} images in {total_steps} steps over {complete_epochs} full epochs plus {steps_in_partial_epoch if steps_in_partial_epoch is not None else 0} extra batches"
    )

    # Initialize the LogitsTable for cap sampling
    # The LogitsTable dimension should be domain_dim - 1
    logits_table_dim = model_cfg.domain_dim - 1
    cap_logits_table = LogitsTable(d=logits_table_dim, n=16384)

    (
        global_step,
        checkpoint_manager,
        train_state,
        mdl,
        data_offset,
    ) = init_train_state(
        model_cfg=model_cfg,
        training_cfg=training_cfg,
        total_steps=total_steps,
        resume_checkpoint_path=args.resume,
        finetune_checkpoint_path=args.finetune,
        start_where_finetune_source_left_off=args.start_where_finetune_source_left_off,
    )

    mesh = setup_sharding(training_cfg.batch_size)
    train_state = train_state.replicate_for_multi_gpu(mesh)
    examples_sharding = NamedSharding(mesh, PartitionSpec("dev"))

    def _compute_loss(eval_params, test_batch, rng):
        return compute_batch_loss(mdl, eval_params, test_batch, rng)

    def _visualize(eval_params, step):
        viz_rng = jax.random.PRNGKey(step)
        visualize_model_samples(
            mdl,
            eval_params,
            args.viz_samples,
            args.viz_batch_size,
            viz_rng,
            step,
            n_steps=args.integration_steps,
        )

    def _nll_setup(eval_params, rng):
        return compute_hemisphere_masses(
            mdl,
            eval_params,
            rng,
            args.integration_steps,
            args.nll_n_projections,
        )

    def _compute_nll(eval_params, test_batch, rng, precomputed_stats):
        return compute_nll(
            mdl,
            eval_params,
            test_batch,
            n_steps=args.integration_steps,
            rng=rng,
            n_projections=args.nll_n_projections,
            precomputed_stats=precomputed_stats,
        )

    save_eval_kwargs = {
        "skip_saving": args.skip_saving,
        "checkpoint_manager": checkpoint_manager,
        "test_dataset": test_dataset,
        "training_cfg": training_cfg,
        "examples_sharding": examples_sharding,
        "vector_column": args.vector_column,
        "compute_loss_fn": _compute_loss,
        "visualize_fn": _visualize,
        "compute_nll_fn": _compute_nll,
        "nll_setup_fn": _nll_setup,
        "nll_batch_size": args.nll_batch_size,
        "max_nll_examples": args.max_nll_examples,
    }

    slow_path_condition, slow_post_step_hook, post_epoch_hook = make_checkpoint_hooks(
        save_checkpoint_and_evaluate_vector_model, save_eval_kwargs
    )

    @partial(jax.jit, static_argnames=("mdl"))
    def loss_fn(params, batch, rng, mdl=None, logits_table=None):
        vecs = batch[args.vector_column]
        flow_batch = {"point_vec": vecs}
        return compute_batch_loss(mdl, params, flow_batch, rng)

    train_state, global_step = train_loop(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        complete_epochs=complete_epochs,
        total_epochs=total_epochs,
        steps_in_partial_epoch=steps_in_partial_epoch,
        initial_step=global_step,
        initial_train_state=train_state,
        get_batch_fn=lambda step: (
            lambda b: {args.vector_column: b[args.vector_column]}
        )(
            get_batch(
                train_dataset,
                training_cfg.batch_size,
                step + data_offset,
                fields=[args.vector_column],
                sharding=examples_sharding,
            )
        ),
        loss_fn=partial(loss_fn, mdl=mdl, logits_table=cap_logits_table),
        post_epoch_hook_fn=post_epoch_hook,
        fast_post_step_hook_fn=fast_post_step_hook,
        slow_post_step_hook_fn=slow_post_step_hook,
        slow_path_condition_fn=slow_path_condition,
    )

    if global_step % steps_per_epoch != 0:
        save_checkpoint_and_evaluate_vector_model(
            train_state, global_step, **save_eval_kwargs
        )
