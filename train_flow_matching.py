"""Train flow matching model."""

import os

# necessary to use more of the GPU's memory. Default is 0.75. It's supposed to be able to
# dynamically allocate more, but there are fragmentation issues since we allocate ginormous arrays.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import datetime
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import matplotlib.pyplot as plt
from distutils.util import strtobool
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec
from math import ceil
from pathlib import Path
from tqdm import tqdm, trange
from typing import Optional

from txt2img_unsupervised.checkpoint import FlowMatchingTrainState
from txt2img_unsupervised.config import (
    FlowMatchingModelConfig,
    TrainingConfig,
)
from txt2img_unsupervised.flow_matching import (
    compute_batch_loss,
    compute_log_probability,
    CapConditionedVectorField,
    LogitsTable,
    sample_loop,
    sample_cap,
    create_mollweide_projection_figure,
)
from txt2img_unsupervised.train_data_loading import get_batch
from txt2img_unsupervised.training_infra import (
    init_common_train_state,
    init_wandb_training,
    IntervalTimer,
    load_dataset,
    plan_steps,
    save_checkpoint,
    setup_common_arguments,
    setup_jax_for_training,
    setup_profiling_server,
    setup_sharding,
    SignalHandler,
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
        default=1,
        help="Number of projections for NLL calculation during evaluation",
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


def mean_cosine_similarity(points: jnp.ndarray) -> float:
    "Compute the mean cosine similarity between all pairs of unit vectors"
    # Compute n x n matrix of singularities. Diagonal is self-similarity (all ones), matrix is
    # symmetric across the diagonal.
    similarities = points @ points.T
    # Set the diagonal to 0.
    similarities = similarities.at[
        jnp.arange(points.shape[0]), jnp.arange(points.shape[0])
    ].set(0)
    # Compute the mean of the off-diagonal elements.
    return jnp.mean(similarities)


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

    # Generate random unit vectors as cap centers for unconditioned sampling
    centers_rng, samples_rng = jax.random.split(rng)

    cap_centers = jax.random.normal(centers_rng, (n_samples, mdl.domain_dim))
    cap_centers = cap_centers / jnp.linalg.norm(cap_centers, axis=1, keepdims=True)

    # Use maximum cap size (2.0) for unconditioned sampling
    cap_d_maxes = jnp.full((n_samples,), 2.0)

    samples = sample_loop(
        mdl,
        params,
        n_samples,
        batch_size,
        samples_rng,
        cap_centers=cap_centers,
        cap_d_maxes=cap_d_maxes,
        n_steps=n_steps,
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


def save_checkpoint_and_evaluate(
    my_train_state,
    global_step: int,
    skip_saving: bool,
    test_dataset,
    training_cfg,
    examples_sharding,
    mdl,
    logits_table,
    vector_column: str = "clip_embedding",
    viz_samples: int = 100,
    viz_batch_size: int = 2048,
    integration_steps: int = 100,
    nll_n_projections: int = 1,
) -> None:
    """Save checkpoint and evaluate on test dataset."""
    save_checkpoint(my_train_state, checkpoint_manager, global_step, skip_saving)

    eval_params = my_train_state.get_eval_params()

    # Generate and visualize samples if domain_dim is 3
    if mdl.domain_dim == 3:
        viz_rng = jax.random.PRNGKey(
            global_step
        )  # Use step as seed for reproducibility
        visualize_model_samples(
            mdl,
            eval_params,
            viz_samples,
            viz_batch_size,
            viz_rng,
            global_step,
            n_steps=integration_steps,
        )

    losses = []
    nlls = []
    test_rng = jax.random.PRNGKey(7357)

    for batch_idx in trange(
        len(test_dataset) // training_cfg.batch_size,
        desc="test batches",
    ):
        test_rng, batch_rng, nll_batch_rng = jax.random.split(test_rng, 3)

        batch = get_batch(
            test_dataset,
            training_cfg.batch_size,
            batch_idx,
            fields=[vector_column],
            sharding=examples_sharding,
        )

        test_batch = {"point_vec": batch[vector_column]}
        loss = compute_batch_loss(
            mdl,
            eval_params,
            test_batch,
            batch_rng,
            logits_table,
        )
        losses.append(loss)

        # Compute NLL using full caps. It's non-obvious what the correct way to test NLL with
        # conditioning is, and the stat without conditioning is clearly useful.
        batch_size = test_batch["point_vec"].shape[0]
        cap_centers = jax.random.normal(nll_batch_rng, (batch_size, mdl.domain_dim))
        cap_centers = cap_centers / jnp.linalg.norm(cap_centers, axis=1, keepdims=True)

        cap_d_maxes = jnp.full((batch_size,), 2.0)

        log_prob = compute_log_probability(
            mdl,
            eval_params,
            test_batch["point_vec"],
            cond_vecs=None,
            n_steps=integration_steps,
            rng=nll_batch_rng,
            n_projections=nll_n_projections,
            cap_centers=cap_centers,
            cap_d_maxes=cap_d_maxes,
        )
        nll = -jnp.mean(log_prob)
        nlls.append(nll)

    test_loss = jnp.mean(jnp.stack(losses))
    test_nll = jnp.mean(jnp.stack(nlls))

    wandb.log(
        {
            "global_step": global_step,
            "test/loss": test_loss,
            "test/nll": test_nll,
        }
    )
    tqdm.write(
        f"Test loss at step {global_step}: {test_loss:.4f}, Test NLL: {test_nll:.4f}"
    )


# Fast path hook that runs operations that don't need the full train state
def fast_post_step_hook(loss, metrics, global_step, norm):
    to_log = {
        "train/loss": loss,
        "grad_global_norm": norm,
        "global_step": global_step,
    }

    if "notfinite_count" in metrics:
        to_log["debug/notfinite_count"] = metrics["notfinite_count"]
    if "clip_count" in metrics:
        to_log["debug/clipped_updates"] = metrics["clip_count"]
    else:
        to_log["debug/clipped_updates"] = 0

    # Log warnings based on metrics
    if not np.isfinite(loss):
        tqdm.write(f"Loss nonfinite 😢 ({loss})")

    if metrics.get("notfinite_count", 0) > 50:
        tqdm.write(f"Too many nonfinite values in gradients, giving up")
        exit(1)

    if metrics.get("clipped_last", False):
        tqdm.write(f"Clipped update due to large gradient norm: {norm}")

    wandb.log(to_log)


# Slow path hook that runs operations that need the full train state
def slow_post_step_hook(loss, state, global_step, norm):
    if signal_handler.exit_requested:
        tqdm.write("Saving checkpoint and exiting early")
        save_checkpoint_and_evaluate(
            state,
            global_step,
            skip_saving=args.skip_saving,
            test_dataset=test_dataset,
            training_cfg=training_cfg,
            examples_sharding=examples_sharding,
            mdl=mdl,
            logits_table=cap_logits_table,
            vector_column=args.vector_column,
            viz_samples=args.viz_samples,
            viz_batch_size=args.viz_batch_size,
            integration_steps=args.integration_steps,
            nll_n_projections=args.nll_n_projections,
        )
        exit(0)

    # If we got a signal to save a checkpoint, do so. If we didn't, save a checkpoint only if it's
    # time.
    if signal_handler.early_checkpoint_requested:
        checkpoint_timer.run_and_reset(
            state,
            global_step,
            test_dataset=test_dataset,
            training_cfg=training_cfg,
            examples_sharding=examples_sharding,
            mdl=mdl,
            logits_table=cap_logits_table,
        )
        signal_handler.reset_checkpoint_flag()
    else:
        checkpoint_timer.check_and_run(state, global_step)

    # Always continue training unless explicitly exited above
    return False


# Function to determine if we need to run the slow path
def slow_path_condition(global_step):
    if signal_handler.exit_requested or signal_handler.early_checkpoint_requested:
        return True
    if checkpoint_timer.check_if_should_run():
        return True
    return False


def post_epoch_hook(state, epoch_idx, global_step):
    checkpoint_timer.run_and_reset(
        state,
        global_step,
        test_dataset=test_dataset,
        training_cfg=training_cfg,
        examples_sharding=examples_sharding,
        mdl=mdl,
        logits_table=cap_logits_table,
    )


def log_test_set_mean_cosine_similarity(test_dataset, vector_column):
    vecs = test_dataset[vector_column]
    mean_sim = mean_cosine_similarity(jax.device_put(vecs))
    wandb.log({"test/mean_cosine_similarity": mean_sim})


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

    checkpoint_timer = IntervalTimer(
        datetime.timedelta(minutes=30),
        lambda state, step, **kwargs: save_checkpoint_and_evaluate(
            state,
            step,
            skip_saving=kwargs.get("skip_saving", args.skip_saving),
            test_dataset=kwargs.get("test_dataset", test_dataset),
            training_cfg=kwargs.get("training_cfg", training_cfg),
            examples_sharding=kwargs.get("examples_sharding", examples_sharding),
            mdl=kwargs.get("mdl", mdl),
            logits_table=kwargs.get("logits_table", cap_logits_table),
            vector_column=kwargs.get("vector_column", args.vector_column),
            viz_samples=kwargs.get("viz_samples", args.viz_samples),
            viz_batch_size=kwargs.get("viz_batch_size", args.viz_batch_size),
            integration_steps=kwargs.get("integration_steps", args.integration_steps),
            nll_n_projections=kwargs.get("nll_n_projections", args.nll_n_projections),
        ),
    )
    signal_handler = SignalHandler()

    @partial(jax.jit, static_argnames=("mdl"))
    def loss_fn(params, batch, rng, mdl=None, logits_table=None):
        # The loss function expects a batch with 'point_vec' key
        flow_batch = {"point_vec": batch[args.vector_column]}
        return compute_batch_loss(mdl, params, flow_batch, rng, logits_table)

    train_state, global_step = train_loop(
        steps_per_epoch=steps_per_epoch,
        total_steps=total_steps,
        complete_epochs=complete_epochs,
        total_epochs=total_epochs,
        steps_in_partial_epoch=steps_in_partial_epoch,
        initial_step=global_step,
        initial_train_state=train_state,
        get_batch_fn=lambda step: (
            lambda b: {
                args.vector_column: b[args.vector_column],
            }
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
        post_step_hook_fn=None,  # Not used with async implementation
        post_epoch_hook_fn=post_epoch_hook,
        fast_post_step_hook_fn=fast_post_step_hook,
        slow_post_step_hook_fn=slow_post_step_hook,
        slow_path_condition_fn=slow_path_condition,
    )

    # Only save a final checkpoint if not at an epoch boundary
    if global_step % steps_per_epoch != 0:
        save_checkpoint_and_evaluate(
            train_state,
            global_step,
            skip_saving=args.skip_saving,
            test_dataset=test_dataset,
            training_cfg=training_cfg,
            examples_sharding=examples_sharding,
            mdl=mdl,
            logits_table=cap_logits_table,
            vector_column=args.vector_column,
            viz_samples=args.viz_samples,
            viz_batch_size=args.viz_batch_size,
            integration_steps=args.integration_steps,
            nll_n_projections=args.nll_n_projections,
        )
