"""Train score matching model."""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import jax
import jax.numpy as jnp
import wandb
import matplotlib.pyplot as plt
from distutils.util import strtobool
from functools import partial
from jax.sharding import NamedSharding, PartitionSpec
from pathlib import Path
from typing import Optional

from txt2img_unsupervised.checkpoint import ScoreMatchingTrainState
from txt2img_unsupervised.config import ScoreMatchingModelConfig, TrainingConfig
from txt2img_unsupervised.flow_matching import create_mollweide_projection_figure
from txt2img_unsupervised.score_matching import (
    compute_batch_loss,
    compute_nll,
    generate_samples,
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
        help="Batch size for sample generation during visualization",
    )
    parser.add_argument(
        "--integration-steps",
        type=int,
        default=100,
        help="Number of integration steps for sampling and NLL calculation",
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
        # NLL calculation is SUPER slow, so we only evaluate on a small subset of test examples.
        # this is bad, but we should be able to increase the number with DPM-Solver++ in the future
        default=250,
        help="Maximum number of examples to evaluate NLL on",
    )

    # Model architecture arguments
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

    # Noise schedule arguments
    parser.add_argument(
        "--sigma-sq-min",
        type=float,
        help="Minimum sigma squared (at t=1, near data)",
    )
    parser.add_argument(
        "--sigma-sq-max",
        type=float,
        help="Maximum sigma squared (at t=0, near noise)",
    )

    args, _unknown = parser.parse_known_args()
    return args


def init_train_state(
    model_cfg: ScoreMatchingModelConfig,
    training_cfg: TrainingConfig,
    total_steps: int,
    resume_checkpoint_path: Optional[Path] = None,
    finetune_checkpoint_path: Optional[Path] = None,
    start_where_finetune_source_left_off: bool = False,
):
    """Set up our initial ScoreMatchingTrainState using the provided configs."""
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
        train_state_class=ScoreMatchingTrainState,
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
    """Generate samples from the model and visualize them using a Mollweide projection."""
    if mdl.domain_dim != 3:
        return

    all_samples = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in range(n_batches):
        batch_rng, rng = jax.random.split(rng)
        this_batch_size = min(batch_size, n_samples - i * batch_size)
        cond_vecs = jnp.zeros((this_batch_size, 0))
        batch_samples = generate_samples(
            mdl, params, batch_rng, cond_vecs, n_steps=n_steps
        )
        all_samples.append(batch_samples)

    samples = jnp.concatenate(all_samples, axis=0)[:n_samples]
    mean_sim = mean_cosine_similarity(samples)

    samples = jax.device_get(samples)

    fig = create_mollweide_projection_figure(
        samples, title=f"Score Matching Model Samples at Step {step}"
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

    model_cfg, training_cfg, _ = init_wandb_training(
        args.resume,
        args.model_config,
        args.training_config,
        args,
        wandb_settings,
        project="txt2img-unsupervised-score",
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
        steps=0,
    )

    print(
        f"Training for {total_steps * training_cfg.batch_size} images in {total_steps} steps over {complete_epochs} full epochs plus {steps_in_partial_epoch if steps_in_partial_epoch is not None else 0} extra batches"
    )

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
        return None

    def _compute_nll(eval_params, test_batch, rng, _setup_context):
        return compute_nll(
            mdl,
            eval_params,
            test_batch,
            n_steps=args.integration_steps,
            rng=rng,
            n_projections=args.nll_n_projections,
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

    @partial(jax.jit, static_argnames=("mdl",))
    def loss_fn(params, batch, rng, mdl=None):
        vecs = batch[args.vector_column]
        score_batch = {"point_vec": vecs}
        return compute_batch_loss(mdl, params, score_batch, rng)

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
        loss_fn=partial(loss_fn, mdl=mdl),
        post_epoch_hook_fn=post_epoch_hook,
        fast_post_step_hook_fn=fast_post_step_hook,
        slow_post_step_hook_fn=slow_post_step_hook,
        slow_path_condition_fn=slow_path_condition,
    )

    if global_step % steps_per_epoch != 0:
        save_checkpoint_and_evaluate_vector_model(
            train_state, global_step, **save_eval_kwargs
        )
