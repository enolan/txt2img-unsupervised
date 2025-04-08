"Coordinate check for flow matching models to check whether muP is working."

from datasets import Dataset
from functools import partial
from pathlib import Path
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
import argparse
import gc
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax
import optax.transforms
from contextlib import nullcontext

from .cap_sampling import LogitsTable
from .flow_matching import CapConditionedVectorField
from . import flow_matching


def process_intermediates(intermediates):
    """
    Convert the raw intermediates dictionary into a more readable format.

    This transforms the nested dictionary structure from captured intermediates
    into a flattened dictionary with clear names, separating the MLP blocks.

    Args:
        intermediates: The raw intermediates dictionary from model execution

    Returns:
        A dictionary with human-readable keys and extracted values
    """
    result = {}
    result["model_output"] = intermediates["__call__"][0]
    result["final_norm_output"] = intermediates["final_norm"]["__call__"][0]
    result["output_projection"] = intermediates["out_proj"]["__call__"][0]

    if "pre_mlp_proj" in intermediates:
        result["pre_mlp_projection"] = intermediates["pre_mlp_proj"]["__call__"][0]

    # Process each MLP block separately
    n_layers = intermediates["mlp_blocks"]["__call__"][0][0].shape[0]
    for layer_idx in range(n_layers):
        result[f"mlp_block_{layer_idx}_output"] = intermediates["mlp_blocks"][
            "__call__"
        ][0][0][layer_idx]
        result[f"mlp_block_{layer_idx}_gate"] = intermediates["mlp_blocks"][
            "gate_proj"
        ]["__call__"][0][layer_idx]
        result[f"mlp_block_{layer_idx}_norm"] = intermediates["mlp_blocks"]["norm"][
            "__call__"
        ][0][layer_idx]
        result[f"mlp_block_{layer_idx}_out_proj"] = intermediates["mlp_blocks"][
            "out_proj"
        ]["__call__"][0][layer_idx]
        result[f"mlp_block_{layer_idx}_value"] = intermediates["mlp_blocks"][
            "value_proj"
        ]["__call__"][0][layer_idx]

    return result


@partial(
    jax.jit,
    static_argnames=["mdl"],
    donate_argnames=["rng"],
)
def compute_loss_no_grad(logits_tbl, mdl, params, rng, pts):
    """Compute loss without gradients for test evaluation."""
    rng, next_rng = jax.random.split(rng)
    loss = flow_matching.compute_batch_loss(
        mdl,
        params,
        {"point_vec": pts},
        rng,
        logits_tbl,
        capture_intermediates=False,
    )
    return loss, next_rng


def compute_test_loss(logits_tbl, mdl, params, rng, test_pts, batch_size):
    """Compute average loss over the test dataset."""
    n_batches = len(test_pts) // batch_size
    total_loss = 0.0

    for i in trange(n_batches, desc="Evaluating test batches"):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch = test_pts[start_idx:end_idx]
        loss, rng = compute_loss_no_grad(logits_tbl, mdl, params, rng, batch)
        total_loss += loss

    return total_loss / n_batches


@partial(
    jax.jit,
    static_argnames=["mdl"],
    donate_argnames=["rng"],
)
def compute_gradients(logits_tbl, mdl, params, rng, pts):
    """
    Compute gradients. This is split from apply_updates so we can do this on GPU and
    apply_updates on CPU.
    """
    rng, next_rng = jax.random.split(rng)

    loss_fn = lambda params: flow_matching.compute_batch_loss(
        mdl,
        params,
        {"point_vec": pts},
        rng,
        logits_tbl,
        capture_intermediates=True,
    )
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, intermediates), grad = grad_fn(params)
    processed_intermediates = jax.tree.map(
        lambda x: jnp.mean(jnp.abs(x)),
        process_intermediates(intermediates["intermediates"]),
    )

    return loss, processed_intermediates, grad, next_rng


@partial(
    jax.jit,
    static_argnames=["opt"],
    donate_argnames=["opt_state", "params"],
)
def grad_update(opt, grad, opt_state, params):
    """Do a gradient descent step given gradients."""
    updates, new_opt_state = opt.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state


str_devices = lambda x: jax.tree.map(lambda y: y.device, x)


def train_step(
    logits_tbl, mdl, opt, params, opt_state, rng, pts, use_cpu_offload=False
):
    """Complete training step, optionally with CPU-GPU split."""
    gpu_params = (
        jax.device_put(params, device=jax.devices("gpu")[0])
        if use_cpu_offload
        else params
    )
    loss, processed_intermediates, grad, next_rng = compute_gradients(
        logits_tbl, mdl, gpu_params, rng, pts
    )

    if use_cpu_offload:
        processed_intermediates = jax.device_put(
            processed_intermediates, jax.devices("cpu")[0]
        )
        grad = jax.device_put(grad, jax.devices("cpu")[0])
        loss = jax.device_put(loss, jax.devices("cpu")[0])

    new_params, new_opt_state = grad_update(opt, grad, opt_state, params)
    return loss, processed_intermediates, new_params, new_opt_state, next_rng


@partial(jax.jit, static_argnames=["model"])
def init_model_params(model, init_key):
    """JIT-compiled model initialization function."""
    return model.init(
        init_key,
        jnp.zeros((1, 3)),
        jnp.zeros((1,)),
        jnp.zeros((1, 3)),
        jnp.zeros((1,)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=Path, required=True, help="Path to a 3D dataset"
    )
    parser.add_argument(
        "--lr-base",
        type=float,
        required=False,
        help="Base learning rate (will be ignored if lr-low and lr-high are provided)",
    )
    parser.add_argument(
        "--lr-low", type=float, required=False, help="Lowest learning rate to test"
    )
    parser.add_argument(
        "--lr-high", type=float, required=False, help="Highest learning rate to test"
    )
    parser.add_argument(
        "--n-lr-points",
        type=int,
        required=False,
        default=5,
        help="Number of learning rate points to test between lr-low and lr-high",
    )
    parser.add_argument("--d-model-low", type=int, required=True)
    parser.add_argument("--d-model-high", type=int, required=True)
    parser.add_argument(
        "--reference-directions", type=int, required=False, default=None
    )
    parser.add_argument("--time-dim", type=int, required=True)
    parser.add_argument("--use-pre-mlp-projection", type=bool, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--mlp-expansion-factor", type=int, required=False, default=4)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=False, default=5)
    parser.add_argument(
        "--cpu-offload-threshold",
        type=int,
        required=False,
        default=2048,
        help="d_model threshold above which optimizer state and weight updates are offloaded to CPU",
    )
    parser.add_argument(
        "--n-train-steps",
        type=int,
        required=False,
        default=10,
        help="Number of training steps to perform for each seed",
    )
    parser.add_argument(
        "--n-test-batches",
        type=int,
        required=False,
        default=10,
        help="Number of test batches to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=False,
        default=".",
        help="Directory to save charts in",
    )

    args = parser.parse_args()

    if args.n_seeds <= 0:
        raise ValueError("n_seeds must be at least 1")
    if args.n_train_steps <= 0:
        raise ValueError("n_train_steps must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Charts will be saved to: {args.output_dir}")

    dsets = (
        Dataset.from_parquet(str(args.dataset_path))
        .with_format("numpy")
        .train_test_split(test_size=args.batch_size * args.n_test_batches)
    )
    dset_train = dsets["train"]
    dset_test = dsets["test"]
    print(
        f"Dataset loaded with {len(dset_train)} training examples and {len(dset_test)} test examples. First example: {dset_train[0]}"
    )
    dset_train = dset_train.select(range(args.batch_size * args.n_train_steps))
    logits_table = LogitsTable(d=3 - 1, n=8192)

    doing_lr_sweep = args.lr_low is not None and args.lr_high is not None
    if doing_lr_sweep and args.lr_base is not None:
        print("Warning: lr-base is ignored when lr-low and lr-high are provided")
    elif not doing_lr_sweep and args.lr_base is None:
        raise ValueError("Either lr-base or both lr-low and lr-high must be provided")

    if doing_lr_sweep:
        lr_values = list(
            np.logspace(np.log10(args.lr_low), np.log10(args.lr_high), args.n_lr_points)
        )
        print(f"Testing learning rates: {lr_values}")
    else:
        lr_values = [args.lr_base]

    print(f"Number of learning rates to test: {len(lr_values)}")
    if len(lr_values) == 0:
        raise ValueError("No learning rates to test!")

    # Generate exponentially spaced d_model values with base 2
    low_exp = math.log2(args.d_model_low)
    high_exp = math.log2(args.d_model_high)

    if not (
        2 ** int(low_exp) == args.d_model_low
        and 2 ** int(high_exp) == args.d_model_high
    ):
        raise ValueError("d-model-low and d-model-high must be powers of 2")

    d_model_values = [2**i for i in range(int(low_exp), int(high_exp) + 1)]

    print(f"Testing d_model values: {d_model_values}")

    activations = []
    losses = np.zeros((len(d_model_values), len(lr_values), args.n_train_steps))
    test_losses = np.zeros((len(d_model_values), len(lr_values)))

    n_combinations = len(d_model_values) * len(lr_values)
    all_keys = jax.random.split(jax.random.PRNGKey(20250319), n_combinations)
    key_idx = 0

    for d_model_idx, d_model in tenumerate(d_model_values, desc="d_model values"):
        for lr_idx, lr in tenumerate(lr_values, desc=f"LR for d_model={d_model}"):
            tqdm.write(
                f"\nTraining with d_model = {d_model}, lr = {lr}, lr_idx = {lr_idx}"
            )

            use_cpu_offload = d_model > args.cpu_offload_threshold
            if use_cpu_offload:
                tqdm.write(f"Using CPU offloading for d_model={d_model}")

            master_key = all_keys[key_idx]
            key_idx += 1
            seed_keys = jax.random.split(master_key, args.n_seeds)

            model = CapConditionedVectorField(
                domain_dim=3,
                reference_directions=args.reference_directions,
                conditioning_dim=None,
                time_dim=args.time_dim,
                use_pre_mlp_projection=args.use_pre_mlp_projection,
                n_layers=args.n_layers,
                d_model=d_model,
                mlp_expansion_factor=args.mlp_expansion_factor,
                mlp_dropout_rate=None,
                input_dropout_rate=None,
            )
            tqdm.write(f"Model: {model}")
            tqdm.write(f"m_d = {model.d_model_scale_factor}")

            opt_fixed_lr = optax.adam(lr)
            opt_scaled_lr = optax.adam(model.scale_lr(lr))

            # Use the partition_map property from the model for proper muP scaling
            opt = optax.transforms.partition(
                {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr},
                model.partition_map,
            )
            init_opt_state = jax.jit(opt.init)

            # List to hold activations for all seeds for this d_model and learning rate
            all_seed_activations = []
            # Array to hold losses for all seeds for this d_model and learning rate
            seed_losses = np.zeros((args.n_seeds, args.n_train_steps))
            # Array to hold test losses for all seeds
            seed_test_losses = np.zeros(args.n_seeds)

            tqdm.write(f"Starting training with {args.n_seeds} seeds")

            for seed_idx in trange(args.n_seeds, desc="Seeds", leave=False):
                tqdm.write(f"Training with seed {seed_idx+1}/{args.n_seeds}")

                init_key, train_key = jax.random.split(seed_keys[seed_idx])

                device_ctx = (
                    jax.default_device(jax.devices("cpu")[0])
                    if use_cpu_offload
                    else nullcontext()
                )
                with device_ctx:
                    tqdm.write("Initializing parameters")
                    params = init_model_params(model, init_key)

                    tqdm.write(f"Initializing optimizer state")
                    opt_state = init_opt_state(params)

                tqdm.write("Training")
                rng = train_key
                activations_this_seed = []
                for i, batch in tenumerate(
                    dset_train.iter(args.batch_size, drop_last_batch=True),
                    desc=f"Seed {seed_idx+1} steps",
                    total=args.n_train_steps,
                ):
                    loss, processed_intermediates, params, opt_state, rng = train_step(
                        logits_table,
                        model,
                        opt,
                        params,
                        opt_state,
                        rng,
                        batch["vec"],
                        use_cpu_offload,
                    )
                    activations_this_seed.append(processed_intermediates)
                    # Convert loss to numpy and store
                    seed_losses[seed_idx, i] = np.array(loss)
                    # Only log losses occasionally to avoid flooding the output
                    log_interval = args.n_train_steps // 10
                    if i % log_interval == 0 or i == args.n_train_steps - 1:
                        tqdm.write(f"Loss: {loss}, Step {i}/{args.n_train_steps}")

                # After training, evaluate on test set
                tqdm.write("Evaluating on test set")
                test_loss = compute_test_loss(
                    logits_table,
                    model,
                    params,
                    rng,
                    dset_test["vec"],
                    args.batch_size,
                )
                seed_test_losses[seed_idx] = np.array(test_loss)
                tqdm.write(f"Test loss: {test_loss}")

                tqdm.write(f"Seed {seed_idx+1} training complete")
                all_seed_activations.append(activations_this_seed)
                del processed_intermediates, params, opt_state, rng
                gc.collect()

            # Average the losses across seeds
            avg_losses = np.mean(seed_losses, axis=0)
            avg_test_loss = np.mean(seed_test_losses)
            losses[d_model_idx, lr_idx] = avg_losses
            test_losses[d_model_idx, lr_idx] = avg_test_loss
            tqdm.write(f"Average test loss: {avg_test_loss}")

            # Only keep activations for the lowest learning rate if doing a sweep
            if lr_idx == 0:
                # Convert to numpy array for easier averaging
                # First, convert each dictionary element to a list of arrays
                num_steps = len(all_seed_activations[0])
                activation_keys = all_seed_activations[0][0].keys()

                # For each step, average across all seeds
                averaged_activations = []
                for step in range(num_steps):
                    step_dict = {}
                    for key in activation_keys:
                        # Average the same key across all seeds for this step
                        values = [
                            seed_activations[step][key]
                            for seed_activations in all_seed_activations
                        ]
                        step_dict[key] = np.mean(values)
                    averaged_activations.append(step_dict)

                tqdm.write(f"Appending activations for d_model={d_model}")
                d_model_activations = averaged_activations

        tqdm.write(f"Storing activations for d_model={d_model}")
        activations.append(d_model_activations)

    # Generate activation charts
    generate_activation_charts(d_model_values, activations, args.n_layers, args)

    # Generate loss charts if doing a learning rate sweep
    if doing_lr_sweep:
        generate_loss_charts(d_model_values, lr_values, losses, test_losses, args)


def generate_activation_charts(d_model_values, activations, n_layers, args):
    """
    Generate charts showing activation values across different model dimensions.

    Args:
        d_model_values: List of d_model values used in training
        activations: List of activation values for each model dimension
        n_layers: Number of MLP layers in the model
        args: Command-line arguments to include in the legend
    """
    # Create a colormap for train steps (only show first 10)
    num_train_steps = min(len(activations[0]), 10)
    colors = plt.cm.viridis(np.linspace(0, 1, num_train_steps))

    params = {
        "dataset": args.dataset_path,
        "lr_base": args.lr_base if args.lr_base is not None else "N/A",
        "lr_range": f"{args.lr_low}-{args.lr_high}"
        if args.lr_low is not None
        else "N/A",
        "n_lr_points": args.n_lr_points if args.lr_low is not None else "N/A",
        "d_model_range": f"{args.d_model_low}-{args.d_model_high}",
        "reference_directions": args.reference_directions,
        "time_dim": args.time_dim,
        "use_pre_mlp_projection": args.use_pre_mlp_projection,
        "n_layers": args.n_layers,
        "mlp_expansion_factor": args.mlp_expansion_factor,
        "batch_size": args.batch_size,
        "n_seeds": args.n_seeds,
        "n_train_steps": args.n_train_steps,
        "n_test_batches": args.n_test_batches,
    }

    def create_chart(key, title, filename):
        """Helper function to create and save a chart for a specific activation type"""
        plt.figure(figsize=(18, 8))

        # Create a layout with two subplots - one for the chart, one for the legend
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio of chart to legend
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Plot activation values on the main axis (only first 10 steps)
        for step in range(num_train_steps):
            values = [model_activations[step][key] for model_activations in activations]
            ax1.plot(
                d_model_values,
                values,
                marker="o",
                color=colors[step],
                label=f"Step {step+1}",
            )

        # Configure the main chart
        ax1.set_xscale("log", base=2)
        ax1.set_ylim(bottom=0)
        ax1.set_xlabel("d_model (log scale)")
        ax1.set_ylabel("Mean Absolute Activation")
        ax1.set_title(title)
        ax1.grid(True, which="both", linestyle="--", alpha=0.6)

        # Create the steps legend on the first axis
        ax1.legend(loc="upper right")

        # Create a separate legend for parameters on the second axis
        ax2.axis("off")  # Turn off axis
        param_labels = [
            f"{param_name}: {param_value}" for param_name, param_value in params.items()
        ]
        ax2.text(0, 0.5, "\n".join(param_labels), va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(args.output_dir / filename, bbox_inches="tight", dpi=300)

    # Chart for pre_mlp_projection
    if "pre_mlp_projection" in activations[0][0]:
        create_chart(
            "pre_mlp_projection",
            "Pre-MLP Projection Activations",
            "pre_mlp_activation_chart.png",
        )

    # Charts for each MLP block's output projection, gate, and value
    for layer_idx in range(n_layers):
        create_chart(
            f"mlp_block_{layer_idx}_out_proj",
            f"MLP Block {layer_idx} Output Projection Activations",
            f"mlp_block_{layer_idx}_out_proj_activation_chart.png",
        )

        create_chart(
            f"mlp_block_{layer_idx}_gate",
            f"MLP Block {layer_idx} Gate Activations",
            f"mlp_block_{layer_idx}_gate_activation_chart.png",
        )

        create_chart(
            f"mlp_block_{layer_idx}_value",
            f"MLP Block {layer_idx} Value Activations",
            f"mlp_block_{layer_idx}_value_activation_chart.png",
        )

    # Chart for model's output projection
    create_chart(
        "output_projection",
        "Model Output Projection Activations",
        "model_output_projection_activation_chart.png",
    )

    print("Charts generated successfully!")


def generate_loss_charts(d_model_values, lr_values, losses, test_losses, args):
    """
    Generate charts showing loss values across different learning rates for each d_model.

    Args:
        d_model_values: List of d_model values used in training
        lr_values: List of learning rate values used in training
        losses: 3D numpy array of shape (n_d_models, n_lr_values, n_steps) containing loss values
        test_losses: 2D numpy array of shape (n_d_models, n_lr_values) containing test loss values
        args: Command-line arguments to include in the legend
    """
    # Select 10 evenly spaced steps (or all steps if fewer than 10)
    num_plots = min(10, args.n_train_steps)
    if num_plots < args.n_train_steps:
        step_indices = np.linspace(0, args.n_train_steps - 1, num_plots, dtype=int)
    else:
        step_indices = np.array(range(args.n_train_steps))

    # Generate charts for the selected steps
    for plot_idx, step_idx in enumerate(step_indices):
        plt.figure(figsize=(12, 8))

        for d_idx, d_model in enumerate(d_model_values):
            # Extract losses for this d_model at all learning rates for this step
            step_losses = losses[d_idx, :, step_idx]
            plt.plot(lr_values, step_losses, marker="o", label=f"d_model={d_model}")

        plt.xscale("log")
        plt.ylim(bottom=0)
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel("Loss")
        plt.title(f"Loss vs Learning Rate at Step {step_idx+1}/{args.n_train_steps}")
        plt.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.legend()

        # Add parameter information as text
        param_info = (
            f"Dataset: {args.dataset_path}\n"
            f"d_model range: {args.d_model_low}-{args.d_model_high}\n"
            f"LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
            f"reference_directions: {args.reference_directions}\n"
            f"time_dim: {args.time_dim}\n"
            f"pre_mlp_projection: {args.use_pre_mlp_projection}\n"
            f"n_layers: {args.n_layers}\n"
            f"mlp_expansion_factor: {args.mlp_expansion_factor}\n"
            f"batch_size: {args.batch_size}\n"
            f"n_seeds: {args.n_seeds}\n"
            f"n_test_batches: {args.n_test_batches}\n"
        )
        plt.figtext(0.01, 0.01, param_info, fontsize=8, va="bottom")

        plt.tight_layout()
        plt.savefig(
            args.output_dir / f"loss_vs_lr_step_{step_idx:06d}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    # Generate a chart for the test losses instead of final training losses
    plt.figure(figsize=(12, 8))

    # Define colors for each d_model
    model_colors = plt.cm.viridis(np.linspace(0, 1, len(d_model_values)))

    # Track minimum loss points and their values
    min_points = []

    # First plot all lines
    for d_idx, d_model in enumerate(d_model_values):
        # Extract test losses for this d_model at all learning rates
        model_test_losses = test_losses[d_idx, :]

        # Find the learning rate with the lowest test loss
        min_loss_idx = np.argmin(model_test_losses)
        min_loss = model_test_losses[min_loss_idx]
        min_lr = lr_values[min_loss_idx]

        # Store the minimum point information
        min_points.append((d_model, min_lr, min_loss, model_colors[d_idx]))

        # Plot the line with a marker
        line = plt.plot(
            lr_values,
            model_test_losses,
            marker="o",
            label=f"d_model={d_model}",
            color=model_colors[d_idx],
        )

        # Highlight the minimum point with a larger marker
        plt.plot(
            min_lr,
            min_loss,
            "o",
            markersize=8,
            color=model_colors[d_idx],
            markeredgecolor="black",
            markeredgewidth=1.5,
        )

    # Create a proper table instead of manually positioning text
    # Prepare the table data
    table_data = []
    for d_model, min_lr, min_loss, color in min_points:
        table_data.append([d_model, f"{min_lr:.2e}", f"{min_loss:.4f}"])

    # Get the current axes for the plot
    ax = plt.gca()

    # Create a table at the top right of the plot using axes coordinates
    # (0,0) is bottom left, (1,1) is top right
    table = ax.table(
        cellText=table_data,
        colLabels=["d_model", "Best LR", "Min Loss"],
        colWidths=[0.1, 0.15, 0.1],
        cellLoc="center",
        loc="upper right",
        bbox=[0.65, 0.65, 0.33, 0.25],  # [x, y, width, height]
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    # Style the header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_text_props(weight="bold")
            cell.set_facecolor("lightgray")
        else:  # Color the d_model values to match the plot lines
            if j == 0:  # d_model column
                cell.set_text_props(color=model_colors[i - 1], weight="bold")
            elif j == 2:  # Min loss column
                cell.set_text_props(weight="bold")

    plt.xscale("log")
    plt.ylim(bottom=0)
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Test Loss")
    plt.title(f"Test Loss vs Learning Rate (Averaged Over Test Set)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()

    # Add parameter information as text
    param_info = (
        f"Dataset: {args.dataset_path}\n"
        f"d_model range: {args.d_model_low}-{args.d_model_high}\n"
        f"LR range: {args.lr_low}-{args.lr_high}, n_points={args.n_lr_points}\n"
        f"reference_directions: {args.reference_directions}\n"
        f"time_dim: {args.time_dim}\n"
        f"pre_mlp_projection: {args.use_pre_mlp_projection}\n"
        f"n_layers: {args.n_layers}\n"
        f"mlp_expansion_factor: {args.mlp_expansion_factor}\n"
        f"batch_size: {args.batch_size}\n"
        f"n_seeds: {args.n_seeds}\n"
        f"n_test_batches: {args.n_test_batches}\n"
    )
    plt.figtext(0.01, 0.01, param_info, fontsize=8, va="bottom")

    plt.tight_layout()
    plt.savefig(args.output_dir / "final_loss_vs_lr.png", bbox_inches="tight", dpi=300)

    print("Loss charts generated successfully!")


if __name__ == "__main__":
    main()
