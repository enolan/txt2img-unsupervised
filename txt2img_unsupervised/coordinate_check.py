"Coordinate check for flow matching models to check whether muP is working."

from datasets import Dataset
from functools import partial
from pathlib import Path
from tqdm import tqdm, trange
import argparse
import gc
import jax
import jax.numpy as jnp
import math
import matplotlib.pyplot as plt
import numpy as np
import optax

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
    static_argnames=["mdl", "opt", "kappa_1"],
    donate_argnames=["params", "opt_state", "rng"],
)
def train_step(logits_tbl, mdl, opt, kappa_1, params, opt_state, rng, pts):
    rng, next_rng = jax.random.split(rng)

    loss_fn = lambda params: flow_matching.compute_batch_loss(
        mdl,
        params,
        {"point_vec": pts},
        rng,
        kappa_1,
        logits_tbl,
        capture_intermediates=True,
    )
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, intermediates), grad = grad_fn(params)
    processed_intermediates = jax.tree.map(
        lambda x: jnp.mean(jnp.abs(x)),
        process_intermediates(intermediates["intermediates"]),
    )
    updates, opt_state = opt.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return loss, processed_intermediates, params, opt_state, next_rng


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
    parser.add_argument("--lr-base", type=float, required=True)
    parser.add_argument("--d-model-low", type=int, required=True)
    parser.add_argument("--d-model-high", type=int, required=True)
    parser.add_argument(
        "--reference-directions", type=int, required=False, default=None
    )
    parser.add_argument("--time-dim", type=int, required=True)
    parser.add_argument("--use-pre-mlp-projection", type=bool, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--mlp-expansion-factor", type=int, required=False, default=4)
    parser.add_argument("--kappa-1", type=float, required=False, default=5.0)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--n-seeds", type=int, required=False, default=5)

    args = parser.parse_args()

    dset = Dataset.from_parquet(str(args.dataset_path)).with_format("numpy")
    # Select just 10 batches worth
    dset = dset.select(range(args.batch_size * 10))
    print(f"Dataset loaded with {len(dset)} examples. First example: {dset[0]}")
    logits_table = LogitsTable(d=3 - 1, n=8192)

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

    # Create an RNG key for each d_model value
    d_model_keys = jax.random.split(jax.random.PRNGKey(20250319), len(d_model_values))

    for d_model_idx, d_model in enumerate(tqdm(d_model_values)):
        tqdm.write(f"\nTraining with d_model = {d_model}")

        d_model_key = d_model_keys[d_model_idx]
        seed_keys = jax.random.split(d_model_key, args.n_seeds)

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

        opt = optax.adam(model.scale_lr(args.lr_base))
        init_opt_state = jax.jit(opt.init)

        # List to hold activations for all seeds for this d_model
        all_seed_activations = []

        for seed_idx in range(args.n_seeds):
            tqdm.write(f"Training with seed {seed_idx+1}/{args.n_seeds}")

            init_key, train_key = jax.random.split(seed_keys[seed_idx])

            tqdm.write("Initializing parameters")
            params = init_model_params(model, init_key)

            tqdm.write("Initializing optimizer state")
            opt_state = init_opt_state(params)

            tqdm.write("Training")
            rng = train_key
            activations_this_seed = []
            for i, batch in enumerate(dset.iter(args.batch_size, drop_last_batch=True)):
                loss, processed_intermediates, params, opt_state, rng = train_step(
                    logits_table,
                    model,
                    opt,
                    args.kappa_1,
                    params,
                    opt_state,
                    rng,
                    batch["vec"],
                )
                activations_this_seed.append(processed_intermediates)
                tqdm.write(f"Loss: {loss}")

            all_seed_activations.append(activations_this_seed)
            del processed_intermediates, params, opt_state, rng
            gc.collect()

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

        activations.append(averaged_activations)

    # Generate charts
    generate_activation_charts(d_model_values, activations, args.n_layers, args)


def generate_activation_charts(d_model_values, activations, n_layers, args):
    """
    Generate charts showing activation values across different model dimensions.

    Args:
        d_model_values: List of d_model values used in training
        activations: List of activation values for each model dimension
        n_layers: Number of MLP layers in the model
        args: Command-line arguments to include in the legend
    """
    # Create a colormap for train steps
    num_train_steps = len(activations[0])
    colors = plt.cm.viridis(np.linspace(0, 1, num_train_steps))

    # Format the model parameters as a dictionary for easier access
    params = {
        "dataset": args.dataset_path,
        "lr_base": args.lr_base,
        "d_model_range": f"{args.d_model_low}-{args.d_model_high}",
        "reference_directions": args.reference_directions,
        "time_dim": args.time_dim,
        "use_pre_mlp_projection": args.use_pre_mlp_projection,
        "n_layers": args.n_layers,
        "mlp_expansion_factor": args.mlp_expansion_factor,
        "kappa_1": args.kappa_1,
        "batch_size": args.batch_size,
        "n_seeds": args.n_seeds,
    }

    def create_chart(key, title, filename):
        """Helper function to create and save a chart for a specific activation type"""
        plt.figure(figsize=(18, 8))  # Increased width from 12 to 18

        # Create a layout with two subplots - one for the chart, one for the legend
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])  # 3:1 ratio of chart to legend
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        # Plot activation values on the main axis
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
        plt.savefig(filename, bbox_inches="tight")

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


if __name__ == "__main__":
    main()
