#!/usr/bin/env python3
"""Compare sampling results with different step counts."""

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import time
from scipy import stats

from txt2img_unsupervised.checkpoint import load_params
from txt2img_unsupervised.flow_matching import sample_loop
from txt2img_unsupervised.training_infra import setup_jax_for_training


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare sampling results with different step counts."
    )
    parser.add_argument(
        "checkpoint_dir", type=Path, help="Directory containing the checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="step_comparison.png",
        help="Output PNG file path",
    )
    parser.add_argument(
        "--step", type=int, help="Specific checkpoint step to load (default: latest)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2048, help="Batch size for sampling"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps-to-compare",
        type=str,
        default="10,25,50,100,200,500",
        help="Comma-separated list of step counts to compare",
    )
    parser.add_argument(
        "--reference-steps",
        type=int,
        default=1000,
        help="Number of steps to use as reference (highest accuracy)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rk4",
        choices=["euler", "midpoint", "rk4"],
        help="ODE solver method",
    )
    return parser.parse_args()


def compute_distances_to_reference(samples_dict, ref_steps):
    """
    Compute cosine distances between samples from each step count and the reference.
    Uses efficient vectorized operations.

    Args:
        samples_dict: Dictionary mapping step counts to sample arrays
        ref_steps: Step count to use as reference

    Returns:
        Dictionary of {steps: distance_array} where distance_array contains cosine distances
    """
    distances = {}
    ref_samples = samples_dict[ref_steps]

    for steps in samples_dict:
        if steps == ref_steps:
            # Distance to self is 0
            distances[steps] = jnp.zeros(ref_samples.shape[0])
            continue

        # Compute cosine distance using vectorized operations
        # Cosine distance = 1 - dot(a, b) for unit vectors
        dots = jnp.sum(samples_dict[steps] * ref_samples, axis=1)
        # Clip to valid range [-1, 1] to handle numerical issues
        dots = jnp.clip(dots, -1.0, 1.0)
        # Cosine distance is in [0, 2] where 0 = same direction, 2 = opposite
        dists = 1.0 - dots

        distances[steps] = dists

    return distances


def main():
    args = parse_arguments()

    # Parse step counts to compare
    step_counts = [int(s) for s in args.steps_to_compare.split(",")]
    if args.reference_steps not in step_counts:
        step_counts.append(args.reference_steps)
    step_counts.sort()

    # Setup JAX
    setup_jax_for_training()
    print(f"Loading checkpoint from {args.checkpoint_dir}")
    params, step, mdl = load_params(args.checkpoint_dir, args.step)
    print(f"Using checkpoint step: {step}")

    # Create identical setup for all sampling runs
    rng = jax.random.PRNGKey(args.seed)
    centers_rng, samples_rng = jax.random.split(rng)

    # Generate cap centers and sizes (needed for CapConditionedVectorField)
    cap_centers = jax.random.normal(centers_rng, (args.n_samples, mdl.domain_dim))
    cap_centers = cap_centers / jnp.linalg.norm(cap_centers, axis=1, keepdims=True)
    cap_d_maxes = jnp.full((args.n_samples,), 2.0)  # Maximum cap size

    # Dictionary to store samples for each step count
    samples_dict = {}
    timing_dict = {}

    # Generate samples with different step counts
    for n_steps in tqdm(
        step_counts, desc="Generating samples with different step counts"
    ):
        # Use the same RNG for all sampling to ensure consistent initial conditions
        start_time = time.time()
        samples = sample_loop(
            mdl,
            params,
            args.n_samples,
            args.batch_size,
            samples_rng,  # Same RNG for all runs
            cap_centers=cap_centers,
            cap_d_maxes=cap_d_maxes,
            n_steps=n_steps,
            method=args.method,
        )
        end_time = time.time()

        samples_dict[n_steps] = jax.device_get(samples)
        timing_dict[n_steps] = end_time - start_time

    # Compute distances between samples and reference
    ref_steps = max(step_counts)
    distances = compute_distances_to_reference(samples_dict, ref_steps)

    # Create a DataFrame for Seaborn plotting
    data = []
    for steps in step_counts:
        if steps == ref_steps:
            continue  # Skip reference (all zeros)

        # Get distance array for this step count and convert to regular numpy array
        dist_array = np.array(jax.device_get(distances[steps]))

        # Add each distance with its step count to the DataFrame
        for dist in dist_array:
            data.append({"Steps": str(steps), "Cosine Distance": float(dist)})

    df = pd.DataFrame(data)

    # Apply Seaborn style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 10))

    # Create a 2-row layout
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])

    # Plot 1: Combined strip plot and box plot using Seaborn
    ax_dist = plt.subplot(gs[0])

    sns.violinplot(
        x="Steps",
        y="Cosine Distance",
        data=df,
        ax=ax_dist,
    )

    # Customize the plot
    ax_dist.set_xlabel("Number of Steps")
    ax_dist.set_ylabel("Cosine Distance to Reference")
    ax_dist.set_title(
        f"Distance Distributions Relative to Reference ({ref_steps} steps)"
    )
    ax_dist.grid(True, axis="y", linestyle="--", alpha=0.7)
    ax_dist.set_yscale("log")

    # Format y-axis to use standard decimal notation on log scale
    from matplotlib.ticker import FuncFormatter

    ax_dist.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.9f}"))

    # Plot 2: Timing information
    ax_time = plt.subplot(gs[1])

    # Prepare timing data
    timing_df = pd.DataFrame(
        {"Steps": step_counts, "Time (seconds)": [timing_dict[s] for s in step_counts]}
    )

    # Plot with Seaborn
    sns.lineplot(x="Steps", y="Time (seconds)", data=timing_df, marker="o", ax=ax_time)

    ax_time.set_xscale("log")
    ax_time.set_xlabel("Number of Steps")
    ax_time.set_ylabel("Time (seconds)")
    ax_time.set_title("Sampling Time vs Number of Steps")
    ax_time.grid(True, which="both", linestyle="--", alpha=0.7)

    # Add a table with detailed information
    table_data = [
        ["Steps", "Time (s)", "Mean Dist to Ref", "90th %ile", "Max Dist to Ref"]
    ]
    for s in step_counts:
        time_value = timing_dict[s]
        if s == ref_steps:
            mean_dist = 0.0
            percentile_90 = 0.0
            max_dist = 0.0
        else:
            mean_dist = float(jnp.mean(distances[s]))
            percentile_90 = float(jnp.percentile(distances[s], 90))
            max_dist = float(jnp.max(distances[s]))
        table_data.append(
            [
                str(s),
                f"{time_value:.2f}",
                f"{mean_dist:.6f}",
                f"{percentile_90:.6f}",
                f"{max_dist:.6f}",
            ]
        )

    # Add text summary
    summary_text = "Summary:\n"
    summary_text += f"- Method: {args.method}\n"
    summary_text += f"- Samples: {args.n_samples}\n"
    summary_text += f"- Reference steps: {ref_steps}\n\n"

    # Add a few key comparisons
    if len(step_counts) > 3:
        mid_step = step_counts[len(step_counts) // 2]
        low_step = step_counts[0]
        low_mean = (
            float(jnp.mean(distances[low_step])) if low_step != ref_steps else 0.0
        )
        mid_mean = (
            float(jnp.mean(distances[mid_step])) if mid_step != ref_steps else 0.0
        )
        summary_text += f"- {low_step} steps: {low_mean:.6f} mean dist ({timing_dict[low_step]:.2f}s)\n"
        summary_text += f"- {mid_step} steps: {mid_mean:.6f} mean dist ({timing_dict[mid_step]:.2f}s)\n"
        speedup = timing_dict[ref_steps] / timing_dict[low_step]
        summary_text += f"- Speedup {low_step} vs {ref_steps}: {speedup:.1f}x\n"

    plt.figtext(0.1, 0.01, summary_text, fontsize=10, va="bottom")

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    plt.savefig(args.output, bbox_inches="tight", dpi=150)
    print(f"Results saved to {args.output}")

    # Also print detailed comparison to console
    print("\nDetailed comparison:")
    # Print table with aligned columns
    col_widths = [
        max(len(row[i]) for row in table_data) for i in range(len(table_data[0]))
    ]
    for row in table_data:
        formatted_row = [f"{cell:{width}}" for cell, width in zip(row, col_widths)]
        print(" | ".join(formatted_row))


if __name__ == "__main__":
    main()
