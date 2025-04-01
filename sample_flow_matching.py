#!/usr/bin/env python3
"""Sample from a flow matching checkpoint and save a visualization."""

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

from txt2img_unsupervised.checkpoint import load_params
from txt2img_unsupervised.flow_matching import (
    sample_loop,
    create_mollweide_projection_figure,
)
from txt2img_unsupervised.training_infra import setup_jax_for_training


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sample from a flow matching model checkpoint and save visualization."
    )
    parser.add_argument(
        "checkpoint_dir", type=Path, help="Directory containing the checkpoint"
    )
    parser.add_argument(
        "--output", type=Path, default="samples.png", help="Output PNG file path"
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
        "--title", type=str, default=None, help="Title for the visualization"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Sets the correct RNG, critical so reference vectors are consistent with training. Also enables
    # compilation cache.
    setup_jax_for_training()

    print(f"Loading checkpoint from {args.checkpoint_dir}")
    params, step, mdl = load_params(args.checkpoint_dir, args.step)
    print(f"Using checkpoint step: {step}")
    print(mdl.tabulate(jax.random.PRNGKey(0), *mdl.dummy_inputs()))

    # Check if we have a 3D model (required for Mollweide projection)
    if mdl.domain_dim != 3:
        print(
            f"Error: Model domain dimension is {mdl.domain_dim}, but must be 3 for Mollweide projection."
        )
        return

    rng = jax.random.PRNGKey(args.seed)
    centers_rng, samples_rng = jax.random.split(rng)

    print(f"Generating {args.n_samples} samples...")
    cap_centers = jax.random.normal(centers_rng, (args.n_samples, mdl.domain_dim))
    cap_centers = cap_centers / jnp.linalg.norm(cap_centers, axis=1, keepdims=True)
    assert cap_centers.shape == (args.n_samples, mdl.domain_dim)
    print(f"{jnp.linalg.norm(cap_centers, axis=1)=}")

    # Set maximum cap size for unconditioned sampling
    cap_d_maxes = jnp.full((args.n_samples,), 2.0)

    samples = sample_loop(
        mdl,
        params,
        args.n_samples,
        args.batch_size,
        samples_rng,
        cap_centers=cap_centers,
        cap_d_maxes=cap_d_maxes,
    )

    samples = jax.device_get(samples)

    print("Creating Mollweide projection visualization...")
    fig = create_mollweide_projection_figure(samples, title=args.title)

    print(f"Saving visualization to {args.output}")
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
