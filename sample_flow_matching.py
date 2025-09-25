#!/usr/bin/env python3
"""Sample from a flow matching checkpoint and save a visualization."""

import argparse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

from txt2img_unsupervised.checkpoint import load_params
from txt2img_unsupervised.flow_matching import (
    create_mollweide_projection_figure,
)
from txt2img_unsupervised.function_weighted_flow_model import (
    BaseDistribution,
    WeightingFunction,
    sample_full_sphere,
    sample_from_cap_backwards_forwards_importance,
    sample_loop,
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
    # Cap conditioning parameters
    parser.add_argument(
        "--latitude", type=float, help="Latitude in degrees for cap center"
    )
    parser.add_argument(
        "--longitude", type=float, help="Longitude in degrees for cap center"
    )
    parser.add_argument(
        "--cap-radius", type=float, help="Angular radius of the cap in degrees"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=8,
        help="Number of integration steps for sampling",
    )

    return parser.parse_args()


def latlon_to_unit_vector(lon, lat):
    """
    Convert longitude/latitude coordinates to 3D unit vector.

    Args:
        lon: Longitude in degrees
        lat: Latitude in degrees

    Returns:
        3D unit vector [x, y, z]
    """
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.array([x, y, z])


def create_mollweide_with_cap(samples, cap_center_latlon, cap_radius_deg, title=None):
    """
    Create a Mollweide projection visualization of 3D points with a cap boundary.

    Args:
        samples: Array of 3D unit vectors with shape [n_samples, 3]
        cap_center_latlon: Tuple of (latitude, longitude) in degrees for cap center
        cap_radius_deg: Angular radius of the cap in degrees
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    assert samples.shape[1] == 3, f"Expected 3D samples, got shape {samples.shape}"

    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplot(111, projection="mollweide")

    # Convert 3D coordinates to longitude/latitude
    # Mollweide projection expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
    longitude = np.arctan2(samples[:, 1], samples[:, 0])  # atan2(y, x) for longitude
    latitude = np.arcsin(samples[:, 2])  # z-coordinate gives latitude (arcsin)

    scatter = ax.scatter(longitude, latitude, s=8, alpha=0.7)

    # Draw cap boundary
    lat_center, lon_center = cap_center_latlon
    cap_radius_rad = np.radians(cap_radius_deg)

    lon_center_rad = np.radians(lon_center)
    lat_center_rad = np.radians(lat_center)

    # Create points along the cap boundary (a small circle on the sphere)
    theta = np.linspace(0, 2 * np.pi, 100)

    # Generate points along the boundary using the spherical law of cosines
    boundary_lats = []
    boundary_lons = []

    for az in theta:
        # Calculate the point at distance cap_radius_rad from center in direction az
        # Using the spherical law of sines and cosines
        slat = np.sin(lat_center_rad) * np.cos(cap_radius_rad) + np.cos(
            lat_center_rad
        ) * np.sin(cap_radius_rad) * np.cos(az)
        slat = np.clip(slat, -1.0, 1.0)
        boundary_lat = np.arcsin(slat)

        dlon = np.arctan2(
            np.sin(az) * np.sin(cap_radius_rad) * np.cos(lat_center_rad),
            np.cos(cap_radius_rad) - np.sin(lat_center_rad) * np.sin(boundary_lat),
        )
        boundary_lon = lon_center_rad + dlon

        # Ensure longitude is within [-pi, pi] for Mollweide projection
        boundary_lon = ((boundary_lon + np.pi) % (2 * np.pi)) - np.pi

        boundary_lats.append(boundary_lat)
        boundary_lons.append(boundary_lon)

    # For large cap radii, the boundary may cross the edge of the projection
    # and need to be drawn as multiple segments to appear correctly
    # Detect jumps in longitude (which indicate edge crossings)
    lon_diffs = np.abs(np.diff(boundary_lons))
    jump_indices = np.where(lon_diffs > np.pi)[0]

    if len(jump_indices) > 0:
        # We have discontinuities - draw the boundary in segments
        segments = []
        start_idx = 0

        # Add jump indices and the last point to complete all segments
        all_indices = list(jump_indices) + [len(boundary_lons) - 1]

        for end_idx in all_indices:
            segment = (
                boundary_lons[start_idx : end_idx + 1],
                boundary_lats[start_idx : end_idx + 1],
            )
            segments.append(segment)
            start_idx = end_idx + 1

        # Draw each segment
        for segment_lons, segment_lats in segments:
            ax.plot(segment_lons, segment_lats, "r-", linewidth=2, alpha=0.7)
    else:
        # No discontinuities - draw the whole boundary at once
        ax.plot(boundary_lons, boundary_lats, "r-", linewidth=2, alpha=0.7)

    # Plot the center of the cap
    ax.plot(lon_center_rad, lat_center_rad, "rx", markersize=10)

    ax.grid(True, alpha=0.3)

    tick_formatter = ticker.FuncFormatter(lambda x, pos: f"{np.degrees(x):.0f}°")
    # Set up longitude (x) ticks every 15 degrees and latitude (y) ticks every 10 degrees -
    # longitude ranges from -180 to +180 and latitude ranges from -90 to +90.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(np.radians(15)))
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(np.radians(10)))
    ax.yaxis.set_major_formatter(tick_formatter)

    if title is not None:
        ax.set_title(title)
    return fig


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

    use_cap_conditioning = all(
        param is not None for param in [args.latitude, args.longitude, args.cap_radius]
    )

    if use_cap_conditioning:
        cap_center = latlon_to_unit_vector(args.longitude, args.latitude)
        max_cos_distance = 1 - np.cos(np.radians(args.cap_radius))

        print(
            f"Using cap conditioning with center at lat={args.latitude}°, lon={args.longitude}°"
        )
        print(
            f"Cap angular radius: {args.cap_radius}° (cosine distance: {max_cos_distance:.6f})"
        )

        # Validate d_max constraints for CAP base distribution
        if mdl.base_distribution == BaseDistribution.CAP:
            if max_cos_distance > 1.0:
                if max_cos_distance < 2.0:
                    raise ValueError(
                        f"Invalid d_max {max_cos_distance:.3f}: for CAP models, d_max must be <= 1.0 or exactly 2.0, not between 1.0 and 2.0"
                    )
                elif max_cos_distance == 2.0:
                    print(
                        "CAP base distribution with d_max = 2.0. Using hemisphere sampling strategy."
                    )
                    samples = sample_full_sphere(
                        mdl,
                        params,
                        samples_rng,
                        args.n_samples,
                        args.batch_size,
                        args.n_steps,
                    )
                else:
                    raise ValueError(
                        f"Invalid d_max {max_cos_distance:.3f}: must be <= 2.0"
                    )
            else:
                # Direct sampling for d_max <= 1.0
                cap_centers = jnp.tile(cap_center, (args.n_samples, 1))
                cap_d_maxes = jnp.full((args.n_samples,), max_cos_distance)
                weighting_function_params = (cap_centers, cap_d_maxes)
                samples = sample_loop(
                    mdl,
                    params,
                    samples_rng,
                    weighting_function_params,
                    args.n_samples,
                    args.batch_size,
                    args.n_steps,
                )
        else:
            # Non-CAP base distribution
            if (
                mdl.weighting_function == WeightingFunction.CONSTANT
                and mdl.base_distribution == BaseDistribution.SPHERE
            ):
                # Use CNF backwards-forwards cap-conditioned sampling for constant weighting
                n_backward = 256
                samples, ess = sample_from_cap_backwards_forwards_importance(
                    model=mdl,
                    params=params,
                    cap_center=cap_center,
                    cap_d_max=max_cos_distance,
                    rng=samples_rng,
                    tbl=None,
                    n_backward_samples=n_backward,
                    n_forward_samples=args.n_samples * 4,
                    batch_size=args.batch_size,
                )
                print(f"Effective sample size (ESS): {ess:.1f}")
            else:
                # Generic sampling via weighting function parameters
                cap_centers = jnp.tile(cap_center, (args.n_samples, 1))
                cap_d_maxes = jnp.full((args.n_samples,), max_cos_distance)
                weighting_function_params = (cap_centers, cap_d_maxes)
                samples = sample_loop(
                    mdl,
                    params,
                    samples_rng,
                    weighting_function_params,
                    args.n_samples,
                    args.batch_size,
                    args.n_steps,
                )
    else:
        # Unconditioned sampling - use d_max=2.0 (full sphere)
        print("Sampling from full sphere")
        samples = sample_full_sphere(
            mdl, params, centers_rng, args.n_samples, args.batch_size, args.n_steps
        )

    samples = jax.device_get(samples)

    title = args.title
    if use_cap_conditioning and title is None:
        title = f"Samples with cap center at lat={args.latitude}°, lon={args.longitude}°, radius={args.cap_radius}°"

    print("Creating Mollweide projection visualization...")
    if use_cap_conditioning:
        fig = create_mollweide_with_cap(
            samples,
            cap_center_latlon=(args.latitude, args.longitude),
            cap_radius_deg=args.cap_radius,
            title=title,
        )
    else:
        fig = create_mollweide_projection_figure(samples, title=title)

    print(f"Saving visualization to {args.output}")
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
