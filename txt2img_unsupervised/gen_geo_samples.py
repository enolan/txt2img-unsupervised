#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pathlib import Path
from typing import Tuple, List, Dict, Any
import urllib.parse
from tqdm import tqdm


def load_geotiff(filepath: str) -> rasterio.DatasetReader:
    """
    Load a GeoTIFF file and return the dataset.

    Args:
        filepath: Path to the GeoTIFF file

    Returns:
        Rasterio dataset
    """
    return rasterio.open(filepath)


def print_metadata(dataset: rasterio.DatasetReader) -> None:
    """
    Print metadata from the GeoTIFF dataset.

    Args:
        dataset: Rasterio dataset
    """
    print(f"Driver: {dataset.driver}")
    print(f"Width: {dataset.width}")
    print(f"Height: {dataset.height}")
    print(f"Bounds: {dataset.bounds}")
    print(f"CRS: {dataset.crs}")
    print(f"Transform: {dataset.transform}")
    print(f"Count: {dataset.count}")  # Number of bands
    print(f"Indexes: {dataset.indexes}")  # Band indexes

    # Print statistics for the first band
    band = 1
    data = dataset.read(band)
    print(f"\nBand {band} statistics:")
    print(f"  Min: {data.min()}")
    print(f"  Max: {data.max()}")

    # Handle NaN/NoData values for mean calculation
    valid_data = data[data > 0]  # Filter out negative values (NoData)
    if len(valid_data) > 0:
        print(f"  Mean of valid data: {valid_data.mean()}")
    print(f"  Non-zero values: {np.count_nonzero(data > 0)}")


def create_sampling_distribution(
    dataset: rasterio.DatasetReader,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a probability distribution for sampling based on population density.

    Args:
        dataset: Rasterio dataset

    Returns:
        Tuple containing:
            - Flattened probability distribution
            - Original data shape for converting indices back to coordinates
    """
    # Read the first band (population density)
    data = dataset.read(1)

    # Replace negative values (NoData) with zeros
    data = np.maximum(data, 0)

    # Create a 1D probability distribution based on population density
    # Flatten the 2D array to 1D for sampling
    flat_data = data.flatten()
    if flat_data.sum() == 0:
        raise ValueError("No valid population data found")

    # Create probability distribution normalized to sum to 1
    prob_dist = flat_data / flat_data.sum()

    return prob_dist, data.shape


def latlon_to_unit_vector(lon_lat: np.ndarray) -> np.ndarray:
    """
    Convert longitude/latitude coordinates to 3D unit vectors.

    Args:
        lon_lat: Array of shape (n, 2) containing [longitude, latitude] pairs in degrees

    Returns:
        Array of shape (n, 3) containing unit vectors [x, y, z]
    """
    # Convert degrees to radians
    lon_rad = np.radians(lon_lat[:, 0])
    lat_rad = np.radians(lon_lat[:, 1])

    # Convert to Cartesian coordinates on unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Stack into single array of shape (n, 3)
    unit_vectors = np.column_stack([x, y, z])

    return unit_vectors


def sample_points(
    dataset: rasterio.DatasetReader,
    prob_dist: np.ndarray,
    data_shape: tuple,
    total_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample all points in a single operation for maximum efficiency.
    Smooths the distribution by sampling positions uniformly within each grid cell.

    Args:
        dataset: Rasterio dataset
        prob_dist: Precomputed probability distribution
        data_shape: Shape of the original data array
        total_samples: Total number of samples to generate

    Returns:
        Tuple containing:
            - Array of shape (total_samples, 3) containing unit vectors
            - Array of shape (total_samples, 2) containing lon/lat coordinates
    """
    # Sample all indices at once based on the probability distribution
    sampled_indices = np.random.choice(
        len(prob_dist), size=total_samples, replace=True, p=prob_dist
    )

    # Convert flat indices to 2D row, col coordinates (centers of grid cells)
    rows = sampled_indices // data_shape[1]
    cols = sampled_indices % data_shape[1]

    # Generate random offsets within each grid cell (-0.5 to 0.5)
    # This smooths the distribution by sampling uniformly within each cell
    row_offsets = np.random.uniform(-0.5, 0.5, total_samples)
    col_offsets = np.random.uniform(-0.5, 0.5, total_samples)

    # Add offsets to create smoothed coordinates
    smoothed_rows = rows + row_offsets
    smoothed_cols = cols + col_offsets

    # Convert row, col to geographic coordinates using the affine transform
    transform = dataset.transform
    xs = transform[0] * smoothed_cols + transform[1] * smoothed_rows + transform[2]
    ys = transform[3] * smoothed_cols + transform[4] * smoothed_rows + transform[5]
    lon_lat = np.column_stack([xs, ys])

    # Convert lat/lon to 3D unit vectors (x, y, z)
    unit_vectors = latlon_to_unit_vector(lon_lat)

    return unit_vectors, lon_lat


def save_to_parquet(
    unit_vectors: np.ndarray, lon_lat: np.ndarray, output_path: str
) -> None:
    """
    Save data to parquet file.

    Args:
        unit_vectors: Array of unit vectors
        lon_lat: Array of longitude/latitude pairs
        output_path: Path to save the parquet file
    """
    total_samples = len(unit_vectors)

    # Create PyArrow arrays directly
    vec_array = pa.array(unit_vectors.tolist())  # Convert to list of arrays
    lon_array = pa.array(lon_lat[:, 0])
    lat_array = pa.array(lon_lat[:, 1])

    # Create table with the arrays
    table = pa.Table.from_arrays(
        [vec_array, lon_array, lat_array], ["vec", "longitude", "latitude"]
    )

    # Write to parquet
    pq.write_table(table, output_path)
    print(f"Saved {total_samples} samples to {output_path}")


def sample_and_save(
    dataset: rasterio.DatasetReader,
    output_path: str,
    total_samples: int,
    display_sample: int = 5,
) -> None:
    """
    Sample points and save to a parquet file.

    Args:
        dataset: Rasterio dataset
        output_path: Path to save the parquet file
        total_samples: Total number of samples to generate
        display_sample: Number of samples to display as examples
    """
    # Create the probability distribution once
    print("Creating sampling distribution...")
    prob_dist, data_shape = create_sampling_distribution(dataset)

    # Process all samples at once with progress reporting
    print(f"Generating {total_samples} samples...")

    # Show a progress bar for the overall operation
    with tqdm(total=3, desc="Processing") as pbar:
        # Step 1: Generate all samples at once
        pbar.set_description("Sampling points")
        unit_vectors, lon_lat = sample_points(
            dataset, prob_dist, data_shape, total_samples
        )
        pbar.update(1)

        # Step 2: Save to parquet
        pbar.set_description("Saving to parquet")
        save_to_parquet(unit_vectors, lon_lat, output_path)
        pbar.update(1)

        # Step 3: Display sample points if requested
        if display_sample > 0:
            pbar.set_description("Generating sample output")
            print(f"\nSample of {min(display_sample, len(unit_vectors))} points:")
            for i in range(min(display_sample, len(unit_vectors))):
                vector = unit_vectors[i]
                lon, lat = lon_lat[i]

                # Create Google Maps link
                maps_url = f"https://www.google.com/maps/place/{lat:.6f},{lon:.6f}"

                print(
                    f"Sample {i+1}: Lat: {lat:.6f}, Lon: {lon:.6f} | Unit vector: [{vector[0]:.6f}, {vector[1]:.6f}, {vector[2]:.6f}]"
                )
                print(f"  Maps: {maps_url}")
            pbar.update(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate samples from geospatial population data"
    )
    parser.add_argument("geotiff_file", type=str, help="Path to the GeoTIFF file")
    parser.add_argument(
        "--output",
        type=str,
        default="geo_samples.parquet",
        help="Output parquet file path",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of sample points to generate",
    )
    parser.add_argument(
        "--display", type=int, default=5, help="Number of sample points to display"
    )
    args = parser.parse_args()

    file_path = Path(args.geotiff_file)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return

    dataset = load_geotiff(str(file_path))
    print_metadata(dataset)

    try:
        sample_and_save(
            dataset=dataset,
            output_path=args.output,
            total_samples=args.num_samples,
            display_sample=args.display,
        )

        print(
            f"\nSamples are unit vectors in 3D space representing points on Earth's surface."
        )
        print(f"Data saved to: {args.output}")
    except Exception as e:
        print(f"Error generating samples: {e}")
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
