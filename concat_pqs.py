"""Concatenate training example parquets into a single file."""

import argparse

from pathlib import Path

from txt2img_unsupervised.load_pq_dir import load_pq_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dset = load_pq_dir(args.input_dir)
    print(f"Loaded {len(dset)} samples from {args.input_dir}")
    dset.to_parquet(args.output, compression="zstd")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
