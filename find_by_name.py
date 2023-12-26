"""Find images from parquet files by name and decode them"""

import argparse
import ldm_autoencoder
import numpy as np
import PIL.Image
import torch

from datasets import Dataset
from ldm_autoencoder import LDMAutoencoder
from load_pq_dir import load_pq_dir
from omegaconf import OmegaConf
from pathlib import Path
from tqdm.contrib import tenumerate
from typing import Any


def find_binsearch(dset: Dataset, name: str) -> int:
    """Find an image in a sorted dataset by name using binary search."""

    lower_bound = 0
    upper_bound = len(dset) - 1

    while lower_bound <= upper_bound:
        midpoint = (lower_bound + upper_bound) // 2
        value = dset[midpoint]["name"]

        if value == name:
            return midpoint
        elif value < name:
            lower_bound = midpoint + 1
        else:
            upper_bound = midpoint - 1

    return None


def decode_names(
    mdl: LDMAutoencoder,
    params: Any,
    dset: Dataset,
    names: list[str],
) -> list[PIL.Image.Image]:
    """Decode images from a dataset by name"""
    idxs = [find_binsearch(dset, name) for name in names]
    print(f"Found indices {idxs} for images {names}")
    dset = dset.select_columns("encoded_img")

    codes = dset[idxs]["encoded_img"]
    assert len(codes.shape) == 2
    assert codes.shape[0] == len(names)
    res_tokens = int(codes.shape[1] ** 0.5)
    assert res_tokens**2 == codes.shape[1]
    res_pixels = res_tokens * 4

    images = ldm_autoencoder.decode_jv(mdl, params, (res_tokens, res_tokens), codes)
    assert images.shape == (len(names), res_pixels, res_pixels, 3)

    return [PIL.Image.fromarray(np.array(img)) for img in images]


def main() -> None:
    """Find images from parquet files by name and decode them"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-cfg", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
    ae_mdl = LDMAutoencoder(ae_cfg)
    ae_params = LDMAutoencoder.params_from_torch(
        torch.load(args.ae_ckpt, map_location="cpu"), cfg=ae_cfg
    )

    dset = load_pq_dir(args.dataset_dir)
    print(f"Loaded dataset with {len(dset)} images. Sorting...")
    dset = dset.sort("name")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Decoding images...")
    imgs = decode_names(ae_mdl, ae_params, dset, args.names)
    for name, img in zip(args.names, imgs):
        img.save(f"{args.output_dir / name}.png")


if __name__ == "__main__":
    main()
