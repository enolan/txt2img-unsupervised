import argparse
import jax
import numpy as np
import PIL.Image
import torch

from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

from txt2img_unsupervised import ldm_autoencoder
from txt2img_unsupervised.ldm_autoencoder import LDMAutoencoder
from txt2img_unsupervised.load_pq_dir import load_pq_dir
from txt2img_unsupervised.sample import batches_split


def main():
    jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")

    parser = argparse.ArgumentParser(description="Sample random images from a dataset.")
    parser.add_argument("--dset-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ae-cfg", type=Path, required=True)
    parser.add_argument("--ae-ckpt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    dset = load_pq_dir(args.dset_dir)
    dset = (
        dset.shuffle(seed=args.seed)
        .select(range(args.n_samples))
        .select_columns("encoded_img")
    )
    if len(dset) < args.n_samples:
        raise ValueError(
            f"Dataset has only {len(dset)} images, expected at least {args.n_samples}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
    ae_mdl = LDMAutoencoder(ae_cfg)
    ae_params = LDMAutoencoder.params_from_torch(
        torch.load(args.ae_ckpt, map_location="cpu"), cfg=ae_cfg
    )

    first_img = dset[0]["encoded_img"]
    res_tokens = int(first_img.shape[0] ** 0.5)
    assert res_tokens**2 == first_img.shape[0]
    res_pixels = res_tokens * 4

    imgs = []
    cur = 0
    with tqdm(total=len(dset), unit="img") as pbar:
        for batch_size in batches_split(batch_size=args.batch_size, n=len(dset)):
            codes = dset[cur : cur + batch_size]["encoded_img"]
            imgs.append(
                ldm_autoencoder.decode_jv(
                    ae_mdl, ae_params, (res_tokens, res_tokens), codes
                )
            )
            cur += batch_size
            pbar.update(batch_size)
    imgs = np.concatenate(imgs, axis=0)
    assert imgs.shape == (len(dset), res_pixels, res_pixels, 3), f"{imgs.shape}"

    for img, i in zip(imgs, range(len(dset))):
        PIL.Image.fromarray(img).save(args.out_dir / f"{i:04d}.png")


if __name__ == "__main__":
    main()
