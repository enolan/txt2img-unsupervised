"""
Given a source of images, scale them, crop them, and encode them with LDM encoder.
"""
import argparse
import gc
import itertools
import jax
import jax.numpy as jnp
import PIL.Image
import torch
from concurrent import futures
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--out_dir", type=str)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--autoencoder_cfg", type=str)
parser.add_argument("in_dirs", type=Path, nargs="+")

args = parser.parse_args()


# Load the autoencoder model
autoencoder_cfg = OmegaConf.load(args.autoencoder_cfg)["model"]["params"]
autoencoder_mdl = LDMAutoencoder(cfg=autoencoder_cfg)
autoencoder_params = autoencoder_mdl.params_from_torch(
    torch.load(args.ckpt, map_location="cpu"), autoencoder_cfg
)


for in_dir in args.in_dirs:
    assert in_dir.is_dir(), f"{in_dir} is not a directory"

out_path = Path(args.out_dir)
out_path.mkdir(exist_ok=True)


def load_img(img_path: Path) -> Optional[PIL.Image.Image]:
    """Load/crop/scale a single image."""
    try:
        img = PIL.Image.open(img_path)
        img.load()
    except (PIL.UnidentifiedImageError, OSError) as e:
        tqdm.write(f"Skipping {img_path}, PIL can't open it due to {e}")
        return None
    w, h = img.size
    if w < 64 or h < 64 or img.mode != "RGB":
        tqdm.write(f"Skipping {img_path}, size {w}x{h}, mode {img.mode}")
        return None
    else:
        if w > h:
            px_to_remove = w - h
            img = img.resize(
                (64, 64),
                PIL.Image.BICUBIC,
                (px_to_remove // 2, 0, w - px_to_remove // 2, h),
            )
        elif h > w:
            px_to_remove = h - w
            img = img.resize(
                (64, 64),
                PIL.Image.BICUBIC,
                (0, px_to_remove // 2, w, h - px_to_remove // 2),
            )
        else:
            img = img.resize((64, 64), PIL.Image.BICUBIC)
    return img


pool = futures.ThreadPoolExecutor(max_workers=32)


def load_imgs(img_paths: list[Path]) -> list[Optional[PIL.Image.Image]]:
    """Load, crop, and scale a list of images, from a list of paths in parallel."""
    tqdm.write(f"Loading {len(img_paths)} images in parallel...", end="")
    imgs = pool.map(load_img, img_paths)
    imgs = list(imgs)
    tqdm.write(" Loaded")
    return imgs


encode_vec = jax.jit(
    jax.vmap(
        lambda img: autoencoder_mdl.apply(
            autoencoder_params,
            method=autoencoder_mdl.encode,
            x=(img.astype(jnp.float32) / 127.5 - 1.0),
        )
    )
)

print("Collecting image paths...", flush=True, end="")
img_paths = [list(in_path.iterdir()) for in_path in args.in_dirs]
img_paths = list(itertools.chain(*img_paths))
print(f" {len(img_paths)} images found.")
img_paths_iter = iter(img_paths)

# We load images with PIL in batch_size chunks, but not every image is valid, so we accumulate
# the PIL images until we have batch_size of them, then convert them to jax arrays and encode.
# This saves recompiling encode_vec for every different shape of input array.
print("Processing images...")
batch_img_paths = []
batch_pil_imgs = []
with tqdm(total=len(img_paths), desc="images") as pbar:
    while True:
        while len(batch_img_paths) < args.batch_size:
            paths_to_load = list(itertools.islice(img_paths_iter, args.batch_size))
            tqdm.write(f"Attempting load of {len(paths_to_load)} images")
            if not paths_to_load:
                break
            pil_imgs = load_imgs(paths_to_load)
            for img, path in zip(pil_imgs, paths_to_load):
                if img is not None:
                    batch_img_paths.append(path)
                    batch_pil_imgs.append(img)
                else:
                    pbar.update(1)
        if not batch_img_paths:
            break
        tqdm.write(
            f"{len(batch_pil_imgs)} loaded/scaled/cropped images ready for encoding"
        )
        this_batch_imgs = batch_pil_imgs[: args.batch_size]
        this_batch_paths = batch_img_paths[: args.batch_size]
        batch_pil_imgs = batch_pil_imgs[args.batch_size :]
        batch_img_paths = batch_img_paths[args.batch_size :]
        tqdm.write(f"encoding {len(this_batch_imgs)} images this batch")
        imgs_jax_arr = jnp.stack([jnp.array(img) for img in this_batch_imgs])
        imgs_encoded = encode_vec(imgs_jax_arr)
        saved = 0
        for encoded_img, img_path in zip(imgs_encoded, this_batch_paths):
            out_file_path = out_path / img_path.name
            jnp.save(out_file_path, encoded_img)
            saved += 1
        tqdm.write(f"saved {saved} images")
        pbar.update(len(imgs_encoded))
        gc.collect()

pool.shutdown()
