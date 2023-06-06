"""
Given a source of images, scale them, crop them, and encode them with LDM encoder.
"""
import argparse
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
parser.add_argument("in_dir", type=str)
parser.add_argument("out_dir", type=str)
parser.add_argument("batch_size", type=int)
parser.add_argument("ckpt", type=str)
parser.add_argument("autoencoder_cfg", type=str)

args = parser.parse_args()


# Load the autoencoder model
autoencoder_cfg = OmegaConf.load(args.autoencoder_cfg)["model"]["params"]
autoencoder_mdl = LDMAutoencoder(cfg=autoencoder_cfg)
autoencoder_params = autoencoder_mdl.params_from_torch(
    torch.load(args.ckpt, map_location="cpu"), autoencoder_cfg
)


in_path = Path(args.in_dir)
assert in_path.is_dir()
out_path = Path(args.out_dir)
out_path.mkdir(exist_ok=True)


def load_img(img_path: Path) -> Optional[PIL.Image.Image]:
    """Load/crop/scale a single image."""
    img = PIL.Image.open(img_path)
    w, h = img.size
    if w < 64 or h < 64 or img.mode != "RGB":
        print(f"Skipping {img_path}, size {w}x{h}, mode {img.mode}")
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
    img.load()
    return img


def load_imgs(img_paths: list[Path]) -> list[Optional[PIL.Image.Image]]:
    """Load, crop, and scale a list of images, from a list of paths in parallel."""
    print(f"Loading {len(img_paths)} images in parallel...")
    with futures.ThreadPoolExecutor(max_workers=16) as pool:
        imgs = pool.map(load_img, img_paths)
    print("Loaded")
    return list(imgs)


# Process the images in chunks
print("Processing...")
encode_vec = jax.jit(
    jax.vmap(
        lambda img: autoencoder_mdl.apply(
            autoencoder_params,
            method=autoencoder_mdl.encode,
            x=(img.astype(jnp.float32) / 127.5 - 1.0),
        )
    )
)

img_paths_iter = in_path.iterdir()
with tqdm(total=len(list(in_path.iterdir()))) as pbar:
    while True:
        img_paths = list(itertools.islice(img_paths_iter, args.batch_size))
        if not img_paths:
            break
        imgs_jax_list = []
        imgs_paths = []
        imgs_pil = load_imgs(img_paths)
        for img_pil, img_path in zip(imgs_pil, img_paths):
            if img_pil is not None:
                imgs_jax_list.append(jnp.array(img_pil))
                imgs_paths.append(img_path)
        print(f"loaded, normed, scaled, and cropped {len(imgs_jax_list)} images")
        imgs_jax_arr = jnp.stack(imgs_jax_list)
        print(f"stacked into shape {imgs_jax_arr.shape}")
        imgs_encoded = encode_vec(imgs_jax_arr)
        print(f"encoded into shape {imgs_encoded.shape}")
        for encoded_img, img_path in zip(imgs_encoded, imgs_paths):
            out_file_path = out_path / img_path.name
            jnp.save(out_file_path, encoded_img)
            print(f"saved encoded image to {out_file_path}")
        pbar.update(len(imgs_encoded))
