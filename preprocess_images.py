"""
Given a source of images, scale them, crop them, and encode them with LDM encoder.
"""
import argparse
import itertools
import jax
import jax.numpy as jnp
import PIL
import torch
from concurrent import futures
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

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


@jax.jit
def norm_scale_and_crop(img):
    h, w, c = img.shape
    assert c == 3, f"image has {c} channels, expected 3"
    # Normalize the image
    img = img.astype(jnp.float32) / 127.5 - 1.0
    # Scale the image so the short axis is 64 pixels
    if h < w:
        img = jax.image.resize(img, (64, 64 * w // h, 3), "cubic")
    else:
        img = jax.image.resize(img, (64 * h // w, 64, 3), "cubic")
    assert img.shape[0] >= 64 and img.shape[1] >= 64
    # Center crop the image
    img = img[
        (img.shape[0] - 64) // 2 : (img.shape[0] + 64) // 2,
        (img.shape[1] - 64) // 2 : (img.shape[1] + 64) // 2,
        :,
    ]
    return img

def load_img(img_path: Path):
    """Load a single image."""
    img = PIL.Image.open(img_path)
    img.load()
    return img

def load_imgs(img_paths: list[Path]):
    """Load a list of images from a list of paths in parallel."""
    print(f"Loading {len(img_paths)} images in parallel...")
    with futures.ProcessPoolExecutor(max_workers=16) as pool:
        imgs = pool.map(load_img, img_paths)
    print("Loaded")
    return list(imgs)

# Process the images in chunks
print("Processing...")
encode_vec = jax.jit(
    jax.vmap(
        lambda img: autoencoder_mdl.apply(
            autoencoder_params, method=autoencoder_mdl.encode, x=img
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
            w, h = img_pil.size
            if w >= 64 and h >= 64 and img_pil.mode == "RGB":
                imgs_jax_list.append(norm_scale_and_crop(jnp.array(img_pil)))
                imgs_paths.append(img_path)
            else:
                print(
                    f"skipping image {img_path} with dimensions {w}x{h} and format {img_pil.mode}"
                )
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
