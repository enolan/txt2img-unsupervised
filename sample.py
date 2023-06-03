"""Sample from the model"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint  # type: ignore[import]
import PIL.Image
import torch
from copy import copy
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from random import randint
from tqdm import tqdm, trange
from transformer_model import ImageModel, gpt_1_config, sample

parser = argparse.ArgumentParser()
parser.add_argument("transformer_checkpoint_dir", type=Path)
parser.add_argument("autoencoder_checkpoint", type=Path)
parser.add_argument("autoencoder_cfg", type=Path)
parser.add_argument("--n", type=int, default=1)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--top-p", type=float, default=0.9)
parser.add_argument("--make-grids", action="store_true")
parser.add_argument("out_dir", type=Path)
args = parser.parse_args()

# check if grids are possible, and if so how many to make of what dimensions. We make 1 or 2 square
# grids.
if args.make_grids:

    def can_make_grid(n: int) -> bool:
        return (n**0.5) % 1 == 0

    if can_make_grid(args.n):
        grid_imgs = [list(range(args.n))]
    elif args.n / 2 % 1 == 0 and can_make_grid(args.n / 2):
        grid_imgs = [list(range(args.n // 2)), list(range(args.n // 2, args.n))]
    else:
        print(f"Can't make grids out of {args.n} images")
        exit(1)


print("Loading transformer model...")
checkpointer = orbax.checkpoint.PyTreeCheckpointer()
restored = checkpointer.restore(args.transformer_checkpoint_dir)
cfg = copy(gpt_1_config)
cfg.dropout = None
im_mdl = ImageModel(**cfg.__dict__)
im_params = restored["params"]

# Set up random seed
if args.seed is not None:
    seed = args.seed
else:
    seed = randint(0, 2**32 - 1)
rng = jax.random.PRNGKey(seed)

print("Sampling encoded images from the transformer model...")
encoded_imgs = []
for _ in trange(args.n):
    rng, rng_sample = jax.random.split(rng)
    encoded_imgs.append(sample(im_mdl, im_params, rng_sample, args.top_p))

print("Loading autoencoder model...")
ae_cfg = OmegaConf.load(args.autoencoder_cfg)["model"]["params"]  # type:ignore[index]
ae_mdl = LDMAutoencoder(ae_cfg)
ae_params = LDMAutoencoder.params_from_torch(
    torch.load(args.autoencoder_checkpoint, map_location="cpu"), cfg=ae_cfg
)

print("Decoding images...")
args.out_dir.mkdir(exist_ok=True)
decode_j = jax.jit(
    lambda codes: ae_mdl.apply(ae_params, method=ae_mdl.decode, x=codes, shape=(16, 16))
)

print("Saving images...")
imgs = []
for i in trange(args.n):
    img = decode_j(encoded_imgs[i])
    img = jnp.clip(-1, img, 1)
    img = ((img + 1) * 127.5).astype(jnp.uint8)
    img = PIL.Image.fromarray(np.array(img))
    img.save(args.out_dir / f"{i:04d}.png")
    imgs.append(img)

if args.make_grids:
    print(f"Making {len(grid_imgs)} grids...")
    for i, grid in enumerate(grid_imgs):
        side_len = int(len(grid) ** 0.5)
        side_len_px = 64 * side_len + 4 * (side_len - 1)  # 4 px padding between images
        grid_img = PIL.Image.new("RGB", (side_len_px, side_len_px), (255, 255, 255))
        for y_img in range(side_len):
            for x_img in range(side_len):
                img = imgs[grid[y_img * side_len + x_img]]
                grid_img.paste(img, (x_img * 68, y_img * 68))
        grid_img.save(args.out_dir / f"grid_{i:04d}.png")
