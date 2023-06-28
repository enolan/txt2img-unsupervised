"""Script for sampling from the model, plus associated utilities."""
import argparse
import dacite
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint  # type: ignore[import]
import PIL.Image
import torch
import transformers
from copy import copy
from einops import repeat
from flax.core import freeze
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from random import randint
from tqdm import tqdm, trange
from transformer_model import ImageModel, ModelConfig, gpt_1_config, sample
from typing import Tuple


def can_make_grid(n: int) -> bool:
    return (n**0.5) % 1 == 0


def batches_split(batch_size: int, n: int) -> list[int]:
    """Split n into batches of size args.batch_size, plus the remainder."""
    split = [batch_size] * (n // batch_size)
    if n % batch_size != 0:
        split.append(n % batch_size)
    return split


sample_v = jax.jit(
    jax.vmap(
        lambda params, clip_embedding, rng: sample(
            im_mdl, params, clip_embedding, rng, args.top_p
        ),
        in_axes=(None, 0, 0),
    )
)


# TODO delete. this is wrong, and redundant with ldm_autoencoder
def decode_to_u8(ae_mdl, ae_params, codes):
    pxls_f32 = ae_mdl.apply(ae_params, method=ae_mdl.decode, x=codes, shape=(16, 16))
    pxls_f32 = jnp.clip(0, (pxls_f32 + 1) * 127.5, 255.0)
    return pxls_f32.astype(jnp.uint8)


decode_jv = jax.jit(
    lambda ae_mdl, ae_params, codeses: jax.vmap(decode_to_u8, in_axes=(None, None, 0))(
        ae_mdl, ae_params, codeses
    ),
    static_argnums=0,
)


def make_grid(
    imgs: list[PIL.Image.Image],
    spacing: int = 4,
    spacing_color: Tuple[int, int, int] = (255, 255, 255),
) -> PIL.Image.Image:
    """Make a square grid of square images of equal size."""
    assert can_make_grid(len(imgs))
    assert all(img.size == imgs[0].size for img in imgs)

    side_len = int(len(imgs) ** 0.5)
    side_len_px = imgs[0].size[0] * side_len + spacing * (side_len - 1)

    grid_img = PIL.Image.new("RGB", (side_len_px, side_len_px), color=spacing_color)
    for y_img in range(side_len):
        for x_img in range(side_len):
            grid_img.paste(
                imgs[y_img * side_len + x_img],
                (
                    x_img * (imgs[0].size[0] + spacing),
                    y_img * (imgs[0].size[0] + spacing),
                ),
            )
    return grid_img


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transformer_checkpoint_dir", type=Path)
    parser.add_argument("autoencoder_checkpoint", type=Path)
    parser.add_argument("autoencoder_cfg", type=Path)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--make-grids", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cond-img", type=Path, default=None)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    # check if grids are possible, and if so how many to make of what dimensions. We make 1 or 2 square
    # grids.
    if args.make_grids:
        if can_make_grid(args.n):
            grid_imgs = [list(range(args.n))]
        elif args.n % 2 == 0 and can_make_grid(args.n / 2):
            grid_imgs = [list(range(args.n // 2)), list(range(args.n // 2, args.n))]
        else:
            print(f"Can't make grids out of {args.n} images")
            exit(1)

    print("Loading transformer model...")
    checkpoint_mngr = orbax.checkpoint.CheckpointManager(
        # Orbax chokes on relative paths for some godforsaken reason
        args.transformer_checkpoint_dir.absolute(),
        orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointer()),
    )
    print(
        f"Loading step {checkpoint_mngr.latest_step()} from {args.transformer_checkpoint_dir}"
    )
    restored = checkpoint_mngr.restore(checkpoint_mngr.latest_step())

    model_cfg = ModelConfig.from_json_dict(checkpoint_mngr.metadata()["model_cfg"])
    model_cfg.activations_dtype = jnp.float32  # Assume we're on the GPU at home
    im_mdl = ImageModel(**model_cfg.__dict__)
    im_params = freeze(restored["params"])
    if model_cfg.clip_conditioning:
        print("Loading CLIP model...")

        clip_mdl_name = "openai/clip-vit-large-patch14"
        clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name, dtype=jnp.float16)
        clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

        print(f"Loading conditioning image {args.cond_img}...")
        cond_img = PIL.Image.open(args.cond_img)

        print("Preprocessing conditioning image...")
        clip_inputs = clip_processor(images=cond_img, return_tensors="np")

        print("Computing CLIP embedding...")
        clip_embedding = clip_mdl.get_image_features(**clip_inputs)[0].astype(jnp.float32)
        print(f"Got CLIP embedding of shape {clip_embedding.shape}, norm {jnp.linalg.norm(clip_embedding)}")
        clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)
    else:
        assert args.cond_img is None, "Can't condition on an image without CLIP conditioning"
        clip_embedding = jnp.zeros((0,), dtype=jnp.float32)
    decode_params = im_mdl.init(
        jax.random.PRNGKey(0), jnp.arange(256), clip_embedding
    )
    im_params = im_params.copy({"cache": decode_params["cache"]})

    # Set up random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = randint(0, 2**32 - 1)
    rng = jax.random.PRNGKey(seed)

    print("Sampling encoded images from the transformer model...")

    with tqdm(total=args.n, unit="img") as pbar:
        encoded_imgs = []
        for batch_size in batches_split(args.batch_size, args.n):
            rng, rng2 = jax.random.split(rng)
            clip_embeddings = repeat(clip_embedding, "d -> n d", n=batch_size)
            encoded_imgs.append(
                sample_v(im_params, clip_embeddings, jax.random.split(rng2, batch_size))
            )
            pbar.update(batch_size)

    print("Loading autoencoder model...")
    ae_cfg = OmegaConf.load(args.autoencoder_cfg)["model"][
        "params"
    ]  # type:ignore[index]
    ae_mdl = LDMAutoencoder(ae_cfg)
    ae_params = LDMAutoencoder.params_from_torch(
        torch.load(args.autoencoder_checkpoint, map_location="cpu"), cfg=ae_cfg
    )

    print("Decoding images...")
    args.out_dir.mkdir(exist_ok=True)

    decoded_imgs = []
    with tqdm(total=args.n, unit="img") as pbar:
        for encoded_img_batch in encoded_imgs:
            decoded_imgs.append(decode_jv(ae_mdl, ae_params, encoded_img_batch))
            pbar.update(len(encoded_img_batch))
    decoded_imgs = np.concatenate(decoded_imgs, axis=0)

    print("Saving images...")
    imgs = []
    for i, img in enumerate(tqdm(decoded_imgs, unit="img")):
        img = PIL.Image.fromarray(np.array(img))
        img.save(args.out_dir / f"{i:04d}.png")
        imgs.append(img)

    if args.make_grids:
        print(f"Making {len(grid_imgs)} grids...")
        for i, indices in enumerate(grid_imgs):
            grid_imgs = [imgs[i] for i in indices]
            make_grid(grid_imgs).save(args.out_dir / f"grid_{i:04d}.png")
