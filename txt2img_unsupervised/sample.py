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
from omegaconf import OmegaConf
from pathlib import Path
from random import randint
from tqdm import tqdm, trange
from typing import Tuple

from . import ldm_autoencoder
from .ldm_autoencoder import LDMAutoencoder
from .transformer_model import ImageModel, ModelConfig, gpt_1_config, sample

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
        lambda params, clip_embedding, cos_sim_lower, cos_sim_upper, rng: sample(
            im_mdl,
            params,
            clip_embedding,
            cos_sim_lower,
            cos_sim_upper,
            rng,
            args.top_p,
        ),
        in_axes=(None, 0, 0, 0, 0),
    )
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


def parse_cond_str(s: str) -> dict:
    """Parse a string for conditioning in either of the following formats:
    "condition" - no cones
    "condition:0.1:0.45" - cones with cosine similarity bounds
    """
    if ":" in s:
        cond, cos_sim_lower, cos_sim_upper = s.split(":")
        cos_sim_lower = jnp.array(float(cos_sim_lower))
        cos_sim_upper = jnp.array(float(cos_sim_upper))
    else:
        cond = s
        cos_sim_lower = cos_sim_upper = None
    return {
        "cond": cond,
        "cos_sim_lower": cos_sim_lower,
        "cos_sim_upper": cos_sim_upper,
    }


def print_cond_debug(d: dict, name: str) -> None:
    print(
        f"Computed CLIP embedding for {name}, shape {d['clip_embedding'].shape}, norm {jnp.linalg.norm(d['clip_embedding'])}, cos sim bounds "
        + (
            f"{d['cos_sim_lower']:.4f} - {d['cos_sim_upper']:.4f}"
            if d["cos_sim_lower"] is not None
            else "not used"
        )
    )


def parse_cond_img(s: str, clip_mdl, clip_processor) -> dict:
    """Parse a string for conditioning on an image."""
    d = parse_cond_str(s)
    path = Path(d["cond"])
    del d["cond"]

    img = PIL.Image.open(path)
    clip_inputs = clip_processor(images=img, return_tensors="jax")
    clip_embedding = clip_mdl.get_image_features(**clip_inputs)[0].astype(jnp.float32)
    clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)

    d["clip_embedding"] = clip_embedding

    print_cond_debug(d, f"image {path}")

    return d


def parse_cond_txt(s: str, clip_mdl, clip_processor) -> dict:
    """Parse a string for conditioning on text."""
    d = parse_cond_str(s)
    txt = d["cond"]
    del d["cond"]

    clip_inputs = clip_processor.tokenizer(txt, return_tensors="jax", padding=True)
    clip_embedding = clip_mdl.get_text_features(**clip_inputs)[0].astype(jnp.float32)
    clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)

    d["clip_embedding"] = clip_embedding
    print_cond_debug(d, f"text {txt}")

    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--make-grids", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cond-img", type=str, nargs="*")
    parser.add_argument("--cond-txt", type=str, nargs="*")
    parser.add_argument("transformer_checkpoint_dir", type=Path)
    parser.add_argument("autoencoder_checkpoint", type=Path)
    parser.add_argument("autoencoder_cfg", type=Path)
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

    # Set up random seed
    if args.seed is not None:
        seed = args.seed
    else:
        seed = randint(0, 2**32 - 1)
    rng = jax.random.PRNGKey(seed)

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
        clip_mdl = transformers.FlaxCLIPModel.from_pretrained(
            clip_mdl_name, dtype=jnp.float16
        )
        clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

        assert (
            args.cond_img is not None or args.cond_txt is not None
        ), "Must specify --cond-img or --cond-txt"
        print("Computing CLIP embeddings...")
        cond_img_dicts = [
            parse_cond_img(s, clip_mdl, clip_processor) for s in (args.cond_img or [])
        ]
        cond_txt_dicts = [
            parse_cond_txt(s, clip_mdl, clip_processor) for s in (args.cond_txt or [])
        ]
        cond_dicts = cond_img_dicts + cond_txt_dicts

        if model_cfg.clip_cones:
            total_conds = len(cond_dicts)
            assert all(
                [d["cos_sim_lower"] is not None for d in cond_dicts]
            ), "Must specify cosine similarity bounds"
            assert (
                total_conds <= model_cfg.clip_cone_count
            ), "Too many CLIP embeddings for the number of cones"

            clip_embeddings_cond = jnp.stack([d["clip_embedding"] for d in cond_dicts])
            cos_sim_lower_cond = jnp.stack([d["cos_sim_lower"] for d in cond_dicts])
            cos_sim_upper_cond = jnp.stack([d["cos_sim_upper"] for d in cond_dicts])
            assert clip_embeddings_cond.shape == (total_conds, 768)
            assert cos_sim_lower_cond.shape == (total_conds,)
            assert cos_sim_upper_cond.shape == (total_conds,)

            fill_cones = model_cfg.clip_cone_count - total_conds
            rng, rng2 = jax.random.split(rng)
            fill_clips = jax.random.normal(rng2, (fill_cones, 768), dtype=jnp.float32)
            fill_clips = fill_clips / jnp.linalg.norm(
                fill_clips, axis=-1, keepdims=True
            )
            print(f"Fill CLIP norms: {jnp.linalg.norm(fill_clips, axis=-1)}")
            fill_cos_sim_lower = jnp.full((fill_cones,), -1.0)
            fill_cos_sim_upper = jnp.full((fill_cones,), 1.0)

            clip_embeddings = jnp.concatenate([clip_embeddings_cond, fill_clips])
            cos_sims_lower = jnp.concatenate([cos_sim_lower_cond, fill_cos_sim_lower])
            cos_sims_upper = jnp.concatenate([cos_sim_upper_cond, fill_cos_sim_upper])

            assert clip_embeddings.shape == (model_cfg.clip_cone_count, 768)
            assert (
                cos_sims_lower.shape
                == cos_sims_upper.shape
                == (model_cfg.clip_cone_count,)
            )
        else:
            assert (
                len(cond_dicts) == 1
            ), "Can only specify one CLIP embedding without clip cones"
            clip_embeddings = cond_dicts[0]["clip_embedding"]
            assert clip_embeddings.shape == (768,)
            assert all(
                [d["cos_sim_lower"] is None for d in cond_dicts]
            ), "Can't specify cosine similarity bounds without clip cones"
            cos_sims_lower = cos_sims_upper = jnp.zeros((0,), dtype=jnp.float32)

    else:
        assert (
            args.cond_img is None and args.cond_txt is None
        ), "Can't specify --cond-img or --cond-txt without CLIP conditioning"

    print("Sampling encoded images from the transformer model...")

    with tqdm(total=args.n, unit="img") as pbar:
        encoded_imgs = []
        for batch_size in batches_split(args.batch_size, args.n):
            rng, rng2 = jax.random.split(rng)
            # Repeat stuff to match batch size
            if im_mdl.clip_conditioning and not im_mdl.clip_cones:
                clip_embeddings_v = repeat(clip_embeddings, "d -> n d", n=batch_size)
                cos_sims_lower_v = cos_sims_upper_v = jnp.zeros(
                    (batch_size, 0), dtype=jnp.float32
                )
            elif im_mdl.clip_conditioning and im_mdl.clip_cones:
                clip_embeddings_v = repeat(
                    clip_embeddings, "cones d -> n cones d", n=batch_size
                )
                cos_sims_lower_v = repeat(
                    cos_sims_lower, "cones -> n cones", n=batch_size
                )
                cos_sims_upper_v = repeat(
                    cos_sims_upper, "cones -> n cones", n=batch_size
                )
            else:
                clip_embeddings = cos_sims_lower = cos_sims_upper = jnp.zeros(
                    (batch_size, 0), dtype=jnp.float32
                )
            encoded_imgs.append(
                sample_v(
                    im_params,
                    clip_embeddings_v,
                    cos_sims_lower_v,
                    cos_sims_upper_v,
                    jax.random.split(rng2, batch_size),
                )
            )
            pbar.update(batch_size)

    print("Loading autoencoder model...")
    ae_res = int(model_cfg.image_tokens**0.5)
    assert ae_res**2 == model_cfg.image_tokens, "Image tokens must be a square number"
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
            decoded_imgs.append(
                ldm_autoencoder.decode_jv(
                    ae_mdl, ae_params, (ae_res, ae_res), encoded_img_batch
                )
            )
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
