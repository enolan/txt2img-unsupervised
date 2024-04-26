"""Script for sampling from the model, plus associated utilities."""
import argparse
import dacite
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import PIL.Image
import torch
import transformers
from copy import copy
from einops import rearrange, repeat
from flax.core import freeze
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
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


@partial(jax.jit, static_argnums=(0,))
def sample_jv(mdl, params, top_p, clip_embeddings, max_cos_distances, rng):
    return jax.vmap(
        lambda clip_embedding, max_cos_distance, rng: sample(
            mdl,
            params,
            clip_embedding,
            max_cos_distance,
            rng,
            top_p,
        )
    )(
        clip_embeddings,
        max_cos_distances,
        jax.random.split(rng, len(clip_embeddings)),
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
    "condition" - no caps
    "condition:0.8" - caps with max cosine distances
    """
    if ":" in s:
        cond, max_cos_distance = s.split(":")
        max_cos_distance = jnp.array(float(max_cos_distance))
    else:
        cond = s
        max_cos_distance = None
    return {
        "cond": cond,
        "max_cos_distance": max_cos_distance,
    }


def print_cond_debug(d: dict, name: str) -> None:
    print(
        f"Computed CLIP embedding for {name}, shape {d['clip_embedding'].shape}, norm {jnp.linalg.norm(d['clip_embedding'])}, max cosine distance: "
        + (
            f"{d['max_cos_distance']:.4f}"
            if d["max_cos_distance"] is not None
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

    d["clip_embedding"] = jax.device_get(clip_embedding)

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

    d["clip_embedding"] = jax.device_get(clip_embedding)
    print_cond_debug(d, f"text {txt}")

    return d


def sample_loop(
    mdl,
    model_cfg,
    params,
    ae_mdl,
    ae_params,
    batch_size,
    cap_centers,
    max_cos_distances,
    rng,
    top_p,
    force_f32=True,
):
    """Sample a bunch of images and return PIL images. cap_centers should have shape
    (n, cap_centers, 768) and max_cos_distances should have shape (n, cap_centers) where n is the
    number of images to sample, unless clip caps is off, in which case the cap centers dimension
    disappears from cap_centers and max_cos_distances should be None."""

    assert model_cfg.clip_conditioning, "unconditioned model is deprecated"
    assert isinstance(cap_centers, np.ndarray)

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    sharding = NamedSharding(mesh, PartitionSpec("dev"))

    if model_cfg.clip_caps:
        assert isinstance(max_cos_distances, np.ndarray)
        assert len(cap_centers.shape) == 3
        assert len(max_cos_distances.shape) == 2
        assert cap_centers.shape[0] == max_cos_distances.shape[0]
        assert cap_centers.shape[1] == model_cfg.clip_cap_count
        assert max_cos_distances.shape[1] == model_cfg.clip_cap_count
    else:
        assert max_cos_distances is None
        assert len(cap_centers.shape) == 2
        assert cap_centers.shape[1] == 768
    assert batch_size % jax.device_count() == 0
    assert (len(cap_centers) % batch_size) % jax.device_count() == 0

    if force_f32:
        # Even with models that are trained with bf16, float32 gives subjectively substantially
        # better results
        mdl = mdl.clone(activations_dtype=jnp.float32)

    # Sample from the transformer model
    batches = batches_split(batch_size, len(cap_centers))
    sampled_codes_arrs = []
    with tqdm(total=len(cap_centers), desc="sampling", unit="img") as pbar:
        ctr = 0
        for batch in batches:
            rng, rng2 = jax.random.split(rng)
            if model_cfg.clip_caps:
                cap_centers_batch = cap_centers[ctr : ctr + batch]
                cap_centers_sharded = jax.device_put(cap_centers_batch, sharding)
                max_cos_distances_batch = max_cos_distances[ctr : ctr + batch]
                max_cos_distances_sharded = jax.device_put(
                    max_cos_distances_batch, sharding
                )
                codes = sample_jv(
                    mdl,
                    params,
                    top_p,
                    cap_centers_sharded,
                    max_cos_distances_sharded,
                    rng2,
                )
                sampled_codes_arrs.append(jax.device_get(codes))
            else:
                cap_centers_batch = cap_centers[ctr : ctr + batch]
                cap_centers_sharded = jax.device_put(cap_centers_batch, sharding)
                codes = sample_jv(
                    mdl,
                    params,
                    top_p,
                    cap_centers_sharded,
                    jnp.zeros((batch, 0), dtype=jnp.float32),
                    rng2,
                )
                sampled_codes_arrs.append(jax.device_get(codes))
            pbar.update(batch)
            ctr += batch

    sampled_codes = np.concatenate(sampled_codes_arrs, axis=0)
    assert sampled_codes.shape == (
        len(cap_centers),
        model_cfg.image_tokens,
    ), f"{sampled_codes.shape} != {(len(cap_centers), model_cfg.image_tokens)}"

    ae_res = int(model_cfg.image_tokens**0.5)
    decoded_imgs = []
    # Decode the sampled codes
    with tqdm(total=len(cap_centers), desc="decoding", unit="img") as pbar:
        ctr = 0
        for batch in batches:
            codes_batch = sampled_codes[ctr : ctr + batch]
            codes_sharded = jax.device_put(codes_batch, sharding)
            imgs = ldm_autoencoder.decode_jv(
                ae_mdl, ae_params, (ae_res, ae_res), codes_sharded
            )
            decoded_imgs.append(jax.device_get(imgs))
            pbar.update(batch)
            ctr += batch

    decoded_imgs = np.concatenate(decoded_imgs, axis=0)
    assert decoded_imgs.shape[0] == len(cap_centers)

    pil_imgs = []
    for img in tqdm(decoded_imgs, desc="PILifying", unit="img"):
        pil_imgs.append(PIL.Image.fromarray(img))
    return pil_imgs


def mk_filler_caps(model_cfg, n_cap_sets, n_used_caps, rng):
    """Make caps with max cosine distance 2 and random centers to fill in all but n_used_caps cap
    slots. These caps *should* have no effect on the output, since they don't restrict the space of
    valid embeddings at all. Filler caps are necessary for prompting models with >1 cap slot.
    """
    assert model_cfg.clip_caps
    assert n_used_caps <= model_cfg.clip_cap_count
    centers = jax.random.normal(
        rng, (n_cap_sets, model_cfg.clip_cap_count - n_used_caps, 768)
    )
    centers = centers / jnp.linalg.norm(centers, axis=-1, keepdims=True)
    max_cos_distances = np.full(
        (n_cap_sets, model_cfg.clip_cap_count - n_used_caps), 2, dtype=jnp.float32
    )
    return np.asarray(centers), max_cos_distances


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
            grid_img_idxs = [list(range(args.n))]
        elif args.n % 2 == 0 and can_make_grid(args.n / 2):
            grid_img_idxs = [list(range(args.n // 2)), list(range(args.n // 2, args.n))]
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
    checkpoint_mngr = ocp.CheckpointManager(
        # Orbax chokes on relative paths for some godforsaken reason
        args.transformer_checkpoint_dir.absolute(),
        item_names=("params",),
    )
    print(
        f"Loading step {checkpoint_mngr.latest_step()} from {args.transformer_checkpoint_dir}"
    )
    im_params = checkpoint_mngr.restore(
        checkpoint_mngr.latest_step(),
        args=ocp.args.Composite(params=ocp.args.StandardRestore()),
    )["params"]

    model_cfg = ModelConfig.from_json_dict(checkpoint_mngr.metadata()["model_cfg"])
    # Samples are substantially better with 32 bits, even for models trained with bf16
    model_cfg.activations_dtype = jnp.float32
    im_mdl = ImageModel(**model_cfg.__dict__)
    if model_cfg.clip_conditioning:
        print("Loading CLIP model...")

        clip_mdl_name = "openai/clip-vit-large-patch14"
        clip_mdl = transformers.FlaxCLIPModel.from_pretrained(
            clip_mdl_name, dtype=jnp.float16
        )
        clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

        print("Computing CLIP embeddings...")
        cond_img_dicts = [
            parse_cond_img(s, clip_mdl, clip_processor) for s in (args.cond_img or [])
        ]
        cond_txt_dicts = [
            parse_cond_txt(s, clip_mdl, clip_processor) for s in (args.cond_txt or [])
        ]
        cond_dicts = cond_img_dicts + cond_txt_dicts

        if model_cfg.clip_caps:
            total_conds = len(cond_dicts)
            assert all(
                [d["max_cos_distance"] is not None for d in cond_dicts]
            ), "Must specify max cosine distance"
            assert (
                total_conds <= model_cfg.clip_cap_count
            ), "Too many CLIP embeddings for the number of caps"

            if total_conds == 0:
                # unconditioned sampling
                clip_embeddings_cond = np.zeros((0, 768))
                max_cos_distances_cond = np.zeros(0)
            else:
                clip_embeddings_cond = np.stack(
                    [d["clip_embedding"] for d in cond_dicts]
                )
                max_cos_distances_cond = np.stack(
                    [d["max_cos_distance"] for d in cond_dicts]
                )
            assert clip_embeddings_cond.shape == (total_conds, 768)
            assert max_cos_distances_cond.shape == (total_conds,)

            rng, rng2 = jax.random.split(rng)
            fill_cap_centers, fill_max_cos_distances = mk_filler_caps(
                model_cfg, 1, total_conds, rng2
            )
            fill_cap_centers = rearrange(fill_cap_centers, "1 cap clip -> cap clip")
            fill_max_cos_distances = rearrange(fill_max_cos_distances, "1 cap -> cap")
            print(f"Fill cap center norms: {np.linalg.norm(fill_cap_centers, axis=-1)}")

            cap_centers = np.concatenate([clip_embeddings_cond, fill_cap_centers])
            max_cos_distances = np.concatenate(
                [max_cos_distances_cond, fill_max_cos_distances]
            )

            assert cap_centers.shape == (model_cfg.clip_cap_count, 768)
            assert max_cos_distances.shape == (model_cfg.clip_cap_count,)
            cap_centers = repeat(cap_centers, "cap clip -> n cap clip", n=args.n)
            max_cos_distances = repeat(max_cos_distances, "cap -> n cap", n=args.n)
        else:
            assert (
                len(cond_dicts) == 1
            ), "Can only specify one CLIP embedding without clip caps"
            clip_embeddings = cond_dicts[0]["clip_embedding"]
            assert clip_embeddings.shape == (768,)
            assert all(
                [d["max_cos_distance"] is None for d in cond_dicts]
            ), "Can't specify max cosine distance without clip caps"
            cap_centers = repeat(clip_embeddings, "clip -> n clip", n=args.n)
            max_cos_distances = None

    else:
        assert (
            args.cond_img is None and args.cond_txt is None
        ), "Can't specify --cond-img or --cond-txt without CLIP conditioning"

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

    imgs = sample_loop(
        mdl=im_mdl,
        model_cfg=model_cfg,
        params=im_params,
        ae_mdl=ae_mdl,
        ae_params=ae_params,
        batch_size=args.batch_size,
        cap_centers=cap_centers,
        max_cos_distances=max_cos_distances,
        rng=rng,
        top_p=args.top_p,
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)

    print("Saving images...")
    for i, img in enumerate(tqdm(imgs, unit="img")):
        img.save(args.out_dir / f"{i:04d}.png")

    if args.make_grids:
        print(f"Making {len(grid_img_idxs)} grids...")
        for i, indices in tqdm(enumerate(grid_img_idxs)):
            grid_imgs = [imgs[i] for i in indices]
            make_grid(grid_imgs).save(args.out_dir / f"grid_{i:04d}.png")
