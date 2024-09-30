"""Script for sampling from the model, plus associated utilities."""
import argparse
import dacite
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import orbax.checkpoint as ocp
import PIL.Image
import PIL.ImageDraw
import subprocess
import torch
import transformers
from copy import copy, deepcopy
from einops import rearrange, repeat
from flax.core import freeze
from functools import partial
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from omegaconf import OmegaConf
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from random import randint
from tqdm import tqdm, trange
from typing import List, Tuple

from . import ldm_autoencoder
from .checkpoint import get_imagemodel_from_checkpoint, load_eval_params
from .ldm_autoencoder import LDMAutoencoder
from .transformer_model import (
    ImageModel,
    LogitFilterMethod,
    ModelConfig,
    gpt_1_config,
    sample,
)


def can_make_grid(n: int) -> bool:
    return (n**0.5) % 1 == 0


def batches_split(batch_size: int, n: int) -> list[int]:
    """Split n into batches of size args.batch_size, plus the remainder."""
    split = [batch_size] * (n // batch_size)
    if n % batch_size != 0:
        split.append(n % batch_size)
    return split


@partial(jax.jit, static_argnums=(0,))
def sample_jv(mdl, params, top_p, clip_embeddings, max_cos_distances, rngs):
    assert (
        clip_embeddings.shape[0] == max_cos_distances.shape[0] == rngs.shape[0]
    ), "Number of clip embeddings, max cosine distances, and random keys should be equal"
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
        rngs,
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

    # Create a checkerboard pattern for the spacing areas. Makes it easier to see the boundaries
    # of the images.
    checkerboard = PIL.Image.new("RGB", (spacing, spacing))
    checkerboard_draw = PIL.ImageDraw.Draw(checkerboard)
    checkerboard_draw.rectangle([0, 0, spacing//2, spacing//2], fill=(255, 255, 255))
    checkerboard_draw.rectangle([spacing//2, 0, spacing, spacing//2], fill=(240, 240, 240))
    checkerboard_draw.rectangle([0, spacing//2, spacing//2, spacing], fill=(240, 240, 240))
    checkerboard_draw.rectangle([spacing//2, spacing//2, spacing, spacing], fill=(255, 255, 255))

    # Fill the grid with the checkerboard pattern
    for y in range(0, side_len_px, spacing):
        for x in range(0, side_len_px, spacing):
            grid_img.paste(checkerboard, (x, y))

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


def parse_cond_img(d: dict, clip_mdl, clip_processor) -> dict:
    """Compute a dictionary of conditioning info for an image."""
    d = deepcopy(d)
    path = Path(d["cond"])
    del d["cond"]

    img = PIL.Image.open(path)
    clip_inputs = clip_processor(images=img, return_tensors="jax")
    clip_embedding = clip_mdl.get_image_features(**clip_inputs)[0].astype(jnp.float32)
    clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)

    d["clip_embedding"] = jax.device_get(clip_embedding)

    print_cond_debug(d, f"image {path}")

    return d


def parse_cond_txt(d: dict, clip_mdl, clip_processor) -> dict:
    """Compute a dictionary of conditioning info for text."""
    d = deepcopy(d)
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
    params,
    ae_mdl,
    ae_params,
    batch_size,
    cap_centers,
    max_cos_distances,
    rng,
    logit_filter_method,
    logit_filter_threshold,
    temperature=1.0,
    force_f32=True,
):
    """Sample a bunch of images and return PIL images. cap_centers should have shape
    (n, cap_centers, 768) and max_cos_distances should have shape (n, cap_centers) where n is the
    number of images to sample, unless clip caps is off, in which case the cap centers dimension
    disappears from cap_centers and max_cos_distances should be None."""

    assert mdl.clip_conditioning, "unconditioned model is deprecated"
    assert isinstance(cap_centers, np.ndarray)
    assert len(cap_centers.shape) == 3
    n_imgs = cap_centers.shape[0]

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    sharding = NamedSharding(mesh, PartitionSpec("dev"))

    if mdl.clip_caps:
        assert isinstance(max_cos_distances, np.ndarray)
        assert len(cap_centers.shape) == 3
        assert len(max_cos_distances.shape) == 2
        assert cap_centers.shape[0] == max_cos_distances.shape[0] == n_imgs
        assert cap_centers.shape[1] == max_cos_distances.shape[1] == mdl.clip_cap_count
    else:
        assert max_cos_distances is None
        assert cap_centers.shape == (n_imgs, 768)
    assert batch_size % jax.device_count() == 0
    assert (len(cap_centers) % batch_size) % jax.device_count() == 0

    if force_f32:
        # Even with models that are trained with bf16, float32 gives subjectively substantially
        # better results
        mdl = mdl.clone(activations_dtype=jnp.float32)

    # Sample from the transformer model
    batches = batches_split(batch_size, len(cap_centers))
    sampled_codes_arrs = []
    rngs = jax.random.split(rng, len(cap_centers))
    with tqdm(total=len(cap_centers), desc="sampling", unit="img") as pbar:
        ctr = 0
        for batch in batches:
            rngs_batch = rngs[ctr : ctr + batch]
            rngs_sharded = jax.device_put(rngs_batch, sharding)
            cap_centers_batch = cap_centers[ctr : ctr + batch]
            cap_centers_sharded = jax.device_put(cap_centers_batch, sharding)
            if mdl.clip_caps:
                max_cos_distances_batch = max_cos_distances[ctr : ctr + batch]
                max_cos_distances_sharded = jax.device_put(
                    max_cos_distances_batch, sharding
                )
                codes = sample(
                    mdl,
                    params,
                    cap_centers_sharded,
                    max_cos_distances_sharded,
                    rngs_sharded,
                    logit_filter_method,
                    logit_filter_threshold,
                    temperature,
                )
            else:
                codes = sample(
                    mdl,
                    params,
                    cap_centers_sharded,
                    jnp.zeros((batch, 0), dtype=jnp.float32),
                    rngs_sharded,
                    logit_filter_method,
                    logit_filter_threshold,
                    temperature,
                )
            sampled_codes_arrs.append(jax.device_get(codes))
            pbar.update(batch)
            ctr += batch

    sampled_codes = np.concatenate(sampled_codes_arrs, axis=0)
    assert sampled_codes.shape == (
        len(cap_centers),
        mdl.image_tokens,
    ), f"{sampled_codes.shape} != {(len(cap_centers), mdl.image_tokens)}"

    ae_res = int(mdl.image_tokens**0.5)
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


def test_sample_loop_batch_equivalence():
    """Test that sampling with batch size 1 is equivalent to sampling with batch size 64.
    This should fail with certain busted cudnn versions."""

    # Set up ImageModel
    model_cfg = ModelConfig(
        image_tokens=256,
        clip_conditioning=True,
        clip_caps=True,
        clip_cap_count=1,
        n_layers=12,
        num_heads=12,
        d_model=768,
        ff_dim=3072,
        dropout=None,
        use_biases=True,
        activations_dtype=jnp.float32,
        activation_function=jax.nn.gelu,
    )
    ae_res = 16

    mdl = ImageModel(**model_cfg.__dict__)

    im_params_rng, ae_params_rng, caps_rng, sample_rng = jax.random.split(
        jax.random.PRNGKey(0), 4
    )

    params = jax.jit(mdl.init)({"params": im_params_rng}, *mdl.dummy_inputs())

    # Set up LDMAutoencoder
    ae_cfg = {
        "n_embed": 8192,
        "embed_dim": 3,
        "ddconfig": {
            "double_z": False,
            "z_channels": 3,
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0,
        },
    }
    ae_cfg = OmegaConf.create(ae_cfg)
    ae_mdl = LDMAutoencoder(ae_cfg)
    ae_params = ae_mdl.init(
        rngs={"params": ae_params_rng},
        x=jnp.zeros((256,), dtype=jnp.int32),
        method=LDMAutoencoder.decode,
        shape=(ae_res, ae_res),
    )

    # Generate inputs
    n_samples = 64
    centers_rng, cos_distances_rng = jax.random.split(caps_rng)
    cap_centers = jax.device_get(
        jax.random.normal(centers_rng, (n_samples, model_cfg.clip_cap_count, 768))
    )
    max_cos_distances = jax.device_get(
        jax.random.uniform(cos_distances_rng, (n_samples, model_cfg.clip_cap_count))
    )

    # Sample
    sample_bs = lambda bs: sample_loop(
        mdl,
        params,
        ae_mdl,
        ae_params,
        bs,
        cap_centers,
        max_cos_distances,
        sample_rng,
        LogitFilterMethod.TOP_P,
        0.9,
    )

    samples_1 = sample_bs(1)
    samples_64 = sample_bs(64)

    # check
    assert (
        len(samples_1) == len(samples_64) == n_samples
    ), "Number of samples should be equal"
    close_samples, far_samples = [], []
    for idx, (s1, s64) in enumerate(zip(samples_1, samples_64)):
        s1_arr, s64_arr = np.asarray(s1), np.asarray(s64)
        if np.allclose(s1_arr, s64_arr, rtol=0, atol=1):
            close_samples.append(idx)
        else:
            far_samples.append(idx)
            print(
                f"Samples at index {idx} are substantially different: {s1_arr} != {s64_arr}"
            )
    if len(far_samples) > 0:
        print(f"Far samples: {far_samples}")
        print(f"Close samples: {close_samples}")
        assert (
            len(close_samples) >= 0.95 * n_samples
        ), "Too many substantially different samples"


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


def mk_png_metadata(
    seed: int,
    logit_filter_method: LogitFilterMethod,
    logit_filter_threshold: float,
    temperature: float,
    force_f32: bool,
    cond_imgs: List[Tuple[str, float]],
    cond_txts: List[Tuple[str, float]],
    checkpoint: str,
    checkpoint_step: int,
) -> PngInfo:
    """Make some metadata to save with PNGs."""
    metadata = PngInfo()

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()
    except subprocess.CalledProcessError:
        commit_hash = "unknown"

    metadata.add_text("seed", str(seed))
    metadata.add_text("logit_filter_method", logit_filter_method.name)
    metadata.add_text("logit_filter_threshold", str(logit_filter_threshold))
    metadata.add_text("temperature", str(temperature))
    metadata.add_text("force_f32", str(force_f32))
    metadata.add_text("cond_imgs", str(cond_imgs))
    metadata.add_text("cond_txts", str(cond_txts))
    metadata.add_text("checkpoint", str(checkpoint))
    metadata.add_text("checkpoint_step", str(checkpoint_step))
    metadata.add_text("commit_hash", commit_hash)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--logit-filter-method", type=str, choices=["top_p", "min_p"], default="top_p"
    )
    parser.add_argument("--logit-filter-threshold", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--make-grids", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cond-img", type=str, nargs="*")
    parser.add_argument("--cond-txt", type=str, nargs="*")
    parser.add_argument(
        "--force-fp32", type=bool, default=True, help="Force float32 precision"
    )
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

    print("Loading transformer model metadata...")
    # We need to minimize VRAM usage, so we don't load the transformer parameters until after
    # computing CLIP embeddings.
    im_mdl = get_imagemodel_from_checkpoint(args.transformer_checkpoint_dir)

    if im_mdl.clip_conditioning:
        print("Loading CLIP model...")

        clip_mdl_name = "openai/clip-vit-large-patch14"
        clip_mdl = transformers.FlaxCLIPModel.from_pretrained(
            clip_mdl_name, dtype=jnp.float16
        )
        clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

        print("Computing CLIP embeddings...")
        cond_img_inputs = [parse_cond_str(s) for s in (args.cond_img or [])]
        cond_txt_inputs = [parse_cond_str(s) for s in (args.cond_txt or [])]
        cond_img_dicts = [
            parse_cond_img(s, clip_mdl, clip_processor) for s in cond_img_inputs
        ]
        cond_txt_dicts = [
            parse_cond_txt(s, clip_mdl, clip_processor) for s in cond_txt_inputs
        ]
        cond_dicts = cond_img_dicts + cond_txt_dicts
        del clip_mdl, clip_processor

        if im_mdl.clip_caps:
            total_conds = len(cond_dicts)
            assert all(
                [d["max_cos_distance"] is not None for d in cond_dicts]
            ), "Must specify max cosine distance"
            assert (
                total_conds <= im_mdl.clip_cap_count
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
                im_mdl, 1, total_conds, rng2
            )
            fill_cap_centers = rearrange(fill_cap_centers, "1 cap clip -> cap clip")
            fill_max_cos_distances = rearrange(fill_max_cos_distances, "1 cap -> cap")
            print(f"Fill cap center norms: {np.linalg.norm(fill_cap_centers, axis=-1)}")

            cap_centers = np.concatenate([clip_embeddings_cond, fill_cap_centers])
            max_cos_distances = np.concatenate(
                [max_cos_distances_cond, fill_max_cos_distances]
            )

            assert cap_centers.shape == (im_mdl.clip_cap_count, 768)
            assert max_cos_distances.shape == (im_mdl.clip_cap_count,)
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
        cond_img_inputs, cond_txt_inputs = [], []

    # Load the transformer model parameters
    im_params, checkpoint_step, im_mdl = load_eval_params(
        args.transformer_checkpoint_dir
    )

    print(
        f"Loaded transformer model step {checkpoint_step} from "
        f"{args.transformer_checkpoint_dir}"
    )

    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    im_params = jax.device_put(im_params, NamedSharding(mesh, PartitionSpec(None)))

    print("Loading autoencoder model...")
    ae_res = int(im_mdl.image_tokens**0.5)
    assert ae_res**2 == im_mdl.image_tokens, "Image tokens must be a square number"
    ae_cfg = OmegaConf.load(args.autoencoder_cfg)["model"][
        "params"
    ]  # type:ignore[index]
    ae_mdl = LDMAutoencoder(ae_cfg)
    ae_params = LDMAutoencoder.params_from_torch(
        torch.load(args.autoencoder_checkpoint, map_location="cpu"), cfg=ae_cfg
    )

    logit_filter_method = LogitFilterMethod[args.logit_filter_method.upper()]

    imgs = sample_loop(
        mdl=im_mdl,
        params=im_params,
        ae_mdl=ae_mdl,
        ae_params=ae_params,
        batch_size=args.batch_size,
        cap_centers=cap_centers,
        max_cos_distances=max_cos_distances,
        rng=rng,
        logit_filter_method=logit_filter_method,
        logit_filter_threshold=args.logit_filter_threshold,
        temperature=args.temperature,
        force_f32=args.force_fp32,
    )

    args.out_dir.mkdir(exist_ok=True, parents=True)

    metadata = mk_png_metadata(
        seed,
        logit_filter_method,
        args.logit_filter_threshold,
        args.temperature,
        args.force_fp32,
        [(d["cond"], float(d["max_cos_distance"])) for d in cond_img_inputs],
        [(d["cond"], float(d["max_cos_distance"])) for d in cond_txt_inputs],
        args.transformer_checkpoint_dir.name,
        checkpoint_step,
    )
    print("Saving images...")
    for i, img in enumerate(tqdm(imgs, unit="img")):
        img.save(args.out_dir / f"{i:04d}.png", pnginfo=metadata)

    if args.make_grids:
        print(f"Making {len(grid_img_idxs)} grids...")
        for i, indices in tqdm(enumerate(grid_img_idxs)):
            grid_imgs = [imgs[i] for i in indices]
            make_grid(grid_imgs).save(
                args.out_dir / f"grid_{i:04d}.png", pnginfo=metadata
            )


if __name__ == "__main__":
    main()
