"""Script for sampling from the model, plus associated utilities."""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import argparse
import gc
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
from typing import List, Tuple, Union

from . import flow_matching, ldm_autoencoder
from .checkpoint import get_model_from_checkpoint, load_params
from .ldm_autoencoder import LDMAutoencoder
from .transformer_model import (
    ImageModel,
    LogitFilterMethod,
    TransformerModelConfig,
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
    checkerboard_draw.rectangle(
        [0, 0, spacing // 2, spacing // 2], fill=(255, 255, 255)
    )
    checkerboard_draw.rectangle(
        [spacing // 2, 0, spacing, spacing // 2], fill=(240, 240, 240)
    )
    checkerboard_draw.rectangle(
        [0, spacing // 2, spacing // 2, spacing], fill=(240, 240, 240)
    )
    checkerboard_draw.rectangle(
        [spacing // 2, spacing // 2, spacing, spacing], fill=(255, 255, 255)
    )

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
    disappears from cap_centers and max_cos_distances should be None. If clip conditioning is off,
    cap_centers should have shape (n, 0)."""

    assert isinstance(cap_centers, np.ndarray)
    n_imgs = cap_centers.shape[0]

    if not mdl.clip_conditioning:
        assert len(cap_centers.shape) == 2
        assert cap_centers.shape[1] == 0
        assert max_cos_distances is None
        max_cos_distances = np.zeros((n_imgs, 0), dtype=jnp.float32)
    elif mdl.clip_caps:
        assert len(cap_centers.shape) == 3
        assert cap_centers.shape[1] == mdl.clip_cap_count
        assert cap_centers.shape[2] == 768
        assert isinstance(max_cos_distances, np.ndarray)
        assert max_cos_distances.shape == (n_imgs, mdl.clip_cap_count)
    else:
        # No cap, single CLIP embedding
        assert len(cap_centers.shape) == 2
        assert cap_centers.shape[1] == 768
        assert max_cos_distances is None
        max_cos_distances = np.zeros((n_imgs, 0), dtype=jnp.float32)

    params_sharded = None
    ae_params_sharded = None
    sharding = None

    def mk_sharding(batch_size):
        # Create a sharding that uses the maximum number of devices such that batch_size is
        # divisible by the number of devices
        nonlocal params_sharded, ae_params_sharded, sharding
        devs_to_use = max(
            i for i in range(1, jax.device_count() + 1) if batch_size % i == 0
        )
        devices = mesh_utils.create_device_mesh(
            (devs_to_use,), devices=jax.devices()[:devs_to_use]
        )
        mesh = Mesh(devices, axis_names=("dev",))
        sharding = NamedSharding(mesh, PartitionSpec("dev"))
        if devs_to_use != jax.device_count():
            # JAX is weird, we have to replicate the params across the same set of devices we shard
            # across, even if the latter is a subset of the former.
            params_sharded = jax.device_put(
                params, NamedSharding(mesh, PartitionSpec(None))
            )
            ae_params_sharded = jax.device_put(
                ae_params, NamedSharding(mesh, PartitionSpec(None))
            )
        else:
            params_sharded = params
            ae_params_sharded = ae_params

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
            mk_sharding(batch)
            rngs_batch = rngs[ctr : ctr + batch]
            rngs_sharded = jax.device_put(rngs_batch, sharding)
            cap_centers_batch = cap_centers[ctr : ctr + batch]
            cap_centers_sharded = jax.device_put(cap_centers_batch, sharding)
            max_cos_distances_batch = max_cos_distances[ctr : ctr + batch]
            max_cos_distances_sharded = jax.device_put(
                max_cos_distances_batch, sharding
            )
            codes = sample(
                mdl,
                params_sharded,
                cap_centers_sharded,
                max_cos_distances_sharded,
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
            mk_sharding(batch)
            codes_batch = sampled_codes[ctr : ctr + batch]
            codes_sharded = jax.device_put(codes_batch, sharding)
            imgs = ldm_autoencoder.decode_jv(
                ae_mdl, ae_params_sharded, (ae_res, ae_res), codes_sharded
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
    model_cfg = TransformerModelConfig(
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


def mk_filler_caps(model, n_cap_sets, n_used_caps, rng):
    """Make caps with max cosine distance 2 and random centers to fill in all but n_used_caps cap
    slots. These caps *should* have no effect on the output, since they don't restrict the space of
    valid embeddings at all. Filler caps are necessary for unconditioned sampling on cap models and
    for prompting models with >1 cap slot with fewer than the full number of caps.
    """
    assert model.clip_caps
    assert n_used_caps <= model.clip_cap_count
    centers = jax.random.normal(
        rng, (n_cap_sets, model.clip_cap_count - n_used_caps, 768)
    )
    centers = centers / jnp.linalg.norm(centers, axis=-1, keepdims=True)
    max_cos_distances = np.full(
        (n_cap_sets, model.clip_cap_count - n_used_caps), 2, dtype=jnp.float32
    )
    return np.asarray(centers), max_cos_distances


def mk_png_metadata(
    seed: int,
    logit_filter_method: LogitFilterMethod,
    logit_filter_threshold: float,
    temperature: float,
    force_f32: bool,
    cond_imgs: List[Tuple[str, Union[float, None]]],
    cond_txts: List[Tuple[str, Union[float, None]]],
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


def two_stage_sample_loop(
    im_mdl,
    im_params,
    flow_mdl,
    flow_params,
    ae_mdl,
    ae_params,
    batch_size,
    cap_centers,
    max_cos_distances,
    rng,
    logit_filter_method=LogitFilterMethod.TOP_P,
    logit_filter_threshold=0.9,
    temperature=1.0,
    force_f32=True,
    n_flow_steps=100,
    n_flow_proposals=128,
    flow_algorithm=None,
    algorithm_params=None,
):
    """Two-stage sampling pipeline that uses a flow model to generate CLIP embeddings constrained to
    caps with importance sampling, then uses an image model to generate codes, then decodes with an
    autoencoder.

    Args:
        im_mdl: ImageModel instance
        im_params: Parameters for the ImageModel (on CPU)
        flow_mdl: VectorField instance
        flow_params: Parameters for the VectorField (on CPU)
        ae_mdl: LDMAutoencoder instance
        ae_params: Parameters for the LDMAutoencoder (on CPU)
        batch_size: Batch size for processing
        cap_centers: Array of cap centers [n_imgs, 768]
        max_cos_distances: Array of max cosine distances [n_imgs]
        rng: JAX random key
        logit_filter_method: Logit filtering method for transformer sampling
        logit_filter_threshold: Threshold for logit filtering
        temperature: Temperature for transformer sampling
        force_f32: Whether to force float32 precision for transformer
        n_flow_steps: Number of ODE integration steps for flow model

    Returns:
        List of PIL images
    """

    assert isinstance(cap_centers, np.ndarray)
    assert isinstance(max_cos_distances, np.ndarray)
    n_imgs = cap_centers.shape[0]
    assert cap_centers.shape == (n_imgs, 768)
    assert max_cos_distances.shape == (n_imgs,)

    if not im_mdl.clip_conditioning:
        raise ValueError(
            "Image model must support CLIP conditioning for two-stage sampling"
        )

    # Validate flow model domain
    if flow_mdl.domain_dim != 768:
        raise ValueError(
            f"Flow model domain_dim must be 768 (CLIP embedding size), got {flow_mdl.domain_dim}"
        )

    # Stage 1: Generate CLIP embeddings using flow model
    print("Stage 1: Generating CLIP embeddings with flow model...")

    # Put flow model parameters on GPU
    devices = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=("dev",))
    flow_params_gpu = jax.device_put(
        flow_params, NamedSharding(mesh, PartitionSpec(None))
    )

    # Prepare importance sampler utilities
    table = flow_matching.LogitsTable(d=flow_mdl.domain_dim - 1, n=16384)
    cond_vec = jnp.zeros((flow_mdl.conditioning_dim,), dtype=jnp.float32)

    # Sample CLIP embeddings from flow model via cap-constrained importance sampling
    flow_rng, rng = jax.random.split(rng)

    # Group all images by identical cap constraints to pool proposal samples
    cap_specs = [
        (tuple(center), float(d_max))
        for center, d_max in zip(cap_centers, max_cos_distances)
    ]

    # Find unique cap specifications and which image indices want each one
    unique_caps = {}
    for img_idx, cap_spec in enumerate(cap_specs):
        if cap_spec not in unique_caps:
            unique_caps[cap_spec] = []
        unique_caps[cap_spec].append(img_idx)

    print(f"Grouped {n_imgs} images into {len(unique_caps)} unique caps")

    # Initialize results array
    generated_clip_embeddings = np.zeros((n_imgs, 768), dtype=np.float32)

    # Generate random keys for each unique cap
    cap_keys = jax.random.split(flow_rng, len(unique_caps))

    # Sample embeddings per unique cap
    with tqdm(total=len(unique_caps), desc="flow sampling", unit="cap") as pbar:
        for cap_idx, ((center_tuple, d_max), img_indices) in enumerate(
            unique_caps.items()
        ):
            center = jnp.array(center_tuple)
            n_samples_for_cap = len(img_indices)

            # Use more proposals when sampling for multiple images with same cap
            # This improves importance sampling statistics
            total_proposals = n_flow_proposals * max(1, n_samples_for_cap // 4)

            # Build per-cap algorithm params
            algo = (
                flow_algorithm
                if flow_algorithm is not None
                else flow_matching.SamplingAlgorithm.SIR
            )
            if algo == flow_matching.SamplingAlgorithm.SIR:
                # Derive per-cap SIR params from base, adjust n_proposal_samples
                base = algorithm_params or flow_matching.SIRParams(
                    n_proposal_samples=total_proposals,
                    n_projections=10,
                    batch_size=512,
                )
                sir_params = flow_matching.SIRParams(
                    n_proposal_samples=total_proposals,
                    n_projections=base.n_projections,
                    batch_size=base.batch_size,
                )
                algo_params_for_cap = sir_params
            elif algo == flow_matching.SamplingAlgorithm.REJECTION:
                algo_params_for_cap = algorithm_params or flow_matching.RejectionParams(
                    proposal_batch_size=256
                )
            elif algo == flow_matching.SamplingAlgorithm.MCMC:
                algo_params_for_cap = algorithm_params or flow_matching.MCMCParams(
                    n_chains=512,
                    n_steps_per_chain=128,
                    step_scale=1 / 3,
                    burnin_steps=16,
                )
            else:
                raise ValueError(f"Unsupported flow sampling algorithm: {algo}")

            samples, ess, _ = flow_matching.generate_cap_constrained_samples(
                flow_mdl,
                flow_params_gpu,
                cap_keys[cap_idx],
                center,
                d_max,
                table,
                cond_vec,
                n_output_samples=n_samples_for_cap,
                flow_n_steps=n_flow_steps,
                algorithm=algo,
                algorithm_params=algo_params_for_cap,
            )

            # Distribute results back to original image indices
            for i, img_idx in enumerate(img_indices):
                generated_clip_embeddings[img_idx] = jax.device_get(samples[i])

            tqdm.write(f"Cap {cap_idx} ESS: {ess}")
            postfix = {
                "n_imgs": n_samples_for_cap,
                "ESS": f"{float(ess):.1f}",
            }
            if algo == flow_matching.SamplingAlgorithm.SIR:
                postfix["proposals"] = total_proposals
            pbar.set_postfix(postfix)
            pbar.update(1)

    cos_distances = 1 - np.sum(generated_clip_embeddings * cap_centers, axis=1)
    assert cos_distances.shape == (n_imgs,)
    print(f"Cosine distances to cap centers: {cos_distances}")
    # Free GPU memory used by flow model
    del flow_params_gpu
    gc.collect()

    # Stage 2: Generate image codes using transformer model
    print("Stage 2: Generating image codes with transformer model...")

    batches = batches_split(batch_size, n_imgs)

    # Put image model parameters on GPU
    im_params_gpu = jax.device_put(im_params, NamedSharding(mesh, PartitionSpec(None)))

    if force_f32:
        im_mdl = im_mdl.clone(activations_dtype=jnp.float32)

    # Prepare inputs for image model based on its conditioning type
    if im_mdl.clip_caps:
        # For cap-conditioned models, use generated embeddings with max_cos_distance=0
        # This means "use this exact embedding"
        cap_centers_for_im = generated_clip_embeddings.reshape(n_imgs, 1, 768)
        cap_centers_for_im = np.tile(cap_centers_for_im, (1, im_mdl.clip_cap_count, 1))
        max_cos_distances_for_im = np.zeros(
            (n_imgs, im_mdl.clip_cap_count), dtype=jnp.float32
        )
    else:
        # For models with CLIP conditioning but no caps, use embeddings directly
        cap_centers_for_im = generated_clip_embeddings
        max_cos_distances_for_im = np.zeros((n_imgs, 0), dtype=jnp.float32)

    # Sample codes from transformer model
    sampled_codes_arrs = []
    trans_rng, rng = jax.random.split(rng)

    with tqdm(total=n_imgs, desc="transformer sampling", unit="img") as pbar:
        ctr = 0
        for batch in batches:
            # Generate fresh random keys for this batch
            batch_rngs = jax.random.split(trans_rng, batch + 1)
            trans_rng = batch_rngs[0]  # Update for next iteration
            rngs_batch = batch_rngs[1:]  # Use the rest for sampling

            cap_centers_batch = cap_centers_for_im[ctr : ctr + batch]
            max_cos_distances_batch = max_cos_distances_for_im[ctr : ctr + batch]

            codes = sample(
                im_mdl,
                im_params_gpu,
                cap_centers_batch,
                max_cos_distances_batch,
                rngs_batch,
                logit_filter_method,
                logit_filter_threshold,
                temperature,
            )
            sampled_codes_arrs.append(jax.device_get(codes))
            pbar.update(batch)
            ctr += batch

    sampled_codes = np.concatenate(sampled_codes_arrs, axis=0)
    assert sampled_codes.shape == (n_imgs, im_mdl.image_tokens)

    # Free GPU memory used by image model
    del im_params_gpu
    gc.collect()

    # Stage 3: Decode codes using autoencoder
    print("Stage 3: Decoding images with autoencoder...")

    # Put autoencoder parameters on GPU
    ae_params_gpu = jax.device_put(ae_params, NamedSharding(mesh, PartitionSpec(None)))

    ae_res = int(im_mdl.image_tokens**0.5)
    decoded_imgs = []

    with tqdm(total=n_imgs, desc="autoencoder decoding", unit="img") as pbar:
        ctr = 0
        for batch in batches:
            codes_batch = sampled_codes[ctr : ctr + batch]

            imgs = ldm_autoencoder.decode_jv(
                ae_mdl, ae_params_gpu, (ae_res, ae_res), codes_batch
            )
            decoded_imgs.append(jax.device_get(imgs))
            pbar.update(batch)
            ctr += batch

    decoded_imgs = np.concatenate(decoded_imgs, axis=0)
    assert decoded_imgs.shape[0] == n_imgs

    # Free GPU memory used by autoencoder
    del ae_params_gpu
    gc.collect()

    # Convert to PIL images
    pil_imgs = []
    for img in tqdm(decoded_imgs, desc="PILifying", unit="img"):
        pil_imgs.append(PIL.Image.fromarray(img))

    return pil_imgs


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
    parser.add_argument(
        "--flow-model",
        type=Path,
        help="Optional flow model checkpoint directory for two-stage sampling",
    )
    parser.add_argument(
        "--n-flow-steps",
        type=int,
        default=100,
        help="Number of ODE integration steps for flow model",
    )
    parser.add_argument(
        "--flow-algorithm",
        type=str,
        choices=["sir", "rejection", "mcmc"],
        default="sir",
        help="Flow sampling algorithm to use",
    )
    # SIR params
    parser.add_argument(
        "--sir-n-projections",
        type=int,
        default=10,
        help="Number of random projections for log-probability estimation (SIR)",
    )
    parser.add_argument(
        "--sir-batch-size",
        type=int,
        default=512,
        help="Batch size for probability evaluation (SIR)",
    )
    # Rejection params
    parser.add_argument(
        "--rejection-proposal-batch-size",
        type=int,
        default=256,
        help="Batch size of proposals for rejection sampling",
    )
    # MCMC params
    parser.add_argument(
        "--mcmc-n-chains",
        type=int,
        default=512,
        help="Number of MCMC chains",
    )
    parser.add_argument(
        "--mcmc-steps-per-chain",
        type=int,
        default=128,
        help="Number of MCMC steps per chain",
    )
    parser.add_argument(
        "--mcmc-step-scale",
        type=float,
        default=1.0 / 3.0,
        help="Step scale as a fraction of cap radius (std dev of geodesic distance)",
    )
    parser.add_argument(
        "--mcmc-burnin-steps",
        type=int,
        default=16,
        help="Number of burn-in steps to discard for MCMC",
    )
    parser.add_argument(
        "--n-flow-proposals",
        type=int,
        default=128,
        help="Number of proposal samples for cap-constrained importance sampling",
    )
    parser.add_argument("transformer_checkpoint_dir", type=Path)
    parser.add_argument("autoencoder_checkpoint", type=Path)
    parser.add_argument("autoencoder_cfg", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    # Turn on JAX compilation cache
    jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")

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
    model_cfg, im_mdl = get_model_from_checkpoint(args.transformer_checkpoint_dir)
    assert isinstance(im_mdl, ImageModel), f"Expected ImageModel, got {type(im_mdl)}"

    if args.flow_model is not None:
        if not im_mdl.clip_conditioning:
            raise ValueError(
                "Image model must support CLIP conditioning for two-stage sampling"
            )

    if im_mdl.clip_conditioning and (args.cond_img or args.cond_txt):
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
        gc.collect()

        # Handle conditioning inputs (both single-stage and two-stage)
        if args.flow_model is not None:
            # Flow models act like they have a single cap constraint
            assert (
                len(cond_dicts) == 1
            ), "Flow models support only one CLIP embedding (like clip_cap_count=1)"
            clip_embedding = cond_dicts[0]["clip_embedding"]
            max_cos_distance = cond_dicts[0]["max_cos_distance"]
            assert clip_embedding.shape == (768,)
            assert (
                max_cos_distance is not None
            ), "Must specify max cosine distance for flow model"

            # Flow model expects cap centers and distances for n images
            cap_centers = repeat(clip_embedding, "clip -> n clip", n=args.n)
            max_cos_distances = np.full((args.n,), max_cos_distance, dtype=np.float32)
        elif im_mdl.clip_caps:
            # Single-stage sampling with caps
            total_conds = len(cond_dicts)
            assert all(
                [d["max_cos_distance"] is not None for d in cond_dicts]
            ), "Must specify max cosine distance"
            assert (
                total_conds <= im_mdl.clip_cap_count
            ), "Too many CLIP embeddings for the number of caps"

            clip_embeddings_cond = np.stack([d["clip_embedding"] for d in cond_dicts])
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

            sort_idxs = np.argsort(max_cos_distances)
            cap_centers = cap_centers[sort_idxs]
            max_cos_distances = max_cos_distances[sort_idxs]

            cap_centers = repeat(cap_centers, "cap clip -> n cap clip", n=args.n)
            max_cos_distances = repeat(max_cos_distances, "cap -> n cap", n=args.n)
        else:
            # Single-stage sampling without caps
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
        # Handle unconditioned sampling
        cond_img_inputs, cond_txt_inputs = [], []

        if args.flow_model is not None:
            # Two-stage unconditioned sampling: use deterministic caps for flow model
            print("Using unconditioned two-stage sampling with flow model...")
            # Use straight north (first dimension = 1, rest = 0) for all cap centers
            cap_centers = np.zeros((args.n, 768), dtype=np.float32)
            cap_centers[:, 0] = 1.0  # Set first dimension to 1 for "straight north"
            # Use max cosine distance of 2 (no restriction)
            max_cos_distances = np.full((args.n,), 2.0, dtype=np.float32)
        elif im_mdl.clip_conditioning:
            # Single-stage unconditioned sampling with CLIP conditioning
            if im_mdl.clip_caps:
                # Use filler caps for cap-conditioned models
                rng, rng2 = jax.random.split(rng)
                cap_centers, max_cos_distances = mk_filler_caps(im_mdl, args.n, 0, rng2)
            else:
                # Models that expect a single CLIP embedding don't support unconditioned sampling
                assert (
                    False
                ), "Unconditioned sampling with single CLIP embedding models doesn't make sense"
        else:
            # Single-stage unconditioned sampling without CLIP conditioning
            cap_centers = np.zeros((args.n, 0), dtype=np.float32)
            max_cos_distances = None

    # Load the transformer model parameters
    im_params, checkpoint_step, im_mdl = load_params(
        args.transformer_checkpoint_dir, device="cpu"
    )

    print(
        f"Loaded transformer model step {checkpoint_step} from "
        f"{args.transformer_checkpoint_dir}"
    )

    # Only put transformer params on GPU if not using two-stage sampling
    if args.flow_model is None:
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

    # Load flow model after CLIP processing to minimize peak memory usage
    flow_mdl = None
    flow_params = None
    if args.flow_model is not None:
        print(f"Loading flow model from {args.flow_model}...")
        flow_params, flow_step, flow_mdl = load_params(args.flow_model, device="cpu")
        print(f"Loaded flow model step {flow_step}")

        # Validate flow model compatibility
        if not isinstance(flow_mdl, flow_matching.VectorField):
            raise ValueError(f"Flow model must be a VectorField, got {type(flow_mdl)}")
        if flow_mdl.domain_dim != 768:
            raise ValueError(
                f"Flow model domain_dim must be 768 (CLIP embedding size), got {flow_mdl.domain_dim}"
            )

    # Choose sampling method based on whether flow model is provided
    if args.flow_model is not None:
        # Map CLI choice to SamplingAlgorithm via Enum value
        flow_algorithm = flow_matching.SamplingAlgorithm(args.flow_algorithm)
        # Base algorithm params from CLI
        if flow_algorithm == flow_matching.SamplingAlgorithm.SIR:
            algorithm_params = flow_matching.SIRParams(
                n_proposal_samples=args.n_flow_proposals,
                n_projections=args.sir_n_projections,
                batch_size=args.sir_batch_size,
            )
        elif flow_algorithm == flow_matching.SamplingAlgorithm.REJECTION:
            algorithm_params = flow_matching.RejectionParams(
                proposal_batch_size=args.rejection_proposal_batch_size
            )
        else:
            algorithm_params = flow_matching.MCMCParams(
                n_chains=args.mcmc_n_chains,
                n_steps_per_chain=args.mcmc_steps_per_chain,
                step_scale=args.mcmc_step_scale,
                burnin_steps=args.mcmc_burnin_steps,
            )
        imgs = two_stage_sample_loop(
            im_mdl=im_mdl,
            im_params=im_params,
            flow_mdl=flow_mdl,
            flow_params=flow_params,
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
            n_flow_steps=args.n_flow_steps,
            n_flow_proposals=args.n_flow_proposals,
            flow_algorithm=flow_algorithm,
            algorithm_params=algorithm_params,
        )
    else:
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

    fmt_cond = lambda d: (
        d["cond"],
        float(d["max_cos_distance"]) if d["max_cos_distance"] is not None else None,
    )
    metadata = mk_png_metadata(
        seed,
        logit_filter_method,
        args.logit_filter_threshold,
        args.temperature,
        args.force_fp32,
        [fmt_cond(d) for d in cond_img_inputs],
        [fmt_cond(d) for d in cond_txt_inputs],
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
