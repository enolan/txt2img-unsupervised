"""Search captrees by CLIP embedding"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import PIL.Image
import torch
import transformers

from einops import repeat
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib import tenumerate

import txt2img_unsupervised.ldm_autoencoder as ldm_autoencoder

from txt2img_unsupervised.ldm_autoencoder import LDMAutoencoder
from txt2img_unsupervised.sample import batches_split
from txt2img_unsupervised.spherical_space_partitioning import CapTree


def cosine_distance(value):
    """Argument parser for cosine distances."""
    fvalue = float(value)
    if fvalue < 0 or fvalue > 2:
        raise argparse.ArgumentTypeError("Cosine distance must be in [0, 2]")
    return fvalue


def positive_int(value):
    """Argument parser for positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Positive integer required")
    return ivalue


def main():
    parser = argparse.ArgumentParser(description="Search captrees by CLIP embedding")
    parser.add_argument("--captree", type=Path, required=True, help="Path to captree")
    parser.add_argument("-n", type=positive_int, default=10, help="Number of results")
    parser.add_argument(
        "--distance",
        type=cosine_distance,
        required=True,
        help="Maximum cosine distance",
    )
    parser.add_argument(
        "--ae-cfg", type=Path, required=True, help="Path to autoencoder config"
    )
    parser.add_argument(
        "--ae-ckpt", type=Path, required=True, help="Path to autoencoder checkpoint"
    )
    parser.add_argument(
        "--out-path", type=Path, required=True, help="Path to save images"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--caption", help="Caption to pass to CLIP")
    group.add_argument("--image", type=Path, help="Image to pass to CLIP")
    args = parser.parse_args()

    print("Loading CLIP...")
    clip_mdl_name = "openai/clip-vit-large-patch14"
    clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name)
    clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

    print("Computing CLIP embedding...")
    if args.caption:
        text = args.caption
        inputs = clip_processor(text, return_tensors="np", padding=True)
        features = clip_mdl.get_text_features(**inputs)
    else:
        image = PIL.Image.open(args.image)
        inputs = clip_processor(images=image, return_tensors="np", padding=True)
        features = clip_mdl.get_image_features(**inputs)
    features = jax.device_get(features)
    embedding = features / np.linalg.norm(features, axis=-1, keepdims=True)

    print("Loading captree...")
    tree = CapTree.load_from_disk(args.captree)

    print("Searching...")
    n_samples = 1024
    cap_centers = repeat(embedding, "1 clip -> samples clip", samples=n_samples)
    cap_max_dists = np.full((n_samples,), args.distance, dtype=np.float32)

    sampled_idxs = tree.sample_in_caps_approx(
        cap_centers, cap_max_dists, density_estimate_samples=n_samples * 4
    )
    sampled_idxs = sampled_idxs[sampled_idxs != -1]

    unique_sampled_idxs = np.unique(sampled_idxs)
    print(
        f"Found {len(unique_sampled_idxs)} unique results after sampling {n_samples} times"
    )
    if len(unique_sampled_idxs) > 0:
        matches = tree.dset[unique_sampled_idxs]
        num_to_decode = min(args.n, len(matches["name"]))

        print("Loading autoencoder")
        ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
        ae_mdl = LDMAutoencoder(ae_cfg)
        ae_params = LDMAutoencoder.params_from_torch(
            torch.load(args.ae_ckpt, map_location="cpu"), cfg=ae_cfg
        )

        ae_res = int(matches["encoded_img"].shape[1] ** 0.5)

        decoded_imgs = []
        cur = 0

        with tqdm(total=num_to_decode, desc="Decoding") as pbar:
            for batch in batches_split(batch_size=16, n=num_to_decode):
                codes = matches["encoded_img"][cur : cur + batch]
                cur += batch
                imgs = ldm_autoencoder.decode_jv(
                    ae_mdl, ae_params, (ae_res, ae_res), codes
                )
                decoded_imgs.append(jax.device_get(imgs))
                pbar.update(len(imgs))

        decoded_imgs = np.concatenate(decoded_imgs, axis=0)

        print("Saving images...")
        args.out_path.mkdir(exist_ok=True, parents=True)
        for i, img in tqdm(enumerate(decoded_imgs), desc="saving images"):
            pil_img = PIL.Image.fromarray(img)
            pil_img.save(args.out_path / f"{matches['name'][i]}.png")
    else:
        print("No matches")
        return


if __name__ == "__main__":
    main()
