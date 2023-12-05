"""Load a dataset from parquet files and decode it to PNGs."""
import argparse
import ldm_autoencoder
import numpy as np
import PIL.Image
import torch
import tqdm
from datasets import Dataset
from itertools import islice
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--pq-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--ae-cfg", type=Path, required=True)
parser.add_argument("--ae-ckpt", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--res", type=int, required=True)
parser.add_argument("-n", type=int, required=True)

args = parser.parse_args()

assert args.res % 4 == 0
res_tokens = args.res // 4

assert args.n > 0

print(f"Loading autoencoder from {args.ae_ckpt}")
ae_cfg = OmegaConf.load(args.ae_cfg)["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)
ae_params = ae_mdl.params_from_torch(
    torch.load(args.ae_ckpt, map_location="cpu"), ae_cfg
)

print(f"Loading dataset from {args.pq_dir}")
dset = Dataset.from_parquet([str(pq) for pq in args.pq_dir.glob("**/*.parquet")])
dset.set_format("numpy")
dset = dset.shuffle()
print(f"Found {len(dset)} images")

print(f"Decoding dataset to {args.output_dir}")
args.output_dir.mkdir(exist_ok=True, parents=True)

print(f"Batch size {args.batch_size}")
dset_iter = dset.iter(batch_size=args.batch_size, drop_last_batch=False)
batches_to_decode = args.n // args.batch_size + 1 if args.n % args.batch_size > 0 else 0
print(f"Decoding {batches_to_decode} batches")
dset_iter = islice(dset_iter, batches_to_decode)

with tqdm(total=args.n, unit="img") as pbar:
    for batch in dset_iter:
        imgs_j = ldm_autoencoder.decode_jv(
            ae_mdl, ae_params, (res_tokens, res_tokens), batch["encoded_img"]
        )
        for img, name in zip(imgs_j, batch["name"]):
            img = PIL.Image.fromarray(np.array(img))
            img.save(args.output_dir / f"{name}.png")
        pbar.update(len(imgs_j))
