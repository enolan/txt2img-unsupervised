"""
Given a source of images, scale them, crop them, and encode them with LDM encoder.
"""
import argparse
import CloseableQueue
import gc
import itertools
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import PIL.Image
import pyarrow as pa
import pyarrow.parquet as pq
import random
import torch
from CloseableQueue import CloseableQueue as CQueue
from concurrent import futures
from copy import copy
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from threading import Lock, Thread
from tqdm import tqdm
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--autoencoder_cfg", type=str)
parser.add_argument("in_dirs", type=Path, nargs="+")

args = parser.parse_args()


# Load the autoencoder model
autoencoder_cfg = OmegaConf.load(args.autoencoder_cfg)["model"]["params"]
autoencoder_mdl = LDMAutoencoder(cfg=autoencoder_cfg)
autoencoder_params = autoencoder_mdl.params_from_torch(
    torch.load(args.ckpt, map_location="cpu"), autoencoder_cfg
)

# Pipelined data processing. We want as much as possible to keep data flowing at all times, and to
# ensure the GPU is doing even batch sizes. We have several queues.
# - one queue for each directory, containing image paths
# - one queue for each directory, containing PIL images
# A single thread writes paths to the directory path queues, filling each in (random) order. A pool
# of threads reads from those queues, also in order, and loads the images. Those threads pass
# scaled/cropped PIL images to the image queue corresponding with the right directory. Then the
# main thread collects images from these queues (in order) and feeds them to the GPU for encoding
# It may collect images from multiple queues and put them in the same batch, but all images from a
# given directory go into one parquet file.

encode_vec = jax.jit(
    jax.vmap(
        lambda img: autoencoder_mdl.apply(
            autoencoder_params,
            method=autoencoder_mdl.encode,
            x=(img.astype(jnp.float32) / 127.5 - 1.0),
        )
    )
)

in_dirs = copy(args.in_dirs)
random.shuffle(in_dirs)  # Makes progress bar ETA more accurate

out_paths = []
for in_dir in in_dirs:
    assert in_dir.is_dir(), f"{in_dir} is not a directory"
    out_path = in_dir.parent.parent / f"{in_dir.parent}-{in_dir.name}.parquet"
    assert not out_path.exists(), f"{out_path} already exists"
    out_paths.append(out_path)

paths_queues = [CQueue(maxsize=args.batch_size * 4) for _ in in_dirs]


def paths_queuer_main(dirs: list[Path]):
    # Queue all the paths for each directory in turn, closing the queues as they're finished.
    for i, dir in enumerate(dirs):
        tqdm.write(f"Queueing paths for {dir} #{i}...")
        CloseableQueue.enqueue(list(dir.iterdir()), paths_queues[i])
        tqdm.write(f"Done queueing paths for {dir} #{i}")


paths_queuer = Thread(
    target=paths_queuer_main, args=(in_dirs,), name="Paths queuer", daemon=True
)
paths_queuer.start()


def load_img(img_path: Path) -> Optional[PIL.Image.Image]:
    """Load/crop/scale a single image."""
    try:
        img = PIL.Image.open(img_path)
        img.load()
    except Exception as e:
        # PIL can throw like eight million different exceptions
        tqdm.write(f"Skipping {img_path}, PIL can't open it due to {e}")
        return None
    w, h = img.size
    if w < 64 or h < 64 or img.mode != "RGB":
        tqdm.write(f"Skipping {img_path}, size {w}x{h}, mode {img.mode}")
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
    return img


imgs_queues = [CQueue(maxsize=args.batch_size * 4) for _ in in_dirs]

# Thread pool for loading/scaling/cropping images

# How many threads are working on each directory
pil_thread_counts = [0 for _ in in_dirs]
pil_thread_counts_locks = [Lock() for _ in in_dirs]


def pil_thread_fn(thread_num: int):
    try:
        for i, q in enumerate(paths_queues):
            with pil_thread_counts_locks[i]:
                pil_thread_counts[i] += 1
                tqdm.write(
                    f"PIL thread {thread_num} starting on {in_dirs[i]} (#{i}) ({pil_thread_counts[i]} total)"
                )
            while True:
                try:
                    img_path = q.get()
                except CloseableQueue.Closed:
                    with pil_thread_counts_locks[i]:
                        pil_thread_counts[i] -= 1
                        tqdm.write(
                            f"PIL thread {thread_num} done with dir {i}, paths queue closed ({pil_thread_counts[i]} left)"
                        )
                        if pil_thread_counts[i] == 0:
                            # There are no more paths to process
                            tqdm.write(
                                f"Last PIL thread ({thread_num}) working on {in_dirs[i]} (#{i}) done, closing corresponding images queue"
                            )
                            imgs_queues[i].close()
                    break
                img = load_img(img_path)
                imgs_queues[i].put((img, img_path))
    except Exception as e:
        # This hangs the process, since the PIL thread won't get closed.
        tqdm.write(
            f"XXX\nXXX\nXXX\nPIL thread {thread_num} crashed due to {e}\nXXX\nXXX\nXXX"
        )


pil_threads = {}

pil_threads_num = 32

for i in range(pil_threads_num):
    t = Thread(target=pil_thread_fn, args=(i,), name=f"PIL thread {i}")
    pil_threads[i] = t
    t.start()


def flush_batches(jax_list, numpy_list, force=False):
    # Flush from jax to numpy when there is more than 256MiB of data on GPU. Movement from
    # GPU->CPU is expensive, so we want to do it infrequently, but often enogh that we don't
    # exhaust GPU memory.
    # TODO this will change with higher res
    bytes_per_enc_img = 256 * 4
    batch_bytes = [len(j) * bytes_per_enc_img for j in jax_list]
    if force or sum(batch_bytes) > 256 * 1024 * 1204:
        tqdm.write(f"Flushing {len(jax_list)} encoded batches to CPU memory")
        numpy_list.extend([np.array(batch_j) for batch_j in jax_list])
        jax_list.clear()


# What image queue we're currently pulling from
img_queue_idx = 0
# Directories that need to be written out to parquet files
dirs_to_write = []
# Encoded images ready to be written out, indexed by dir (JAX and NumPy)
encoded_batches_j = {0: []}
encoded_batches_np = {0: []}
# The paths of the images that have been encoded
encoded_imgs_paths = {0: []}

with tqdm(total=len(in_dirs), desc="directories") as dirs_pbar:
    with tqdm(desc="files") as files_pbar:
        while True:
            # Collect a batch of PIL images
            batch = []
            # Each element of this is a pair of a dir index and the index of the first image in the
            # batch from that dir
            batch_dir_indices = []
            while len(batch) < args.batch_size:
                try:
                    img, img_path = imgs_queues[img_queue_idx].get()
                    if img is not None:
                        batch.append(img)
                        encoded_imgs_paths[img_queue_idx].append(img_path)
                        if (
                            batch_dir_indices == []
                            or batch_dir_indices[-1][0] != img_queue_idx
                        ):
                            batch_dir_indices.append((img_queue_idx, len(batch) - 1))
                    else:
                        # This image was skipped
                        files_pbar.update(1)
                except CloseableQueue.Closed:
                    # This queue is closed, move on to the next one
                    tqdm.write(f"Image queue {img_queue_idx} closed, moving on")
                    # We're ready to write the parquet file for this directory as soon as any
                    # remaining data in batches is processed.
                    dirs_to_write.append(img_queue_idx)
                    if img_queue_idx == len(imgs_queues) - 1:
                        # We've reached the end of the queues
                        tqdm.write("All image queues closed, we're almost done")
                        break
                    else:
                        img_queue_idx += 1
                        encoded_batches_j[img_queue_idx] = []
                        encoded_batches_np[img_queue_idx] = []
                        encoded_imgs_paths[img_queue_idx] = []
                        tqdm.write(
                            f"Moving to image queue {img_queue_idx} ({in_dirs[img_queue_idx]})"
                        )
            tqdm.write(
                f"Got batch of {len(batch)} images, queue sizes {[q.qsize() for q in imgs_queues]}"
            )

            if len(batch) > 0:
                # Encode the batch
                batch_j = jnp.stack([jnp.array(img) for img in batch])
                encoded = encode_vec(batch_j)
                # Add the parts of the batch to the appropriate lists
                for i in range(len(batch_dir_indices)):
                    dir_idx, batch_start = batch_dir_indices[i]
                    if i == len(batch_dir_indices) - 1:
                        batch_end = len(batch)
                    else:
                        batch_end = batch_dir_indices[i + 1][1]
                    encoded_batches_j[dir_idx].append(encoded[batch_start:batch_end])
                    tqdm.write(
                        f"Assigning images {batch_start}:{batch_end} to dir {dir_idx}"
                    )
                files_pbar.update(len(batch))

                # Flush batches to CPU memory if necessary
                for i in encoded_batches_j.keys():
                    # Hmm this enforces a *per-directory* limit, not a global limit :/
                    flush_batches(encoded_batches_j[i], encoded_batches_np[i])

                # Finalize directories that are done, writing the encoded images to parquet files.
                for i in dirs_to_write:
                    flush_batches(
                        encoded_batches_j[i], encoded_batches_np[i], force=True
                    )
                    if len(encoded_batches_np[i]) == 0:
                        tqdm.write(
                            f"Skipping directory {i} because it has no encoded images"
                        )
                    else:
                        encoded_imgs = np.concatenate(encoded_batches_np[i])
                        assert len(encoded_imgs) == len(
                            encoded_imgs_paths[i]
                        ), f"{len(encoded_imgs)} encoded images but {len(encoded_imgs_paths[i])} paths"
                        tqdm.write(
                            f"Writing {len(encoded_imgs)} encoded images to {out_paths[i]}"
                        )
                        pd_rows = [
                            {"encoded_img": img, "name": path.name}
                            for img, path in zip(encoded_imgs, encoded_imgs_paths[i])
                        ]
                        df = pd.DataFrame(pd_rows)
                        tqdm.write(
                            f"Writing {len(df)} rows to {out_paths[i]}, cols {df.columns}"
                        )
                        pq.write_table(pa.Table.from_pandas(df), out_paths[i])
                        tqdm.write(f"Done writing {out_paths[i]}")
                    del encoded_batches_np[i]
                    del encoded_batches_j[i]
                    del encoded_imgs_paths[i]
                    dirs_pbar.update(1)
                dirs_to_write.clear()
            else:
                # We've reached the end of the queues
                tqdm.write("All done :) Joining PIL threads")
                for t in pil_threads.values():
                    t.join()
                break
