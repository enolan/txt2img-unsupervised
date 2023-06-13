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
    out_path = in_dir.parent / f"{in_dir.name}.parquet"
    assert not out_path.exists(), f"{out_path} already exists"
    out_paths.append(out_path)

for in_dir, out_path in tqdm(zip(in_dirs, out_paths), total=len(in_dirs)):
    # Pipelined data processing:
    # - enqueue image paths (single thread)
    # - dequeue image paths, load images, enqueue PIL images (many threads)
    # - dequeue PIL images, encode with autoencoder, write to parquet when done (main thread)

    # Queue for image paths
    paths_queue = CQueue(maxsize=args.batch_size * 4)
    # Queue for PIL images
    imgs_queue = CQueue(maxsize=args.batch_size * 4)

    tqdm.write("Collecting image paths...")
    img_paths = list(in_dir.iterdir())
    tqdm.write(f" {len(img_paths)} images found.")
    random.shuffle(img_paths)  # Makes progress bar ETA more accurate

    # Enqueue image paths asynchonously
    path_queuer = Thread(
        target=lambda: CloseableQueue.enqueue(img_paths, paths_queue),
        name="Path enqueuer",
    )
    path_queuer.start()

    # Thread pool for loading/scaling/cropping images
    pil_threads = {}
    pil_threads_lock = Lock()

    def pil_thread_fn(thread_num: int):
        try:
            while True:
                try:
                    img_path = paths_queue.get()
                except CloseableQueue.Closed:
                    with pil_threads_lock:
                        del pil_threads[thread_num]
                        tqdm.write(
                            f"PIL thread {thread_num} exiting, paths queue closed ({len(pil_threads)} left)"
                        )
                        if not pil_threads:
                            # There are no more paths to process
                            tqdm.write(
                                f"Last PIL thread ({thread_num}) exiting, closing imgs queue"
                            )
                            imgs_queue.close()
                    break
                img = load_img(img_path)
                imgs_queue.put((img_path, img))
        except Exception as e:
            tqdm.write(
                f"XXX\nXXX\nXXX\nPIL thread {thread_num} crashed due to {e}\nXXX\nXXX\nXXX"
            )

    for i in range(os.cpu_count() * 2):
        t = Thread(target=pil_thread_fn, args=(i,), name=f"PIL thread {i}")
        pil_threads[i] = t
        t.start()

    # Read PIL images from queue, encode, and write to parquet

    def fetch_batch(n: int, pbar) -> list[PIL.Image.Image]:
        """Grab a batch of n items from the queue, or less if the queue is closed."""
        batch = []
        while True:
            if len(batch) >= n:
                return batch
            else:
                try:
                    path, pil_img = imgs_queue.get()
                    if pil_img is not None:
                        batch.append((path, pil_img))
                    else:
                        # If the image was unreadable/unusable, it counts as done
                        pbar.update(1)
                except CloseableQueue.Closed:
                    return batch

    with tqdm(total=len(img_paths), desc=str(in_dir)) as pbar:
        encoded_batches_j = []
        encoded_batches_np = []
        encoded_paths = []

        def flush_batches(jax_list, numpy_list, force=False):
            # Flush from jax to numpy when there is more than 256MiB of data on GPU. Movement from
            # GPU->CPU is expensive, so we want to do it infrequently, but often enogh that we don't
            # exhaust GPU memory.
            batch_size_bytes = (
                args.batch_size * 256 * 4
            )  # TODO this will change with higher res
            if force or len(jax_list) * batch_size_bytes > 256 * 1024 * 1204:
                tqdm.write(f"Flushing {len(jax_list)} batches to CPU memory")
                numpy_list.extend([np.array(batch_j) for batch_j in jax_list])
                jax_list.clear()

        while True:
            batch = fetch_batch(args.batch_size, pbar)
            if not batch:
                tqdm.write("Images queue done!")
                for thread in pil_threads.values():
                    thread.join()
                break
            else:
                tqdm.write(
                    f"Got batch of {len(batch)} images, (qsize now {imgs_queue.qsize()}) encoding..."
                )
                batch_j = jnp.stack([jnp.array(img) for path, img in batch])
                tqdm.write(f"Batch shape: {batch_j.shape}")
                encoded = encode_vec(batch_j)
                encoded_batches_j.append(encoded)
                encoded_paths.extend([path for path, img in batch])
                flush_batches(encoded_batches_j, encoded_batches_np)
                pbar.update(len(batch))
        flush_batches(encoded_batches_j, encoded_batches_np, force=True)
        tqdm.write("Done encoding, joining path queuer...")
        path_queuer.join()
        tqdm.write("Writing parquet...")
        encoded_imgs = np.concatenate(encoded_batches_np)
        pd_rows = [
            {"encoded_img": encoded, "name": str(path)}
            for encoded, path in zip(encoded_imgs, encoded_paths)
        ]
        df = pd.DataFrame(pd_rows)
        tqdm.write(f"Writing {len(df)} rows to {out_path}, cols {df.columns}")
        pq.write_table(pa.Table.from_pandas(df), out_path)
        tqdm.write("Done writing parquet!")
