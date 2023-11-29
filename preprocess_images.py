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
import queue
import random
import torch
import transformers
from CloseableQueue import CloseableQueue as CQueue
from concurrent import futures
from copy import copy
from einops import rearrange
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from pathlib import Path
from threading import Lock, Semaphore, Thread
from tqdm import tqdm
from typing import Optional

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int)
parser.add_argument("--pil-threads", type=int, default=os.cpu_count() // 2)
parser.add_argument("--ckpt", type=str)
parser.add_argument("--autoencoder-cfg", type=str)
parser.add_argument("--res", type=int, required=True)
parser.add_argument("in_dirs", type=Path, nargs="+")

args = parser.parse_args()

pxl_res = args.res
token_count = pxl_res // 4 * pxl_res // 4
assert pxl_res % 4 == 0

# Load the autoencoder model
autoencoder_cfg = OmegaConf.load(args.autoencoder_cfg)["model"]["params"]
autoencoder_mdl = LDMAutoencoder(cfg=autoencoder_cfg)
autoencoder_params = autoencoder_mdl.params_from_torch(
    torch.load(args.ckpt, map_location="cpu"), autoencoder_cfg
)

# Load CLIP
clip_mdl_name = "openai/clip-vit-large-patch14"

# By experiment, float16 activations are 2.47x faster and 99.99% of the outputs (on an imgur
# convenience sample of 41,377 images) have at least 0.99 cosine similarity with the float32
# activation outupts. So we use float16 activations. F16 doesn't exist on TPUs though, so we'll
# need to do new tests if we want to run the preprocessor in Google land.

# perf notes on my 2080 (weights-activations):
# f32-f32 batch size 128 48601it [15:55, 50.87it/s]
# f32-f32 batch size 256 OOM
# f32-f32 batch size 64 48601it [16:00, 50.59it/s]

# f32-f16 batch size 64 48601it [06:32, 123.86it/s] !!!
# f32-f16 batch size 128 48601it [06:26, 125.88it/s]
# f32-f16 batch size 256 48601it [06:50, 118.30it/s]

# f16-f16 batch size 128 48601it [06:24, 126.50it/s]
# f16-f16 batch size 256 48601it [06:48, 118.96it/s]
clip_mdl = transformers.FlaxCLIPModel.from_pretrained(clip_mdl_name, dtype=jnp.float16)
clip_res = 224
clip_processor = transformers.AutoProcessor.from_pretrained(clip_mdl_name)

# Pipelined data processing. We want as much as possible to keep data flowing at all times, and to
# ensure the GPU is doing even batch sizes. We have several queues.
# - one queue for each directory, containing image paths
# - one queue for each directory, containing PIL images
# A single thread writes paths to the directory path queues, filling each in (random) order. A pool
# of threads reads from those queues, also in order, and loads the images. Those threads pass
# scaled/cropped PIL images to the image queue corresponding with the right directory. Then the
# main thread collects images from these queues (in order) and feeds them to the GPU for encoding
# & CLIP embedding. It may collect images from multiple queues and put them in the same batch, but
# all images from a given directory go into one parquet file.

# NB we get slightly different CLIP embeddings from this pipeline (cos. sim. = ~0.97) than if we
# run the images throught transformers' CLIP preprocessor. I think this is just the different order
# we do preprocessing in, but idk. Hopefully not significant.


def encode(ae_params, clip_params, img_ae: jax.Array, img_clip: jax.Array) -> jax.Array:
    """Encode an image with the LDM encoder and compute its CLIP embedding. Takes the model
    parameters, an image of the right resolution for the LDM encoder and another one for CLIP.
    Returns a tuple of (LDM encoding, CLIP embedding)."""
    assert img_ae.shape == (pxl_res, pxl_res, 3)
    assert img_clip.shape == (
        clip_res,
        clip_res,
        3,
    ), f"Expected ({clip_res}, {clip_res}, 3), got {img_clip.shape}"
    assert img_ae.dtype == jnp.uint8
    assert img_clip.dtype == jnp.uint8

    img_ae = img_ae.astype(jnp.float32) / 127.5 - 1.0
    img_enc = rearrange(img_ae, "w h c -> h w c")
    img_enc = autoencoder_mdl.apply(ae_params, method=autoencoder_mdl.encode, x=img_ae)

    # They scale them to [0, 1], then subtract their computed mean and divide by sd
    clip_means = jnp.array(clip_processor.image_processor.image_mean)
    clip_stds = jnp.array(clip_processor.image_processor.image_std)
    img_clip = img_clip.astype(jnp.float32) / 255.0
    img_clip = (img_clip - clip_means) / clip_stds
    img_clip = rearrange(img_clip, "w h c -> 1 c h w")
    img_clip_emb = clip_mdl.get_image_features(
        params=clip_params, pixel_values=img_clip
    )[0].astype(jnp.float32)
    img_clip_emb = img_clip_emb / jnp.linalg.norm(img_clip_emb)

    assert img_enc.shape == (
        token_count,
    ), f"encoded image shape is {img_enc.shape}, should be ({token_count},)"
    assert img_clip_emb.shape == (
        768,
    ), f"CLIP embedding shape is {img_clip_emb.shape}, should be (768,)"
    return img_enc, img_clip_emb


encode_vec = jax.jit(jax.vmap(encode, in_axes=(None, None, 0, 0)))

in_dirs = copy(args.in_dirs)
random.shuffle(in_dirs)  # Makes progress bar ETA more accurate

out_paths = []
for in_dir in in_dirs:
    in_dir = in_dir.absolute()
    assert in_dir.is_dir(), f"{in_dir} is not a directory"
    out_path = in_dir.parent.parent / f"{in_dir.parent}-{in_dir.name}.parquet"
    assert not out_path.exists(), f"{out_path} already exists"
    out_paths.append(out_path)
print(f"in_dirs: {in_dirs}, out_paths: {out_paths}")

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
    if w < pxl_res or h < pxl_res or img.mode != "RGB":
        tqdm.write(f"Skipping {img_path}, size {w}x{h}, mode {img.mode}")
        return None
    else:
        if w > h:
            px_to_remove = w - h
            x1 = px_to_remove // 2
            x2 = w - px_to_remove // 2
            y1 = 0
            y2 = h
        elif h > w:
            px_to_remove = h - w
            x1 = 0
            x2 = w
            y1 = px_to_remove // 2
            y2 = h - px_to_remove // 2
        else:
            x1 = 0
            x2 = w
            y1 = 0
            y2 = h
        img_for_enc = img.resize(
            (pxl_res, pxl_res), PIL.Image.BICUBIC, (x1, y1, x2, y2)
        )
        img_for_clip = img.resize(
            (clip_res, clip_res), PIL.Image.BICUBIC, (x1, y1, x2, y2)
        )
    return img_for_enc, img_for_clip


imgs_queues = [CQueue() for _ in in_dirs]
# Global limit on the number of waiting PIL images in queues
queued_img_semaphore = Semaphore(args.batch_size * 4)

# There's a potential deadlock with a single global limit, since the main thread can be waiting on
# the head image queue but PIL threads can be waiting on the global limit, which isn't released
# until the main thread dequeues stuff. Hacky, but we can avoid this by having a "backup" semaphore
# for each directory that's only used when the global limit is reached. They start empty and are
# incremented to 1 if the main thread blocks long enough waiting for that dir.
per_dir_semaphores = [Semaphore(0) for _ in in_dirs]

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
                imgs = load_img(img_path)
                while True:
                    if queued_img_semaphore.acquire(timeout=0.1):
                        imgs_queues[i].put((imgs, img_path))
                        break
                    elif per_dir_semaphores[i].acquire(blocking=False):
                        imgs_queues[i].put((imgs, img_path))
                        break
    except Exception as e:
        # This hangs the process, since the PIL thread won't get closed.
        tqdm.write(
            f"XXX\nXXX\nXXX\nPIL thread {thread_num} crashed due to {e}\nXXX\nXXX\nXXX"
        )


pil_threads = {}

for i in range(args.pil_threads):
    t = Thread(target=pil_thread_fn, args=(i,), name=f"PIL thread {i}", daemon=True)
    pil_threads[i] = t
    t.start()


def flush_batches(jax_list, numpy_list, force=False):
    # Flush from jax to numpy when there is more than 256MiB of data on GPU. Movement from
    # GPU->CPU is expensive, so we want to do it infrequently, but often enogh that we don't
    # exhaust GPU memory.
    # TODO this will change with higher res
    bytes_per_enc_img = token_count * 4
    bytes_per_clip_embed = 768 * 4
    batch_bytes = [
        len(j) * (bytes_per_enc_img + bytes_per_clip_embed) for j in jax_list
    ]
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
embedded_batches_j = {0: []}
embedded_batches_np = {0: []}
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
                if img_queue_idx is None:
                    break
                try:
                    while True:
                        try:
                            imgs, img_path = imgs_queues[img_queue_idx].get(timeout=1)
                            break
                        except queue.Empty:
                            # See comment above re potential deadlock
                            tqdm.write(
                                f"Images for dir #{img_queue_idx} coming in slowly, poking relief semaphore"
                            )
                            per_dir_semaphores[img_queue_idx].release()
                    if imgs is not None:
                        batch.append(imgs)
                        encoded_imgs_paths[img_queue_idx].append(img_path)
                        if (
                            batch_dir_indices == []
                            or batch_dir_indices[-1][0] != img_queue_idx
                        ):
                            batch_dir_indices.append((img_queue_idx, len(batch) - 1))
                        queued_img_semaphore.release()
                    else:
                        # This image was skipped
                        queued_img_semaphore.release()
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
                        img_queue_idx = None
                    else:
                        img_queue_idx += 1
                        encoded_batches_j[img_queue_idx] = []
                        encoded_batches_np[img_queue_idx] = []
                        embedded_batches_j[img_queue_idx] = []
                        embedded_batches_np[img_queue_idx] = []
                        encoded_imgs_paths[img_queue_idx] = []
                        tqdm.write(
                            f"Moving to image queue {img_queue_idx} ({in_dirs[img_queue_idx]})"
                        )
            tqdm.write(
                f"Got batch of {len(batch)} images, queue sizes {[q.qsize() for q in imgs_queues]}"
            )

            if len(batch) > 0:
                # Encode the batch
                batch_for_encoder = jnp.stack([jnp.array(imgs[0]) for imgs in batch])
                batch_for_clip = jnp.stack([jnp.array(imgs[1]) for imgs in batch])
                encoded, embedded = encode_vec(
                    autoencoder_params,
                    clip_mdl.params,
                    batch_for_encoder,
                    batch_for_clip,
                )
                # Add the parts of the batch to the appropriate lists
                for i in range(len(batch_dir_indices)):
                    dir_idx, batch_start = batch_dir_indices[i]
                    if i == len(batch_dir_indices) - 1:
                        batch_end = len(batch)
                    else:
                        batch_end = batch_dir_indices[i + 1][1]
                    encoded_batches_j[dir_idx].append(encoded[batch_start:batch_end])
                    embedded_batches_j[dir_idx].append(embedded[batch_start:batch_end])
                    tqdm.write(
                        f"Assigning images {batch_start}:{batch_end} to dir {dir_idx}"
                    )
                files_pbar.update(len(batch))

                # Flush batches to CPU memory if necessary
                for i in encoded_batches_j.keys():
                    # Hmm this enforces a *per-directory* limit, not a global limit :/
                    flush_batches(encoded_batches_j[i], encoded_batches_np[i])
                    flush_batches(embedded_batches_j[i], embedded_batches_np[i])

            # Finalize directories that are done, writing the encoded images to parquet files.
            for i in dirs_to_write:
                tqdm.write(f"Finalizing directory #{i}...")
                flush_batches(encoded_batches_j[i], encoded_batches_np[i], force=True)
                flush_batches(embedded_batches_j[i], embedded_batches_np[i], force=True)
                assert len(encoded_batches_np[i]) == len(embedded_batches_np[i])
                if len(encoded_batches_np[i]) == 0:
                    tqdm.write(
                        f"Skipping directory {i} because it has no encoded images"
                    )
                else:
                    encoded_imgs = np.concatenate(encoded_batches_np[i])
                    print(
                        f"encoded_imgs dtype: {encoded_imgs.dtype}, shape: {encoded_imgs.shape}"
                    )
                    embedded_imgs = np.concatenate(embedded_batches_np[i])
                    print(
                        f"embedded_imgs dtype: {embedded_imgs.dtype}, shape: {embedded_imgs.shape}"
                    )
                    assert len(encoded_imgs) == len(
                        encoded_imgs_paths[i]
                    ), f"{len(encoded_imgs)} encoded images but {len(encoded_imgs_paths[i])} paths"
                    tqdm.write(
                        f"Writing {len(encoded_imgs)} encoded images to {out_paths[i]} for dir #{i}"
                    )
                    pd_rows = [
                        {
                            "encoded_img": img,
                            "clip_embedding": embedding,
                            "name": path.name,
                        }
                        for img, embedding, path in zip(
                            encoded_imgs, embedded_imgs, encoded_imgs_paths[i]
                        )
                    ]
                    df = pd.DataFrame(pd_rows)
                    tqdm.write(
                        f"Writing {len(df)} rows to {out_paths[i]}, cols {df.columns}"
                    )
                    pq.write_table(pa.Table.from_pandas(df), out_paths[i])
                    tqdm.write(f"Done writing {out_paths[i]}")
                del encoded_batches_np[i]
                del embedded_batches_np[i]
                del encoded_batches_j[i]
                del embedded_batches_j[i]
                del encoded_imgs_paths[i]
                dirs_pbar.update(1)
            dirs_to_write.clear()
            if len(batch) == 0:
                # We've reached the end of the queues
                tqdm.write("All done :) Joining PIL threads")
                for t in pil_threads.values():
                    t.join()
                break
