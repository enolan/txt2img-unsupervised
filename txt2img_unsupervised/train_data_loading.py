"""Functions for loading training data."""

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datasets import Dataset
from einops import rearrange
from functools import lru_cache
from sortedcontainers import SortedList
from time import monotonic
from tqdm import tqdm
from typing import Optional, Tuple

_MAX_SHUFFLED_DATASET_CACHE_SIZE = 4


@lru_cache(maxsize=_MAX_SHUFFLED_DATASET_CACHE_SIZE)
def _prepare_shuffled_dataset(
    dataset: Dataset, epoch: int, clip_conditioning: bool, clip_caps: bool
) -> Dataset:
    """
    Reproducibly shuffle a dataset and select the appropriate columns.

    Args:
        dataset: The dataset to shuffle.
        epoch: The epoch number.
        clip_conditioning: Whether to include clip conditioning data.
        clip_caps: Whether to include clip caps data.

    Returns:
        A shuffled dataset with selected columns.
    """
    tqdm.write(f"Cache miss: shuffling dataset {id(dataset)} for epoch {epoch}")

    cols = ["encoded_img"]
    if clip_conditioning and not clip_caps:
        cols.append("clip_embedding")
    elif clip_conditioning and clip_caps:
        cols.extend(["cap_center", "cap_max_cos_distance"])

    return dataset.shuffle(seed=epoch).select_columns(cols).with_format("numpy")


def get_batch(
    dataset: Dataset,
    batch_size: int,
    global_step: int,
    clip_conditioning: bool,
    clip_caps: bool,
    clip_cap_count: Optional[int] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Get a batch of examples from a Dataset. Handles shuffling internally. If the length of the
    dataset is not a multiple of the batch size, the extra examples are discarded.

    Args:
        dataset: The dataset to get the batch from.
        batch_size: The number of examples to get.
        global_step: The global step number.
        clip_conditioning: Whether to get clip conditioning data.
        clip_caps: Whether to get clip caps data.
        clip_cap_count: The number of clip caps to get, if any.
        sharding: Optional sharding for the returned arrays.

    Returns:
        A tuple of encoded images, clip embeddings, and max cosine distances. (Latter two may be
        empty arrays if clip conditioning is not used.)
    """
    extra_examples = len(dataset) % batch_size
    effective_dataset_size = len(dataset) - extra_examples
    idx_start = global_step * batch_size

    epoch = idx_start // effective_dataset_size

    dataset = _prepare_shuffled_dataset(dataset, epoch, clip_conditioning, clip_caps)

    idx_start_in_epoch = idx_start % effective_dataset_size
    batch_dict = dataset[idx_start_in_epoch : idx_start_in_epoch + batch_size]

    imgs = batch_dict["encoded_img"]
    assert len(imgs.shape) == 2
    assert imgs.shape[0] == batch_size
    if clip_conditioning and clip_caps:
        if clip_cap_count is None:
            raise ValueError(
                "clip_cap_count must be provided if clip_conditioning and clip_caps are True"
            )
        clip_embeddings = batch_dict["cap_center"]
        assert (
            clip_embeddings.shape[0] == batch_size
        ), f"cap_center leading dimension should be {batch_size}, got {clip_embeddings.shape[0]}"
        if len(clip_embeddings.shape) != 2:
            raise ValueError(
                f"cap_center should be 2D, got shape {clip_embeddings.shape}"
            )
        max_cos_distances = batch_dict["cap_max_cos_distance"]
        assert (
            max_cos_distances.shape[0] == batch_size
        ), f"max_cos_distances leading dimension should be {batch_size}, got {max_cos_distances.shape[0]}"

        # Rearrange the cap data from parquet into the shape expected by the model and select the
        # appropriate subset of the caps for the current epoch. For reasons, parquet does not
        # support multidimensional arrays.
        if clip_embeddings.shape[1] % 768 != 0:
            raise ValueError(
                f"Cap centers shape {clip_embeddings.shape} not divisible by 768"
            )

        dset_cap_count = clip_embeddings.shape[1] // 768
        if dset_cap_count % clip_cap_count != 0:
            raise ValueError(
                f"Dataset cap count {dset_cap_count} not divisible by clip cap count {clip_cap_count}"
            )

        if dset_cap_count == 1:
            if len(max_cos_distances.shape) != 1:
                raise ValueError(
                    f"Max cosine distances should be a scalar when dset cap count is 1, got {max_cos_distances.shape[1:]}"
                )
            max_cos_distances = rearrange(max_cos_distances, "b -> b 1")
        else:
            if max_cos_distances.shape != (batch_size, dset_cap_count):
                raise ValueError(
                    f"Max cosine distances should be {dset_cap_count}D when dset cap count is {dset_cap_count}, got {max_cos_distances.shape[1:]}"
                )

        clip_embeddings = rearrange(
            clip_embeddings, "b (n c) -> b n c", n=dset_cap_count, c=768
        )

        # We feed all the non-overlapping contiguous subsequences of the dataset caps to the
        # model sequentially - one subsequence per epoch. Ideally we'd go through all the
        # permutations in a smart order - non-overlapping sets first, but that's more complicated.
        distinct_cap_sets = dset_cap_count // clip_cap_count
        this_cap_set = epoch % distinct_cap_sets
        this_cap_set_start = this_cap_set * clip_cap_count
        this_cap_set_end = this_cap_set_start + clip_cap_count

        clip_embeddings = clip_embeddings[:, this_cap_set_start:this_cap_set_end, :]
        max_cos_distances = max_cos_distances[:, this_cap_set_start:this_cap_set_end]
        assert clip_embeddings.shape == (batch_size, clip_cap_count, 768)
        assert max_cos_distances.shape == (batch_size, clip_cap_count)

    elif clip_conditioning and not clip_caps:
        clip_embeddings = batch_dict["clip_embedding"]
        assert clip_embeddings.shape == (batch_size, 768)
        max_cos_distances = jnp.zeros((batch_size, 0))
    else:
        clip_embeddings = jnp.zeros((batch_size, 0))
        max_cos_distances = jnp.zeros((batch_size, 0))

    return (
        jax.device_put(imgs, sharding),
        jax.device_put(clip_embeddings, sharding),
        jax.device_put(max_cos_distances, sharding),
    )


def _mk_mock_batch_dataset(dset_caps: int):
    gen = np.random.default_rng(5)
    dset_len = 10
    assert dset_caps > 0

    cap_centers = gen.random((dset_len, 768 * dset_caps))
    if dset_caps == 1:
        max_cos_distances = gen.random((dset_len,))
    else:
        max_cos_distances = gen.random((dset_len, dset_caps))

    return Dataset.from_dict(
        {
            "encoded_img": gen.integers(low=0, high=256, size=(dset_len, 256)),
            "clip_embedding": gen.random((dset_len, 768)),
            "cap_center": cap_centers,
            "cap_max_cos_distance": max_cos_distances,
        }
    ).with_format("numpy")


@pytest.mark.parametrize(
    "dset_caps,clip_cap_count,global_step,clip_conditioning,clip_caps",
    [
        (4, 2, 0, True, True),
        (4, 2, 10, True, True),
        (4, 4, 10, True, True),
        (1, 1, 0, True, False),
        (1, 1, 10, False, False),
    ],
)
def test_get_batch_shapes(
    dset_caps: int,
    clip_cap_count: int,
    global_step: int,
    clip_conditioning: bool,
    clip_caps: bool,
):
    dset = _mk_mock_batch_dataset(dset_caps)
    imgs, clip_embs, cap_max_cos_dists = get_batch(
        dset, 3, global_step, clip_conditioning, clip_caps, clip_cap_count
    )
    assert imgs.shape == (3, 256)

    if clip_conditioning and clip_caps:
        assert clip_embs.shape == (3, clip_cap_count, 768)
        assert cap_max_cos_dists.shape == (3, clip_cap_count)
    elif clip_conditioning and not clip_caps:
        assert clip_embs.shape == (3, 768)
        assert cap_max_cos_dists.shape == (3, 0)
    else:
        assert clip_embs.shape == (3, 0)
        assert cap_max_cos_dists.shape == (3, 0)


@pytest.mark.parametrize("global_step", [0, 4])
def test_consecutive_batches_consistency(global_step):
    """Test that consecutive batches are consistent with a double-sized batch. Note this is only
    true in general when the larger batch size is a factor of the dataset size."""
    dset = _mk_mock_batch_dataset(dset_caps=4)
    batch_size = 5
    clip_conditioning = True
    clip_caps = True
    clip_cap_count = 2

    imgs1, clip_embs1, cap_max_cos_dists1 = get_batch(
        dset, batch_size, global_step, clip_conditioning, clip_caps, clip_cap_count
    )
    imgs2, clip_embs2, cap_max_cos_dists2 = get_batch(
        dset, batch_size, global_step + 1, clip_conditioning, clip_caps, clip_cap_count
    )

    imgs_combined = np.concatenate([imgs1, imgs2])
    clip_embs_combined = np.concatenate([clip_embs1, clip_embs2])
    cap_max_cos_dists_combined = np.concatenate(
        [cap_max_cos_dists1, cap_max_cos_dists2]
    )

    imgs_double, clip_embs_double, cap_max_cos_dists_double = get_batch(
        dset,
        batch_size * 2,
        global_step // 2,
        clip_conditioning,
        clip_caps,
        clip_cap_count,
    )

    # Assert that the combined batches are equal to the double-sized batch
    np.testing.assert_array_equal(imgs_combined, imgs_double)
    np.testing.assert_array_equal(clip_embs_combined, clip_embs_double)
    np.testing.assert_array_equal(cap_max_cos_dists_combined, cap_max_cos_dists_double)


def test_different_epochs_get_different_shuffles():
    dset = _mk_mock_batch_dataset(dset_caps=4)
    batch_size = 2
    clip_conditioning = True
    clip_caps = True
    clip_cap_count = 2

    # Get all batches for each epoch
    all_imgs_epoch0 = []
    all_imgs_epoch1 = []
    for step in range(len(dset) // batch_size):
        imgs1, _, _ = get_batch(
            dset,
            batch_size,
            global_step=step,
            clip_conditioning=clip_conditioning,
            clip_caps=clip_caps,
            clip_cap_count=clip_cap_count,
        )
        imgs2, _, _ = get_batch(
            dset,
            batch_size,
            global_step=step + len(dset) // batch_size,
            clip_conditioning=clip_conditioning,
            clip_caps=clip_caps,
            clip_cap_count=clip_cap_count,
        )
        imgs1, imgs2 = jax.device_get((imgs1, imgs2))
        all_imgs_epoch0.extend(imgs1)
        all_imgs_epoch1.extend(imgs2)

    all_imgs_epoch0 = np.array(all_imgs_epoch0)
    all_imgs_epoch1 = np.array(all_imgs_epoch1)
    # Ensure all images are present in each epoch, but in a different order
    assert set(map(tuple, all_imgs_epoch0)) == set(map(tuple, all_imgs_epoch1))

    assert not np.all(all_imgs_epoch0 == all_imgs_epoch1)


@pytest.mark.parametrize("model_caps", [1, 2])
def test_clip_cap_selection_across_epochs(model_caps):
    dset = _mk_mock_batch_dataset(dset_caps=4)
    batch_size = 2

    imgs_1, cap_centers_1, cap_max_cos_dists_1 = get_batch(
        dset,
        batch_size,
        global_step=0,
        clip_conditioning=True,
        clip_caps=True,
        clip_cap_count=model_caps,
    )
    imgs_2, cap_centers_2, cap_max_cos_dists_2 = get_batch(
        dset,
        batch_size,
        global_step=len(dset) // batch_size,
        clip_conditioning=True,
        clip_caps=True,
        clip_cap_count=model_caps,
    )

    # Ensure different subsets of caps are selected
    assert not np.array_equal(cap_centers_1, cap_centers_2)
    assert not np.array_equal(cap_max_cos_dists_1, cap_max_cos_dists_2)
