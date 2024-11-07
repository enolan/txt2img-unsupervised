"""Functions for loading training data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datasets import Dataset
from einops import rearrange
from functools import lru_cache
from tqdm import tqdm
from typing import Optional, Tuple

_MAX_SHUFFLED_DATASET_CACHE_SIZE = 4


@lru_cache(maxsize=_MAX_SHUFFLED_DATASET_CACHE_SIZE)
def _prepare_shuffled_dataset(
    dataset: Dataset, epoch: int, clip_conditioning: bool
) -> Dataset:
    """
    Reproducibly shuffle a dataset and select the appropriate columns.

    Args:
        dataset: The dataset to shuffle.
        epoch: The epoch number.
        clip_conditioning: Whether to include clip conditioning data.

    Returns:
        A shuffled dataset with selected columns.
    """
    tqdm.write(f"Cache miss: shuffling dataset {id(dataset)} for epoch {epoch}")

    cols = ["encoded_img"]
    if clip_conditioning:
        cols.append("clip_embedding")

    return dataset.shuffle(seed=epoch).select_columns(cols).with_format("numpy")


def get_batch(
    dataset: Dataset,
    batch_size: int,
    global_step: int,
    clip_conditioning: bool,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Tuple[jax.Array, jax.Array]:
    """
    Get a batch of examples from a Dataset. Handles shuffling internally. If the length of the
    dataset is not a multiple of the batch size, the extra examples are discarded.

    Args:
        dataset: The dataset to get the batch from.
        batch_size: The number of examples to get.
        global_step: The global step number.
        clip_conditioning: Whether to get clip conditioning data.
        sharding: Optional sharding for the returned arrays.

    Returns:
        A tuple of encoded images and clip embeddings. (Clip embeddings will be empty if clip
        conditioning is not used.)
    """
    extra_examples = len(dataset) % batch_size
    effective_dataset_size = len(dataset) - extra_examples
    idx_start = global_step * batch_size

    epoch = idx_start // effective_dataset_size

    dataset = _prepare_shuffled_dataset(dataset, epoch, clip_conditioning)

    idx_start_in_epoch = idx_start % effective_dataset_size
    batch_dict = dataset[idx_start_in_epoch : idx_start_in_epoch + batch_size]

    imgs = batch_dict["encoded_img"]
    assert len(imgs.shape) == 2
    assert imgs.shape[0] == batch_size
    if clip_conditioning:
        clip_embeddings = batch_dict["clip_embedding"]
        assert clip_embeddings.shape == (batch_size, 768)
    else:
        clip_embeddings = jnp.zeros((batch_size, 0))

    return (
        jax.device_put(imgs, sharding),
        jax.device_put(clip_embeddings, sharding),
    )


def _mk_mock_batch_dataset():
    gen = np.random.default_rng(5)
    dset_len = 10

    return Dataset.from_dict(
        {
            "encoded_img": gen.integers(low=0, high=256, size=(dset_len, 256)),
            "clip_embedding": gen.random((dset_len, 768)),
        }
    ).with_format("numpy")


@pytest.mark.parametrize(
    "global_step,clip_conditioning",
    [
        (0, True),
        (10, True),
        (10, True),
        (0, True),
        (10, False),
    ],
)
def test_get_batch_shapes(
    global_step: int,
    clip_conditioning: bool,
):
    dset = _mk_mock_batch_dataset()
    imgs, clip_embs = get_batch(dset, 3, global_step, clip_conditioning)
    assert imgs.shape == (3, 256)

    if clip_conditioning:
        assert clip_embs.shape == (3, 768)
    else:
        assert clip_embs.shape == (3, 0)


@pytest.mark.parametrize("global_step", [0, 4])
def test_consecutive_batches_consistency(global_step):
    """Test that consecutive batches are consistent with a double-sized batch. Note this is only
    true in general when the larger batch size is a factor of the dataset size."""
    dset = _mk_mock_batch_dataset()
    batch_size = 5
    clip_conditioning = True

    imgs1, clip_embs1 = get_batch(dset, batch_size, global_step, clip_conditioning)
    imgs2, clip_embs2 = get_batch(dset, batch_size, global_step + 1, clip_conditioning)

    imgs_combined = np.concatenate([imgs1, imgs2])
    clip_embs_combined = np.concatenate([clip_embs1, clip_embs2])

    imgs_double, clip_embs_double = get_batch(
        dset,
        batch_size * 2,
        global_step // 2,
        clip_conditioning,
    )

    # Assert that the combined batches are equal to the double-sized batch
    np.testing.assert_array_equal(imgs_combined, imgs_double)
    np.testing.assert_array_equal(clip_embs_combined, clip_embs_double)


def test_different_epochs_get_different_shuffles():
    dset = _mk_mock_batch_dataset()
    batch_size = 2
    clip_conditioning = True

    # Get all batches for each epoch
    all_imgs_epoch0 = []
    all_imgs_epoch1 = []
    for step in range(len(dset) // batch_size):
        imgs1, _ = get_batch(
            dset,
            batch_size,
            global_step=step,
            clip_conditioning=clip_conditioning,
        )
        imgs2, _ = get_batch(
            dset,
            batch_size,
            global_step=step + len(dset) // batch_size,
            clip_conditioning=clip_conditioning,
        )
        imgs1, imgs2 = jax.device_get((imgs1, imgs2))
        all_imgs_epoch0.extend(imgs1)
        all_imgs_epoch1.extend(imgs2)

    all_imgs_epoch0 = np.array(all_imgs_epoch0)
    all_imgs_epoch1 = np.array(all_imgs_epoch1)
    # Ensure all images are present in each epoch, but in a different order
    assert set(map(tuple, all_imgs_epoch0)) == set(map(tuple, all_imgs_epoch1))

    assert not np.all(all_imgs_epoch0 == all_imgs_epoch1)
