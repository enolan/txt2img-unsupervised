"""Functions for loading training data."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from datasets import Dataset
from einops import rearrange
from functools import lru_cache
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

_MAX_SHUFFLED_DATASET_CACHE_SIZE = 4


@lru_cache(maxsize=_MAX_SHUFFLED_DATASET_CACHE_SIZE)
def _prepare_shuffled_dataset(
    dataset: Dataset, epoch: int, columns: Tuple[str, ...]
) -> Dataset:
    """
    Reproducibly shuffle a dataset and select the specified columns.

    Args:
        dataset: The dataset to shuffle.
        epoch: The epoch number.
        columns: The columns to select from the dataset.

    Returns:
        A shuffled dataset with selected columns.
    """
    tqdm.write(f"Cache miss: shuffling dataset {id(dataset)} for epoch {epoch}")
    return dataset.shuffle(seed=epoch).select_columns(columns).with_format("numpy")


def get_batch(
    dataset: Dataset,
    batch_size: int,
    global_step: int,
    fields: List[str],
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Dict[str, jax.Array]:
    """
    Get a batch of examples from a Dataset. Handles shuffling internally. If the length of the
    dataset is not a multiple of the batch size, the extra examples are discarded.

    Args:
        dataset: The dataset to get the batch from.
        batch_size: The number of examples to get.
        global_step: The global step number.
        fields: List of field names to select from the dataset.
        sharding: Optional sharding for the returned arrays.

    Returns:
        A dictionary mapping field names to arrays of batch data.
    """
    extra_examples = len(dataset) % batch_size
    effective_dataset_size = len(dataset) - extra_examples
    idx_start = global_step * batch_size

    epoch = idx_start // effective_dataset_size

    dataset = _prepare_shuffled_dataset(dataset, epoch, tuple(fields))

    idx_start_in_epoch = idx_start % effective_dataset_size
    batch_dict = dataset[idx_start_in_epoch : idx_start_in_epoch + batch_size]

    # Move all fields to the device with the specified sharding
    result = {}
    for field in fields:
        assert field in batch_dict, f"Field {field} not found in dataset"
        data = batch_dict[field]
        assert (
            data.shape[0] == batch_size
        ), f"Expected batch size {batch_size}, got {data.shape[0]}"
        result[field] = jax.device_put(data, sharding)

    return result


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
    "global_step,fields",
    [
        (0, ["encoded_img", "clip_embedding"]),
        (10, ["encoded_img", "clip_embedding"]),
        (10, ["encoded_img", "clip_embedding"]),
        (0, ["encoded_img", "clip_embedding"]),
        (10, ["encoded_img"]),
    ],
)
def test_get_batch_shapes(
    global_step: int,
    fields: List[str],
):
    dset = _mk_mock_batch_dataset()
    batch = get_batch(dset, 3, global_step, fields)

    assert "encoded_img" in batch
    assert batch["encoded_img"].shape == (3, 256)

    if "clip_embedding" in fields:
        assert "clip_embedding" in batch
        assert batch["clip_embedding"].shape == (3, 768)


@pytest.mark.parametrize("global_step", [0, 4])
def test_consecutive_batches_consistency(global_step):
    """Test that consecutive batches are consistent with a double-sized batch. Note this is only
    true in general when the larger batch size is a factor of the dataset size."""
    dset = _mk_mock_batch_dataset()
    batch_size = 5
    fields = ["encoded_img", "clip_embedding"]

    batch1 = get_batch(dset, batch_size, global_step, fields)
    batch2 = get_batch(dset, batch_size, global_step + 1, fields)

    # Combine the batches
    combined_batch = {}
    for field in fields:
        combined_batch[field] = np.concatenate([batch1[field], batch2[field]])

    # Get a double-sized batch
    double_batch = get_batch(dset, batch_size * 2, global_step // 2, fields)

    # Assert that the combined batches are equal to the double-sized batch
    for field in fields:
        np.testing.assert_array_equal(combined_batch[field], double_batch[field])


def test_different_epochs_get_different_shuffles():
    dset = _mk_mock_batch_dataset()
    batch_size = 2
    fields = ["encoded_img"]

    # Get all batches for each epoch
    all_imgs_epoch0 = []
    all_imgs_epoch1 = []
    for step in range(len(dset) // batch_size):
        batch1 = get_batch(
            dset,
            batch_size,
            global_step=step,
            fields=fields,
        )
        batch2 = get_batch(
            dset,
            batch_size,
            global_step=step + len(dset) // batch_size,
            fields=fields,
        )
        imgs1, imgs2 = jax.device_get((batch1["encoded_img"], batch2["encoded_img"]))
        all_imgs_epoch0.extend(imgs1)
        all_imgs_epoch1.extend(imgs2)

    all_imgs_epoch0 = np.array(all_imgs_epoch0)
    all_imgs_epoch1 = np.array(all_imgs_epoch1)
    # Ensure all images are present in each epoch, but in a different order
    assert set(map(tuple, all_imgs_epoch0)) == set(map(tuple, all_imgs_epoch1))

    assert not np.all(all_imgs_epoch0 == all_imgs_epoch1)
