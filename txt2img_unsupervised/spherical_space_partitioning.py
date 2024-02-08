"""A data structure for organizing unit vectors into a tree of spherical caps. This is primarily
to enable efficient sampling from arbitrary caps."""

import base64
import concurrent.futures
import hypothesis as hyp
import hypothesis.extra.numpy as hyp_np
import infinidata
import jax
import jax.numpy as jnp
import json
import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

from datetime import timedelta
from einops import rearrange
from functools import partial
from hypothesis import given, strategies as st
from pathlib import Path
from sortedcontainers import SortedList
from tqdm import tqdm

from .load_pq_dir import load_pq_to_infinidata


def find_k_means(dset, batch_size, k, iters):
    # Mini-batch k-means, see "Web-Scale K-Means Clustering" by D. Sculley 2010
    # This is a spherical k-means, using cosine similarity inside of euclidean distance. We assume
    # all vectors have unit norm.

    assert len(dset) >= k

    # Initialize centroids
    tqdm.write(f"Initializing centroids with {k} random samples")
    centroids = dset.shuffle(seed=np.random.randint(0, 2**63 - 1))[:k][
        "clip_embedding"
    ]

    # To make batch sizes even, we drop the last batch if it's smaller than batch_size, since
    # shuffling means we can see every example either way.
    drop_last_batch = len(dset) > batch_size

    dset_iter = dset.shuffle(seed=np.random.randint(0, 2**63 - 1)).batch_iter(
        batch_size=batch_size, drop_last_batch=drop_last_batch, threads=8, readahead=8
    )

    per_center_counts = np.zeros(k, dtype=np.int32)

    with tqdm(total=iters, desc="k-means iterations", leave=None) as pbar:
        while pbar.n < iters:
            try:
                batch = next(dset_iter)
            except StopIteration:
                dset_iter = dset.shuffle(
                    seed=np.random.randint(0, 2**63 - 1)
                ).batch_iter(
                    batch_size=batch_size,
                    drop_last_batch=drop_last_batch,
                    threads=8,
                    readahead=8,
                )
                batch = next(dset_iter)

            this_batch_size = len(batch["clip_embedding"])

            # Find nearest centroids
            (
                nearest_centroids,
                avg_variance,
            ) = find_nearest_centroids_and_average_variance(
                batch["clip_embedding"], centroids
            )

            pbar.set_postfix({"avg_variance": avg_variance})

            # Ship back to CPU
            nearest_centroids = np.array(nearest_centroids)

            # Update centroids
            for i in range(this_batch_size):
                nearest_centroid = nearest_centroids[i]
                per_center_counts[nearest_centroid] += 1
                lr = 1 / per_center_counts[nearest_centroid]
                centroids[nearest_centroid] = (1 - lr) * centroids[
                    nearest_centroid
                ] + lr * batch["clip_embedding"][i]
                # normalize the centroid location to put it back on the unit sphere
                norm = np.linalg.norm(centroids[nearest_centroid])
                if norm > 0:
                    centroids[nearest_centroid] /= norm
                else:
                    # This should be very rare
                    tqdm.write("WARNING: centroid is the zero vector, reinitializing")
                    centroids[nearest_centroid] = batch["clip_embedding"][i]
                    per_center_counts[nearest_centroid] = 1

            centroids_set = set(tuple(centroid) for centroid in centroids)
            assert len(centroids_set) == k, "Centroids are not unique"
            assert np.all(~np.isnan(centroids)), "Centroids contain NaNs"

            pbar.update(1)

    tqdm.write(f"Done finding centroids, final batch avg variance: {avg_variance}")

    return centroids


@jax.jit
def find_nearest_centroids_and_average_variance(xs, nearest_centroids):
    """Compute the nearest centroid for each vector in xs, and the average variance of clusters."""
    tqdm.write("tracing find_nearest_centroids_and...")
    nearest_centroids, distances = jax.vmap(find_nearest_centroid, in_axes=(0, None))(
        xs, nearest_centroids
    )
    return nearest_centroids, jnp.mean(distances)


def assign_centroids(dset, centroids, batch_size, n_distances=0):
    """Assign each CLIP embedding to its nearest centroid and compute the n highest overall cosine
    distances along with the associated vectors and centroids."""
    assert centroids.shape[1] == dset[0]["clip_embedding"].shape[0]
    assert np.all(~np.isnan(centroids))

    tqdm.write(f"Assigning {len(dset)} examples to {len(centroids)} centroids.")

    @jax.jit
    def nearest_centroids_and_distances(xs, centroids):
        return jax.vmap(find_nearest_centroid, in_axes=(0, None))(xs, centroids)

    # each element of high_distances is a SortedList containing tuples of cosine distance and index
    # into dset
    high_distances = [SortedList() for _ in range(len(centroids))]
    # each element of centroid_assignments is a list of indices into dset
    centroid_assignments = [[] for _ in range(len(centroids))]

    with tqdm(
        total=len(dset), desc="Assigning centroids", unit="vecs", leave=None
    ) as pbar:
        for batch_idx, batch in enumerate(
            dset.batch_iter(
                batch_size=batch_size, drop_last_batch=False, threads=8, readahead=8
            )
        ):
            this_batch_size = len(batch["clip_embedding"])

            nearest_centroids, distances = nearest_centroids_and_distances(
                batch["clip_embedding"], centroids
            )

            assert nearest_centroids.shape == (this_batch_size,)
            assert distances.shape == (this_batch_size,)

            nearest_centroids = np.array(nearest_centroids)
            distances = np.array(distances)

            # Add the indices of the vectors in this batch to the appropriate sublists in
            # nearest_centroids
            for i in range(this_batch_size):
                centroid_assignments[nearest_centroids[i]].append(
                    batch_idx * batch_size + i
                )

            for i in range(len(centroids)):
                # For each centroid, update high_distances by adding any distances that are higher
                # than the current lowest value in high_distances[i], or, if there are fewer than
                # n_distances in the SortedList, all of the distances.
                threshold = (
                    high_distances[i][0][0]
                    if len(high_distances[i]) >= n_distances
                    else 0
                )

                this_indices = np.arange(this_batch_size)[nearest_centroids == i]

                this_distances = distances[this_indices]
                this_distances_idxs_above_threshold = this_distances >= threshold

                distances_to_add = this_distances[this_distances_idxs_above_threshold]
                indices_to_add = this_indices[this_distances_idxs_above_threshold]

                high_distances[i].update(
                    zip(distances_to_add, batch_idx * batch_size + indices_to_add)
                )

                # Delete all but the n_distances highest distances
                del high_distances[i][:-n_distances]

            pbar.update(this_batch_size)

    assert len(centroid_assignments) == len(high_distances) == len(centroids)
    assert sum(len(cluster) for cluster in centroid_assignments) == len(dset)
    for i in range(len(centroids)):
        assert len(high_distances[i]) == min(n_distances, len(centroid_assignments[i]))
    return centroid_assignments, high_distances


def test_assign_centroids():
    """Test that assign_centroids works."""
    centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
    vecs = np.array([[1, -0.2], [1, 0.1], [-0.1, 1], [0.1, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dset = infinidata.TableView({"clip_embedding": vecs})

    assignments, high_distances = assign_centroids(dset, centroids, 2, n_distances=2)

    assert assignments == [[0, 1], [2, 3]]

    assert np.isclose(high_distances[0][0][0], cosine_distance(centroids[0], vecs[1]))
    assert high_distances[0][0][1] == 1

    assert np.isclose(high_distances[0][1][0], cosine_distance(centroids[0], vecs[0]))
    assert high_distances[0][1][1] == 0

    assert np.isclose(high_distances[1][0][0], cosine_distance(centroids[1], vecs[2]))
    assert np.isclose(high_distances[1][1][0], cosine_distance(centroids[1], vecs[3]))
    # They have the same distance, so they can appear in either order
    assert high_distances[1][0][1] in [2, 3]
    assert high_distances[1][1][1] in [2, 3]


def test_assign_centroids_top_2():
    centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
    vecs = np.array(
        [[1, -0.2], [1, 0.1], [0.9, 0], [1, 0.3], [-0.3, 1], [0.1, 1]],
        dtype=np.float32,
    )
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dset = infinidata.TableView({"clip_embedding": vecs})

    assignments, high_distances = assign_centroids(dset, centroids, 2, n_distances=2)

    assert assignments == [[0, 1, 2, 3], [4, 5]]

    # The most distant vector for centroid 0 is vecs[3], followed by vecs[0]
    assert high_distances[0][1][0] == cosine_distance(centroids[0], vecs[3])
    assert high_distances[0][1][1] == 3

    assert high_distances[0][0][0] == cosine_distance(centroids[0], vecs[0])
    assert high_distances[0][0][1] == 0

    # The most distant vector for centroid 1 is vecs[4], followed by vecs[5]
    assert high_distances[1][1][0] == cosine_distance(centroids[1], vecs[4])
    assert high_distances[1][1][1] == 4

    assert high_distances[1][0][0] == cosine_distance(centroids[1], vecs[5])
    assert high_distances[1][0][1] == 5


@st.composite
def _unit_vecs(draw, shape):
    """Strategy for drawing unique unit vectors."""
    # Generating unique unit vectors using Hypothesis is, AFAICT, impossible to do well in a
    # non insane way. This is an insane way, and not a good one, but it's at least fast. The more
    # natural methods reject very high proportions of candidates and are consequently very slow.
    # They also generate very small sets of vectors. This version shrinks towards axis aligned
    # vectors, just to make it easier to read the output. Shrinking a set of unit vectors beyond
    # that doesn't make much sense. The natural methods provide no useful shrinking. It will spin
    # for five minutes failing to shrink the example at all if you have a test failure.
    shape = draw(shape)

    # (sometimes) prepend a series of axis aligned vectors. Shrinking goes toward doing it more.
    aa_prefix_len = shape[1] - draw(st.integers(0, shape[1]))
    aa_prefix = np.eye(aa_prefix_len, dtype=np.float32)
    aa_prefix = np.pad(
        aa_prefix, ((0, 0), (0, shape[1] - aa_prefix_len)), mode="constant"
    )
    aa_prefix = np.concatenate([aa_prefix, -aa_prefix], axis=0)
    aa_prefix = aa_prefix[: shape[0]]
    assert aa_prefix.shape[0] <= shape[0]
    assert aa_prefix.shape[1] == shape[1]
    assert np.all(np.isclose(np.linalg.norm(aa_prefix, axis=1), 1))

    shape_remaining = (shape[0] - aa_prefix.shape[0], shape[1])

    # Generate the remainder in the typical numpy way. this is effectively unshrinkable, and
    # hopefully this method stops Hypothesis from trying.
    bytes = draw(st.binary(min_size=8, max_size=8))
    seed = np.frombuffer(bytes, dtype=np.uint64)
    rng = np.random.default_rng(seed)

    vecs = rng.standard_normal(shape_remaining).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    hyp.assume(np.all(np.isfinite(vecs)))
    hyp.assume(np.all(np.isclose(np.linalg.norm(vecs, axis=1), 1)))

    vecs = np.concatenate([aa_prefix, vecs], axis=0)
    assert vecs.shape == shape

    # The vectors need to be unique, and very close vectors are close enough to being duplicates to
    # induce the same problems. So we quantize them to 16 bits and detect duplicates based on that.
    vecs_dedup_quantized = {
        np.clip(vec * 2**15, -1 * 2**15, 2**15 - 1).astype(np.int16).tobytes()
        for vec in vecs
    }

    hyp.assume(len(vecs_dedup_quantized) == len(vecs))

    return vecs


@hyp.settings(
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.too_slow,
        hyp.HealthCheck.filter_too_much,
    ],
    deadline=timedelta(seconds=30),
)
@given(
    _unit_vecs(
        st.tuples(st.integers(5, 256), st.integers(2, 4)),
    ),
    st.integers(1, 4),
    st.integers(1, 10),
    st.integers(1, 8),
    st.random_module(),
)
def test_assign_centroids_high_distances(
    vecs, n_centroids, n_distances, batch_size, _rand
):
    """The high distances returned by assign_centroids are correct."""
    centroids = vecs[:n_centroids]
    vecs = vecs[n_centroids:]
    dset = infinidata.TableView({"clip_embedding": vecs})

    assignments, high_distances = assign_centroids(
        dset, centroids, batch_size=batch_size, n_distances=n_distances
    )

    assert len(assignments) == len(high_distances) == len(centroids)
    for i in range(len(assignments)):
        this_cluster_vecs = vecs[assignments[i]]
        this_cluster_distances = cosine_distance_many_to_one(
            this_cluster_vecs, centroids[i]
        )
        sorted_this_cluster_indices = np.argsort(this_cluster_distances)
        sorted_dset_indices = np.array(assignments[i])[sorted_this_cluster_indices]

        assert np.array_equal(
            sorted_dset_indices[-n_distances:],
            np.array([idx for _, idx in high_distances[i]]),
        ) and np.allclose(
            this_cluster_distances[sorted_this_cluster_indices[-n_distances:]],
            np.array([distance for distance, _ in high_distances[i]]),
            rtol=0,
            atol=1e-5,
        ), f"cluster {i} failed, actual high distances idxs: {sorted_dset_indices[-n_distances:]}, values: {this_cluster_distances[sorted_this_cluster_indices[-n_distances:]]}, returned from assign_centroids: {high_distances[i]}"


def cosine_distance(x, y):
    """Cosine distance between x and y. Assumes x and y are unit vectors."""
    assert x.shape == y.shape
    return jnp.clip(1 - jnp.dot(x, y), 0, 2)


@partial(jax.jit, inline=True)
def cosine_distance_many_to_one(xs, y):
    """Cosine distance between each x in xs and y. Assumes xs and y are unit vectors."""
    assert len(xs.shape) == 2
    assert y.shape == xs.shape[1:]
    return jnp.clip(1 - jnp.dot(xs, y), 0, 2)


@partial(jax.jit, inline=True)
def vector_in_cap(v, cap_center, max_cos_distance):
    """Check if v is in the cap defined by cap_center and max_cos_distance."""
    assert len(v.shape) == 1
    assert v.shape == cap_center.shape
    assert max_cos_distance.shape == ()
    return cosine_distance(v, cap_center) <= max_cos_distance


@partial(jax.jit, inline=True)
def vectors_in_cap(vs, cap_center, max_cos_distance):
    """Check which vectors in vs are in the cap defined by cap_center and max_cos_distance."""
    assert len(vs.shape) == 2
    assert len(cap_center.shape) == 1
    assert vs.shape[1] == cap_center.shape[0]
    assert max_cos_distance.shape == ()
    print(f"Tracing vectors_in_cap for shape {vs.shape}")
    return jax.vmap(vector_in_cap, in_axes=(0, None, None))(
        vs, cap_center, max_cos_distance
    )


@partial(jax.jit, static_argnames=("need_cumsum", "need_bools"))
def vectors_in_caps(
    vs, cap_centers, max_cos_distances, need_cumsum=False, need_bools=True
):
    """Check which vectors in vs are in the caps defined by cap_centers and max_cos_distances.
    If need_cumsum is True returns a cumulative sum over the vectors, if need_bools is True returns
    a bool mask. If both are, returns a tuple.
    mask and cumsum both have shape (len(vs), len(cap_centers))
    """
    assert len(vs.shape) == 2
    assert len(cap_centers.shape) == 2
    assert vs.shape[1] == cap_centers.shape[1]
    assert len(max_cos_distances.shape) == 1
    assert cap_centers.shape[0] == max_cos_distances.shape[0]
    assert need_cumsum or need_bools
    print(f"Tracing vectors_in_caps for vs {vs.shape[0]}, caps {cap_centers.shape[0]}")

    mask = jax.vmap(vectors_in_cap, in_axes=(None, 0, 0))(
        vs, cap_centers, max_cos_distances
    )
    mask = rearrange(mask, "c v -> v c")
    assert mask.shape == (len(vs), len(cap_centers)), f"mask.shape: {mask.shape}"
    if need_cumsum:
        cumsum = jnp.cumsum(mask, axis=0)
    if need_cumsum and not need_bools:
        return cumsum
    elif need_bools and not need_cumsum:
        return mask
    else:
        return mask, cumsum


def pad_to_multiple(arr, x):
    """Pad arr to a multiple of x along its 0th dimension. Returns the padded array and the amount
    of padding added."""
    pad = x - (len(arr) % x) if len(arr) % x != 0 else 0
    if pad != 0:
        padded_arr = np.pad(
            arr, tuple([(0, pad)] + [(0, 0) for dim in arr.shape[1:]]), mode="constant"
        )
    else:
        padded_arr = arr
    assert padded_arr.shape[0] % x == 0
    assert padded_arr.shape[0] - pad == arr.shape[0]
    assert padded_arr.shape[1:] == arr.shape[1:]
    return padded_arr, pad


def vectors_in_caps_padded(
    vs, cap_centers, max_cos_distances, need_cumsum=False, need_bools=True
):
    """Compute vectors in caps, padding the dimension up to multiples of small powers of two."""
    assert len(vs.shape) == 2
    assert len(cap_centers.shape) == 2
    assert vs.shape[1] == cap_centers.shape[1]
    assert len(max_cos_distances.shape) == 1
    assert cap_centers.shape[0] == max_cos_distances.shape[0]
    assert need_cumsum or need_bools

    vs_padded, vs_pad = pad_to_multiple(vs, 32)
    cap_centers_padded, cap_centers_pad = pad_to_multiple(cap_centers, 64)
    max_cos_distances_padded, max_cos_distances_pad = pad_to_multiple(
        max_cos_distances, 64
    )
    assert cap_centers_pad == max_cos_distances_pad

    res = vectors_in_caps(
        vs_padded,
        cap_centers_padded,
        max_cos_distances_padded,
        need_cumsum=need_cumsum,
        need_bools=need_bools,
    )
    # This sort of slicing is really slow on GPU so we ship the result back to CPU.
    if need_cumsum != need_bools:
        return np.array(res)[: vs.shape[0], : cap_centers.shape[0]]
    elif need_cumsum and need_bools:
        mask, cumsum = np.array(res[0]), np.array(res[1])
        return (
            mask[: vs.shape[0], : cap_centers.shape[0]],
            cumsum[: vs.shape[0], : cap_centers.shape[0]],
        )
    else:
        assert False, "This should be unreachable"


def vectors_in_cap_even_batch(
    vs, cap_center, max_cos_distance, max_batch_size_log_sqrt2=31
):
    """A version of vectors in cap that pads vs up to a power of sqrt(2). Reduces the number of
    versions of the function that need to be compiled."""
    assert len(vs.shape) == 2
    assert len(cap_center.shape) == 1
    assert vs.shape[1] == cap_center.shape[0]
    assert (type(max_cos_distance) is float) or max_cos_distance.shape == ()
    batch_sizes = np.round(np.sqrt(2) ** np.arange(max_batch_size_log_sqrt2)).astype(
        np.int32
    )

    out = np.zeros(len(vs), dtype=np.bool_)
    cur = 0
    while cur < len(vs):
        # Most of the time this will complete in one GPU call, but if it doesn't, we loop until
        # we've done all the calculations.
        this_batch_size_idx = np.ceil(
            np.log(len(vs) - cur) / np.log(np.sqrt(2))
        ).astype(np.int32)
        this_batch_size_idx = min(this_batch_size_idx, max_batch_size_log_sqrt2 - 1)
        this_batch_size = batch_sizes[this_batch_size_idx]

        if len(vs) - cur < this_batch_size:
            # If the data remaining is less than the selected batch size, we pad with zeros.
            this_batch = np.empty((this_batch_size, vs.shape[1]), dtype=vs.dtype)
            this_batch[: len(vs) - cur] = vs[cur:]
            batch_len_unpadded = len(vs) - cur
        else:
            # Otherwise we can use the data as is, slicing to the batch size.
            this_batch = vs[cur : cur + this_batch_size]
            batch_len_unpadded = this_batch_size

        out[cur : cur + batch_len_unpadded] = np.array(
            vectors_in_cap(this_batch, cap_center, max_cos_distance)
        )[:batch_len_unpadded]
        cur += this_batch_size

    return out


@hyp.settings(
    deadline=timedelta(seconds=30),
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
    max_examples=500,
)
@given(_unit_vecs(st.tuples(st.integers(2, 256), st.integers(2, 4))))
def test_vectors_in_cap_even_batch(vecs):
    """Test that vectors_in_cap_even_batch is equivalent to vectors_in_cap."""
    cap_center = vecs[0]
    max_cos_distance = 0.5
    vecs = vecs[1:]

    out_even_batch = vectors_in_cap_even_batch(
        vecs, cap_center, max_cos_distance, max_batch_size_log_sqrt2=10
    )
    out_classic = vectors_in_cap(vecs, cap_center, max_cos_distance)

    np.testing.assert_allclose(out_even_batch, out_classic)


@partial(jax.jit, inline=True)
def cap_intersection_status(center_a, max_cos_distance_a, center_b, max_cos_distance_b):
    """Calculate whether cap a contains cap b, or they intersect, or neither."""
    assert center_a.shape == center_b.shape
    assert max_cos_distance_a.shape == max_cos_distance_b.shape == ()
    centers_angular_distance = jnp.arccos(jnp.clip(jnp.dot(center_a, center_b), -1, 1))
    angular_radius_a = jnp.arccos(1 - max_cos_distance_a)
    angular_radius_b = jnp.arccos(1 - max_cos_distance_b)
    contained = centers_angular_distance + angular_radius_b <= angular_radius_a
    intersect = centers_angular_distance <= angular_radius_a + angular_radius_b
    return (contained, intersect & ~contained)


@partial(jax.jit, inline=True)
def cap_intersection_status_one_to_many(
    center_a, max_cos_distance_a, centers_b, max_cos_distances_b
):
    """Calculate whether cap a contains any of the caps in b, or they intersect, or neither."""
    return jax.vmap(cap_intersection_status, in_axes=(None, None, 0, 0))(
        center_a, max_cos_distance_a, centers_b, max_cos_distances_b
    )


@jax.jit
def cap_intersection_status_many_to_many(
    centers_a, max_cos_distances_a, centers_b, max_cos_distances_b
):
    """Calculate whether each cap in a contains any of the caps in b, or they intersect, or neither."""
    print(
        f"Tracing cap_intersection_status_many_to_many for shapes {centers_a.shape}, {centers_b.shape}"
    )
    return jax.vmap(cap_intersection_status_one_to_many, in_axes=(0, 0, None, None))(
        centers_a, max_cos_distances_a, centers_b, max_cos_distances_b
    )


def cap_intersection_status_many_to_many_padded(
    centers_a, max_cos_distances_a, centers_b, max_cos_distances_b
):
    """Run cap_intersection_status_many_to_many, padding the dimensions up to multiples of 8."""
    assert centers_a.shape[0] == max_cos_distances_a.shape[0]
    assert centers_b.shape[0] == max_cos_distances_b.shape[0]
    centers_a_padded, pad_a = pad_to_multiple(centers_a, 8)
    centers_b_padded, pad_b = pad_to_multiple(centers_b, 8)
    max_cos_distances_a_padded, _ = pad_to_multiple(max_cos_distances_a, 8)
    max_cos_distances_b_padded, _ = pad_to_multiple(max_cos_distances_b, 8)

    contained, intersecting = cap_intersection_status_many_to_many(
        centers_a_padded,
        max_cos_distances_a_padded,
        centers_b_padded,
        max_cos_distances_b_padded,
    )
    contained, intersecting = np.array(contained), np.array(intersecting)

    contained = contained[: centers_a.shape[0], : centers_b.shape[0]]
    intersecting = intersecting[: centers_a.shape[0], : centers_b.shape[0]]
    return contained, intersecting


def find_nearest_centroid(x, centroids):
    """Find the nearest centroid to x."""
    distances = jax.vmap(cosine_distance, in_axes=(None, 0))(x, centroids)
    nearest = jnp.argmin(distances)
    return nearest, distances[nearest]


@jax.jit
def find_max_cosine_distance(x, xs, old_max):
    """Find the maximum cosine distance between x and any vector in xs or old_max if it's greater.
    Assumes x and xs are unit vectors."""
    distances = cosine_distance_many_to_one(xs, x)
    return jnp.maximum(old_max, jnp.max(distances))


class CapTree:
    """A tree of spherical caps containing unit vectors at the leaves. We split a cap into k
    children by running k-means on the unit vectors in the cap. Each centroid and the vectors
    assigned to it becomes a child cap. Then, we remove vectors from each cluster starting from the
    one with the highest cosine distance to its centroid until we've removed len(self)/k vectors.
    If the overall highest cosine distance has not decreased by at least outlier_removal_level -
    interpreted as a fraction - we put everything back and keep the original k-means clusters.
    Otherwise, we put those vectors into their own cluster. The idea here is to reduce the spatial
    size of the k-means clusters by removing outliers, which should speed up queries by reducing
    the number of clusters that search has to visit. This is at the expense, of course, of making
    a single really big cluster. We then repeat the split process recursively until each leaf has
    at most max_leaf_size vectors."""

    # I'd really love to live in a world where "what is the dot product of x and y?" is a question
    # with only one answer, but alas we do not live in that world.
    EPSILON = 0.005

    def __init__(
        self,
        dset,
        batch_size,
        k,
        iters,
        max_leaf_size=None,
        outlier_removal_level=0,
        dup_check=False,
        center=None,
        max_cos_distance=2.0,
        found_duplicates=[],
    ):
        assert len(dset) > 0, "CapTree must be initialized with a non-empty dataset"
        self.dset = dset
        self.dset_thin = dset.select_columns({"clip_embedding"})
        self.dsets_contiguous = True
        self.len = len(dset)
        self.batch_size = batch_size
        self.k = k
        if max_leaf_size is None:
            self.max_leaf_size = k**2
        else:
            self.max_leaf_size = max_leaf_size
            assert self.max_leaf_size >= k, "max_leaf_size must be at least k"
        self.outlier_removal_level = outlier_removal_level
        assert self.outlier_removal_level >= 0 and self.outlier_removal_level <= 1
        self.iters = iters
        self.dup_check = dup_check
        if center is None:
            center = np.zeros(dset[0]["clip_embedding"].shape[0], dtype=np.float32)
            center[0] = 1.0
        self.center = center
        self.max_cos_distance = max_cos_distance
        self.children = []
        self.child_cap_centers = None
        self.child_cap_max_cos_distances = None
        self.found_duplicates = found_duplicates
        self.ready_for_queries = False

    def __len__(self):
        return self.len

    def split_once(self):
        """Split this cap into children."""

        centroids = find_k_means(self.dset_thin, self.batch_size, self.k, self.iters)
        max_outliers_removed = int(self.len / self.k)
        assignments, high_distances = assign_centroids(
            self.dset_thin,
            centroids,
            self.batch_size,
            n_distances=max_outliers_removed + 1,
        )
        centroids, assignments, max_distances = self._remove_outliers(
            centroids,
            assignments,
            high_distances,
            max_outliers_removed,
            batch_size=4096,
        )
        assert len(assignments) == len(max_distances) == len(centroids)
        self.children = [
            CapTree(
                self.dset.new_view(np.array(assignments[i])),
                self.batch_size,
                self.k,
                self.iters,
                dup_check=self.dup_check,
                center=centroids[i],
                max_cos_distance=max_distances[i],
                found_duplicates=self.found_duplicates,
            )
            for i in range(len(centroids))
            if len(assignments[i]) > 0
        ]
        self.ready_for_queries = False
        self.dsets_contiguous = False
        self.child_cap_centers = np.array([child.center for child in self.children])
        self.child_cap_max_cos_distances = np.array(
            [child.max_cos_distance for child in self.children]
        )

        if len(self.children) == 1:
            # There may be very rare cases where this isn't caused by duplicate vectors, but in
            # general it is. It happens when there are more than max_leaf_size duplicates and
            # causes an infinite loop if not caught.
            tqdm.write(
                f"found node with only one child, probably duplicate vectors. node len {len(self)}"
            )
            self.children[0]._check_for_duplicates(force=True)

    def _remove_outliers(
        self, centroids, assignments, high_distances, max_outliers_removed, batch_size
    ):
        """Remove outliers from the clusters. assignments is a list of lists of indices, each inner
        list contains the indices assigned to a particular cluster. high_distances is a list of
        SortedLists of (cosine distance, index) pairs, one for each cluster. The new outlier
        cluster will have at most max_outliers_removed vectors in it. Return the new centroids,
        assignments, and max distances."""
        assert len(assignments) == len(high_distances)
        assert len(assignments) > 0
        assert max_outliers_removed >= 0 and max_outliers_removed < len(self)

        # We need to save the original max distances in case we end up not making an outlier
        # cluster.
        original_max_distances = []
        for i in range(len(assignments)):
            if len(assignments[i]) > 0:
                original_max_distances.append(high_distances[i][-1][0])
            else:
                tqdm.write(f"WARNING: empty cluster: {i}")
                original_max_distances.append(0)

        # aggregate the high distances from each cluster
        high_distances_for_agg = [
            [(distance, cluster_idx, idx) for distance, idx in cluster]
            for cluster_idx, cluster in enumerate(high_distances)
        ]
        assert len(high_distances_for_agg) == len(assignments)
        assert sum(len(cluster) for cluster in high_distances_for_agg) <= sum(
            len(cluster_assignments) for cluster_assignments in assignments
        )
        high_distances_all = SortedList(
            [item for cluster in high_distances_for_agg for item in cluster]
        )
        assert len(high_distances_all) == sum(
            len(cluster) for cluster in high_distances_for_agg
        )

        # remove outliers
        max_distance = high_distances_all[-1][0]
        target_max_distance = max_distance * (1 - self.outlier_removal_level)
        removed_vectors = []
        while len(removed_vectors) < max_outliers_removed:
            # Find the highest cosine distance vector
            highest_cos_distance, cluster_idx, idx = high_distances_all.pop(-1)
            high_distances[cluster_idx].remove((highest_cos_distance, idx))
            removed_vectors.append(idx)

        # return new clusters
        if (
            len(removed_vectors) == 0
            or high_distances_all[-1][0] >= target_max_distance
        ):
            # We didn't remove any vectors, or we didn't remove enough vectors to get the max
            # distance below the target. In either case, we don't make an outlier cluster and
            # return the original ones.

            # TODO if we only shrunk one cluster, do we abort and return the original clusters?
            return (
                centroids,
                assignments,
                [
                    original_max_distances[cluster_idx]
                    for cluster_idx in range(len(assignments))
                ],
            )
        else:
            # We removed enough vectors to get the max distance below the target. We make an
            # outlier cluster and return the new clusters.

            # We find the outlier centroid by averaging a sample of the removed vectors.
            outlier_dset = self.dset_thin.new_view(np.array(removed_vectors))
            outlier_centroid = np.mean(
                outlier_dset.shuffle(seed=np.random.randint(0, 2**63 - 1))[:16384][
                    "clip_embedding"
                ],
                axis=0,
            )
            outlier_centroid_norm = np.linalg.norm(outlier_centroid)
            if outlier_centroid_norm > 0:
                outlier_centroid /= outlier_centroid_norm
            else:
                tqdm.write("WARNING: outlier centroid has norm <= 0")
                outlier_centroid = np.zeros_like(outlier_centroid)
                outlier_centroid[0] = 1.0
            assert outlier_centroid.shape == centroids.shape[1:]

            # Compute the maximum cosine distance for the outlier cluster
            outlier_max_cos_distance = 0
            with tqdm(desc="Finding outlier max cos distance", leave=False) as pbar:
                for batch in outlier_dset.batch_iter(
                    batch_size=batch_size, drop_last_batch=False, threads=8, readahead=8
                ):
                    outlier_max_cos_distance = find_max_cosine_distance(
                        outlier_centroid,
                        batch["clip_embedding"],
                        outlier_max_cos_distance,
                    )
                    pbar.update(len(batch["clip_embedding"]))

            # Generate new clusters with the outliers removed
            outlier_set = set(removed_vectors)
            filtered_clusters = [[] for _ in range(len(assignments))]

            for cluster_idx, cluster in enumerate(assignments):
                for idx in cluster:
                    if idx not in outlier_set:
                        filtered_clusters[cluster_idx].append(idx)

            # remove any clusters that are now empty
            new_assignments = []
            new_max_distances = []
            new_centroids = []
            for cluster_idx in range(len(filtered_clusters)):
                if len(filtered_clusters[cluster_idx]) > 0:
                    new_assignments.append(filtered_clusters[cluster_idx])
                    new_max_distances.append(high_distances[cluster_idx][-1][0])
                    new_centroids.append(centroids[cluster_idx])
                else:
                    assert len(high_distances[cluster_idx]) == 0
            new_assignments.append(removed_vectors)
            new_max_distances.append(outlier_max_cos_distance)
            new_centroids = np.stack(new_centroids + [outlier_centroid], axis=0)
            assert new_centroids.shape == (len(new_assignments), centroids.shape[1])
            assert len(new_assignments) == len(new_max_distances) == len(new_centroids)
            assert sum(len(cluster) for cluster in new_assignments) == sum(
                len(cluster) for cluster in assignments
            )

            return new_centroids, new_assignments, new_max_distances

    def split_rec(self):
        """Split this cap and all children recursively until each leaf has at most max_leaf_size
        vectors."""
        assert self.children == [], "Can only split once"

        if len(self) > self.max_leaf_size:
            self.split_once()
            for child in tqdm(self.children, desc="Splitting children", leave=None):
                child.split_rec()
        else:
            self._check_for_duplicates()

    def _check_for_duplicates(self, force=False):
        """Check for duplicate vectors in this leaf. Check is skipped unless dup_check or force is
        True. Will fix them and write the names to found_duplicates if found."""
        if self.dup_check or force:
            assert self.children == [], "Can only check for duplicates in leaves"

            vecs_dict = {}
            for batch_idx, batch in enumerate(
                tqdm(
                    self.dset.batch_iter(
                        batch_size=4096, drop_last_batch=False, threads=8, readahead=8
                    ),
                    desc="Deduplicating",
                    leave=None,
                    total=len(self.dset) // 4096
                    + (1 if len(self.dset) % 4096 != 0 else 0),
                )
            ):
                clips = batch["clip_embedding"]
                nans = np.any(np.isnan(clips), axis=1)
                assert nans.shape == (len(clips),)
                for i in np.arange(len(clips))[nans]:
                    tqdm.write(
                        f"WARNING: found NaN in CLIP embedding for {self.dset[i]['name']}"
                    )
                # Near duplicates can be just as bad as exact ones, so we quantize the values
                # for the dict key.
                keys = [
                    vec.tobytes() if ~nans[i] else "NaN"
                    for i, vec in enumerate(
                        np.clip(clips * 2**15, -1 * 2**15, 2**15 - 1).astype(
                            np.int16
                        )
                    )
                ]
                for i, key in enumerate(keys):
                    vecs_dict.setdefault(key, []).append(batch_idx * 4096 + i)
            if len(self.dset) > len(vecs_dict):
                tqdm.write(
                    f"Found {len(self.dset) - len(vecs_dict)} duplicates/nan vectors"
                )
                self.found_duplicates.extend(
                    [
                        self.dset[idx]["name"] if "name" in self.dset[idx] else "🤷"
                        for idx in idxs
                    ]
                    for key, idxs in vecs_dict.items()
                    if len(idxs) > 1 or key == "NaN"
                )

                # Create a new deduplicated TableView
                unique_idxs = np.stack([idxs[0] for idxs in vecs_dict.values()])
                self.dset = self.dset.new_view(unique_idxs)
                self.dset_thin = self.dset.select_columns({"clip_embedding"})
                tqdm.write(f"New leaf size: {len(self.dset)}")

    def _to_summary_inner(self, centers=False):
        """Make a JSON representation of a node. If centers is True, include the centers of the
        caps. A captree can be reconstructed from this representation if centers is True and the
        contents of the dsets in the leaves are saved as well."""
        out = {
            "max_cos_distance": float(self.max_cos_distance),
            "len": len(self),
        }
        if centers:
            out["center"] = base64.b64encode(self.center.tobytes()).decode("ascii")
        if len(self.children) > 0:
            out["children"] = [
                child._to_summary_inner(centers) for child in self.children
            ]
        return out

    def to_summary(self, centers=False):
        """Make a summary representation of the tree. Returns a dict to be encoded as JSON. Same
        deal as above re reconstructing the tree."""
        return {
            "structure": self._to_summary_inner(centers),
            "total_vectors": len(self),
            "batch_size": self.batch_size,
            "k": self.k,
            "max_leaf_size": self.max_leaf_size,
            "iters": self.iters,
            "dup_check": self.dup_check,
            "depth": self.depth(),
            "largest_leaf_size": self.largest_leaf_size(),
            "smallest_leaf_size": self.smallest_leaf_size(),
            "mean_depth": np.mean(list(self.leaf_depths())),
        }

    def depth(self):
        """Depth of the tree."""
        if len(self.children) == 0:
            return 1
        else:
            return 1 + max(child.depth() for child in self.children)

    def delete_idxs(self, idxs):
        """Delete the entries at the specified indices from the tree. Input should be a numpy
        array. Returns True if the tree would be empty after this operation, False otherwise.
        Since captrees can't be empty, caller is responsible for handling this case. Doing this
        repeatedly with small sets of indices is very inefficient, it's best to do large batches.
        Otherwise you'll have a really big stack of TableViews."""

        assert isinstance(idxs, np.ndarray)
        assert len(idxs.shape) == 1
        assert self.dsets_contiguous

        if len(idxs) == 0:
            return False
        assert idxs.min() >= 0
        assert idxs.max() < len(self)

        if len(self) == len(idxs):
            return True

        start_len = len(self)

        if len(self.children) == 0:
            sorted_idxs = np.sort(idxs)
            assert len(np.unique(sorted_idxs)) == len(
                sorted_idxs
            ), "indices to delete must be unique"
            # Build a list of slices, containing the contiguous regions we're keeping.
            slices = []

            start = 0
            for idx in sorted_idxs:
                if idx > start:
                    slices.append(slice(start, idx))
                start = idx + 1
            if start < len(self):
                slices.append(slice(start, len(self)))

            # We try to avoid making a bunch of new views if we can. Each slice is a whole
            # TableView, but if you're doing index arrays you can have as many of those in the same
            # TableView as you like. The factor of 16 is totally made up.
            if len(slices) * 16 < (len(self.dset) - len(sorted_idxs)):
                dsets = [self.dset.new_view(s) for s in slices]
                self.dset = infinidata.TableView.concat(dsets)
                self.dset_thin = self.dset.select_columns({"clip_embedding"})
            else:
                idxs_to_keep = np.ones(len(self), dtype=np.bool_)
                idxs_to_keep[sorted_idxs] = False
                idxs_to_keep = np.arange(len(self))[idxs_to_keep]
                self.dset = self.dset.new_view(idxs_to_keep)
                self.dset_thin = self.dset_thin.new_view(idxs_to_keep)
            self.len = len(self.dset)
            assert len(self.dset) == len(self.dset_thin) == len(self)
            assert len(self) == start_len - len(sorted_idxs)
        else:
            children_to_delete = []
            for i, child in enumerate(self.children):
                child_idxs_to_delete_mask = (idxs >= self.child_start_idxs[i]) & (
                    idxs < self.child_start_idxs[i + 1]
                )
                child_idxs_to_delete = (
                    idxs[child_idxs_to_delete_mask] - self.child_start_idxs[i]
                )
                if np.any(child_idxs_to_delete_mask):
                    child_start_len = len(child)
                    child_start_dset_len = len(child.dset)
                    assert child_start_len == child_start_dset_len
                    if child.delete_idxs(child_idxs_to_delete):
                        children_to_delete.append(i)
                    else:
                        assert len(child) == child_start_len - len(child_idxs_to_delete)
                        assert len(child.dset) == child_start_dset_len - len(
                            child_idxs_to_delete
                        )
            if len(children_to_delete) == len(self.children):
                assert (
                    False
                ), "We somehow deleted all the children despite having fewer indices to delete than are in this subtree"
            if len(children_to_delete) > 0:
                deleted_so_far = 0
                for child_idx in children_to_delete:
                    del self.children[child_idx - deleted_so_far]
                    deleted_so_far += 1
                self.child_cap_centers = np.array(
                    [child.center for child in self.children]
                )
                self.child_cap_max_cos_distances = np.array(
                    [child.max_cos_distance for child in self.children]
                )
            self.dsets_contiguous = False
            self._make_contiguous()
        self.len = len(self.dset)
        assert len(self) == start_len - len(idxs)
        return False

    def leaf_depths(self):
        """Generator of the depths of the leaves of the tree."""
        if len(self.children) == 0:
            yield 1
        else:
            for child in self.children:
                for depth in child.leaf_depths():
                    yield depth + 1

    def largest_leaf_size(self):
        """Number of vectors in the largest leaf. This is different from the max_leaf_size, which
        is a limit."""
        if len(self.children) == 0:
            return len(self)
        else:
            return max(child.largest_leaf_size() for child in self.children)

    def smallest_leaf_size(self):
        """Minimum number of vectors in a leaf."""
        if len(self.children) == 0:
            return len(self)
        else:
            return min(child.smallest_leaf_size() for child in self.children)

    def _check_invariants(self):
        """Check invariants of the tree."""
        assert len(self) > 0
        assert len(self) == len(self.dset)
        if len(self.children) > 0:
            assert sum(len(child) for child in self.children) == len(self)
            assert (
                self.child_cap_centers.shape[0]
                == self.child_cap_max_cos_distances.shape[0]
                == len(self.children)
            )
            for i in range(len(self.child_cap_centers)):
                assert np.all(self.child_cap_centers[i] == self.children[i].center)
                assert (
                    self.child_cap_max_cos_distances[i]
                    == self.children[i].max_cos_distance
                )
            for thicc_batch, thin_batch in tqdm(
                zip(
                    self.dset.batch_iter(
                        self.batch_size, drop_last_batch=False, readahead=8, threads=8
                    ),
                    self.dset_thin.batch_iter(
                        self.batch_size, drop_last_batch=False, readahead=8, threads=8
                    ),
                ),
                total=len(self.dset) // self.batch_size + 1,
                desc="Checking dset/dset_thin equality",
                leave=False,
            ):
                assert list(thin_batch.keys()) == ["clip_embedding"]
                np.testing.assert_array_equal(
                    thicc_batch["clip_embedding"], thin_batch["clip_embedding"]
                )
            if self.dsets_contiguous:
                for child_idx, child in enumerate(
                    tqdm(self.children, desc="Checking dset contiguity", leave=False)
                ):
                    assert child.dsets_contiguous
                    child_start_idx = self.child_start_idxs[child_idx]
                    child_stop_idx = self.child_start_idxs[child_idx] + len(child)
                    parent_sliced_view = self.dset.new_view(
                        slice(child_start_idx, child_stop_idx)
                    )

                    for parent_batch, child_batch in zip(
                        parent_sliced_view.batch_iter(
                            self.batch_size,
                            drop_last_batch=False,
                            readahead=8,
                            threads=8,
                        ),
                        child.dset.batch_iter(
                            self.batch_size,
                            drop_last_batch=False,
                            readahead=8,
                            threads=8,
                        ),
                    ):
                        assert list(parent_batch.keys()) == list(child_batch.keys())
                        for k in parent_batch.keys():
                            assert np.array_equal(parent_batch[k], child_batch[k])
            if self.ready_for_queries:
                assert self.dsets_contiguous
                for child in self.children:
                    assert child.ready_for_queries

        else:
            assert self.center.shape == self.dset[0]["clip_embedding"].shape
            assert self.child_cap_centers == self.child_cap_max_cos_distances == None

        assert self.max_cos_distance <= 2
        assert np.isclose(np.linalg.norm(self.center), 1.0)
        assert self.center.dtype == np.float32

        self._check_inside_cap(self.center, self.max_cos_distance)

        for subtree in tqdm(
            self.children, leave=False, desc="Checking subtree invariants"
        ):
            subtree._check_invariants()

    def _check_inside_cap(self, cap_center, max_cos_distance):
        """Check that all vectors in this node are inside the given cap."""
        if len(self.children) == 0:
            distances = cosine_distance_many_to_one(
                self.dset[:]["clip_embedding"], cap_center
            )
            assert distances.shape == (len(self),)
            valid_distances_mask = distances <= max_cos_distance + self.EPSILON
            invalid_distances = distances[~valid_distances_mask]
            assert np.all(
                valid_distances_mask
            ), f"invalid distances: {invalid_distances}"
        else:
            for child in tqdm(
                self.children, leave=False, desc="Checking subtree vecs are in cap"
            ):
                child._check_inside_cap(cap_center, max_cos_distance)

    def items(self):
        """Generator for all the original rows in the dataset."""
        # The primary purpose of this function is testing
        if len(self.children) == 0:
            for row in self.dset:
                yield row
        else:
            for child in self.children:
                for row in child.items():
                    yield row

    def leaves(self):
        """Generator for all the leaves of the tree."""
        if len(self.children) == 0:
            yield self
        else:
            for child in self.children:
                for leaf in child.leaves():
                    yield leaf

    def shuffle_leaves(self):
        """Shuffle the leaves of the tree."""
        for leaf in self.leaves():
            leaf.dset = leaf.dset.shuffle()

    def sample_in_caps_approx(
        self,
        query_centers,
        query_max_cos_distances,
        density_estimate_samples=8,
        batch_size=4096,
    ):
        """Sample from multiple caps at once. Much better throughput than calling
        sample_in_cap_approx repeatedly. Returns a numpy array of indices into the dataset, where
        -1 indicates the cap was empty. The distribution of returned indices is biased against caps
        that contain few matching vectors, and becomes less biased as density_estimate_samples
        increases.
        """
        assert (
            self.ready_for_queries
        ), "Load tree from disk or call tree.prepare_for_queries."
        assert len(query_centers.shape) == 2
        assert query_centers.shape[1] == self.center.shape[0]
        assert len(query_max_cos_distances.shape) == 1
        assert len(query_max_cos_distances) == len(query_centers)

        if len(self.children) == 0:
            return self.sample_in_caps(
                query_centers, query_max_cos_distances, batch_size=batch_size
            )

        query_cnt = len(query_centers)
        sizes = np.array([child.len for child in self.children])

        # Do geometric tests to find the caps that are fully contained in the query caps, as well
        # as the ones that intersect but do not contain them.
        contained, intersecting = cap_intersection_status_many_to_many_padded(
            query_centers,
            query_max_cos_distances,
            self.child_cap_centers,
            self.child_cap_max_cos_distances,
        )

        assert contained.shape == intersecting.shape == (query_cnt, len(self.children))
        contained, intersecting = np.array(contained), np.array(intersecting)

        densities = np.zeros((query_cnt, len(self.children)), dtype=np.float32)
        densities[contained] = 1.0

        # For the intersecting caps, we attempt to estimate the density of matching vectors in the
        # subtrees by sampling.
        subtrees_to_sample = jnp.any(intersecting, axis=0)
        assert subtrees_to_sample.shape == (len(self.children),)
        subtrees_to_sample_idxs = np.arange(len(self.children))[subtrees_to_sample]

        # Gather density_estimate_samples samples for each subtree that intersects any of the query
        # caps.
        sampled_vecs = np.full(
            (
                len(subtrees_to_sample_idxs),
                density_estimate_samples,
                self.center.shape[0],
            ),
            -1,
            dtype=np.float32,
        )
        positive_samples = np.zeros(
            (query_cnt, len(subtrees_to_sample_idxs), density_estimate_samples),
            dtype=bool,
        )
        sampled_idxs = np.full(
            (len(subtrees_to_sample_idxs), density_estimate_samples),
            -1,
            dtype=np.int64,
        )
        for i, j in enumerate(subtrees_to_sample_idxs):
            # Choose the vectors we're sampling from each subtree
            sampled_idxs_this = np.random.randint(
                sizes[j], size=density_estimate_samples
            )
            sampled_by_index_array = self.children[j].dset_thin[sampled_idxs_this][
                "clip_embedding"
            ]
            sampled_vecs[i] = sampled_by_index_array
            sampled_idxs[i, :] = sampled_idxs_this
        assert not np.any(sampled_idxs == -1)
        sampled_vecs = jnp.array(sampled_vecs)

        for i, j in enumerate(subtrees_to_sample_idxs):
            # Iterate over the subtrees we need to sample from and do the calculation
            query_caps_to_test = np.arange(query_cnt)[intersecting[:, j]]
            in_caps = vectors_in_caps_padded(
                sampled_vecs[i],
                query_centers[query_caps_to_test],
                query_max_cos_distances[query_caps_to_test],
            )
            assert in_caps.shape == (density_estimate_samples, len(query_caps_to_test))
            matching_cnts = jnp.sum(in_caps, axis=0)
            assert matching_cnts.shape == (
                len(query_caps_to_test),
            ), f"{matching_cnts.shape}"
            positive_samples[query_caps_to_test, i] = in_caps.T
            densities[query_caps_to_test, j] = matching_cnts / density_estimate_samples

        estimated_matching_sizes = densities * sizes
        assert estimated_matching_sizes.shape == (query_cnt, len(self.children))

        # Find the queries that at this point have 0 estimated matching vectors but are not
        # eliminated by the geometric test. We fall back to exact sampling for those. They match
        # a very small fraction of the dataset, or none of it.
        zero_estimated_matches = np.sum(estimated_matching_sizes, axis=1) == 0
        need_exact = zero_estimated_matches & np.any(intersecting, axis=1)
        assert need_exact.shape == (query_cnt,)
        need_exact_cnt = np.sum(need_exact)
        assert zero_estimated_matches.shape == need_exact.shape == (query_cnt,)

        if need_exact_cnt > 0:
            contained_for_exact = contained[need_exact]
            intersecting_for_exact = intersecting[need_exact]
            exact_results = self.sample_in_caps(
                query_centers=query_centers[need_exact, :],
                query_max_cos_distances=query_max_cos_distances[need_exact],
                batch_size=batch_size,
                mask=(contained_for_exact, intersecting_for_exact),
            )
            exact_idxs = np.arange(query_cnt)[need_exact]
        else:
            exact_results = np.array([], dtype=np.int64)
            exact_idxs = np.array([], dtype=np.int64)
        assert exact_results.shape == exact_idxs.shape == (need_exact_cnt,)
        assert (
            exact_results.dtype == exact_idxs.dtype == np.int64
        ), f"{exact_results.dtype}, {exact_idxs.dtype}"

        known_empty = np.all(~intersecting & ~contained, axis=1)
        assert known_empty.shape == (query_cnt,)
        assert not np.any(known_empty & need_exact)

        # For the queries that we didn't fall back for, we sample from the distribution estimated
        # based on the sizes and the sampled vectors.

        # First we sample which subtree to sample from for each query.
        approximate_query_idxs = np.arange(query_cnt)[~need_exact & ~known_empty]
        sampled_subtrees = np.full((query_cnt,), -1, dtype=np.int64)
        for i in approximate_query_idxs:
            sampled_subtrees[i] = np.random.choice(
                np.arange(len(self.children)),
                p=estimated_matching_sizes[i] / np.sum(estimated_matching_sizes[i]),
            )

        # Sample within the subtrees. If the subtree we chose is one which we sampled from for
        # density estimation, we use one of the samples we already have, otherwise we sample from
        # the subtree by generating an index.

        # Map from subtree index to index in sampled_vecs
        subtree_to_sampled_vecs = {}
        for i, j in enumerate(subtrees_to_sample_idxs):
            subtree_to_sampled_vecs[j] = i

        sampled_idxs_out = np.full((query_cnt,), -1, dtype=np.int64)
        for i in approximate_query_idxs:
            if contained[i, sampled_subtrees[i]]:
                # If we chose a subtree that is fully contained in the query we sample an index
                # index that subtree uniformly
                sampled_idxs_out[i] = self.child_start_idxs[
                    sampled_subtrees[i]
                ] + np.random.randint(sizes[sampled_subtrees[i]])
            else:
                # If it's not fully contained we sample from the matching vectors we sampled for
                # density estimation.
                positive_samples_this_subtree = positive_samples[
                    i, subtree_to_sampled_vecs[sampled_subtrees[i]]
                ]
                positive_samples_idxs = np.arange(density_estimate_samples)[
                    positive_samples_this_subtree
                ]
                sampled_idxs_this_subtree = sampled_idxs[
                    subtree_to_sampled_vecs[sampled_subtrees[i]]
                ]
                sampled_idx_in_samples = np.random.choice(positive_samples_idxs)
                sampled_idxs_out[i] = (
                    self.child_start_idxs[sampled_subtrees[i]]
                    + sampled_idxs_this_subtree[sampled_idx_in_samples]
                )

        sampled_idxs_out[need_exact] = exact_results
        return sampled_idxs_out

    def sample_in_caps(
        self, query_centers, query_max_cos_distances, batch_size=8192, mask=None
    ):
        """Exact method for sampling from multiple caps at once. Returns a numpy array of indices.
        -1 indicates the cap was empty."""
        assert self.ready_for_queries
        assert len(query_centers.shape) == 2
        assert query_centers.shape[1] == self.center.shape[0]
        assert len(query_max_cos_distances.shape) == 1
        assert len(query_max_cos_distances) == len(query_centers)

        matching_subtrees_all = self._subtrees_in_caps(
            query_centers, query_max_cos_distances, batch_size=batch_size, mask=mask
        )

        assert len(matching_subtrees_all) == len(query_centers)

        out = np.full(len(query_centers), -1, dtype=np.int64)

        # processing matches at the end can be a bottleneck, so we do it in parallel with threads.
        # fetching leaf vectors from the dset and calling jax both release the GIL for a while,
        # though neither are super efficient about it.
        def process_match(query_num, matching_subtrees):
            # If there are matches, we pick one, otherwise we leave it as -1
            if len(matching_subtrees) > 0:
                total_matches = sum(cnt for _, cnt in matching_subtrees)
                assert total_matches > 0

                # Sample uniformly from the matching vectors
                sampled_idx = np.random.randint(total_matches)
                cur = 0
                for path, cnt in matching_subtrees:
                    cur += cnt
                    if sampled_idx < cur:
                        break
                idx_in_subtree = sampled_idx - (cur - cnt)

                # Find the start index of the subtree we're sampling from
                subtree_start = 0
                subtree = self
                for step in path:
                    subtree_start += subtree.child_start_idxs[step]
                    subtree = subtree.children[step]
                if len(subtree) == cnt:
                    # If the entire subtree matches we just sample from it
                    return subtree_start + idx_in_subtree
                else:
                    # Otherwise, we need to check the vectors in the subtree. This is redundant,
                    # since we did this in _subtrees_in_caps. Not sure if this is worse than saving
                    # the matching vector masks in _subtrees_in_caps.
                    valid_distances_mask = vectors_in_cap_even_batch(
                        subtree.dset_thin[:]["clip_embedding"],
                        query_centers[query_num],
                        query_max_cos_distances[query_num],
                    )
                    valid_distances_idxs = np.arange(len(subtree.dset))[
                        valid_distances_mask
                    ]
                    if len(valid_distances_idxs) == 0:
                        tqdm.write(
                            f"WARNING: _subtrees_in_caps found {cnt} matches but sample_in_caps "
                            + "found 0. Returning none."
                        )
                        return -1
                    elif len(valid_distances_idxs) != cnt:
                        tqdm.write(
                            f"WARNING: _subtrees_in_caps found {cnt} matches but sample_in_caps "
                            + f"found {len(valid_distances_idxs)}. Sampling from those."
                        )
                        return (
                            subtree_start
                            + valid_distances_idxs[
                                np.random.randint(len(valid_distances_idxs))
                            ]
                        )
                    elif len(valid_distances_idxs) == cnt:
                        return subtree_start + valid_distances_idxs[idx_in_subtree]
                    else:
                        assert False, "unreachable"
            else:
                return -1

        futs = {
            self.threadpool.submit(process_match, i, matching_subtrees): i
            for i, matching_subtrees in enumerate(matching_subtrees_all)
        }
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()

        return out

    def _subtrees_in_caps(
        self, query_centers, query_max_cos_distances, batch_size, mask=None
    ):
        """Find all the subtrees that are either fully contained in the input caps or are leaves and
        have vectors that are in the input caps. Returns a list of paths to subtrees and matching
        vector counts."""
        out = [[] for _ in range(len(query_centers))]

        if len(self.children) > 0:
            # For a node, we use the geometric test, then check the children that intersect the
            # query caps but don't contain them, recursively.
            if mask is None:
                contained, intersecting = cap_intersection_status_many_to_many_padded(
                    query_centers,
                    query_max_cos_distances,
                    self.child_cap_centers,
                    self.child_cap_max_cos_distances,
                )
            else:
                contained, intersecting = mask

            assert (
                contained.shape
                == intersecting.shape
                == (len(query_centers), len(self.children))
            )
            contained, intersecting = np.array(contained), np.array(intersecting)

            # Record the paths to the subtrees that are fully contained in the query caps.
            contained_query_idxs, contained_subtree_idxs = np.nonzero(contained)
            for i, j in zip(contained_query_idxs, contained_subtree_idxs):
                out[i].append(([j], self.children[j].len))

            # For the subtrees that intersect but aren't contained in the query caps, we need to
            # check inside them.
            subtrees_to_check = np.any(intersecting, axis=0)
            assert subtrees_to_check.shape == (len(self.children),)
            # A list of (leaf_idx, leaf_size, query_mask) tuples to be tested later
            leaves_to_check = []
            for i in np.arange(len(self.children))[subtrees_to_check]:
                queries_this_subtree = np.arange(len(query_centers))[intersecting[:, i]]
                assert len(queries_this_subtree) > 0
                if len(self.children[i].children) == 0:
                    # We check any leaves here without recursing to take advantage of infinidata's
                    # batch_iter_concat and provide larger batches to the GPU.
                    leaves_to_check.append(
                        (i, self.children[i].len, intersecting[:, i])
                    )
                else:
                    matching_subtrees = self.children[i]._subtrees_in_caps(
                        query_centers[queries_this_subtree],
                        query_max_cos_distances[queries_this_subtree],
                        mask=None,
                        batch_size=batch_size,
                    )
                    assert len(matching_subtrees) == len(queries_this_subtree)
                    for j, k in enumerate(queries_this_subtree):
                        out[k].extend(
                            ([i] + path, cnt) for path, cnt in matching_subtrees[j]
                        )

            if len(leaves_to_check) > 0:
                leaf_match_cnts = np.zeros(
                    (len(leaves_to_check), len(query_centers)), dtype=np.int64
                )
                dsets = [self.children[i].dset_thin for i, _, _ in leaves_to_check]
                leaf_sizes = [size for _, size, _ in leaves_to_check]
                leaf_idx = 0  # Which leaf we're processing
                idx_in_leaf = 0  # Index in that leaf
                # Caps that need to be checked for the current batch
                queries_in_batch = np.zeros(len(query_centers), dtype=bool)
                for batch_idx, batch in enumerate(
                    infinidata.TableView.batch_iter_concat(
                        dsets,
                        batch_size=batch_size,
                        threads=8,
                        readahead=8,
                    )
                ):
                    batch = jnp.array(batch["clip_embedding"])
                    queries_in_batch[:] = False
                    # list of (leaf index, row count) pairs used to assign results to leaves
                    leaf_assignments = []
                    idx_in_batch = 0
                    # Loop over leaves within the batch, collecting which caps we need to check for
                    # the batch
                    while idx_in_batch < len(batch):
                        queries_in_batch |= leaves_to_check[leaf_idx][2]
                        rows_used = min(
                            # rest of the batch
                            len(batch) - idx_in_batch,
                            # rest of the leaf
                            leaf_sizes[leaf_idx] - idx_in_leaf,
                        )
                        idx_in_batch += rows_used
                        idx_in_leaf += rows_used
                        leaf_assignments.append((leaf_idx, rows_used))
                        if idx_in_leaf == leaf_sizes[leaf_idx]:
                            idx_in_leaf = 0
                            leaf_idx += 1
                        assert (
                            leaf_idx == len(leaf_sizes) and idx_in_batch == len(batch)
                        ) or idx_in_leaf <= leaf_sizes[leaf_idx]
                    query_idxs_in_batch = np.arange(len(query_centers))[
                        queries_in_batch
                    ]
                    # Check the batch
                    in_caps_cumsum = vectors_in_caps_padded(
                        batch,
                        query_centers[queries_in_batch],
                        query_max_cos_distances[queries_in_batch],
                        need_cumsum=True,
                        need_bools=False,
                    )
                    assert in_caps_cumsum.shape == (
                        len(batch),
                        len(query_idxs_in_batch),
                    )
                    in_caps_cumsum = np.array(in_caps_cumsum)

                    # Use leaf_assignments to update the correct counts
                    cur = 0
                    for assigned_leaf_idx, rows_used in leaf_assignments:
                        # Use the difference in the cumulative sum to get the number of matches
                        # this is marginally faster that sending the boolen array to the CPU and
                        # doing a sum over a slice.
                        match_cnts_by_query_cumsum = in_caps_cumsum[
                            cur + rows_used - 1
                        ] - (in_caps_cumsum[cur - 1] if cur > 0 else 0)
                        leaf_match_cnts[
                            assigned_leaf_idx, query_idxs_in_batch
                        ] += match_cnts_by_query_cumsum
                        cur += rows_used
                for i, (subtree, _len, queries) in enumerate(leaves_to_check):
                    query_idxs = np.arange(len(query_centers))[queries]
                    matching_leaves = leaf_match_cnts[i, query_idxs] > 0
                    for query_idx in query_idxs[matching_leaves]:
                        out[query_idx].append(
                            ([subtree], leaf_match_cnts[i, query_idx])
                        )
        else:
            # For a leaf, we check the vectors in the leaf against the query caps.
            in_caps = vectors_in_caps(
                self.dset_thin[:]["clip_embedding"],
                query_centers,
                query_max_cos_distances,
            )
            assert in_caps.shape == (len(self.dset), len(query_centers))
            matching_cnts = np.asarray(np.sum(in_caps, axis=0))
            matching_subtrees = np.arange(len(query_centers))[matching_cnts > 0]
            for subtree_idx in matching_subtrees:
                out[subtree_idx].append(([], matching_cnts[subtree_idx]))
        return out

    def __getitem__(self, idx):
        """Get a vector by index. There is no meaningful ordering."""
        return self.dset[idx]

    def _make_contiguous(self):
        """Make the dset of this tree contain the dsets of the children in order. Not recursive."""
        if len(self.children) > 0:
            if not self.dsets_contiguous:
                self.dset = infinidata.TableView.concat(
                    [child.dset for child in self.children]
                )
                self.dset_thin = self.dset.select_columns({"clip_embedding"})
                self.child_start_idxs = np.cumsum(
                    [0] + [len(child) for child in self.children]
                )
        self.dsets_contiguous = True

    def prepare_for_queries(self):
        """Prepare the tree for fast querying. This is unnecessary if the tree was loaded from
        disk. Note that loading from disk yields a tree that is much faster to query that building
        one and then calling this function, since this only mucks with the indices in RAM and does
        not actually move anything around."""
        # Make the dsets contiguous so that each child's dset is a slice of this tree's dset and
        # those slices are in order.
        if not self.ready_for_queries:
            for child in self.children:
                child.prepare_for_queries()
            self._make_contiguous()
            self.threadpool = concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            )
            self.ready_for_queries = True
        else:
            print("Tree already ready for queries, skipping prep.")

    def save_to_disk(self, dir, thin=False):
        """Save the tree to disk."""
        dir.mkdir(exist_ok=False, parents=True)
        summary = self.to_summary(centers=True)
        with open(dir / "structure.json", "w") as f:
            json.dump(summary, f, indent=2)

        # We concatenate all the leaves into one big dataset before saving to disk. This is a bit
        # more complicated (especially loading) but lets the parquet compression work much better.
        if thin:
            dsets = [leaf.dset_thin for leaf in self.leaves()]
        else:
            dsets = [leaf.dset for leaf in self.leaves()]
        dset_all = infinidata.TableView.concat(dsets)

        for k, v in dset_all[0].items():
            assert (
                len(v.shape) < 2
            ), f"serializing multidimensional arrays is not supported yet, got shape {v.shape} for column {k}"

        df = pd.DataFrame([dset_all[0]])
        pq_schema = pa.Schema.from_pandas(df)

        with open(dir / "data.parquet", "wb") as f:
            pq_writer = pq.ParquetWriter(f, pq_schema, compression="zstd")
            with tqdm(total=len(dset_all), desc="Writing parquet", unit="rows") as pbar:
                for batch in dset_all.batch_iter(
                    batch_size=4096, drop_last_batch=False, threads=8, readahead=8
                ):
                    rows = len(batch[list(batch.keys())[0]])
                    df_rows = []
                    for i in range(rows):
                        df_rows.append({k: v[i] for k, v in batch.items()})
                    pq_writer.write_table(
                        pa.Table.from_pandas(pd.DataFrame(df_rows), schema=pq_schema)
                    )
                    pbar.update(rows)
            pq_writer.close()

    def _empty_from_summary(self, root, node_json):
        """Fill in everything in a node based on the JSON representation (with centers) except the
        dset."""
        self.batch_size = root.batch_size
        self.k = root.k
        self.max_leaf_size = root.max_leaf_size
        self.iters = root.iters
        self.dup_check = root.dup_check
        self.found_duplicates = root.found_duplicates

        self.ready_for_queries = False
        self.is_contiguous = False

        self.len = node_json["len"]
        assert (
            "center" in node_json
        ), "centers must be included in summary if you want to load a tree"
        self.center = np.frombuffer(
            base64.b64decode(node_json["center"]), dtype=np.float32
        )
        self.max_cos_distance = node_json["max_cos_distance"]
        self.children = []

        if "children" in node_json:
            for child_json in node_json["children"]:
                child = self.__class__.__new__(self.__class__)
                child._empty_from_summary(root, child_json)
                self.children.append(child)
            self.dset = None

    def _fixup_traversal_arrays(self):
        """Fix up the arrays used for traversing the tree."""
        if len(self.children) > 0:
            self.child_cap_centers = np.array([child.center for child in self.children])
            self.child_cap_max_cos_distances = np.array(
                [child.max_cos_distance for child in self.children]
            )
            for child in self.children:
                child._fixup_traversal_arrays()
        else:
            self.child_cap_centers = self.child_cap_max_cos_distances = None

    def _fixup_inner_dsets(self):
        """Fill in the dsets of the children of this tree with the appropriate slices of this
        tree's dset. Assumes this dset is ordered with the children's dsets concatenated.
        """
        self.child_start_idxs = np.cumsum([0] + [len(child) for child in self.children])
        cur_idx = 0
        for child in self.children:
            child.dset = self.dset.new_view(slice(cur_idx, cur_idx + len(child)))
            child.dset_thin = child.dset.select_columns({"clip_embedding"})
            cur_idx += len(child)
            child._fixup_inner_dsets()
        self.dsets_contiguous = True

    @classmethod
    def load_from_disk(cls, dir, save_cache=True):
        """Load a tree from disk."""
        with open(dir / "structure.json", "r") as f:
            summary = json.load(f)

        out = cls.__new__(cls)

        out.batch_size = summary["batch_size"]
        out.k = summary["k"]
        out.max_leaf_size = summary["max_leaf_size"]
        out.iters = summary["iters"]
        out.dup_check = summary["dup_check"]
        out.found_duplicates = []
        out.dset = None
        out.len = summary["total_vectors"]

        out._empty_from_summary(out, summary["structure"])

        out.dset = load_pq_to_infinidata(dir / "data.parquet", save_cache=save_cache)
        out.dset_thin = out.dset.select_columns({"clip_embedding"})

        out._fixup_inner_dsets()

        out._fixup_traversal_arrays()
        out.prepare_for_queries()

        return out


@hyp.settings(
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.too_slow,
        hyp.HealthCheck.filter_too_much,
    ],
    deadline=timedelta(seconds=30),
)
@given(
    _unit_vecs(
        st.tuples(st.integers(4, 1024), st.integers(2, 4)),
    )
)
def test_remove_outliers_with_level_1_doesnt_make_outlier_cluster(vecs):
    """Test that setting outlier_removal_level to 1 causes no outlier removal."""
    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=16, outlier_removal_level=1)
    tree.split_once()
    # There should mostly be 4 clusters but occassionally we'll end up with fewer. If it makes an
    # outlier cluster, there will mostly be 5, though we can still end up with less.
    assert (
        len(tree.children) <= 4
    ), f"Got more than 4 children: {len(tree.children)}. Centers: {tree.child_cap_centers}, max cos distances: {tree.child_cap_max_cos_distances}, values: {[child.dset[:]['clip_embedding'] for child in tree.children]}, distances: {[cosine_distance_many_to_one(child.dset[:]['clip_embedding'], child.center) for child in tree.children]}"


def test_remove_outliers_with_level_0_makes_outlier_cluster():
    """Test that it makes an outlier cluter when outlier_removal_level is 0."""
    # There are various weird degenerate cases where it shouldn't make a cluster even when
    # outlier_removal_level is 0, so this is a regular unit test and not a hypothesis test.
    vecs = np.random.normal(size=(64, 4)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dset = infinidata.TableView({"clip_embedding": vecs})

    tree = CapTree(dset, batch_size=32, k=4, iters=16, outlier_removal_level=0)
    tree._check_invariants()
    tree.split_once()
    assert len(tree.children) == 5, f"Only got {len(tree.children)} children"
    tree._check_invariants()


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(1, 256), st.integers(2, 4)),
    ),
    st.integers(3, 8),
    st.integers(3, 16),
    st.floats(0.0, 1.0),
    st.booleans(),
    st.booleans(),
    st.random_module(),
)
def test_tree_invariants(
    vecs, k, max_leaf_size, outlier_removal_level, do_split, do_query_prep, _rand
):
    """Test that the tree invariants hold for any set of vectors."""
    hyp.assume(k <= max_leaf_size)
    vecs_set = set(tuple(vec) for vec in vecs)
    hyp.assume(len(vecs_set) == len(vecs))

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(
        dset,
        batch_size=32,
        k=k,
        iters=4,
        max_leaf_size=max_leaf_size,
        outlier_removal_level=outlier_removal_level,
    )
    if do_split:
        tree.split_rec()
    if do_query_prep:
        tree.prepare_for_queries()

    tree._check_invariants()

    # check that the elements in the tree are the ones we put in
    tree_vecs_set = set(tuple(row["clip_embedding"]) for row in tree.items())
    assert tree_vecs_set == vecs_set


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.too_slow,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(st.tuples(st.integers(2, 1024), st.integers(2, 4))),
    st.integers(1, 4).flatmap(
        lambda num_queries: st.tuples(
            st.just(num_queries),
            hyp_np.arrays(np.float64, (num_queries,), elements=st.floats(0.01, 2.0)),
        )
    ),
    st.integers(3, 8),
    st.booleans(),
    st.integers(1, 1024),
    st.random_module(),
)
def test_tree_subtrees_in_caps_sizes_are_correct(
    vecs, query_info, k, do_split, batch_size, _rand
):
    """Test that _subtrees_in_caps returns subtrees with the correct sizes."""
    num_queries, max_cos_distances = query_info
    hyp.assume(num_queries < len(vecs))
    queries = vecs[:num_queries]
    dset = infinidata.TableView({"clip_embedding": vecs[num_queries:]})
    tree = CapTree(dset, batch_size=32, k=k, iters=4, max_leaf_size=k + 4)
    if do_split:
        tree.split_rec()

    matching_subtrees_all = tree._subtrees_in_caps(
        queries, max_cos_distances, batch_size=batch_size
    )
    assert len(matching_subtrees_all) == num_queries
    for i, matching_subtrees in enumerate(matching_subtrees_all):
        max_cos_distance_narrow = max_cos_distances[i] - 0.01
        max_cos_distance_wide = max_cos_distances[i] + 0.01
        for match_num, (path, size) in enumerate(matching_subtrees):
            cur_subtree = tree
            for step in path:
                cur_subtree = cur_subtree.children[step]

            if len(cur_subtree.children) == 0:
                assert size <= len(cur_subtree)
            else:
                assert size == len(cur_subtree)

            subtree_vecs = cur_subtree.dset[:]["clip_embedding"]
            cosine_distances = cosine_distance_many_to_one(subtree_vecs, queries[i])
            vecs_in_cap_narrow = cosine_distances <= max_cos_distance_narrow
            vecs_in_cap_wide = cosine_distances <= max_cos_distance_wide
            assert size <= np.sum(vecs_in_cap_wide)
            assert size >= np.sum(vecs_in_cap_narrow)
        total_matches = sum(size for _, size in matching_subtrees)
        all_cosine_distances = cosine_distance_many_to_one(
            tree.dset[:]["clip_embedding"], queries[i]
        )
        assert total_matches <= np.sum(all_cosine_distances <= max_cos_distance_wide)
        assert total_matches >= np.sum(all_cosine_distances <= max_cos_distance_narrow)


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(st.tuples(st.integers(1, 1024), st.integers(2, 4))),
    st.integers(3, 8),
    st.floats(0.0, 1.0),
    st.booleans(),
    st.integers(1, 4),
    st.integers(1, 1024),
    st.random_module(),
)
def test_tree_subtrees_in_caps_finds_all_at_2(
    vecs, k, outlier_removal_level, do_split, n_queries, batch_size, _rand
):
    """Test that _subtrees_in_caps finds all vectors when max_cos_distance is 2."""
    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(
        dset, batch_size=32, k=k, iters=4, outlier_removal_level=outlier_removal_level
    )

    # Make n orthogonal axis aligned query vectors
    hyp.assume(n_queries < vecs.shape[1])
    query_centers = np.zeros((n_queries, vecs.shape[1]), dtype=np.float32)
    max_cos_distances = np.full(n_queries, 2.0, dtype=np.float32)
    for i in range(n_queries):
        query_centers[i, i] = 1.0
    np.testing.assert_allclose(np.linalg.norm(query_centers, axis=1), 1.0)

    if do_split:
        tree.split_rec()

    matching_subtrees_all = tree._subtrees_in_caps(
        query_centers, max_cos_distances, batch_size=batch_size
    )
    assert len(matching_subtrees_all) == n_queries

    for i, matching_subtrees in enumerate(matching_subtrees_all):
        matching_subtrees = sorted(matching_subtrees)
        assert np.sum([size for _, size in matching_subtrees]) == len(vecs)
        if len(matching_subtrees) == 1:
            # This happens when the top level is a leaf
            path, size = matching_subtrees[0]
            assert path == []
            assert size == len(vecs)
        else:
            # This happens when the top level is a node. Due to floating point error, it doesn't always
            # find the smallest set of subtrees that cover all the vectors, but it should always find
            # all the vectors.
            for path, size in matching_subtrees:
                cur_subtree = tree
                for step in path:
                    cur_subtree = cur_subtree.children[step]
                assert size == len(cur_subtree), f"bad size for subtree {path}"


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(st.tuples(st.integers(1, 256), st.integers(2, 4))),
    st.integers(3, 8),
    st.integers(1, 4),
    st.random_module(),
)
def test_tree_sample_batch_finds_all(vecs, k, batch_size, _rand):
    """Test that exact sampling finds all vectors in the tree when sampling from tiny caps centered
    on them."""

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=k, max_leaf_size=2 * k, iters=4)
    tree.split_rec()
    tree.prepare_for_queries()

    tol = 0.005

    vec_idx = 0

    while vec_idx < len(vecs):
        vecs_this_batch = vecs[vec_idx : vec_idx + batch_size]
        samples = tree.sample_in_caps(
            vecs_this_batch, np.repeat(tol, len(vecs_this_batch))
        )

        for i in range(len(vecs_this_batch)):
            sample_idx = samples[i]
            assert sample_idx != -1
            sample_vec = tree[sample_idx]["clip_embedding"]
            vec = vecs_this_batch[i]
            distance = cosine_distance(sample_vec, vec)
            assert distance <= tol
            # Hypothesis sometimes generates vectors that are *very* close together, so even with a
            # very small cap we can sample a different vector.
            # np.testing.assert_array_equal(sample["clip_embedding"], vec)
        vec_idx += len(vecs_this_batch)


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    st.tuples(st.integers(2, 4), st.integers(1, 4)).flatmap(
        lambda d_and_num_queries: st.tuples(
            _unit_vecs(st.tuples(st.integers(1, 1024), st.just(d_and_num_queries[0]))),
            _unit_vecs(
                st.tuples(st.just(d_and_num_queries[1]), st.just(d_and_num_queries[0]))
            ),
            hyp_np.arrays(
                np.float64, (d_and_num_queries[1],), elements=st.floats(0.01, 2.0)
            ),
        )
    ),
    st.integers(3, 8),
    st.random_module(),
)
def test_tree_sample_batch_in_bounds(vecs_and_queries, k, _rand):
    """Test that exact sampling retrieves vectors in the specified caps."""

    vecs, query_centers, max_cos_distances = vecs_and_queries
    vecs_set = set(tuple(vec) for vec in vecs)
    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=k, iters=4, max_leaf_size=2 * k)
    tree.split_rec()
    tree.prepare_for_queries()

    max_cos_distances = max_cos_distances.astype(np.float32)
    samples = tree.sample_in_caps(query_centers, max_cos_distances)

    for i in range(len(query_centers)):
        sample_idx = samples[i]
        if sample_idx != -1:
            sample_vec = tree[sample_idx]["clip_embedding"]
            assert cosine_distance(sample_vec, query_centers[i]) <= max_cos_distances[i]
            assert tuple(sample_vec) in vecs_set
        else:
            assert (
                cosine_distance_many_to_one(vecs, query_centers[i]).min()
                > max_cos_distances[i]
            )


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(st.tuples(st.integers(1, 256), st.integers(2, 4))),
    st.integers(3, 8),
    st.integers(1, 4),
    st.random_module(),
)
def test_tree_sample_batch_approx_finds_all(vecs, k, batch_size, _rand):
    """Test that approximate sampling finds all vectors in the tree when sampling from tiny caps
    centered on them."""

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=k, max_leaf_size=2 * k, iters=4)
    tree.split_rec()
    tree.prepare_for_queries()

    # Need a high tolerance since otherwise it falls back to exact sampling which defeats the
    # purpose of this test
    tol = 0.1

    vec_idx = 0

    while vec_idx < len(vecs):
        vecs_this_batch = vecs[vec_idx : vec_idx + batch_size]
        samples = tree.sample_in_caps_approx(
            vecs_this_batch, np.repeat(tol, len(vecs_this_batch))
        )

        for i in range(len(vecs_this_batch)):
            sample_idx = samples[i]
            assert sample_idx != -1
            sample_vec = tree[sample_idx]["clip_embedding"]
            vec = vecs_this_batch[i]
            distance = cosine_distance(sample_vec, vec)
            assert distance <= tol
            # Hypothesis sometimes generates vectors that are *very* close together, so even with a
            # very small cap we can sample a different vector.
            # np.testing.assert_array_equal(sample["clip_embedding"], vec)
        vec_idx += len(vecs_this_batch)


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    st.tuples(st.integers(2, 4), st.integers(1, 4)).flatmap(
        lambda d_and_num_queries: st.tuples(
            _unit_vecs(st.tuples(st.integers(1, 256), st.just(d_and_num_queries[0]))),
            _unit_vecs(
                st.tuples(st.just(d_and_num_queries[1]), st.just(d_and_num_queries[0]))
            ),
            hyp_np.arrays(
                np.float64, (d_and_num_queries[1],), elements=st.floats(0.01, 2.0)
            ),
        )
    ),
    st.integers(3, 8),
    st.random_module(),
)
def test_tree_sample_batch_approx_in_bounds(vecs_and_queries, k, _rand):
    """Test that approximate sampling retrieves vectors in the specified caps."""

    vecs, query_centers, max_cos_distances = vecs_and_queries
    vecs_set = set(tuple(vec) for vec in vecs)
    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=k, iters=4, max_leaf_size=2 * k)
    tree.split_rec()
    tree.prepare_for_queries()

    max_cos_distances = max_cos_distances.astype(np.float32)
    samples = tree.sample_in_caps_approx(query_centers, max_cos_distances)

    for i in range(len(query_centers)):
        sample_idx = samples[i]
        if sample_idx != -1:
            sample_vec = tree[sample_idx]["clip_embedding"]
            assert (
                cosine_distance(sample_vec, query_centers[i])
                <= max_cos_distances[i] + 0.01
            )
            assert tuple(sample_vec) in vecs_set
        else:
            assert (
                cosine_distance_many_to_one(vecs, query_centers[i]).min()
                > max_cos_distances[i]
            )


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[
        hyp.HealthCheck.data_too_large,
        hyp.HealthCheck.filter_too_much,
    ],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(1, 256), st.integers(2, 4)),
    )
)
def test_tree_save_load(vecs):
    """Test that saving and loading a tree preserves the tree structure and vectors."""

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=16, max_leaf_size=8)
    tree.split_rec()
    tree_vecs_set = set(tuple(row["clip_embedding"]) for row in tree.items())

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "tree_save_test"
        tree.save_to_disk(temp_path)
        tree2 = CapTree.load_from_disk(temp_path)

        tree2._check_invariants()
        tree2_vecs_set = set(tuple(row["clip_embedding"]) for row in tree2.items())
        assert tree_vecs_set == tree2_vecs_set


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
)
@given(
    st.integers(1, 256).flatmap(
        lambda n_vecs: st.tuples(
            _unit_vecs(st.tuples(st.integers(n_vecs, 256), st.integers(2, 4))),
            st.sets(st.integers(0, n_vecs - 1), min_size=1, max_size=n_vecs),
        )
    ),
    st.booleans(),
    st.random_module(),
)
def test_tree_delete_idxs(vecs_and_idxs, do_split, _rand):
    """Test that deleting indices from a tree leaves a valid tree with the correct stuff removed."""
    vecs, idxs_to_delete = vecs_and_idxs
    vecs_set = set(tuple(vec) for vec in vecs)
    assert len(vecs) == len(vecs_set)

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=3, iters=4, max_leaf_size=4)

    if do_split:
        tree.split_rec()
    tree.prepare_for_queries()

    # The meaning of indices passed to delete_idxs is relative to their order in the tree, not
    # their order in any original dataset, so to predict which rows should be deleted we need to
    # look at the dataset after the tree's built.
    vecs_from_tree = tree.dset[:]["clip_embedding"]
    vecs_set_after_delete = vecs_set - set(
        tuple(vecs_from_tree[i]) for i in idxs_to_delete
    )

    assert len(vecs_set_after_delete) == len(vecs) - len(idxs_to_delete)

    tree._check_invariants()
    should_delete = tree.delete_idxs(np.array(list(idxs_to_delete)))

    if not should_delete:
        assert len(tree) == len(vecs) - len(idxs_to_delete)
        tree._check_invariants()
        vecs_from_modified_tree = set(
            tuple(row["clip_embedding"]) for row in tree.items()
        )
        assert len(vecs_from_modified_tree) == len(tree)
        assert vecs_from_modified_tree == vecs_set_after_delete
    else:
        assert len(idxs_to_delete) == len(vecs)
