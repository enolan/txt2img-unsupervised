import argparse
import base64
import datasets
import hypothesis as hyp
import hypothesis.extra.numpy as hyp_np
import infinidata
import jax
import jax.numpy as jnp
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

from datasets import Dataset
from datetime import datetime, timedelta
from functools import partial
from hypothesis import given, strategies as st
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib import tenumerate
from txt2img_unsupervised.load_pq_dir import (
    load_pq_dir_to_infinidata,
    load_pq_to_infinidata,
)


def find_k_means(dset, batch_size, k, iters):
    # Mini-batch k-means, see "Web-Scale K-Means Clustering" by D. Sculley 2010
    # This is a spherical k-means, using cosine similarity inside of euclidean distance. We assume
    # all vectors have unit norm.

    assert len(dset) > k
    assert batch_size > k

    # Initialize centroids
    tqdm.write(f"Initializing centroids with {k} random samples")
    centroids = dset.shuffle()[:k]["clip_embedding"]

    # To make batch sizes even, we drop the last batch if it's smaller than batch_size, since
    # shuffling means we can see every example either way.
    drop_last_batch = len(dset) > batch_size

    dset_iter = dset.shuffle().batch_iter(
        batch_size=batch_size, drop_last_batch=drop_last_batch, threads=8, readahead=8
    )

    per_center_counts = np.zeros(k, dtype=np.int32)

    with tqdm(total=iters, desc="k-means iterations", leave=None) as pbar:
        while pbar.n < iters:
            try:
                batch = next(dset_iter)
            except StopIteration:
                dset_iter = dset.shuffle().batch_iter(
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
                per_center_counts[nearest_centroids[i]] += 1
                nearest_centroid = nearest_centroids[i]
                lr = 1 / per_center_counts[nearest_centroid]
                centroids[nearest_centroid] = (1 - lr) * centroids[
                    nearest_centroid
                ] + lr * batch["clip_embedding"][i]
                centroids[nearest_centroid] /= np.linalg.norm(
                    centroids[nearest_centroid]
                )

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


def assign_centroids(dset, centroids, batch_size):
    """Assign each CLIP embedding to its nearest centroid and compute the greatest cosine distance
    for each centroid."""
    assert centroids.shape[1] == dset[0]["clip_embedding"].shape[0]

    tqdm.write(f"Assigning {len(dset)} examples to {len(centroids)} centroids.")

    @partial(jax.jit, donate_argnums=(2,))
    def nearest_centroids_and_max_distances(xs, centroids, max_distances):
        nearest_centroids, distances = jax.vmap(
            find_nearest_centroid, in_axes=(0, None)
        )(xs, centroids)

        # (len(xs), len(centroids)) array of distances or zeros
        distances = jnp.where(
            jnp.arange(len(centroids)) == nearest_centroids[:, None],
            distances[:, None],
            0,
        )
        assert distances.shape == (len(xs), len(centroids))

        # tack on the existing max distances
        distances = jnp.concatenate((distances, max_distances[None, :]), axis=0)
        assert distances.shape == (len(xs) + 1, len(centroids))

        # take the max distance for each centroid
        max_distances = jnp.max(distances, axis=0)
        assert max_distances.shape == (len(centroids),)

        return nearest_centroids, max_distances

    max_distances = jnp.zeros(len(centroids), dtype=np.float32)
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

            nearest_centroids, max_distances = nearest_centroids_and_max_distances(
                batch["clip_embedding"], centroids, max_distances
            )

            assert nearest_centroids.shape == (this_batch_size,)
            assert max_distances.shape == (len(centroids),)

            nearest_centroids = np.array(nearest_centroids)

            for i in range(this_batch_size):
                centroid_assignments[nearest_centroids[i]].append(
                    batch_idx * batch_size + i
                )
            pbar.update(this_batch_size)
    max_distances = np.array(max_distances)
    return centroid_assignments, max_distances


def test_assign_centroids():
    """Test that assign_centroids works."""
    centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
    vecs = np.array([[1, -0.2], [1, 0.1], [-0.1, 1], [0.1, 1]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dset = infinidata.TableView({"clip_embedding": vecs})

    assignments, max_distances = assign_centroids(dset, centroids, 2)

    assert assignments == [[0, 1], [2, 3]]

    distance_0 = cosine_distance(centroids[0], vecs[0])
    distance_1 = cosine_distance(centroids[1], vecs[2])
    assert np.isclose(max_distances[0], distance_0)
    assert np.isclose(max_distances[1], distance_1)


def cosine_distance(x, y):
    """Cosine distance between x and y. Assumes x and y are unit vectors."""
    return 1 - jnp.dot(x, y)


@partial(jax.jit, inline=True)
def cosine_distance_many_to_one(xs, y):
    """Cosine distance between each x in xs and y. Assumes xs and y are unit vectors."""
    return 1 - jnp.dot(xs, y)


def caps_overlap(center1, max_cos_distance1, center2, max_cos_distance2):
    """Check if two caps overlap."""
    return (
        cosine_distance(center1, center2)
        <= max_cos_distance1 + max_cos_distance2 + CapTree.EPSILON
    )


def caps_overlap_v(center1, max_cost_distance1, centers, max_cos_distances):
    """Check if a cap overlaps with any of a set of caps. Returns an array of bools."""
    return jax.vmap(caps_overlap, in_axes=(None, None, 0, 0))(
        center1, max_cost_distance1, centers, max_cos_distances
    )


@jax.jit
def sample_overlapping_caps_weighted(
    rng, center1, max_cos_distance1, centers, max_cos_distances, weights
):
    """Given a query cap and a set of target caps plus a set of weights for the targets, sample
    from the subset of the targets that overlap the query with probability proportional to the
    weights."""
    assert len(centers) == len(max_cos_distances) == len(weights)
    assert len(centers) > 0

    overlapping_caps_mask = caps_overlap_v(
        center1, max_cos_distance1, centers, max_cos_distances
    )
    assert overlapping_caps_mask.shape == (len(centers),)

    logits = jnp.where(overlapping_caps_mask, jnp.log(weights), -1e6)
    assert logits.shape == (len(centers),)

    selected = jax.random.categorical(rng, logits)
    assert selected.shape == ()
    return selected, overlapping_caps_mask


def find_nearest_centroid(x, centroids):
    """Find the nearest centroid to x."""
    distances = jax.vmap(cosine_distance, in_axes=(None, 0))(x, centroids)
    nearest = jnp.argmin(distances)
    return nearest, distances[nearest]


class CapTree:
    """A tree of spherical caps containing unit vectors at the leaves. We split a cap into k
    children by running k-means on the unit vectors in the cap. Each centroid and the vectors
    assigned to it becomes a child cap."""

    # I'd really love to live in a world where "what is the dot product of x and y?" is a question
    # with only one answer, but alas we do not live in that world.
    EPSILON = 0.005

    def __init__(
        self,
        dset,
        batch_size,
        k,
        iters,
        dup_check=False,
        center=None,
        max_cos_distance=2.0,
        found_duplicates=[],
    ):
        assert len(dset) > 0, "CapTree must be initialized with a non-empty dataset"
        self.dset = dset
        self.len = len(dset)
        self.batch_size = batch_size
        self.k = k
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

    def __len__(self):
        return self.len

    def split_once(self):
        """Split this cap into children."""

        dset_thin = self.dset.select_columns({"clip_embedding"})
        centroids = find_k_means(dset_thin, self.batch_size, self.k, self.iters)
        assignments, max_distances = assign_centroids(
            dset_thin, centroids, self.batch_size
        )
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
        self.child_cap_centers = np.array([child.center for child in self.children])
        self.child_cap_max_cos_distances = np.array(
            [child.max_cos_distance for child in self.children]
        )

        self.dset = None

        if len(self.children) == 1:
            # There may be very rare cases where this isn't caused by duplicate vectors, but in
            # general it is. It happens when there are more than k^2 duplicates and causes an
            # infinite loop if not caught.
            tqdm.write("found node with only one child, probably duplicate vectors")
            self.children[0]._check_for_duplicates(force=True)

    def split_rec(self):
        """Split this cap and all children recursively until each leaf has at most k^2 vectors."""
        assert self.children == [], "Can only split once"

        if len(self) > self.k**2:
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
            for row in tqdm(self.dset, desc="Deduplicating", leave=None):
                k = row["clip_embedding"].tobytes()
                if k in vecs_dict:
                    vecs_dict[k].append(row)
                else:
                    vecs_dict[k] = [row]
            if len(self.dset) > len(vecs_dict):
                tqdm.write(f"Found {len(self.dset) - len(vecs_dict)} duplicates")
                self.found_duplicates.extend(
                    [row["name"] for row in rows]
                    for rows in vecs_dict.values()
                    if len(rows) > 1
                )

                # Creat a new, deduplicated Dataset
                dset_dict = {}
                for col in self.dset.column_names:
                    dset_dict[col] = np.array(
                        [rows[0][col] for rows in vecs_dict.values()]
                    )
                self.dset = infinidata.TableView(dset_dict)
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
            "iters": self.iters,
            "dup_check": self.dup_check,
            "depth": self.depth(),
            "max_leaf_size": self.max_leaf_size(),
            "min_leaf_size": self.min_leaf_size(),
            "mean_depth": np.mean(list(self.leaf_depths())),
        }

    def depth(self):
        """Depth of the tree."""
        if len(self.children) == 0:
            return 1
        else:
            return 1 + max(child.depth() for child in self.children)

    def leaf_depths(self):
        """Generator of the depths of the leaves of the tree."""
        if len(self.children) == 0:
            yield 1
        else:
            for child in self.children:
                for depth in child.leaf_depths():
                    yield depth + 1

    def max_leaf_size(self):
        """Maximum number of vectors in a leaf."""
        if len(self.children) == 0:
            return len(self)
        else:
            return max(child.max_leaf_size() for child in self.children)

    def min_leaf_size(self):
        """Minimum number of vectors in a leaf."""
        if len(self.children) == 0:
            return len(self)
        else:
            return min(child.min_leaf_size() for child in self.children)

    def _check_invariants(self):
        """Check invariants of the tree."""
        assert len(self) > 0
        if len(self.children) > 0:
            assert self.dset is None
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
        else:
            assert self.dset is not None
            assert self.len == len(self.dset)
            assert self.center.shape == self.dset[0]["clip_embedding"].shape
            assert self.child_cap_centers == self.child_cap_max_cos_distances == None

        assert self.max_cos_distance <= 2
        assert np.isclose(np.linalg.norm(self.center), 1.0)
        assert self.center.dtype == np.float32

        self._check_inside_cap(self.center, self.max_cos_distance)

        for subtree in tqdm(
            self.children, leave=False, desc="Checking subtree invariants"
        ):
            assert subtree.max_cos_distance <= self.max_cos_distance + self.EPSILON
            assert (
                cosine_distance(self.center, subtree.center)
                <= self.max_cos_distance + self.EPSILON
            )
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

    def sample_in_cap(self, center, max_cos_distance):
        """Sample uniformly from the vectors in this cap that are inside the input cap."""
        if len(self.children) == 0:
            # leaf
            distances = cosine_distance_many_to_one(
                self.dset[:]["clip_embedding"], center
            )
            valid_distances_mask = distances <= max_cos_distance
            valid_idxs = np.arange(len(self.dset))[valid_distances_mask]
            if len(valid_idxs) == 0:
                return None
            else:
                random_valid_idx = np.random.randint(len(valid_idxs))
                random_idx = int(valid_idxs[random_valid_idx])
                return self.dset[random_idx]
        else:
            # inner node
            rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
            sizes = np.array([len(child) for child in self.children])
            assert np.all(sizes > 0)
            while True:
                # Loop until we eliminate all children from consideration (setting their sizes to
                # 0) or we find a sample.
                sub_rng, rng = jax.random.split(rng)
                sampled_child, overlapping_caps_mask = sample_overlapping_caps_weighted(
                    sub_rng,
                    center,
                    max_cos_distance + self.EPSILON,
                    self.child_cap_centers,
                    self.child_cap_max_cos_distances,
                    sizes,
                )
                sizes = np.where(overlapping_caps_mask, sizes, 0)
                if np.all(sizes == 0):
                    return None
                inner_sample = self.children[sampled_child].sample_in_cap(
                    center, max_cos_distance
                )
                if inner_sample is None:
                    sizes[sampled_child] = 0
                else:
                    return inner_sample
                if np.all(sizes == 0):
                    return None

    def save_to_disk(self, dir):
        """Save the tree to disk."""
        dir.mkdir(exist_ok=False, parents=True)
        summary = self.to_summary(centers=True)
        with open(dir / "structure.json", "w") as f:
            json.dump(summary, f, indent=2)

        # We concatenate all the leaves into one big dataset before saving to disk. This is a bit
        # more complicated (especially loading) but lets the parquet compression work much better.
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
        self.iters = root.iters
        self.dup_check = root.dup_check
        self.found_duplicates = root.found_duplicates

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

    @classmethod
    def load_from_disk(cls, dir):
        """Load a tree from disk."""
        with open(dir / "structure.json", "r") as f:
            summary = json.load(f)

        out = cls.__new__(cls)

        out.batch_size = summary["batch_size"]
        out.k = summary["k"]
        out.iters = summary["iters"]
        out.dup_check = summary["dup_check"]
        out.found_duplicates = []
        out.dset = None
        out.len = summary["total_vectors"]

        out._empty_from_summary(out, summary["structure"])

        dset_full = load_pq_to_infinidata(dir / "data.parquet")

        leaf_ranges = []
        leaf_idx = 0

        for leaf in out.leaves():
            leaf_ranges.append((leaf_idx, leaf_idx + len(leaf)))
            leaf_idx += len(leaf)

        assert leaf_ranges[-1][1] == len(dset_full)

        for i, leaf in enumerate(out.leaves()):
            leaf.dset = dset_full.new_view(slice(*leaf_ranges[i]))

        out._fixup_traversal_arrays()

        return out


@st.composite
def _unit_vecs(draw, shape):
    """Strategy for drawing unique unit vectors."""
    vecs = draw(
        hyp_np.arrays(
            np.float32, shape, elements=st.floats(-1.0, 1.0, width=32), unique=True
        )
    )
    hyp.assume(np.all(np.linalg.norm(vecs, axis=1) > 0))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[hyp.HealthCheck.data_too_large],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(1, 1024), st.integers(2, 4)),
    )
)
def test_tree_invariants(vecs):
    """Test that the tree invariants hold for any set of vectors."""

    vecs_set = set(tuple(vec) for vec in vecs)
    hyp.assume(len(vecs_set) == len(vecs))

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=64)
    tree._check_invariants()
    tree.split_rec()
    tree._check_invariants()

    # check that the elements in the tree are the ones we put in
    tree_vecs_set = set(tuple(row["clip_embedding"]) for row in tree.items())
    assert tree_vecs_set == vecs_set


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[hyp.HealthCheck.data_too_large],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(2, 1025), st.integers(2, 4)),
    ),
    st.floats(0, 2.0, width=32),
)
def test_tree_sample_in_bounds(vecs, max_cos_distance):
    """Test that sampling retrieves vectors in the specified cap."""

    query_center = vecs[0]
    vecs = vecs[1:]
    vecs_set = set(tuple(vec) for vec in vecs)

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=16)
    tree.split_rec()
    print(f"items: {len(tree)} max depth: {tree.depth()}")

    sample = tree.sample_in_cap(query_center, max_cos_distance)
    if sample is not None:
        assert (
            cosine_distance(sample["clip_embedding"], query_center)
            <= max_cos_distance + tree.EPSILON
        )
        assert tuple(sample["clip_embedding"]) in vecs_set


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[hyp.HealthCheck.data_too_large],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(1, 1024), st.integers(2, 4)),
    )
)
def test_tree_sample_finds_all(vecs):
    """Test that sampling retrieves all vectors in the tree when sampling from tiny caps centered
    on each vector."""

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=16)
    tree.split_rec()

    for vec in vecs:
        sample = tree.sample_in_cap(vec, 0.00001)
        distance = cosine_distance(sample["clip_embedding"], vec)
        assert distance <= 0.00001
        # Hypothesis sometimes generates vectors that are *very* close together, so even with a
        # very small cap we can sample a different vector.
        # np.testing.assert_array_equal(sample["clip_embedding"], vec)


@hyp.settings(
    deadline=timedelta(seconds=30),
    max_examples=500,
    suppress_health_check=[hyp.HealthCheck.data_too_large],
)
@given(
    _unit_vecs(
        st.tuples(st.integers(1, 1024), st.integers(2, 4)),
    )
)
def test_tree_save_load(vecs):
    """Test that saving and loading a tree preserves the tree structure and vectors."""

    dset = infinidata.TableView({"clip_embedding": vecs})
    tree = CapTree(dset, batch_size=32, k=4, iters=16)
    tree.split_rec()
    tree_vecs_set = set(tuple(row["clip_embedding"]) for row in tree.items())

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "tree_save_test"
        tree.save_to_disk(temp_path)
        tree2 = CapTree.load_from_disk(temp_path)

        tree2._check_invariants()
        tree2_vecs_set = set(tuple(row["clip_embedding"]) for row in tree2.items())
        assert tree_vecs_set == tree2_vecs_set


def main():
    parser = argparse.ArgumentParser(
        description="Build a captree from a set of txt2img-unsupervised parquet files."
    )
    parser.add_argument("--pq-dir", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--k-means-iters", type=int, default=200)
    parser.add_argument("--summary-file", type=Path, default=None)
    parser.add_argument("--write-dup-blacklist", type=Path, default=None)
    parser.add_argument("--read-dup-blacklist", type=Path, default=None)
    parser.add_argument("--paranoid", action="store_true")
    parser.add_argument("--save-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.save_dir.exists():
        print(f"Save dir {args.save_dir} exists, exiting")
        exit(1)

    get_timestamp = lambda: datetime.utcnow().isoformat()
    print(f"Time at start: {get_timestamp()}")

    dset_all = load_pq_dir_to_infinidata(args.pq_dir).shuffle(seed=19900515)
    print(f"Loaded dataset with {len(dset_all)} rows")
    dset = dset_all.new_view(slice(0, int(len(dset_all) * 0.99)))
    print(f"Train set size: {len(dset)}")
    print(f"Time after split: {get_timestamp()}")

    if args.read_dup_blacklist is not None:
        with open(args.read_dup_blacklist, "r") as f:
            found_duplicates = json.load(f)
        blacklist = [
            item
            for sublist in (names[1:] for names in found_duplicates)
            for item in sublist
        ]
        print(f"Found {len(blacklist)} blacklisted images")
        pre_count = len(dset)
        blacklist = set(blacklist)
        dset = dset.remove_matching_strings("name", blacklist)
        print(f"Removed {pre_count - len(dset)} blacklisted images")
        print(f"Time after blacklist: {get_timestamp()}")
    if args.subset is not None:
        dset = dset.new_view(slice(args.subset))

    print(f"Time after subset/before building tree: {get_timestamp()}")

    tree = CapTree(
        dset,
        args.batch_size,
        args.k,
        args.k_means_iters,
        True if args.write_dup_blacklist is not None else False,
    )
    tree.split_rec()

    print(f"Time after building tree: {get_timestamp()}")

    tree.shuffle_leaves()

    print(f"Time after shuffling: {get_timestamp()}")

    if args.paranoid:
        # This is a pretty slow check, but I don't 100% trust the code
        tree._check_invariants()

    # minimum possible depth, given branching factor is k and leaves can have at most k^2 vectors
    min_depth = int(np.ceil(np.log(len(dset)) / np.log(args.k**2)))
    print(f"Tree depth: {tree.depth()}, minimum possible depth: {min_depth}")

    if args.summary_file is not None:
        args.summary_file.write_text(json.dumps(tree.to_summary(), indent=2))

    if len(tree.found_duplicates) > 0:
        dup_set_count = len(tree.found_duplicates)
        dup_total_count = sum(len(dup_set) for dup_set in tree.found_duplicates)
        if args.write_dup_blacklist is None:
            print(
                f"Found {dup_set_count} sets of duplicates, containing {dup_total_count} total images! You should rerun and generate a blacklist with --write-dup-blacklist!"
            )
            print("Counts will be wrong and sampling will be non-uniform!")
        else:
            print(
                f"Found {len(tree.found_duplicates)} sets of duplicates, containing {dup_total_count} total images. writing to {args.write_dup_blacklist}"
            )
            with open(args.write_dup_blacklist, "w") as f:
                json.dump(tree.found_duplicates, f, indent=2)

    print(f"Time at end: {get_timestamp()}")

    print(f"Saving to {args.save_dir}")
    tree.save_to_disk(args.save_dir)


if __name__ == "__main__":
    main()
