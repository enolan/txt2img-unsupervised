import argparse
import hypothesis as hyp
import hypothesis.extra.numpy as hyp_np
import jax
import jax.numpy as jnp
import json
import numpy as np
import tqdm
from datasets import Dataset
from datetime import timedelta
from functools import partial
from hypothesis import given, strategies as st
from pathlib import Path
from tqdm import tqdm

from txt2img_unsupervised.load_pq_dir import load_pq_dir

def find_k_means(dset, batch_size, k, iters):
    # Mini-batch k-means, see "Web-Scale K-Means Clustering" by D. Sculley 2010
    # This is a spherical k-means, using cosine similarity inside of euclidean distance. We assume
    # all vectors have unit norm.

    # We assume dset is shuffled, and in numpy mode
    assert len(dset) > k
    assert batch_size > k

    # Initialize centroids
    tqdm.write(f"Initializing centroids with {k} random samples")
    centroids = dset[:k]["clip_embedding"]
    tqdm.write(f"Skipping initial samples, {len(dset) - k} remaining")

    # To make batch sizes even, we drop the last batch if it's smaller than batch_size, since
    # shuffling means we can see every example either way.
    drop_last_batch = len(dset) > batch_size
    # Skip initial samples
    dset_iter = dset.select(range(k, len(dset))).iter(
        batch_size=batch_size, drop_last_batch=drop_last_batch
    )

    per_center_counts = np.zeros(k, dtype=np.int32)

    with tqdm(total=iters, desc="k-means iterations", leave=None) as pbar:
        while pbar.n < iters:
            try:
                batch = next(dset_iter)
            except StopIteration:
                dset = dset.shuffle()
                dset_iter = dset.iter(
                    batch_size=batch_size, drop_last_batch=drop_last_batch
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

    with tqdm(total=len(dset), desc="Assigning centroids", leave=None) as pbar:
        for batch_idx, batch in enumerate(dset.iter(batch_size=batch_size)):
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
    vecs = np.array([[1, -0.2], [1, 0.1], [-0.1, 1], [0.1, 1]])
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    dset = Dataset.from_dict({"clip_embedding": vecs}).with_format("numpy")

    assignments, max_distances = assign_centroids(dset, centroids, 2)

    assert assignments == [[0, 1], [2, 3]]

    distance_0 = cosine_distance(centroids[0], vecs[0])
    distance_1 = cosine_distance(centroids[1], vecs[2])
    assert np.isclose(max_distances[0], distance_0)
    assert np.isclose(max_distances[1], distance_1)


def cosine_distance(x, y):
    """Cosine distance between x and y. Assumes x and y are unit vectors."""
    return 1 - jnp.dot(x, y)


@jax.jit
def cosine_distance_many_to_one(xs, y):
    """Cosine distance between each x in xs and y. Assumes xs and y are unit vectors."""
    return 1 - jnp.dot(xs, y)


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
        self.found_duplicates = found_duplicates

    def __len__(self):
        return self.len

    def split_once(self):
        """Split this cap into children."""

        # If we don't flatten the indices the process uses gobs of RAM and OOMs. This doesn't make
        # any sense any is probably related to a bug in datasets. If we do flatten the indices it
        # makes a ton of cache files which need to be deleted afterward.
        # num_proc is an asspulled guess. The correct value is the smallest that saturates SSD
        # bandwidth.
        self.dset = self.dset.flatten_indices(num_proc=8)
        dset_thin = self.dset.select_columns(["clip_embedding"])
        centroids = find_k_means(dset_thin, self.batch_size, self.k, self.iters)
        assignments, max_distances = assign_centroids(
            dset_thin, centroids, self.batch_size
        )
        self.children = [
            CapTree(
                self.dset.select(assignments[i]),
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
                self.dset = Dataset.from_dict(dset_dict).with_format("numpy")
                tqdm.write(f"New leaf size: {len(self.dset)}")

    def _to_summary_inner(self):
        """Make a python dict summary representation of the tree for visualization."""
        return {
            "max_cos_distance": float(self.max_cos_distance),
            "size": len(self),
        } | (
            {"children": [child._to_summary_inner() for child in self.children]}
            if len(self.children) > 0
            else {}
        )

    def to_summary(self):
        """Make a JSON summary representation of the tree for visualization."""
        return json.dumps(
            {
                "structure": self._to_summary_inner(),
                "depth": self.depth(),
                "total_vectors": len(self),
                "k": self.k,
                "max_leaf_size": self.max_leaf_size(),
                "min_leaf_size": self.min_leaf_size(),
                "mean_depth": np.mean(list(self.leaf_depths())),
            },
            indent=2,
        )

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
        else:
            assert self.dset is not None
            assert self.len == len(self.dset)
            assert self.center.shape == self.dset[0]["clip_embedding"].shape

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
                self.dset["clip_embedding"], cap_center
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
        # The primary purpose of this function is testing, so we get everything from the leaves.
        if len(self.children) == 0:
            for row in self.dset:
                yield row
        else:
            for child in self.children:
                for row in child.items():
                    yield row


@hyp.settings(
    suppress_health_check=[hyp.HealthCheck.data_too_large],
    deadline=timedelta(seconds=30),
    max_examples=500,
)
@given(
    hyp_np.arrays(
        np.float32,
        st.tuples(st.integers(1, 512), st.integers(2, 4)),
        elements=st.floats(-1.0, 1.0, width=32),
        unique=True,
    )
)
def test_tree_invariants(vecs):
    """Test that the tree invariants hold for any set of vectors."""
    hyp.assume(np.all(np.linalg.norm(vecs, axis=1) > 0))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    vecs_set = set(tuple(vec) for vec in vecs)
    hyp.assume(len(vecs_set) == len(vecs))

    dset = Dataset.from_dict({"clip_embedding": vecs}).with_format("numpy")
    tree = CapTree(dset, batch_size=32, k=4, iters=64)
    tree._check_invariants()
    tree.split_rec()
    tree._check_invariants()

    # check that the elements in the tree are the ones we put in
    tree_vecs = set(tuple(row["clip_embedding"]) for row in tree.items())
    assert tree_vecs == vecs_set


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
    args = parser.parse_args()

    dset_all = load_pq_dir(args.pq_dir)
    dset_split = dset_all.train_test_split(test_size=0.01, seed=19900515)
    dset = dset_split["train"]

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
        dset = dset.filter(
            lambda name: name not in blacklist, input_columns="name", num_proc=16
        )
        print(f"Removed {pre_count - len(dset)} blacklisted images")

    if args.subset is not None:
        dset = dset.select(range(args.subset))

    tree = CapTree(
        dset,
        args.batch_size,
        args.k,
        args.k_means_iters,
        True if args.write_dup_blacklist is not None else False,
    )
    tree.split_rec()

    if args.paranoid:
        # This is a pretty slow check, but I don't 100% trust the code
        tree._check_invariants()

    # minimum possible depth, given branching factor is k and leaves can have at most k^2 vectors
    min_depth = int(np.ceil(np.log(len(dset)) / np.log(args.k**2)))
    print(f"Tree depth: {tree.depth()}, minimum possible depth: {min_depth}")

    if args.summary_file is not None:
        args.summary_file.write_text(tree.to_summary())

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


if __name__ == "__main__":
    main()