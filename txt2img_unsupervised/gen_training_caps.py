"""Generate training data for the CLIP cap conditioned version of the model.
The input to this process is a collection of images and their computed CLIP embeddings, the output
is a collection of pairs of spherical caps and images, with the images' CLIP embeddings being
contained within the caps. The model is later trained to predict images based on caps. The goal
here is that at inference time, users can generate images with the constraint that the images' CLIP
embeddings will be within a cap.
"""

import argparse
import hypothesis as hyp
import hypothesis.extra.numpy as hyp_np
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datetime import timedelta
from functools import partial
from hypothesis import given, strategies as st
from infinidata import TableView
from pathlib import Path
from tqdm import tqdm

from .spherical_space_partitioning import _unit_vecs, CapTree


@partial(jax.jit, inline=True)
def slerp(p1, p2, t):
    """Spherical linear interpolation between two points on the unit sphere. Return ~0 if
    p1 == -p2. Otherwise returns a unit vector."""
    assert p1.shape == p2.shape
    assert len(p1.shape) == 1
    omega = jnp.arccos(jnp.clip(jnp.dot(p1, p2), -1.0, 1.0))

    def linear_interp(_):
        return (1.0 - t) * p1 + t * p2

    def spherical_interp(omega):
        sin_omega = jnp.sin(omega) + 1e-8
        w1 = jnp.sin((1.0 - t) * omega) / sin_omega
        w2 = jnp.sin(t * omega) / sin_omega
        return w1 * p1 + w2 * p2

    interp = jax.lax.cond(
        jnp.isclose(omega, 0.0, rtol=0, atol=1e-6),
        linear_interp,
        spherical_interp,
        omega,
    )
    assert interp.shape == p1.shape
    return interp


@hyp.settings(deadline=timedelta(seconds=5), max_examples=500)
@given(
    _unit_vecs(st.tuples(st.just(2), st.integers(2, 16))),
    st.floats(0, 1),
)
def test_slerp_on_sphere(vecs, t):
    """Test that slerp returns a point on the unit sphere."""
    p1, p2 = vecs
    hyp.assume(not np.allclose(p1, -p2))
    slerp_vec = np.array(slerp(p1, p2, t))
    assert np.isfinite(slerp_vec).all()
    np.testing.assert_allclose(np.linalg.norm(slerp_vec), 1.0, rtol=0, atol=1e-3)


@hyp.settings(deadline=timedelta(seconds=5), max_examples=500)
@given(
    st.integers(2, 16).flatmap(
        lambda dim: st.tuples(
            _unit_vecs(st.tuples(st.just(1), st.just(dim))),
            _unit_vecs(st.tuples(st.just(1), st.just(dim))),
        )
    ),
    st.booleans(),
)
def test_slerp_endpoints(pts, high):
    """Test that slerp returns the correct endpoint when t is 0 or 1."""
    p1, p2 = pts[0][0], pts[1][0]
    hyp.assume(not np.allclose(p1, -p2))
    slerp_vec = np.array(slerp(p1, p2, 1.0 if high else 0.0))
    np.testing.assert_allclose(slerp_vec, p2 if high else p1, rtol=0, atol=0.01)


def gen_slerp_pt(p1, p2, rng):
    t = jax.random.uniform(rng, shape=())
    return slerp(p1, p2, t)


def gen_slerp_pts(ps, rng):
    assert len(ps.shape) == 2
    assert ps.shape[0] % 2 == 0
    p1s = ps[::2]
    p2s = ps[1::2]
    assert p1s.shape == p2s.shape
    rngs = jax.random.split(rng, len(p1s))
    return jax.vmap(gen_slerp_pt, in_axes=(0, 0, 0))(p1s, p2s, rngs)


def gen_max_cos_distance(rng):
    """Generate maximum cosine distances for caps, biased towards smaller values since those are
    more informative to the model and more representative of what a user would want to specify.
    """
    rand = jax.random.uniform(rng, shape=())
    # 95% of probability mass goes to U[0, 1], 5% goes to U[1, 2].
    rescaled = jax.lax.cond(
        rand < 0.95, lambda r: r / 0.95, lambda r: 1 + (r - 0.95) / 0.05, rand
    )
    return rescaled


@jax.jit
def gen_slerp_caps(ps, rng):
    assert ps.shape[0] % 2 == 0

    interp_rng, max_cos_distance_rng = jax.random.split(rng, 2)
    caps_to_gen = ps.shape[0] // 2

    interp_pts = gen_slerp_pts(ps, interp_rng)
    max_cos_distances = jax.vmap(gen_max_cos_distance)(
        jax.random.split(max_cos_distance_rng, caps_to_gen)
    )
    assert interp_pts.shape[0] == caps_to_gen == max_cos_distances.shape[0]
    return interp_pts, max_cos_distances


def gen_training_examples_from_tree(
    captree,
    rng,
    batch_size,
    stop_after=None,
    density_estimate_samples=512,
    with_replacement=False,
):
    """Iterate over a tree and sample caps, then sample images within those caps. Modifies the
    captree, removing images as it goes."""
    assert batch_size > 0
    assert batch_size % 2 == 0
    if with_replacement:
        assert stop_after is not None
    sampled_rows = []

    with tqdm(
        unit="caps", total=len(captree) - 2 if stop_after is None else stop_after
    ) as pbar:
        while len(captree) > 2 and (stop_after is None or pbar.n < stop_after):
            dset_idx = 0

            rng, shuf_rng = jax.random.split(rng)
            dset_shuffle_mapping = np.array(
                jax.random.choice(
                    shuf_rng, len(captree), shape=(len(captree),), replace=False
                )
            ).astype(np.int64)
            shuffle_inverse = np.argsort(dset_shuffle_mapping)
            shuffled_dset = captree.dset_thin.new_view(dset_shuffle_mapping)

            # We track stuff sampled this iteration through the dataset separately from stuff that
            # is part of the global sample, since deduplication is done at the end of each
            # iteration.
            sampled_idxs_this_run = []
            sampled_cap_centers_this_run = []
            sampled_max_cos_distances_this_run = []

            for batch in shuffled_dset.batch_iter(
                batch_size=batch_size, drop_last_batch=False, readahead=1
            ):
                embeds = batch["clip_embedding"]
                if len(embeds) % 2 != 0:
                    embeds = embeds[:-1]
                this_batch_size = len(embeds)

                rng, rng2 = jax.random.split(rng)
                this_cap_centers, this_max_cos_distances = gen_slerp_caps(embeds, rng2)
                this_cap_centers, this_max_cos_distances = np.array(
                    this_cap_centers
                ), np.array(this_max_cos_distances)

                # The vectors from the dset that we used to generate the caps
                cap_sources_a = shuffle_inverse[
                    np.arange(this_batch_size)[::2] + dset_idx
                ]
                cap_sources_b = shuffle_inverse[
                    np.arange(this_batch_size)[1::2] + dset_idx
                ]
                assert (
                    cap_sources_a.shape
                    == cap_sources_b.shape
                    == (embeds.shape[0] // 2,)
                )

                this_sampled_idxs = captree.sample_in_caps_approx(
                    this_cap_centers,
                    this_max_cos_distances,
                    density_estimate_samples=density_estimate_samples,
                )

                empty_cap_mask = this_sampled_idxs == -1

                # Did we sample a vector that was used to generate the corresponding cap?
                sample_self_mask = (this_sampled_idxs == cap_sources_a) | (
                    this_sampled_idxs == cap_sources_b
                )
                assert sample_self_mask.shape == empty_cap_mask.shape

                usable_mask = ~empty_cap_mask & ~sample_self_mask

                _unique_idxs, unique_idxs_idxs = np.unique(
                    this_sampled_idxs[usable_mask], return_index=True
                )

                sampled_idxs_this_run.append(
                    this_sampled_idxs[usable_mask][unique_idxs_idxs]
                )
                sampled_cap_centers_this_run.append(
                    this_cap_centers[usable_mask][unique_idxs_idxs]
                )
                sampled_max_cos_distances_this_run.append(
                    this_max_cos_distances[usable_mask][unique_idxs_idxs]
                )
                pbar.update(len(unique_idxs_idxs))
                pbar.set_postfix(
                    {
                        "hit%": np.mean(usable_mask) * 100,
                        "empty%": np.mean(empty_cap_mask) * 100,
                        "self%": np.mean(sample_self_mask) * 100,
                        "batch dup%": (
                            100
                            * (np.count_nonzero(usable_mask) - len(unique_idxs_idxs))
                            / np.count_nonzero(usable_mask)
                        )
                        if np.count_nonzero(usable_mask) > 0
                        else "NaN",
                        "median max cos distance": np.median(
                            this_max_cos_distances[usable_mask][unique_idxs_idxs]
                        ),
                    }
                )
                dset_idx += this_batch_size

                if stop_after is not None and pbar.n >= stop_after:
                    break
            tqdm.write(
                "Complete pass through dataset, reshuffling"
                if with_replacement
                else "Completed pass through dataset, removing sampled images and reshuffling."
            )
            sampled_idxs_this_run_arr = np.concatenate(sampled_idxs_this_run)
            sampled_cap_centers_this_run_arr = np.concatenate(
                sampled_cap_centers_this_run
            )
            sampled_max_cos_distances_this_run_arr = np.concatenate(
                sampled_max_cos_distances_this_run
            )
            assert (
                sampled_idxs_this_run_arr.shape[0]
                == sampled_cap_centers_this_run_arr.shape[0]
                == sampled_max_cos_distances_this_run_arr.shape[0]
            )

            unique_idxs, unique_idxs_idxs = np.unique(
                sampled_idxs_this_run_arr, return_index=True
            )

            unique_pct = (
                f"{100 * len(unique_idxs_idxs) / len(sampled_idxs_this_run_arr):.2f}"
                if len(sampled_idxs_this_run_arr) > 0
                else "NaN"
            )
            tqdm.write(
                f"Unique results: {len(unique_idxs_idxs)} = {unique_pct}% of total"
            )
            sampled_rows_orig = captree.dset.new_view(unique_idxs)
            sampled_rows_merged = []
            for rows_orig_batch_idx, rows_orig_batch in enumerate(
                tqdm(
                    sampled_rows_orig.batch_iter(
                        batch_size=batch_size,
                        drop_last_batch=False,
                        readahead=8,
                        threads=8,
                    ),
                    total=len(sampled_rows_orig) // batch_size,
                    desc="building merged tableview",
                )
            ):
                start_idx = rows_orig_batch_idx * batch_size
                stop_idx = start_idx + len(rows_orig_batch["clip_embedding"])
                cap_centers_this_batch = sampled_cap_centers_this_run_arr[
                    unique_idxs_idxs
                ][start_idx:stop_idx]
                cap_max_cos_distances_this_batch = (
                    sampled_max_cos_distances_this_run_arr[unique_idxs_idxs][
                        start_idx:stop_idx
                    ]
                )
                assert (
                    rows_orig_batch["clip_embedding"].shape[0]
                    == cap_centers_this_batch.shape[0]
                    == cap_max_cos_distances_this_batch.shape[0]
                )
                new_dict = {
                    "cap_center": cap_centers_this_batch,
                    "cap_max_cos_distance": cap_max_cos_distances_this_batch,
                } | rows_orig_batch
                sampled_rows_merged.append(TableView(new_dict))

            if len(sampled_rows_merged) > 0:
                sampled_rows.append(TableView.concat(sampled_rows_merged))
            if not with_replacement:
                captree.delete_idxs(unique_idxs)
            tqdm.write(f"Rows remaining in captree: {len(captree)}")
            pbar.n = sum(len(tv) for tv in sampled_rows)
            pbar.refresh()

    return TableView.concat(sampled_rows)


def save_training_data(dset, out_path):
    """Save the generated data to a parquet file."""
    df = pd.DataFrame([dset[0]])
    pq_schema = pa.Schema.from_pandas(df)

    with open(out_path, "wb") as f:
        pq_writer = pq.ParquetWriter(f, pq_schema, compression="zstd")
        with tqdm(total=len(dset), desc="Writing parquet", unit="rows") as pbar:
            for batch in dset.batch_iter(
                batch_size=8192, drop_last_batch=False, readahead=8, threads=8
            ):
                rows = len(batch["clip_embedding"])
                df_rows = []
                for i in range(rows):
                    df_rows.append({k: v[i] for k, v in batch.items()})
                pq_writer.write_table(
                    pa.Table.from_pandas(pd.DataFrame(df_rows), schema=pq_schema)
                )
                pbar.update(rows)
        pq_writer.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for the CLIP cap conditioned model."
    )
    parser.add_argument("--tree-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--density-estimate-samples", type=int, default=512)
    parser.add_argument("--replacement", action="store_true")
    parser.add_argument("--no-save-cache", action="store_false", dest="save_cache")
    args = parser.parse_args()

    tree = CapTree.load_from_disk(args.tree_path, save_cache=args.save_cache)
    if args.seed is not None:
        rng = jax.random.PRNGKey(args.seed)
    else:
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))
    if args.out.exists():
        print(f"Output path {args.out} exists, exiting")
        exit(1)

    caps_dset = gen_training_examples_from_tree(
        tree,
        rng,
        args.batch_size,
        stop_after=args.stop_after,
        density_estimate_samples=args.density_estimate_samples,
        with_replacement=args.replacement,
    )
    save_training_data(caps_dset, args.out)


if __name__ == "__main__":
    main()
