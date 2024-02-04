"""Generate training data for the CLIP cap conditioned version of the model.
The input to this process is a collection of images and their computed CLIP embeddings, the output
is a collection of pairs of spherical caps and images, with the images' CLIP embeddings being
contained within the caps. The model is later trained to predict images based on caps. The goal
here is that at inference time, users can generate images with the constraint that the images' CLIP
embeddings will be within a cap.
"""

import hypothesis as hyp
import hypothesis.extra.numpy as hyp_np
import jax
import jax.numpy as jnp
import numpy as np

from datetime import timedelta
from functools import partial
from hypothesis import given, strategies as st
from tqdm import tqdm

from txt2img_unsupervised.spherical_space_partitioning import _unit_vecs


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


def gen_training_examples_from_tree(captree, rng, batch_size, stop_after=None):
    """Iterate over a tree and sample caps, then sample images within those caps."""
    assert batch_size > 0
    assert batch_size % 2 == 0

    cap_centers = []
    max_cos_distances = []
    sampled_idxs = []

    dset_idx = 0
    with tqdm(
        unit="caps", total=len(captree) // 2 if stop_after is None else stop_after
    ) as pbar:
        for batch in captree.dset_thin.shuffle().batch_iter(
            batch_size=batch_size, drop_last_batch=False
        ):
            embeds = batch["clip_embedding"]

            rng, rng2 = jax.random.split(rng)
            this_cap_centers, this_max_cos_distances = gen_slerp_caps(embeds, rng2)
            this_cap_centers, this_max_cos_distances = np.array(
                this_cap_centers
            ), np.array(this_max_cos_distances)

            this_sampled_idxs = (
                captree.sample_in_caps_approx(
                    this_cap_centers,
                    this_max_cos_distances,
                    density_estimate_samples=512,
                )
                + dset_idx
            )

            match_mask = this_sampled_idxs != -1

            cap_centers.append(this_cap_centers[match_mask])
            max_cos_distances.append(this_max_cos_distances[match_mask])
            sampled_idxs.append(this_sampled_idxs[match_mask])
            pbar.update(np.count_nonzero(match_mask))
            pbar.set_postfix({"hit rate": np.mean(match_mask)})
            dset_idx += batch_size

            if stop_after is not None and pbar.n >= stop_after:
                break

    cap_centers, max_cos_distances, sampled_idxs = (
        np.concatenate(cap_centers),
        np.concatenate(max_cos_distances),
        np.concatenate(sampled_idxs),
    )
    print(f"Unique results: {len(np.unique(sampled_idxs))} / {len(sampled_idxs)}")
    return cap_centers, max_cos_distances, sampled_idxs
