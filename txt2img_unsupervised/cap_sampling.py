"""Functions for efficiently sampling from spherical caps and sampling spherical caps that contain
a given point."""

# Terminology note: an n-sphere is a sphere embedded in n+1 dimensions. A 1-sphere is a circle, a
# 2-sphere is a regular sphere, a 3-sphere is a sphere in 4 dimensions, etc.

# For embedding guided content generation, we need to pair each training example with a spherical
# cap that contains its embedding, so the model can learn to generate things conditioned on their
# embedding being within a given cap. It's important that the caps not tell the model anything
# beyond the fact that the embedding is inside the cap. The code in this file supports a method of
# generating the caps that is run *per-example at training time*. It's easier to prove that the
# caps don't contain any information they're not supposed to if you generate the caps in a
# preprocessing step, sampling the caps and then sampling the examples with embeddings inside them.
# But it's *way* slower and has its own bias problems (mostly that with uniformly distributed
# centers small caps are disproportionately empty, so you have to bias the selection of the cap
# centers). The code for the preprocessing version is in gen_training_caps.py and
# spherical_space_partitioning.py

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special
from functools import partial
from typing import Tuple


def log_surface_area_of_slice(d, h):
    """Calculate the log of the surface area of a (d-1)-sphere slice at height h on a d-sphere.
    N.B. A d-sphere is a sphere in d+1 dimensional sphere e.g. a regular sphere is a 2-sphere.
    """

    d = np.float64(d)
    h = np.float64(h)

    radius_slice = np.sqrt(1 - h**2)

    # Divide by zero is expected for h = 1 and h = -1.
    with np.errstate(divide="ignore"):
        log_surface_area = (
            np.log(2)
            + (d / 2) * np.log(np.pi)
            - scipy.special.loggamma(d / 2)
            + (d - 1) * np.log(radius_slice)
        )
    assert log_surface_area.dtype == np.float64
    return log_surface_area


def test_log_surface_area_of_slice():
    """Test that log_surface_area_of_slice is correct in lower dimensions."""

    # 2-sphere, slices are circles
    np.testing.assert_allclose(log_surface_area_of_slice(2, 0), np.log(2 * np.pi))
    np.testing.assert_allclose(log_surface_area_of_slice(2, 1), -np.inf)
    np.testing.assert_allclose(log_surface_area_of_slice(2, -1), -np.inf)
    np.testing.assert_allclose(
        log_surface_area_of_slice(2, 0.5), np.log(np.sqrt(0.75) * 2 * np.pi)
    )
    np.testing.assert_allclose(
        log_surface_area_of_slice(2, -0.5), np.log(np.sqrt(0.75) * 2 * np.pi)
    )

    # 3-sphere, slices are spheres
    np.testing.assert_allclose(log_surface_area_of_slice(3, 0), np.log(4 * np.pi))
    np.testing.assert_allclose(log_surface_area_of_slice(3, 1), -np.inf)
    np.testing.assert_allclose(log_surface_area_of_slice(3, -1), -np.inf)
    np.testing.assert_allclose(log_surface_area_of_slice(3, 0.5), np.log(3 * np.pi))
    np.testing.assert_allclose(log_surface_area_of_slice(3, -0.5), np.log(3 * np.pi))


def test_log_surface_area_symmetry():
    """Test that log_surface_area_of_slice is symmetric around 0."""
    negative_heights = np.linspace(-1.0, 0.0, 100, dtype=np.float64)
    positive_heights = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    negative_height_log_surface_areas = np.array(
        [log_surface_area_of_slice(767, h) for h in negative_heights]
    )
    positive_height_log_surface_areas = np.array(
        [log_surface_area_of_slice(767, h) for h in positive_heights]
    )
    np.testing.assert_allclose(
        negative_height_log_surface_areas,
        positive_height_log_surface_areas[::-1],
    )


@jax.tree_util.register_pytree_node_class
class LogitsTable:
    """Precomputed table of the log measures of slices of an arbitrary dimensional sphere. Used for
    sampling points in caps, specifically sampling the distance of the output to the center of the
    cap. I'm sure a direct way of sampling this exists but my brain is too smooth for math.
    """

    def __init__(self, d, n):
        """Generate a table of log surface areas of slices of a d-sphere for sampling from caps."""
        # I'm paranoid about floating point issues and this is a precomputation step so I don't
        # care if it's a bit slow.
        slice_heights = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        slice_logits = np.array(
            [log_surface_area_of_slice(d, h) for h in slice_heights]
        )
        slice_logits = slice_logits.astype(np.float32)

        assert slice_logits.shape == (n,)

        self.d = d
        self.buckets = n
        self.table = jnp.array(slice_logits)

    def tree_flatten(self):
        return (self.d, self.table), self.buckets

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out.d, out.table = children
        out.buckets = aux_data
        return out

    def _height_to_idx(self, h):
        """Convert a slice height to an index in the table."""
        h = jnp.float32(h)
        val = (h + 1) / 2
        idx = (val * (self.buckets - 1)).astype(jnp.int32)
        return idx

    def _idx_to_height(self, idx):
        """Convert an index in the table to a slice height (in [-1, 1])."""
        val = idx / (self.buckets - 1)
        return 2 * val - 1

    @partial(jax.jit, inline=True, static_argnames=("interpolate",))
    def sample_cap_cos_distance(self, rng, d_max, interpolate=True):
        """Sample a cosine distance from the center of a cap with a given max cosine distance,
        uniformly with respect to the surface area of the cap. If you take this distance and then
        uniformly sample a point with that distance from the center, the result will be uniformly
        distributed over the surface of the cap.
        """
        # Imagine the cap is centered on the bottom of the sphere. d_max corresponds to the height
        # of the top of the cap.
        d_max_idx = self._height_to_idx(d_max - 1)
        filtered_logits = jnp.where(
            jnp.arange(self.buckets) <= d_max_idx, self.table, -jnp.inf
        )

        if not interpolate:
            sampled_height = self._idx_to_height(
                jax.random.categorical(rng, filtered_logits)
            )
        else:
            # Linear interpolation between the two nearest table entries
            filtered_probs = jax.nn.softmax(filtered_logits)
            cum_probs = jnp.cumsum(filtered_probs)
            rand = jax.random.uniform(rng, shape=())
            idx_high = jnp.searchsorted(cum_probs, rand, side="right")

            # Do interpolation unless we're at the first entry, since we can't interpolate using an
            # entry that isn't in the table.
            def do_interp():
                idx_low = idx_high - 1
                cumprob_low = cum_probs[idx_low]
                cumprob_high = cum_probs[idx_high]
                interp_frac = (rand - cumprob_low) / (cumprob_high - cumprob_low)
                return self._idx_to_height(idx_low) + interp_frac * (
                    self._idx_to_height(idx_high) - self._idx_to_height(idx_low)
                )

            sampled_height = jax.lax.cond(
                idx_high == 0,
                lambda: -1.0,
                do_interp,
            )
        # We've sampled a height in [-1, d_max - 1]. Convert to a cosine distance.
        return sampled_height + 1

    @partial(jax.jit, inline=True)
    def log_cap_size(self, d_max):
        """Calculate the log of the size of a cap with a given max cosine distance to its center, as
        a fraction of the total surface area of the sphere.
        """
        # The table contains logged areas of slices, if we want logged fractions of the total we
        # need to take the softmax.
        log_area_fracs = jax.nn.log_softmax(self.table)
        d_max_idx = self._height_to_idx(d_max - 1)
        filtered_log_area_fracs = jnp.where(
            jnp.arange(self.buckets) <= d_max_idx, log_area_fracs, -jnp.inf
        )
        return jax.nn.logsumexp(filtered_log_area_fracs)


def test_height_idx_mapping():
    """Test that the height to index and index to height functions are inverses of each other."""
    table = LogitsTable(d=767, n=8192)

    # Test some key values
    test_heights = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for h in test_heights:
        idx = table._height_to_idx(h)
        h_recovered = table._idx_to_height(idx)
        print(f"h: {h:.1f} -> idx: {idx} -> h_recovered: {h_recovered:.6f}")
        np.testing.assert_allclose(h, h_recovered, atol=0.001, rtol=0)

    # Test edge cases
    assert table._height_to_idx(-1.0) == 0
    assert table._height_to_idx(1.0) == table.buckets - 1


_test_sample_cap_cos_distance_jv = jax.jit(
    lambda table, batch_size, rng, d_max, interpolate: jax.vmap(
        lambda rng: table.sample_cap_cos_distance(rng, d_max, interpolate)
    )(jax.random.split(rng, batch_size)),
    static_argnames=("batch_size", "interpolate"),
)


@pytest.mark.parametrize("interpolate", [False, True])
def test_distribution_is_symmetric(interpolate: bool):
    """Test that sampling heights returns ~50% less than 0 and ~50% greater than 0."""
    table = LogitsTable(767, 8192)

    iters = 128
    batch_size = 1024
    samples = iters * batch_size

    rng = jax.random.PRNGKey(3)
    cos_dists = []
    for _ in range(iters):
        rng, subrng = jax.random.split(rng)
        dists = _test_sample_cap_cos_distance_jv(
            table, batch_size, subrng, 2.0, interpolate
        )
        cos_dists.append(dists)
    cos_dists = jax.device_get(jnp.concatenate(cos_dists))

    assert cos_dists.shape == (samples,)

    mean = np.mean(cos_dists)
    less_than_one = np.sum(cos_dists < 1)
    greater_than_one = np.sum(cos_dists > 1)

    np.testing.assert_allclose(mean, 1.0, atol=0.001, rtol=0)
    np.testing.assert_allclose(less_than_one / samples, 0.5, atol=0.003, rtol=0)
    np.testing.assert_allclose(greater_than_one / samples, 0.5, atol=0.003, rtol=0)


@pytest.mark.parametrize("d_max", [1e-5, 0.1, 0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("interpolate", [False, True])
def test_sample_cap_cos_distance_below_d_max(d_max, interpolate):
    """Test that sampling with a max cosine distance returns values in the range [0, d_max]."""
    table = LogitsTable(767, 8192)

    batch_size = 8192
    iters = 256
    samples = batch_size * iters

    rng = jax.random.PRNGKey(420_69)
    samples = []
    for _ in range(iters):
        rng, subrng = jax.random.split(rng)
        samples.append(
            _test_sample_cap_cos_distance_jv(
                table, batch_size, subrng, d_max, interpolate
            )
        )
    samples = np.concatenate(samples)
    assert np.isfinite(samples).all()

    np.testing.assert_array_less(samples, d_max)


def sample_d_sphere(rng, d):
    """Sample from a d-sphere using the normal-distribution-plus-normalization method."""
    vec = jax.random.normal(rng, (d + 1,))
    vec = vec / jnp.linalg.norm(vec)
    return vec


def random_pt_with_cosine_similarity(
    rng: jax.Array, pt: jax.Array, sim: jax.Array
) -> jax.Array:
    """Generate a random point on the unit sphere that has a given cosine similarity with a given
    point."""

    # Generate a random point v on the sphere
    v = sample_d_sphere(rng, pt.shape[0] - 1)

    # Orthogonalize v with respect to pt
    v_orthogonal = v - jnp.dot(pt, v) * pt
    v_orthogonal = v_orthogonal / jnp.linalg.norm(v_orthogonal)

    # Find the orthogonal component
    orthogonal_length = jnp.sqrt(1 - sim**2)

    # Scale v_orthogonal to achieve the desired cosine similarity with u
    new_point = sim * pt + orthogonal_length * v_orthogonal

    # Ensure the new point is on the sphere
    new_point = new_point / jnp.linalg.norm(new_point)

    return new_point


def test_random_pt_with_cosine_similarity() -> None:
    n_inputs = 512
    inputs = jax.random.normal(jax.random.PRNGKey(0), shape=(n_inputs, 768))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)
    norms = jax.device_get(jnp.linalg.norm(inputs, axis=1))
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    rng = jax.random.PRNGKey(1)
    for pt in inputs:
        rng, new_pt_rng, sim_rng = jax.random.split(rng, 3)
        tgt_sim = jax.random.uniform(sim_rng, minval=-1.0, maxval=1.0)
        new_pt = random_pt_with_cosine_similarity(new_pt_rng, pt, tgt_sim)
        assert jnp.isclose(jnp.linalg.norm(new_pt), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.dot(pt, new_pt), tgt_sim, atol=1e-5)


def sample_from_cap(rng, table, v, d_max):
    """Sample a point inside a cap, defined by a center vector v and a maximum cosine distance d_max
    to that center."""

    cos_rng, pt_rng = jax.random.split(rng)
    cos_theta = 1 - table.sample_cap_cos_distance(cos_rng, d_max)

    # Generate a point on the sphere with the desired cosine similarity to the center
    pt = random_pt_with_cosine_similarity(pt_rng, v, cos_theta)
    assert pt.shape == v.shape
    return pt


def sample_cap(
    table: LogitsTable, rng: jax.Array, v: jax.Array, bias_d_max: bool = False
) -> Tuple[jax.Array, jax.Array]:
    """Given a unit vector v, sample a spherical cap that contains it. With bias_d_max = False, the
    max cosine distance will be drawn from U[0, 2]. With it set to true, it'll be a mixture of two
    uniform distributions: 95% U[0, 1.0] and 5% U[1.0, 2.0]. Important properties:
    * Given two vectors, a cap that contains both of them is equally likely to be sampled from
      either.
    * Knowing the cap a vector is in gives you no more information than the fact that the vector is
      inside the cap. I.e. your Bayesian update is just to set the probability of everything outside
      the cap to zero and rescale the probabilities of everything inside the cap to sum to one.

    These properties are necessary for embedding guided content generation to work.

    The algorithm is this:
    * Sample a max cosine distance d_max from the distribution described above. A cap with that
      d_max that contains v can have a center anywhere with cosine distance to v <= d_max. That's
      the definition of a spherical cap - the set of valid centers is the cap centered on v with
      d_max = d_max.
    * Sample a point uniformly from that cap.


    Argument that the properties are satisfied:
    * The former implies the latter. If a cap is equally likely to be sampled starting from any
      vector that it contains, then the conditional distribution of the vector is uniform.
    * Our algorithm samples cap centers uniformly, conditioned on d_max.
    * The former is true. First, d_max's distribution does not depend on v, so it's true of d_max.
      Assume two points, u & v. Further assume we've sampled a d_max, the same value for both u & v.
      It must be <= the cosine distance between u & v, or the cap would not include both points. For
      each point, the distribution of the cap center is uniform over the possible centers, so the
      distribution is also uniform over the cap centers that are <= d_max away from both points. The
      size of a cap is a function of d_max, and does not depend on its center, so the density
      starting from either u or v is equal.
    """
    # Some more thoughts from o1-preview:
    # https://chatgpt.com/share/672831d7-c668-800c-bff8-b749f667120e

    d_max_rng, ctr_rng = jax.random.split(rng, 2)

    # Sample max cosine distance
    if bias_d_max:
        rand = jax.random.uniform(d_max_rng, minval=0.0, maxval=1.0)
        d_max = jax.lax.cond(
            rand < 0.95, lambda r: r / 0.95, lambda r: 1 + (r - 0.95) / 0.05, rand
        )
    else:
        d_max = jax.random.uniform(d_max_rng, minval=0.0, maxval=2.0)

    # Sample the center of the output cap from the cap centered at v with max cosine distance
    # d_max.
    pt = sample_from_cap(ctr_rng, table, v, d_max)
    return pt, d_max


@pytest.mark.parametrize("bias_d_max", [False, True])
def test_sample_cap(bias_d_max: bool) -> None:
    table = LogitsTable(767, 8192)

    # Generate some inputs
    n_inputs = 8192

    inputs = jax.random.normal(jax.random.PRNGKey(0), shape=(n_inputs, 768))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)

    sample_cap_j = jax.jit(lambda rng, v: sample_cap(table, rng, v, bias_d_max))

    rng = jax.random.PRNGKey(90210)
    for v in inputs:
        rng, gen_rng = jax.random.split(rng, 2)
        new_pt, d_max = sample_cap_j(gen_rng, v)
        assert jnp.all(jnp.isfinite(new_pt))
        assert jnp.isfinite(d_max)
        assert jnp.isclose(jnp.linalg.norm(new_pt), 1.0, atol=1e-5)
        assert 0.0 <= d_max <= 2.0
        dist = 1 - jnp.dot(v, new_pt)
        assert dist <= d_max


def test_sample_from_cap():
    table = LogitsTable(767, 8192)

    # Generate some caps
    n_caps = 128
    vec_rng, size_rng, samples_rng = jax.random.split(jax.random.PRNGKey(0), 3)
    # Generate some central vectors
    vectors = jax.random.normal(vec_rng, shape=(n_caps, 768))
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    # Generate some max cosine distances
    max_cos_distances = jax.random.uniform(
        size_rng, minval=0.0, maxval=2.0, shape=(n_caps,), dtype=jnp.float32
    )

    assert vectors.shape == (n_caps, 768)
    assert max_cos_distances.shape == (n_caps,)

    sample_jv = jax.jit(
        lambda rng, v, d_max: jax.vmap(
            lambda rng: sample_from_cap(rng, table, v, d_max),
            in_axes=(0,),
        )(jax.random.split(rng, 128))
    )

    for i in range(n_caps):
        samples_rng, subrng = jax.random.split(samples_rng)
        samples = sample_jv(subrng, vectors[i], max_cos_distances[i])
        assert samples.shape == (128, 768)
        np.testing.assert_allclose(
            jnp.linalg.norm(samples, axis=1), 1.0, atol=1e-5, rtol=0
        )
        dists = 1 - jnp.dot(samples, vectors[i])
        assert jnp.all(dists <= max_cos_distances[i])
