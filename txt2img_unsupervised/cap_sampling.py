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
from functools import partial
from typing import Tuple


@jax.tree_util.register_pytree_node_class
class LogitsTable:
    """Precomputed table for sampling heights on a d-sphere.

    The logits approximate the log of the band area density with respect to height ``h`` (the
    coordinate along an axis through the sphere), not the zero-thickness slice area. For a
    d-sphere, the band area density per unit ``h`` is proportional to ``(1 - h^2)^{(d-2)/2}``.

    Using band-area density (not slice area) is critical: a uniform distribution on the sphere has
    uniform ``h`` in 3D, and the above formula generalizes that to arbitrary dimensions. Using slice
    area alone would bias samples toward the equator.
    """

    def __init__(self, d, n):
        """Generate a table of log band-area densities for a d-sphere.

        We discretize heights ``h ∈ [-1, 1]`` into ``n`` buckets and compute logits proportional to
        the area of the spherical band per unit ``h`` at each height. For a d-sphere, the density is
        proportional to ``(1 - h^2)^{(d-2)/2}`` times a constant factor that cancels in softmax.
        """
        # Precompute on float64 for numerical stability; cast to float32 afterward.
        slice_heights = np.linspace(-1.0, 1.0, n, dtype=np.float64)

        # Logits ∝ (d-2)/2 * log(1 - h^2)
        # Handle endpoints robustly: set logits to -inf at |h|=1 where the band measure is 0.
        with np.errstate(invalid="ignore"):
            band_logits = ((d - 2) / 2.0) * np.log(1.0 - slice_heights**2)
            band_logits[np.isnan(band_logits)] = -np.inf

        band_logits = band_logits.astype(np.float32)

        assert band_logits.shape == (n,)

        self.d = d
        self.buckets = n
        self.table = jax.nn.log_softmax(jnp.array(band_logits))

    def tree_flatten(self):
        # For some reason you can't just return the array as the first element of the returned
        # tuple, you have to return a tuple there.
        return ((self.table,), (self.d, self.buckets))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        out = cls.__new__(cls)
        out.table = children[0]
        out.d, out.buckets = aux_data
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

    def weighted(self, weights: jax.Array):
        """Return a new LogitsTable whose logits are reweighted by a per-bucket weight array.

        The provided ``weights`` array must have the same shape as the underlying table logits
        (i.e., ``(self.buckets,)``). Values <= 0 are treated as zeros (excluded) by mapping them to
        ``-inf`` in log-space. The returned table's logits are renormalized with ``log_softmax``.
        """
        assert weights.shape == self.table.shape
        # Map non-positive weights to -inf in log-space, positive weights to log(weights)
        log_w = jnp.where(weights > 0.0, jnp.log(weights), -jnp.inf)
        out = LogitsTable.__new__(LogitsTable)
        out.table = jax.nn.log_softmax(self.table + log_w)
        out.d = self.d
        out.buckets = self.buckets
        return out

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

            # Do interpolation unless we're at the first entry or last entry, since we can't
            # interpolate using an entry that isn't in the table.
            def do_interp():
                idx_low = idx_high - 1
                cumprob_low = cum_probs[idx_low]
                cumprob_high = cum_probs[idx_high]
                interp_frac = (rand - cumprob_low) / (cumprob_high - cumprob_low)
                return self._idx_to_height(idx_low) + interp_frac * (
                    self._idx_to_height(idx_high) - self._idx_to_height(idx_low)
                )

            def handle_edge_case():
                return jax.lax.cond(
                    idx_high == 0,
                    lambda: -1.0,
                    lambda: 1.0,
                )

            sampled_height = jax.lax.cond(
                jnp.logical_or(idx_high == 0, idx_high == self.buckets),
                handle_edge_case,
                do_interp,
            )
        # We've sampled a height in [-1, d_max - 1]. Convert to a cosine distance.
        return sampled_height + 1

    @partial(jax.jit, inline=True)
    def log_cap_size(self, d_max):
        """Calculate the log of the size of a cap with a given max cosine distance to its center, as
        a fraction of the total surface area of the sphere.
        """
        d_max_idx = self._height_to_idx(d_max - 1)
        filtered_log_area_fracs = jnp.where(
            jnp.arange(self.buckets) <= d_max_idx, self.table, -jnp.inf
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


def process_d_max_dist(
    d_max_dist: list[tuple[float, float]] = None
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Process a d_max_dist parameter into normalized weights, range starts, and range ends. Used
    when sampling maximum cosine distances from mixtures of uniform distributions.

    Args:
        d_max_dist: List of (weight, max_value) tuples defining the mixture distribution.
                   If None, defaults to [(1.0, 2.0)] (uniform U[0, 2])

    Returns:
        Tuple of (weights, range_starts, range_ends) as JAX arrays
    """
    # Set default distribution if none provided
    if d_max_dist is None:
        d_max_dist = [(1.0, 2.0)]

    if not d_max_dist:
        raise ValueError("d_max_dist cannot be empty")

    # Extract weights and create component ranges
    weights = jnp.array([weight for weight, _ in d_max_dist], dtype=jnp.float32)
    max_values = jnp.array([max_val for _, max_val in d_max_dist], dtype=jnp.float32)

    # Normalize weights to ensure they sum to 1
    weights = weights / jnp.sum(weights)

    # Create ranges: [0, max_values[0]), [max_values[0], max_values[1]), etc.
    range_starts = jnp.concatenate([jnp.array([0.0]), max_values[:-1]])
    range_ends = max_values

    return weights, range_starts, range_ends


def sample_cap(
    table: LogitsTable,
    rng: jax.Array,
    v: jax.Array,
    d_max_dist: list[tuple[float, float]] = None,
) -> Tuple[jax.Array, jax.Array]:
    """Given a unit vector v, sample a spherical cap that contains it. The max cosine distance will be
    drawn from a mixture of uniform distributions specified by d_max_dist. If d_max_dist is None,
    defaults to uniform U[0, 2]. Important properties:
    * Given two vectors, a cap that contains both of them is equally likely to be sampled from
      either.
    * Knowing the cap a vector is in gives you no more information than the fact that the vector is
      inside the cap. I.e. your Bayesian update is just to set the probability of everything outside
      the cap to zero and rescale the probabilities of everything inside the cap to sum to one.

    These properties are necessary for embedding guided content generation to work.

    The algorithm is this:
    * Sample a max cosine distance d_max from the mixture distribution described by d_max_dist.
      A cap with that d_max that contains v can have a center anywhere with cosine distance to v <= d_max.
      That's the definition of a spherical cap - the set of valid centers is the cap centered on v with
      d_max = d_max.
    * Sample a point uniformly from that cap.

    Args:
        table: LogitsTable for the sphere dimension
        rng: JAX random key
        v: Unit vector that must be contained in the sampled cap
        d_max_dist: List of (weight, max_value) tuples defining the mixture distribution.
                   If None, defaults to [(1.0, 2.0)] (uniform U[0, 2])

    Examples:
        [(1.0, 2.0)] = 100% U[0.0, 2.0] (uniform over full range)
        [(0.95, 1.0), (0.05, 2.0)] = 95% U[0.0, 1.0] + 5% U[1.0, 2.0]
        [(0.4, 0.8), (0.4, 1.2), (0.2, 2.0)] = 40% U[0.0, 0.8] + 40% U[0.8, 1.2] + 20% U[1.2, 2.0]

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

    # sample d_max
    weights, range_starts, range_ends = process_d_max_dist(d_max_dist)
    component_idx = jax.random.categorical(d_max_rng, jnp.log(weights))
    d_max = jax.random.uniform(
        d_max_rng, minval=range_starts[component_idx], maxval=range_ends[component_idx]
    )

    # sample the center of the output cap from the cap centered at v with max cosine distance d_max
    pt = sample_from_cap(ctr_rng, table, v, d_max)
    return pt, d_max


@pytest.mark.parametrize(
    "d_max_dist",
    [
        None,  # Default: uniform U[0, 2]
        [(1.0, 2.0)],  # Explicit uniform U[0, 2]
        [(0.95, 1.0), (0.05, 2.0)],  # Biased distribution
        [(0.45, 0.8), (0.45, 1.2), (0.1, 2.0)],  # Triangular distribution
    ],
)
def test_sample_cap(d_max_dist) -> None:
    "Test the distribution of samples generated with sample_cap"
    table = LogitsTable(767, 8192)

    # Use a single input vector for testing d_max distribution
    input_vec = jax.random.normal(jax.random.PRNGKey(0), shape=(768,))
    input_vec = input_vec / jnp.linalg.norm(input_vec)

    sample_cap_jv = jax.jit(
        jax.vmap(
            lambda rng: sample_cap(table, rng, input_vec, d_max_dist), in_axes=(0,)
        )
    )

    # Collect d_max samples in batches for efficiency
    rng = jax.random.PRNGKey(90210)
    batch_size = 1024
    n_batches = 64
    n_samples = batch_size * n_batches

    all_new_pts = []
    all_d_max_samples = []

    for _ in range(n_batches):
        rng, batch_rng = jax.random.split(rng)
        batch_rngs = jax.random.split(batch_rng, batch_size)

        new_pts, d_max_batch = sample_cap_jv(batch_rngs)

        all_new_pts.append(new_pts)
        all_d_max_samples.append(d_max_batch)

    # Concatenate all batches
    all_new_pts = jnp.concatenate(all_new_pts, axis=0)
    all_d_max_samples = jnp.concatenate(all_d_max_samples, axis=0)

    # Basic sanity checks on the vectorized results
    assert jnp.all(jnp.isfinite(all_new_pts))
    assert jnp.all(jnp.isfinite(all_d_max_samples))
    assert jnp.allclose(jnp.linalg.norm(all_new_pts, axis=1), 1.0, atol=1e-5)
    assert jnp.all(all_d_max_samples >= 0.0)
    assert jnp.all(all_d_max_samples <= 2.0)
    dists = 1 - jnp.dot(all_new_pts, input_vec)
    assert jnp.all(dists <= all_d_max_samples)

    d_max_samples = np.array(all_d_max_samples)

    weights, range_starts, range_ends = process_d_max_dist(d_max_dist)

    # Convert JAX arrays to numpy for the test
    weights = np.array(weights)
    range_starts = np.array(range_starts)
    range_ends = np.array(range_ends)

    # Test the distribution by checking proportions in each range
    for i, (expected_weight, range_start, range_end) in enumerate(
        zip(weights, range_starts, range_ends)
    ):
        # Count samples in this range
        in_range = np.sum((d_max_samples >= range_start) & (d_max_samples < range_end))
        observed_proportion = in_range / n_samples
        expected_proportion = expected_weight

        # Allow for some statistical variation (3 sigma test)
        expected_std = np.sqrt(
            expected_proportion * (1 - expected_proportion) / n_samples
        )
        tolerance = 3 * expected_std

        np.testing.assert_allclose(
            observed_proportion,
            expected_proportion,
            atol=tolerance,
            rtol=0,
            err_msg=f"Component {i}: expected {expected_proportion:.3f}, got {observed_proportion:.3f}",
        )

    # Also test that all samples are within the overall expected range
    overall_max = np.max(range_ends)
    assert np.all(d_max_samples >= 0.0)
    assert np.all(d_max_samples <= overall_max)

    # For uniform components, test that the distribution within each range is roughly uniform
    for i, (expected_weight, range_start, range_end) in enumerate(
        zip(weights, range_starts, range_ends)
    ):
        range_samples = d_max_samples[
            (d_max_samples >= range_start) & (d_max_samples < range_end)
        ]
        if len(range_samples) > 100:  # Only test if we have enough samples
            # Divide the range into bins and test uniformity
            n_bins = 10
            hist, bin_edges = np.histogram(
                range_samples, bins=n_bins, range=(range_start, range_end)
            )
            expected_count_per_bin = len(range_samples) / n_bins

            # Chi-square-like test for uniformity within the range
            for bin_count in hist:
                # Allow for reasonable statistical variation
                expected_std = np.sqrt(expected_count_per_bin)
                tolerance = 3 * expected_std
                np.testing.assert_allclose(
                    bin_count,
                    expected_count_per_bin,
                    atol=tolerance,
                    err_msg=f"Component {i} uniformity test failed",
                )


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


def sphere_log_inverse_surface_area(d):
    """Compute the log inverse surface area (density of a uniform distribution) of a sphere
    embedded in d dimensions."""
    # Surface area of unit sphere in d dimensions: A_d = 2 * π^(d/2) / Γ(d/2)
    # Log probability density: log_p0 = -log(A_d) = -log(2) - (d/2)*log(π) + log(Γ(d/2))
    return -(jnp.log(2.0) + (d / 2) * jnp.log(jnp.pi) - jax.lax.lgamma(d / 2))


def test_sphere_log_inverse_surface_area_3d():
    np.testing.assert_allclose(
        sphere_log_inverse_surface_area(3),
        jnp.log(1.0 / (4 * jnp.pi)),
        atol=1e-5,
        rtol=0,
    )
