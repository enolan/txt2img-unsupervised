"""Functions for efficiently sampling from cones."""

# Uniformly sampling from a cone is actually quite difficult.

# Terminology note: an n-sphere is a sphere embedded in n+1 dimensions. A 1-sphere is a circle, a
# 2-sphere is a regular sphere, a 3-sphere is a sphere in 4 dimensions, etc.

# For context, remember a cone is defined as a central point on an n-sphere, along with a minimum
# and maximum cosine similarity to the central point. We use cones to define target areas of CLIP
# space to condition image generation on. The cosine similarities are equivalent to angles, but it
# makes more sense to think of CLIP embeddings in terms of cosine similarities since that's CLIP's
# training objective. CLIP embeddings are 768-dimensional (exist on a 767-sphere), which is what
# makes things hard.

# Sampling uniformly from an n-sphere is easy. We sample from a normal distribution and then normalize
# the vector. Sampling uniformly from an n-dimensional cone is easy too, if you don't care how many
# years it takes. Simply sample from the sphere and reject any samples outside your cone. The
# problem with working in high dimensions is that the volume of a sphere is *heavily* concentrated
# close to 0 cosine similarity. The probability of sampling a point with cosine similiarity between
# 0.5 & 1.0 is ~3.0029e-50. So that's untenable obviously.

# So here's one approach, which we don't actually use. We slice our 767-sphere into n 766-spheres,
# compute the volume of each slice, and save a table. To sample from a cone centered at the north
# pole, we sample from the slices proportional to their volume, after setting the probabilities of
# any slices outside our range to 0. Then we sample a point inside the corresponding slice using
# the normal-distribution-plus-normalization method, and we have a point in a cone whose center is
# the north pole. After that we rotate the point so that the center of the cone is wherever we want
# it. This process approaches the continuous distribution as n approaches infinity.

# Doing that rotation in high dimensions is weirdly hard, so we use a different approach. We sample
# a slice, the height of the slice is the cosine similarity between every point on that slice and
# the north pole. Since spheres are rotationally symmetric, sampling a cosine similarity to the
# north pole is equivalent to sampling a cosine similarity to any other point on the sphere. So we
# sample a cosine similarity to the north pole, and from there we can sample a point with that
# cosine similarity to the central point of our cone using the random_pt_with_cosine_similarity
# function below. To constrain the cosine similarities to be inside our cone, we filter the table
# of slice probabilities to only include the slices with cosine similarities in our range.

# Continuitycels: Nooooooo you can't use a discrete approximation to a continuous distribution!
# Me, a Chad: haha big table go brrrrr
# Continuitycels: *seething*

import jax
import jax.numpy as jnp
import numpy as np
import scipy.special
from copy import copy
from dataclasses import dataclass
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
    """Table of slice logits for sampling from a sphere."""

    def __init__(self, d, n):
        """Generate a table of logits for sampling from a cone."""
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
        """Convert an index in the table to a slice height."""
        val = idx / (self.buckets - 1)
        h = 2 * val - 1
        return h

    def filter_slice_logits(self, lower_bound, upper_bound):
        """Filter the table of slice logits to only include slices within the given bounds."""

        out = copy(self)

        # This seems like a kind of inefficient way to mask stuff, oh well.
        idxs = jnp.arange(self.buckets)
        lower_mask = idxs < self._height_to_idx(lower_bound)
        upper_mask = idxs > self._height_to_idx(upper_bound)
        out.table = jnp.where(lower_mask, -np.inf, out.table)
        out.table = jnp.where(upper_mask, -np.inf, out.table)
        return out

    def sample_slice_from_table(self, rng):
        """Sample a slice using a table of slice logits. Returns slice height."""
        slice_idx = jax.random.categorical(rng, self.table)
        return self._idx_to_height(slice_idx)


def test_filter_logits_identity():
    """Test that filtering a table to the whole range is the identity."""
    table = LogitsTable(768, 1000)
    assert np.array_equal(table.table, table.filter_slice_logits(-1.0, 1.0).table)


def test_logits_are_symmetric():
    """Test that the logits table is symmetrical around 0."""
    table = LogitsTable(768, 1000)
    np.testing.assert_array_equal(table.table, table.table[::-1])

    table = LogitsTable(768, 1001)
    np.testing.assert_array_equal(table.table, table.table[::-1])


_test_sample_jv = jax.jit(
    lambda table, batch_size, rng: jax.vmap(
        lambda rng: table.sample_slice_from_table(rng)
    )(jax.random.split(rng, batch_size)),
    static_argnums=(1,),
)


def test_distribution_is_symmetric():
    """Test that sampling heights returns ~50% less than 0 and ~50% greater than 0."""
    table = LogitsTable(768, 8192)

    batch_size = 1024
    iters = 1024
    samples = batch_size * iters

    rng = jax.random.PRNGKey(3)
    heights = []
    for _ in range(iters):
        rng, subrng = jax.random.split(rng)
        heights.append(_test_sample_jv(table, batch_size, subrng))
    heights = np.concatenate(heights)

    assert heights.shape == (samples,)

    mean = np.mean(heights)
    less_than_zero = np.sum(heights < 0)
    greater_than_zero = np.sum(heights > 0)

    # The empirical mean is slightly less than 0. I suspect ghosts. Or floating point issues.
    # ~ -5e-5.
    np.testing.assert_allclose(mean, 0.0, atol=1e-4, rtol=0)
    np.testing.assert_allclose(less_than_zero / samples, 0.5, atol=0.001, rtol=0)
    np.testing.assert_allclose(greater_than_zero / samples, 0.5, atol=0.001, rtol=0)


def _test_filtering_range(lower_bound, upper_bound):
    """Test that filtering a table to some range works."""
    table = LogitsTable(767, 8192)
    filtered_table = table.filter_slice_logits(lower_bound, upper_bound)

    batch_size = 1024
    iters = 1024
    samples = batch_size * iters

    rng = jax.random.PRNGKey(420_69)
    samples = []
    for _ in range(iters):
        rng, subrng = jax.random.split(rng)
        samples.append(_test_sample_jv(filtered_table, batch_size, subrng))
    samples = np.array(samples)

    # Rounding error can make samples slightly out of range.
    np.testing.assert_array_less(lower_bound, samples + 1 / table.buckets)
    np.testing.assert_array_less(samples - 1 / table.buckets, upper_bound)


def test_filtering_minus_4_to_plus_6():
    """Test that filtering a table to -0.4 to +0.6 works."""
    _test_filtering_range(-0.4, 0.6)


def test_filtering_95_to_100():
    """Test that filtering a table to 0.95 to 1.0 works."""
    _test_filtering_range(0.95, 1.0)


# TODO this section might all end up as dead code
# def sample_d_sphere(rng, d):
#     """Sample from a d-sphere using the normal-distribution-plus-normalization method."""
#     vec = jax.random.normal(rng, (d,))
#     vec = vec / jnp.linalg.norm(vec)
#     return vec


# def sample_in_slice(rng, slice_height, d):
#     """Sample from a (d-1)-sphere slice using the normal-distribution-plus-normalization method,
#     then place that point at the given height on the d-sphere."""
#     pt_in_slice = sample_d_sphere(rng, d - 1)
#     pt_in_slice = pt_in_slice * np.sqrt(1 - slice_height**2)
#     pt_in_slice = np.append(pt_in_slice, slice_height)
#     return pt_in_slice


# def sample_from_north_cone(rng, table, lower_bound, upper_bound):
#     """Sample from a cone centered straight up."""
#     slice_rng, pt_rng = jax.random.split(rng)
#     table = table.filter_slice_logits(lower_bound, upper_bound)
#     slice = table.sample_slice_from_table(slice_rng)
#     pt = sample_in_slice(pt_rng, slice, table.d)
#     return pt


def random_pt_with_cosine_similarity(
    rng: jax.Array, pt: jax.Array, sim: jax.Array
) -> jax.Array:
    """Generate a random point on the unit sphere that has a given cosine similarity with a given
    point."""

    # Generate a random point v on the sphere
    v = jax.random.normal(rng, shape=pt.shape, dtype=pt.dtype)
    v = v / jnp.linalg.norm(v)

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
    # Generate some inputs
    n_inputs = 512
    inputs = jax.random.normal(jax.random.PRNGKey(0), shape=(n_inputs, 768))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)
    assert jnp.linalg.norm(inputs[0]) == 1.0

    rng = jax.random.PRNGKey(1)
    for pt in inputs:
        rng, new_pt_rng, sim_rng = jax.random.split(rng, 3)
        tgt_sim = jax.random.uniform(sim_rng, minval=-1.0, maxval=1.0)
        new_pt = random_pt_with_cosine_similarity(new_pt_rng, pt, tgt_sim)
        assert jnp.isclose(jnp.linalg.norm(new_pt), 1.0, atol=1e-5)
        assert jnp.isclose(jnp.dot(pt, new_pt), tgt_sim, atol=1e-5)


def sample_from_cone(rng, table, v, lower_bound, upper_bound):
    """Sample from a cone centered at v."""

    cos_rng, pt_rng = jax.random.split(rng)
    # Sample a cosine similarity inside the bounds
    table = table.filter_slice_logits(lower_bound, upper_bound)
    cos_sim = table.sample_slice_from_table(cos_rng)

    # Sample a point with that cosine similarity
    pt = random_pt_with_cosine_similarity(pt_rng, v, cos_sim)
    return pt


def generate_clip_cone(
    table: LogitsTable, rng: jax.Array, clip: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Given a CLIP embedding, generate a random cone that contains it."""
    lower_bound_rng, sim_rng, new_pt_rng = jax.random.split(rng, 3)

    lower_bound = jax.random.uniform(lower_bound_rng, minval=-1.0, maxval=1.0)
    upper_bound = 1.0

    # Generate the new point
    new_pt_sim = table.filter_slice_logits(
        lower_bound, upper_bound
    ).sample_slice_from_table(sim_rng)
    new_pt = random_pt_with_cosine_similarity(new_pt_rng, clip, new_pt_sim)

    return new_pt, lower_bound, upper_bound


def test_generate_clip_cone() -> None:
    # Generate some inputs
    n_inputs = 128

    table = LogitsTable(767, 8192)

    inputs = jax.random.normal(jax.random.PRNGKey(0), shape=(n_inputs, 768))
    inputs = inputs / jnp.linalg.norm(inputs, axis=1, keepdims=True)

    rng = jax.random.PRNGKey(90210)
    for clip in inputs:
        rng, gen_rng = jax.random.split(rng, 2)
        new_pt, lower_bound, upper_bound = generate_clip_cone(table, gen_rng, clip)
        assert jnp.isclose(jnp.linalg.norm(new_pt), 1.0, atol=1e-5)
        sim = jnp.dot(clip, new_pt)
        assert (sim + 2 / 8192) >= lower_bound
        assert (sim - 2 / 8192) <= upper_bound


def test_sample_from_cone():
    """Test that sample_from_cone works."""
    # Generate some cones
    n_cones = 128
    vec_rng, size_rng, lower_bound_rng, samples_rng = jax.random.split(
        jax.random.PRNGKey(0), 4
    )
    # Generate some central vectors
    vectors = jax.random.normal(vec_rng, shape=(n_cones, 768))
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    # Generate some cone sizes
    sizes = jax.random.uniform(
        size_rng, minval=2 / 8192, maxval=2.0, shape=(n_cones,), dtype=jnp.float32
    )

    # Generate some cosine similarity bounds
    max_lower_bounds = 1 - sizes
    lower_bounds = jax.random.uniform(
        lower_bound_rng,
        minval=-1.0,
        maxval=max_lower_bounds,
        shape=(n_cones,),
        dtype=jnp.float32,
    )
    upper_bounds = lower_bounds + sizes

    assert vectors.shape == (n_cones, 768)
    assert lower_bounds.shape == upper_bounds.shape == (n_cones,)

    # Sample from the cones
    table = LogitsTable(767, 8192)
    tolerance = 2 / table.buckets

    sample_jv = jax.jit(
        lambda rng, v, lower_bound, upper_bound: jax.vmap(
            lambda rng: sample_from_cone(rng, table, v, lower_bound, upper_bound),
            in_axes=(0,),
        )(jax.random.split(rng, 128))
    )

    for i in range(n_cones):
        samples_rng, subrng = jax.random.split(samples_rng)
        samples = sample_jv(subrng, vectors[i], lower_bounds[i], upper_bounds[i])
        assert samples.shape == (128, 768)
        np.testing.assert_allclose(
            jnp.linalg.norm(samples, axis=1), 1.0, atol=1e-5, rtol=0
        )
        sims = jnp.dot(samples, vectors[i])
        assert jnp.all((sims + tolerance) >= lower_bounds[i])
        assert jnp.all((sims - tolerance) <= upper_bounds[i])
