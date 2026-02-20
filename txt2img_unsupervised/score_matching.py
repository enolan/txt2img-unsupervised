"""
Score matching model for spherical data.

This implementation adapts denoising diffusion models to the unit sphere using vMF noise.

## Theory and Design

### Forward Process
The forward process is Brownian motion on the sphere starting from clean data. At time t, the
conditional distribution p_t(x|x_1) is approximately vMF centered at x_1 with concentration
κ(t) = 1/σ²(t). (The true heat kernel on the sphere isn't exactly vMF, but vMF is a good
approximation, especially at high concentration.)

### Score Function
The score of the vMF distribution is ∇log p_t(x|x_1) = κ(t) · P_x(x_1), where P_x denotes
projection onto the tangent space at x. This points toward x_1 along the geodesic, with magnitude
κ(t) * sin(θ) where θ is the angle between x and x_1. Note: the magnitude is zero both at the mode
(θ=0) and at the antipode (θ=π), peaking at θ=π/2.

### Time Convention
Following flow matching convention: t ∈ [0,1] where t=0 is pure noise, t=1 is data.

### Noise Schedule
We use a linear schedule: σ²(t) = σ²_max - (σ²_max - σ²_min) * t
- At t=0: σ²(0) = σ²_max (high noise, nearly uniform on sphere)
- At t=1: σ²(1) = σ²_min ≈ 0 (minimal noise, nearly clean data)

### Probability Flow ODE
For sampling, we integrate the probability flow ODE from t=0 (noise) to t=1 (data):

    dx/dt = ½ |dσ²/dt| · ∇log p_t(x)

With the linear schedule, |dσ²/dt| = (σ²_max - σ²_min) is constant, giving:

    dx/dt = ½ * (σ²_max - σ²_min) · s_θ(x, t)

where s_θ is the learned score. After each integration step, project back onto the sphere.

### Model Reuse
We reuse the VectorField class from flow_matching.py since estimating a score (tangent vector at x)
has the same shape as estimating a velocity field.
"""

import math

from dataclasses import dataclass, field, replace
from functools import partial
from typing import FrozenSet, Literal, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from datasets import Dataset
from einops import repeat
from jax import Array
from scipy import stats

from txt2img_unsupervised import vmf
from txt2img_unsupervised.cap_sampling import (
    cap_conditioning_dim,
    sphere_log_inverse_surface_area,
)
from txt2img_unsupervised.config import CapConditioningMode
from txt2img_unsupervised.flow_matching import (
    VectorField,
    geodesic_step,
    reverse_path_and_compute_divergence,
    sample_sphere,
    create_train_state,
    generate_samples_inner,
    _train_loop_for_tests_generic,
)


@dataclass(frozen=True)
class NoiseSchedule:
    """Configuration for the vMF noise schedule.

    The schedule defines σ²(t) = σ²_max - (σ²_max - σ²_min) * t (linear)
    and κ(t) = 1/σ²(t).

    Attributes:
        sigma_sq_min: σ² at t=1 (near data). Should be small but positive for numerical stability.
        sigma_sq_max: σ² at t=0 (near uniform). Larger values give more diffuse initial distribution.
    """

    sigma_sq_min: float = 1e-4
    sigma_sq_max: float = 2.0


class ScoreMatchingModel(nn.Module):
    """Score matching model bundling a VectorField with noise schedule and conditioning mode.

    Wraps a VectorField submodule. Currently __call__ forwards directly to the VectorField,
    but will eventually preprocess cap specifications into conditioning vectors for
    CONDITIONED_SCORE mode.
    """

    # VectorField hyperparameters
    domain_dim: int
    reference_directions: Optional[int]
    time_dim: Optional[int]
    use_pre_mlp_projection: bool
    n_layers: int
    d_model: int
    mlp_expansion_factor: int
    mlp_dropout_rate: Optional[float]
    input_dropout_rate: Optional[float]
    mlp_always_inject: FrozenSet[Literal["x", "t", "cond"]] = field(
        default_factory=frozenset
    )
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    d_model_base: int = 512
    variance_base: float = 1 / 512
    alpha_input: float = 1.0
    alpha_output: float = 2.0 / math.pi

    # Score matching specific
    schedule: NoiseSchedule = NoiseSchedule()
    cap_conditioning: CapConditioningMode = CapConditioningMode.UNCONDITIONED
    relative_cap_encoding: bool = False

    @property
    def conditioning_dim(self) -> int:
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return 0
        elif self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            return cap_conditioning_dim(
                self.domain_dim, self.reference_directions, self.relative_cap_encoding
            )
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            return 0
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")

    @nn.nowrap
    def mk_vector_field(self) -> VectorField:
        """Create a VectorField with parameters derived from this model's config."""
        return VectorField(
            domain_dim=self.domain_dim,
            reference_directions=self.reference_directions,
            conditioning_dim=self.conditioning_dim,
            time_dim=self.time_dim,
            use_pre_mlp_projection=self.use_pre_mlp_projection,
            n_layers=self.n_layers,
            d_model=self.d_model,
            mlp_expansion_factor=self.mlp_expansion_factor,
            mlp_dropout_rate=self.mlp_dropout_rate,
            input_dropout_rate=self.input_dropout_rate,
            mlp_always_inject=self.mlp_always_inject,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            d_model_base=self.d_model_base,
            variance_base=self.variance_base,
            alpha_input=self.alpha_input,
            alpha_output=self.alpha_output,
        )

    def setup(self):
        if self.cap_conditioning != CapConditioningMode.UNCONDITIONED:
            raise NotImplementedError(
                f"Cap conditioning mode {self.cap_conditioning} is not yet implemented"
            )
        self.vector_field = self.mk_vector_field()

    @nn.nowrap
    def dummy_inputs(self):
        """Create dummy inputs for model initialization."""
        return self.mk_vector_field().dummy_inputs()

    @nn.nowrap
    def mk_partition_map(self, use_muon: bool):
        """Create a partition map for optimizer configuration with muP scaling."""
        return {
            "params": {
                "vector_field": self.mk_vector_field().mk_partition_map(use_muon)[
                    "params"
                ]
            }
        }

    @nn.nowrap
    def scale_lr(self, lr: float) -> float:
        """Scaled learning rate for hidden layers."""
        return self.mk_vector_field().scale_lr(lr)

    def __call__(self, x, t, cond_vec):
        return self.vector_field(x, t, cond_vec)


def sigma_squared(t: Array, schedule: NoiseSchedule) -> Array:
    """Compute σ²(t) using a linear schedule.

    σ²(t) = σ²_max - (σ²_max - σ²_min) * t

    At t=0: σ²(0) = σ²_max (high noise)
    At t=1: σ²(1) = σ²_min (low noise, near data)
    """
    return schedule.sigma_sq_max - (schedule.sigma_sq_max - schedule.sigma_sq_min) * t


def kappa(t: Array, schedule: NoiseSchedule) -> Array:
    """Compute κ(t) = 1/σ²(t), the vMF concentration parameter."""
    return 1.0 / sigma_squared(t, schedule)


def ode_coefficient(t: Array, schedule: NoiseSchedule) -> Array:
    """Compute the ODE coefficient: ½|dσ²/dt| = ½(σ²_max - σ²_min).

    With a linear schedule, this is constant throughout the integration,
    ensuring non-zero velocity at all times (unlike cosine schedule where
    the coefficient is zero at t=0 and t=1).
    """
    return 0.5 * (schedule.sigma_sq_max - schedule.sigma_sq_min)


def sample_t_log_uniform_kappa(
    rng: Array, shape: tuple, schedule: NoiseSchedule
) -> Array:
    """Sample t so that κ(t) is log-uniformly distributed, giving density p(t) ∝ 1/σ²(t).

    This partially compensates for the σ⁴(t) bias that the scaled score parameterization
    introduces, without being as aggressive as full κ²-weighting (which concentrates
    almost all training at t→1 where gradient signal is tiny).

    Samples log(κ) ~ U[log(κ_min), log(κ_max)] where κ_min = 1/σ²_max and
    κ_max = 1/σ²_min, then inverts to get t.

    Args:
        rng: JAX random key
        shape: Shape of the output array
        schedule: Noise schedule configuration

    Returns:
        Sampled t values with the given shape, in [0, 1]
    """
    u = jax.random.uniform(rng, shape)
    s_min = schedule.sigma_sq_min
    s_max = schedule.sigma_sq_max
    # log(κ) ~ U[log(1/s_max), log(1/s_min)], so σ² = 1/κ is log-uniform on [s_min, s_max]
    sigma_sq = s_min * (s_max / s_min) ** u
    return (s_max - sigma_sq) / (s_max - s_min)


def sample_noisy_point(
    rng: Array, x_1: Array, t: Array, schedule: NoiseSchedule
) -> Array:
    """Sample x_t ~ vMF(x_1, κ(t)) for each (x_1, t) pair.

    Args:
        rng: JAX random key
        x_1: Clean data points [batch_size, dim]
        t: Time values [batch_size]
        schedule: Noise schedule configuration

    Returns:
        Noisy samples x_t [batch_size, dim]
    """
    assert x_1.ndim == 2
    assert t.ndim == 1
    assert x_1.shape[0] == t.shape[0]

    batch_size = x_1.shape[0]
    kappa_values = kappa(t, schedule)

    # Use vmf.sample's batched mode: mu (n, d), kappa (n,) -> (n, d)
    return vmf.sample(rng, x_1, kappa_values, n_samples=batch_size)


def tangent_projection(x: Array, v: Array) -> Array:
    """Project v onto the tangent space at x: P_x(v) = v - (v·x)x.

    Args:
        x: Points on the unit sphere [batch_size, dim]
        v: Vectors to project [batch_size, dim]

    Returns:
        Projected vectors in tangent space [batch_size, dim]
    """
    dot_products = jnp.sum(v * x, axis=-1, keepdims=True)
    return v - dot_products * x


def compute_target_score(
    x_t: Array, x_1: Array, t: Array, schedule: NoiseSchedule
) -> Array:
    """Compute the ground-truth score: κ(t) * P_{x_t}(x_1).

    The score of the vMF distribution p_t(x|x_1) is the gradient of log p_t with respect to x,
    constrained to the tangent space. For vMF(μ=x_1, κ), this is κ * P_x(x_1).

    Args:
        x_t: Noisy points [batch_size, dim]
        x_1: Clean data points [batch_size, dim]
        t: Time values [batch_size]
        schedule: Noise schedule configuration

    Returns:
        Target score vectors [batch_size, dim]
    """
    kappa_values = kappa(t, schedule)
    projected = tangent_projection(x_t, x_1)
    return kappa_values[:, None] * projected


def denoising_score_matching_loss(
    model: ScoreMatchingModel,
    params,
    x_1: Array,
    x_t: Array,
    t: Array,
    conditioning_data: Array,
    rng: Optional[Array] = None,
) -> Array:
    """Compute MSE loss between predicted and target scaled score.

    We train the network to predict the "scaled score" σ²(t) * s(x,t) = P_x(x_1),
    which is bounded (magnitude ≤ 1) regardless of t. This avoids the issue where
    the raw score κ(t) * P_x(x_1) can be very large when t→1.

    At sampling time, we recover the actual score by dividing by σ²(t).

    Args:
        model: Score matching model
        params: Model parameters
        x_1: Clean data [batch_size, dim]
        x_t: Noisy samples [batch_size, dim]
        t: Times [batch_size]
        conditioning_data: Conditioning vectors [batch_size, cond_dim]
        rng: Random key for dropout (if model uses it)

    Returns:
        Scalar loss value
    """
    # Target is the scaled score: σ²(t) * score = σ²(t) * κ(t) * P_x(x_1) = P_x(x_1)
    # This is just the tangent projection, bounded in [-1, 1]
    target_scaled_score = tangent_projection(x_t, x_1)

    rngs_dict = {"dropout": rng} if rng is not None else {}
    predicted_scaled_score = model.apply(
        params, x_t, t, conditioning_data, rngs=rngs_dict
    )

    per_sample_loss = jnp.sum(
        (predicted_scaled_score - target_scaled_score) ** 2, axis=1
    )
    return jnp.mean(per_sample_loss)


@partial(jax.jit, static_argnames=("model",))
def compute_batch_loss(
    model: ScoreMatchingModel,
    params,
    batch: dict,
    rng: Array,
) -> Array:
    """Extract data from batch, sample t and x_t, compute loss.

    Args:
        model: Score matching model
        params: Model parameters
        batch: Batch of data containing "point_vec" and optionally "cond_vec"
        rng: JAX random key

    Returns:
        The computed loss value
    """
    x_1 = batch["point_vec"]
    batch_size = x_1.shape[0]

    noise_rng, time_rng = jax.random.split(rng)

    conditioning_data = batch.get("cond_vec", jnp.zeros((batch_size, 0)))

    t = sample_t_log_uniform_kappa(time_rng, (batch_size,), model.schedule)
    x_t = sample_noisy_point(noise_rng, x_1, t, model.schedule)

    return denoising_score_matching_loss(model, params, x_1, x_t, t, conditioning_data)


@partial(jax.jit, static_argnames=("model",), donate_argnames=("state", "rng"))
def train_step(model: ScoreMatchingModel, state, batch: dict, rng: Array):
    """Train for a single step.

    Args:
        model: Score matching model
        state: Training state
        batch: Batch of data
        rng: JAX random key

    Returns:
        Updated state, loss value, gradient norm, and updated random key
    """
    rng, next_rng = jax.random.split(rng)

    def loss_fn(params):
        return compute_batch_loss(model, params, batch, rng)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)

    state = state.apply_gradients(grads=grads)

    return state, loss, grad_norm, next_rng


@partial(jax.jit, inline=True, static_argnames=("model",))
def _compute_velocity_for_sampling(
    model: ScoreMatchingModel,
    params,
    cond_vecs: Array,
    x: Array,
    t,
    rng: Optional[Array] = None,
) -> Array:
    """Compute ODE velocity for sampling.

    The network outputs the scaled score: σ²(t) * s(x, t).
    The actual score is: s(x, t) = network_output / σ²(t).
    The ODE velocity is: coeff * s(x, t) = coeff / σ²(t) * network_output.

    Args:
        model: Score matching model
        params: Model parameters
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        x: Current points [batch_size, dim]
        t: Current time (scalar or [batch_size])
        rng: Random key for dropout

    Returns:
        ODE velocity [batch_size, dim]
    """
    rngs_dict = {"dropout": rng} if rng is not None else {}

    if isinstance(t, jax.Array):
        if t.ndim == 0:
            t_vec = jnp.full((x.shape[0],), t)
        elif t.ndim == 1:
            assert t.shape[0] == x.shape[0]
            t_vec = t
        else:
            raise ValueError("t must be a scalar or 1D array")
    else:
        t_vec = jnp.full((x.shape[0],), t)

    # Network outputs scaled score: σ²(t) * s(x, t)
    scaled_score = model.apply(params, x, t_vec, cond_vecs, rngs=rngs_dict)

    # ODE velocity = coeff * s = coeff * (scaled_score / σ²) = (coeff / σ²) * scaled_score
    coeff = ode_coefficient(t_vec, model.schedule)
    sigma_sq = sigma_squared(t_vec, model.schedule)
    return (coeff / sigma_sq[:, None]) * scaled_score


@partial(jax.jit, static_argnames=("model",), inline=True)
def _velocity_fn_for_ode(model, params, cond_vecs, x, t, rng):
    """ODE velocity function for score matching sampling and NLL computation.

    Delegates to _compute_velocity_for_sampling. Defined at module level so JIT
    caching works across calls.
    """
    return _compute_velocity_for_sampling(model, params, cond_vecs, x, t, rng)


def generate_samples(
    model: ScoreMatchingModel,
    params,
    rng: Array,
    cond_vecs: Array,
    n_steps: int = 100,
    method: str = "tsit5",
) -> Array:
    """Generate samples by integrating the probability flow ODE.

    Integrates dx/dt = ode_coefficient(t) * s_θ(x, t) from t=0 (noise) to t=1 (data).

    Args:
        model: Score matching model
        params: Model parameters
        rng: JAX random key
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        method: ODE solver method

    Returns:
        Generated samples [batch_size, domain_dim]
    """
    assert len(cond_vecs.shape) == 2
    batch_size = cond_vecs.shape[0]

    x, _eval_counts = generate_samples_inner(
        rng,
        n_steps,
        batch_size,
        method,
        _velocity_fn_for_ode,
        model,
        params,
        cond_vecs,
        model.domain_dim,
    )
    return x


def compute_log_probability(
    model: ScoreMatchingModel,
    params,
    samples: Array,
    cond_vecs: Array,
    n_steps: int = 100,
    rng=None,
    n_projections: int = 10,
    method: str = "tsit5",
) -> Array:
    """Compute the log probability of samples under the score matching model.

    Uses the probability flow ODE in reverse (from data to noise) while accumulating the
    divergence of the velocity field, then combines with the uniform base density.

    Args:
        model: Score matching model
        params: Model parameters
        samples: Points on the sphere to evaluate [batch_size, dim]
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        rng: JAX random key for stochastic divergence estimation
        n_projections: Number of random projections for divergence estimation
        method: ODE solver method ('rk4' or 'tsit5')

    Returns:
        Log probabilities of the samples [batch_size]
    """
    batch_size = samples.shape[0]
    assert samples.shape == (batch_size, model.domain_dim)
    assert cond_vecs.shape == (batch_size, model.conditioning_dim)

    if rng is None:
        rng = jax.random.PRNGKey(0)

    x0, div_sum = reverse_path_and_compute_divergence(
        _velocity_fn_for_ode,
        model,
        params,
        cond_vecs,
        samples,
        n_steps,
        rng,
        n_projections,
        method=method,
    )

    log_p0 = sphere_log_inverse_surface_area(model.domain_dim)
    return log_p0 - div_sum


def compute_nll(
    model: ScoreMatchingModel,
    params,
    batch: dict,
    n_steps: int = 100,
    rng=None,
    n_projections: int = 10,
    method: str = "tsit5",
) -> Array:
    """Compute negative log-likelihood for a batch of data.

    Args:
        model: Score matching model
        params: Model parameters
        batch: Dict with "point_vec" key containing data [batch_size, dim]
        n_steps: Number of integration steps
        rng: JAX random key
        n_projections: Number of random projections for divergence estimation
        method: ODE solver method ('rk4' or 'tsit5')

    Returns:
        NLL per example [batch_size]
    """
    samples = batch["point_vec"]
    batch_size = samples.shape[0]
    cond_vecs = jnp.zeros((batch_size, model.conditioning_dim))
    return -compute_log_probability(
        model,
        params,
        samples,
        cond_vecs,
        n_steps=n_steps,
        rng=rng,
        n_projections=n_projections,
        method=method,
    )


# =============================================================================
# Tests
# =============================================================================


def test_noise_schedule():
    """Test that the noise schedule functions compute correct values."""
    schedule = NoiseSchedule(sigma_sq_min=1e-4, sigma_sq_max=2.0)

    # Test boundary conditions
    t_0 = jnp.array(0.0)
    t_1 = jnp.array(1.0)

    # At t=0: σ²(0) = σ²_max
    sigma_sq_0 = sigma_squared(t_0, schedule)
    np.testing.assert_allclose(sigma_sq_0, schedule.sigma_sq_max, rtol=1e-6)

    # At t=1: σ²(1) = σ²_min
    sigma_sq_1 = sigma_squared(t_1, schedule)
    np.testing.assert_allclose(sigma_sq_1, schedule.sigma_sq_min, rtol=1e-3)

    # Test κ(t) = 1/σ²(t)
    t_mid = jnp.array(0.5)
    kappa_mid = kappa(t_mid, schedule)
    sigma_sq_mid = sigma_squared(t_mid, schedule)
    np.testing.assert_allclose(kappa_mid, 1.0 / sigma_sq_mid, rtol=1e-6)

    # Test ODE coefficient is constant and positive (linear schedule)
    t_values = jnp.linspace(0.0, 1.0, 100)
    coeffs = jax.vmap(lambda t: ode_coefficient(t, schedule))(t_values)
    expected_coeff = 0.5 * (schedule.sigma_sq_max - schedule.sigma_sq_min)

    # All coefficients should be equal to the expected constant
    np.testing.assert_allclose(coeffs, expected_coeff, rtol=1e-6)
    assert jnp.all(coeffs > 0), "ODE coefficient should be positive"


def test_sample_t_log_uniform_kappa():
    """Test that sample_t_log_uniform_kappa produces a log-uniform distribution over κ."""
    schedule = NoiseSchedule(sigma_sq_min=0.01, sigma_sq_max=2.0)
    n = 100_000
    samples = sample_t_log_uniform_kappa(jax.random.PRNGKey(0), (n,), schedule)

    # Output should be in [0, 1]
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)

    # CDF: F(t) = log(σ²_max / σ²(t)) / log(σ²_max / σ²_min)
    # (equivalently, log(κ(t) / κ_min) / log(κ_max / κ_min))
    s_min, s_max = schedule.sigma_sq_min, schedule.sigma_sq_max
    log_ratio = jnp.log(s_max / s_min)
    cdf = lambda t: jnp.log(s_max / sigma_squared(t, schedule)) / log_ratio
    np.testing.assert_allclose(float(cdf(0.0)), 0.0, atol=1e-10)
    np.testing.assert_allclose(float(cdf(1.0)), 1.0, atol=1e-10)

    # KS test: compare empirical samples against the theoretical CDF
    ks_stat, p_value = stats.kstest(
        np.array(samples), lambda t: np.array(cdf(jnp.array(t)))
    )
    assert p_value > 0.01, f"KS test failed: stat={ks_stat:.4f}, p={p_value:.4f}"

    # Samples should be skewed toward t=1 (low noise), but less extremely than κ²
    assert jnp.mean(samples) > 0.5, "Samples should be skewed toward t=1"


def test_target_score_at_mode():
    """Test that score is zero when x_t = x_1 (at the mode)."""
    schedule = NoiseSchedule()

    # When x_t = x_1, the tangent projection P_{x_t}(x_1) = x_1 - (x_1·x_1)x_1 = 0
    batch_size = 10
    dim = 3
    rng = jax.random.PRNGKey(42)

    x_1 = sample_sphere(rng, batch_size, dim)
    x_t = x_1  # Same point
    t = jax.random.uniform(rng, (batch_size,))

    target = compute_target_score(x_t, x_1, t, schedule)

    # Score should be zero (within numerical precision)
    np.testing.assert_allclose(target, 0.0, atol=1e-5)


def test_target_score_direction():
    """Test that score points toward x_1 in tangent space."""
    schedule = NoiseSchedule()

    batch_size = 100
    dim = 16

    key1, key2 = jax.random.split(jax.random.PRNGKey(123))
    x_1 = sample_sphere(key1, batch_size, dim)

    t = jnp.full((batch_size,), 0.5)
    x_t = sample_noisy_point(key2, x_1, t, schedule)

    target = compute_target_score(x_t, x_1, t, schedule)

    # The geodesic direction from x_t toward x_1 is P_{x_t}(x_1) / ||P_{x_t}(x_1)||
    geodesic_dir = tangent_projection(x_t, x_1)
    geodesic_dir_norm = jnp.linalg.norm(geodesic_dir, axis=1, keepdims=True)

    # For non-coincident points, check alignment
    nonzero_mask = geodesic_dir_norm[:, 0] > 1e-6
    if jnp.any(nonzero_mask):
        geodesic_dir_unit = geodesic_dir / jnp.maximum(geodesic_dir_norm, 1e-8)
        target_unit = target / jnp.maximum(
            jnp.linalg.norm(target, axis=1, keepdims=True), 1e-8
        )

        # Dot product should be positive (pointing in same direction)
        alignment = jnp.sum(geodesic_dir_unit * target_unit, axis=1)
        assert jnp.all(alignment[nonzero_mask] > 0.99), "Score should point toward x_1"


def test_sample_noisy_point():
    """Test that sample_noisy_point produces valid samples."""
    schedule = NoiseSchedule()

    batch_size = 1000
    dim = 3

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(456), 3)
    x_1 = sample_sphere(key1, batch_size, dim)
    # With linear schedule: σ²(0.95) ≈ 0.1, so κ(0.95) ≈ 10
    t = jnp.full((batch_size,), 0.95)

    x_t = sample_noisy_point(key2, x_1, t, schedule)

    norms = jnp.linalg.norm(x_t, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    similarities = jnp.sum(x_t * x_1, axis=1)
    mean_similarity = jnp.mean(similarities)
    assert mean_similarity > 0.8, f"Mean similarity {mean_similarity} should be > 0.8"

    # Higher t = higher κ = more concentrated samples
    t_high = jnp.full((batch_size,), 0.99)
    x_t_high = sample_noisy_point(key3, x_1, t_high, schedule)
    similarities_high = jnp.sum(x_t_high * x_1, axis=1)
    mean_similarity_high = jnp.mean(similarities_high)

    assert (
        mean_similarity_high > mean_similarity
    ), "Higher κ should give samples closer to x_1"


def _train_loop_for_tests(
    model: ScoreMatchingModel,
    dataset,
    batch_size: int,
    learning_rate: float,
    epochs: int,
    test_dataset=None,
):
    """Training loop for score matching tests. Wraps the generic flow_matching loop."""

    def compute_nll_wrapper(
        model, params, batch, n_steps, rng, n_projections, precomputed_stats=None
    ):
        return compute_nll(
            model,
            params,
            batch,
            n_steps=n_steps,
            rng=rng,
            n_projections=n_projections,
        )

    return _train_loop_for_tests_generic(
        model,
        dataset,
        batch_size,
        learning_rate,
        epochs,
        test_dataset=test_dataset,
        train_step_fn=train_step,
        compute_batch_loss_fn=compute_batch_loss,
        compute_nll_fn=compute_nll_wrapper,
        compute_nll_on_step_0=False,
    )


_baseline_model = ScoreMatchingModel(
    domain_dim=3,
    reference_directions=128,
    n_layers=6,
    d_model=512,
    time_dim=None,
    mlp_expansion_factor=4,
    input_dropout_rate=None,
    mlp_dropout_rate=None,
    use_pre_mlp_projection=True,
)


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_trivial(domain_dim):
    """Train a model where all data is a single fixed point."""
    model = replace(_baseline_model, domain_dim=domain_dim, n_layers=2)

    batch_size = 256
    first_dim_vec = jnp.zeros(domain_dim)
    first_dim_vec = first_dim_vec.at[0].set(1.0)
    points = repeat(first_dim_vec, "v -> b v", b=batch_size * 100)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    test_dset = Dataset.from_dict({"point_vec": points[:batch_size]}).with_format("np")

    state, loss, test_loss, test_nll = _train_loop_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-2,
        epochs=2,
        test_dataset=test_dset,
    )
    print(f"Final loss: {loss:.6f}")

    samples = generate_samples(
        model,
        state.params,
        jax.random.PRNGKey(0),
        cond_vecs=jnp.zeros((20, 0)),
        n_steps=1000,
    )
    cos_sims = samples @ points[0]

    print(f"Sample cosine similarities for domain_dim={domain_dim}:")
    for i, (sample, cos_sim) in enumerate(zip(samples, cos_sims)):
        if domain_dim == 3 or i < 3:
            sample_str = ", ".join([f"{x:9.6f}" for x in sample[: min(3, domain_dim)]])
            if domain_dim > 3:
                sample_str += ", ..."
            print(f"Sample: [{sample_str}]  Cosine similarity: {cos_sim:9.6f}")

    high_sims = cos_sims > 0.99
    assert (
        high_sims.mean() >= 0.9
    ), f"Only {high_sims.mean()*100:.1f}% of samples close to target"

    # Single-point distribution: model should assign very high density, giving very negative NLL
    print(f"Model NLL: {test_nll:.4f}")
    assert (
        test_nll < -5
    ), f"NLL {test_nll:.4f} should be << 0 for single-point distribution"


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_vmf(domain_dim):
    """Train a model with data from a von Mises-Fisher distribution."""
    model = replace(
        _baseline_model,
        domain_dim=domain_dim,
        n_layers=2,
        d_model=256,
        schedule=NoiseSchedule(sigma_sq_min=0.01),
    )

    batch_size = 256
    n_samples = 32768

    mean_direction = np.zeros(domain_dim)
    mean_direction[0] = 1.0
    kappa_data = 2

    vmf_dist = stats.vonmises_fisher(mean_direction, kappa_data)
    data_rng = np.random.default_rng(42)
    points = vmf_dist.rvs(n_samples, random_state=data_rng)

    dset = Dataset.from_dict({"point_vec": points}).with_format("np")

    test_n_samples = 128
    test_points = vmf_dist.rvs(test_n_samples, random_state=data_rng)
    test_dset = Dataset.from_dict({"point_vec": test_points}).with_format("np")

    differential_entropy = vmf_dist.entropy()
    print(f"vMF distribution entropy: {differential_entropy:.6f}")

    state, train_loss = _train_loop_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-3,
        epochs=3,
    )
    print(f"Final train loss: {train_loss:.6f}")

    n_test_samples = 100
    samples = generate_samples(
        model,
        state.params,
        jax.random.PRNGKey(42),
        cond_vecs=jnp.zeros((n_test_samples, 0)),
        n_steps=100,
    )

    samples_np = np.array(samples)
    log_probs = vmf_dist.logpdf(samples_np)
    sample_nll = -np.mean(log_probs)

    print(f"Sample NLL: {sample_nll:.6f}")
    print(f"Differential entropy: {differential_entropy:.6f}")

    assert (
        sample_nll < differential_entropy + 1.0
    ), f"Sample NLL {sample_nll} too high compared to entropy {differential_entropy}"

    # Compare model NLL to theoretical entropy (computed post-hoc to avoid per-epoch overhead)
    test_nlls = compute_nll(
        model,
        state.params,
        dict(test_dset[:]),
        n_steps=256,
        rng=jax.random.PRNGKey(99),
        n_projections=32,
    )
    model_nll = float(jnp.mean(test_nlls))
    print(f"Model NLL: {model_nll:.4f}, vMF entropy: {differential_entropy:.4f}")
    assert np.isfinite(model_nll), f"NLL is not finite: {model_nll}"
    assert (
        model_nll < differential_entropy + 0.5
    ), f"Model NLL {model_nll:.4f} too high compared to entropy {differential_entropy:.4f}"


@pytest.mark.parametrize("domain_dim", [3, 16])
def test_zero_init_uniform_nll(domain_dim):
    """Verify that a freshly initialized model (zero-init output) gives correct uniform NLL."""
    model = replace(
        _baseline_model,
        domain_dim=domain_dim,
        n_layers=2,
        d_model=32,
        schedule=NoiseSchedule(sigma_sq_min=0.01),
    )

    params_rng, data_rng = jax.random.split(jax.random.PRNGKey(12345))
    state = create_train_state(params_rng, model, 1e-3)

    points = sample_sphere(data_rng, 256, domain_dim)
    batch = {"point_vec": points}

    test_nlls = compute_nll(
        model,
        state.params,
        batch,
        n_steps=256,
        rng=jax.random.PRNGKey(99),
        n_projections=32,
    )
    model_nll = float(jnp.mean(test_nlls))
    uniform_entropy = float(-sphere_log_inverse_surface_area(domain_dim))
    print(f"Model NLL: {model_nll:.4f}, uniform entropy: {uniform_entropy:.4f}")
    np.testing.assert_allclose(model_nll, uniform_entropy, atol=1e-2, rtol=0)
