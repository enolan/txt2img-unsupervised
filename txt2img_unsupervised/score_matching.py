"""
Score matching model for spherical data using a Variational Diffusion Model (VDM) design.

This implementation adapts denoising diffusion models to the unit sphere using vMF noise,
with a learned noise schedule and proper variational lower bound (VLB) loss.

## Theory and Design

### Forward Process
The forward process adds noise to data on the sphere. At time t, the conditional distribution
p_t(x|x_1) is vMF centered at x_1 with concentration κ(t), where κ is determined by a learned
monotonic schedule: log κ(t) = γ(t).

### Score Function
The score of the vMF distribution is ∇log p_t(x|x_1) = κ(t) · P_x(x_1), where P_x denotes
projection onto the tangent space at x.

### Time Convention
Following flow matching convention: t ∈ [0,1] where t=0 is pure noise, t=1 is data.

### Noise Schedule
The schedule is a learned monotonic function γ(t) = log κ(t) implemented by LearnedNoiseSchedule.
The endpoints γ(0) = log κ_min and γ(1) = log κ_max are also learned.

### VLB Loss
The variational lower bound decomposes into three terms:
  L_prior = KL(vMF(κ_min) ‖ Uniform)
  L_diff  = ½ E_t[ γ'(t) · κ(t) · ‖w_θ − w_true‖² ]
  L_recon = -log C_d(κ_max) - κ_max · A_d(κ_max)
where w = P_{x_t}(x_1) is the scaled score (tangent projection).

### Probability Flow ODE
For sampling, we integrate the probability flow ODE from t=0 (noise) to t=1 (data):
    dx/dt = ½ γ'(t) · w_θ(x, t)
The model's __call__ directly returns this ODE velocity.

### Model Reuse
We reuse the VectorField class from flow_matching.py since estimating a tangent vector at x
has the same shape as estimating a velocity field.
"""

import math

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, FrozenSet, Literal, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
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
    generate_samples_inner,
)
from txt2img_unsupervised.learned_schedule import LearnedNoiseSchedule
from txt2img_unsupervised.vmf import (
    kl_vmf_uniform,
)


class ScoreMatchingModel(nn.Module):
    """Score matching model bundling a VectorField with a learned noise schedule.

    The neural network (vector_field) predicts the scaled score w_θ = P_{x_t}(x_1),
    and __call__ returns the ODE velocity ½γ'(t)·w_θ for direct use by the ODE solver.
    """

    # VectorField hyperparameters
    domain_dim: int
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

    # Learned schedule hyperparameters
    schedule_hidden_dim: int = 32
    schedule_n_quadrature_points: int = 1024
    init_log_kappa_min: float = -0.693  # log(0.5)
    init_log_kappa_max: float = 9.210  # log(10000)

    cap_conditioning: CapConditioningMode = CapConditioningMode.UNCONDITIONED

    @property
    def conditioning_dim(self) -> int:
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return 0
        elif self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            return cap_conditioning_dim(self.domain_dim)
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            return 0
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")

    @nn.nowrap
    def mk_vector_field(self) -> VectorField:
        """Create a VectorField with parameters derived from this model's config."""
        return VectorField(
            domain_dim=self.domain_dim,
            conditioning_dim=self.conditioning_dim,
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
        self.schedule = LearnedNoiseSchedule(
            hidden_dim=self.schedule_hidden_dim,
            n_quadrature_points=self.schedule_n_quadrature_points,
            init_log_kappa_min=self.init_log_kappa_min,
            init_log_kappa_max=self.init_log_kappa_max,
        )

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
                ],
                "schedule": "schedule",
            }
        }

    @nn.nowrap
    def scale_lr(self, lr: float) -> float:
        """Scaled learning rate for hidden layers."""
        return self.mk_vector_field().scale_lr(lr)

    def __call__(self, x, t, cond_vec):
        """Run the model, returning the ODE velocity ½γ'(t)·w_θ(x, t).

        Args:
            x: Points on the sphere [batch_size, domain_dim]
            t: Time values [batch_size]
            cond_vec: Conditioning vectors [batch_size, cond_dim]

        Returns:
            ODE velocity vectors [batch_size, domain_dim] in the tangent space at x.
        """
        w = self.vector_field(x, t, cond_vec)
        gamma_prime = self.schedule.log_kappa_derivative(t)
        return 0.5 * gamma_prime[:, None] * w

    def compute_vlb_loss(self, x_1, t, cond_vec):
        """Compute the VLB loss combining prior, diffusion, and reconstruction terms.

        Args:
            x_1: Clean data points [batch_size, dim]
            t: Uniformly sampled time values [batch_size]
            cond_vec: Conditioning vectors [batch_size, cond_dim]

        Returns:
            Scalar VLB loss value.
        """
        batch_size = x_1.shape[0]
        dim = self.domain_dim

        # Get schedule values
        log_kappa_t = self.schedule(t)
        kappa_t = jnp.exp(log_kappa_t)
        gamma_prime = self.schedule.log_kappa_derivative(t)

        # Sample noisy points x_t ~ vMF(x_1, κ(t)).
        # stop_gradient: the sampling process uses rejection sampling (lax.while_loop)
        # which doesn't support reverse-mode differentiation. Gradients w.r.t. the schedule
        # flow through the loss weighting (gamma_prime * kappa_t) and boundary losses instead.
        noise_rng = self.make_rng("noise")
        x_t = vmf.sample(
            noise_rng, x_1, jax.lax.stop_gradient(kappa_t), n_samples=batch_size
        )

        # Predict scaled score and compute target
        w_theta = self.vector_field(x_t, t, cond_vec)
        w_true = tangent_projection(x_t, x_1)

        # Diffusion loss: ½ E_t[ γ'(t) · κ(t) · ‖w_θ − w_true‖² ]
        per_sample_sq_err = jnp.sum((w_theta - w_true) ** 2, axis=1)
        diffusion_loss = 0.5 * jnp.mean(gamma_prime * kappa_t * per_sample_sq_err)

        # Prior loss: KL(vMF(κ_min) ‖ Uniform)
        kappa_min = jnp.exp(self.schedule.log_kappa_min)
        prior_loss = kl_vmf_uniform(kappa_min, dim)

        # Reconstruction loss: -log C_d(κ_max) - κ_max · A_d(κ_max)
        kappa_max = jnp.exp(self.schedule.log_kappa_max)
        recon_loss = vmf_recon_loss(kappa_max, dim)

        return diffusion_loss + prior_loss + recon_loss


def vmf_recon_loss(kappa, dim):
    """Compute the VLB reconstruction loss: -log C_d(κ) - κ·A_d(κ).

    This is the expected negative log-likelihood of the data under the final-time
    vMF distribution, i.e. -E_{x~vMF(x_1, κ)}[log p(x|x_1)] = -(log C_d(κ) + κ·A_d(κ)).

    Args:
        kappa: Concentration at t=1 (JAX array).
        dim: Ambient space dimension (int).

    Returns:
        Reconstruction loss, same shape as kappa.
    """
    log_surface_area = -sphere_log_inverse_surface_area(dim)
    # H(vMF) = log|S^{d-1}| - KL(vMF || Uniform). Reusing the numerically stable
    # KL implementation avoids both small-κ cancellation in high dimensions and
    # large-κ cancellation in low dimensions.
    return log_surface_area - kl_vmf_uniform(kappa, dim)


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


@partial(jax.jit, static_argnames=("model",))
def compute_batch_loss(
    model: ScoreMatchingModel,
    params,
    batch: dict,
    rng: Array,
) -> Array:
    """Extract data from batch, sample t, compute VLB loss.

    Args:
        model: Score matching model
        params: Model parameters
        batch: Batch of data containing "point_vec" and optionally "cond_vec"
        rng: JAX random key

    Returns:
        The computed VLB loss value
    """
    x_1 = batch["point_vec"]
    batch_size = x_1.shape[0]

    noise_rng, time_rng, dropout_rng = jax.random.split(rng, 3)

    cond_vecs = batch.get("cond_vec", jnp.zeros((batch_size, 0)))

    t = jax.random.uniform(time_rng, (batch_size,))

    return model.apply(
        params,
        x_1,
        t,
        cond_vecs,
        rngs={"dropout": dropout_rng, "noise": noise_rng},
        method=model.compute_vlb_loss,
    )


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

    model.__call__ directly returns the ODE velocity ½γ'(t)·w_θ.

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

    return model.apply(params, x, t_vec, cond_vecs, rngs=rngs_dict)


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

    Integrates dx/dt = ½γ'(t)·w_θ(x, t) from t=0 (noise) to t=1 (data).

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


def test_target_score_at_mode():
    """Test that tangent projection is zero when x_t = x_1 (at the mode)."""
    batch_size = 10
    dim = 3
    rng = jax.random.PRNGKey(42)

    x_1 = sample_sphere(rng, batch_size, dim)
    x_t = x_1  # Same point

    target = tangent_projection(x_t, x_1)
    np.testing.assert_allclose(target, 0.0, atol=1e-5)


def test_sample_noisy_point():
    """Test that vMF sampling produces valid samples with correct concentration behavior."""
    batch_size = 1000
    dim = 3

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(456), 3)
    x_1 = sample_sphere(key1, batch_size, dim)
    kappa_low = 10.0
    kappa_high = 100.0

    x_t_low = vmf.sample(
        key2, x_1, jnp.full((batch_size,), kappa_low), n_samples=batch_size
    )
    x_t_high = vmf.sample(
        key3, x_1, jnp.full((batch_size,), kappa_high), n_samples=batch_size
    )

    norms = jnp.linalg.norm(x_t_low, axis=1)
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    similarities_low = jnp.mean(jnp.sum(x_t_low * x_1, axis=1))
    similarities_high = jnp.mean(jnp.sum(x_t_high * x_1, axis=1))

    assert similarities_low > 0.8, f"Mean similarity {similarities_low} should be > 0.8"
    assert (
        similarities_high > similarities_low
    ), "Higher κ should give samples closer to x_1"


def _make_model(domain_dim, **kwargs):
    """Create a ScoreMatchingModel with test-friendly defaults."""
    defaults = dict(
        domain_dim=domain_dim,
        n_layers=2,
        d_model=512,
        mlp_expansion_factor=4,
        input_dropout_rate=None,
        mlp_dropout_rate=None,
        use_pre_mlp_projection=True,
    )
    defaults.update(kwargs)
    return ScoreMatchingModel(**defaults)


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_trivial(domain_dim):
    """Train a model where all data is a single fixed point."""
    from txt2img_unsupervised.training_infra import train_for_tests

    model = _make_model(domain_dim)

    batch_size = 256
    first_dim_vec = jnp.zeros(domain_dim)
    first_dim_vec = first_dim_vec.at[0].set(1.0)
    points = repeat(first_dim_vec, "v -> b v", b=batch_size * 100)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    test_dset = Dataset.from_dict({"point_vec": points[:batch_size]}).with_format("np")

    loss_fn = partial(compute_batch_loss, model)

    def nll_fn(model, params, batch, rng):
        return compute_nll(model, params, batch, n_steps=256, rng=rng, n_projections=32)

    result = train_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-2,
        loss_fn=loss_fn,
        fields=["point_vec"],
        epochs=2,
        test_dataset=test_dset,
        nll_fn=nll_fn,
    )
    eval_params = result.state.get_eval_params()

    samples = generate_samples(
        model,
        eval_params,
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
    print(f"Model NLL: {result.test_nll:.4f}")
    assert (
        result.test_nll < -5
    ), f"NLL {result.test_nll:.4f} should be << 0 for single-point distribution"


def _sample_sphere_np(n, dim, rng):
    """Sample uniformly from the unit sphere using numpy."""
    points = rng.standard_normal((n, dim))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


@dataclass
class _TargetDistribution:
    """A target distribution for training tests.

    Attributes:
        rvs: Sample function (n, rng) -> [n, dim] array.
        logpdf: Log-probability function (points) -> [n] array. May return -inf.
        entropy: Differential entropy of the distribution.
        interior_mask: Returns True for points well inside the support, far from any boundary,
            where we expect the model's density to be accurate.
        exterior_mask: Returns True for points well outside the support, far from any boundary,
            where we expect the model to assign very low density.
    """

    rvs: Callable[[int, np.random.Generator], np.ndarray]
    logpdf: Callable[[np.ndarray], np.ndarray]
    entropy: float
    interior_mask: Callable[[np.ndarray], np.ndarray]
    exterior_mask: Callable[[np.ndarray], np.ndarray]


def _make_target_distribution(name, domain_dim):
    """Create a target distribution for training tests."""
    _all_true = lambda x: np.ones(x.shape[0], dtype=bool)
    _all_false = lambda x: np.zeros(x.shape[0], dtype=bool)

    if name == "vmf":
        mean_direction = np.zeros(domain_dim)
        mean_direction[0] = 1.0
        vmf_dist = stats.vonmises_fisher(mean_direction, 2)
        return _TargetDistribution(
            rvs=vmf_dist.rvs,
            logpdf=vmf_dist.logpdf,
            entropy=vmf_dist.entropy(),
            interior_mask=_all_true,
            exterior_mask=_all_false,
        )
    elif name == "uniform":
        log_inv_sa = float(sphere_log_inverse_surface_area(domain_dim))
        return _TargetDistribution(
            rvs=lambda n, rng: _sample_sphere_np(n, domain_dim, rng),
            logpdf=lambda x: np.full(x.shape[0], log_inv_sa),
            entropy=-log_inv_sa,
            interior_mask=_all_true,
            exterior_mask=_all_false,
        )
    elif name == "hemisphere":
        log_inv_sa = float(sphere_log_inverse_surface_area(domain_dim))
        hemi_logp = log_inv_sa + np.log(2)
        margin = 0.05

        def hemi_rvs(n, rng):
            points = _sample_sphere_np(n, domain_dim, rng)
            points[points[:, 0] < 0] *= -1
            return points

        def hemi_logpdf(x):
            result = np.full(x.shape[0], hemi_logp)
            result[x[:, 0] <= 0] = -np.inf
            return result

        return _TargetDistribution(
            rvs=hemi_rvs,
            logpdf=hemi_logpdf,
            entropy=-hemi_logp,
            interior_mask=lambda x: x[:, 0] > margin,
            exterior_mask=lambda x: x[:, 0] < -margin,
        )
    else:
        raise ValueError(f"Unknown distribution: {name}")


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
@pytest.mark.parametrize("dist_name", ["vmf", "uniform", "hemisphere"])
def test_train_distribution(domain_dim, dist_name):
    """Train a model on a distribution and verify learned densities match at uniform test points."""
    from txt2img_unsupervised.training_infra import train_for_tests

    dist = _make_target_distribution(dist_name, domain_dim)

    model = _make_model(
        domain_dim,
        d_model=256,
        init_log_kappa_min=-0.693,
        init_log_kappa_max=jnp.log(100.0).item(),  # κ_max = 100
    )

    batch_size = 256
    n_samples = 32768

    data_rng = np.random.default_rng(42)
    points = dist.rvs(n_samples, data_rng)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")

    test_n_samples = 128
    test_points = dist.rvs(test_n_samples, data_rng)
    test_dset = Dataset.from_dict({"point_vec": test_points}).with_format("np")

    print(f"Distribution entropy: {dist.entropy:.6f}")

    loss_fn = partial(compute_batch_loss, model)

    def nll_fn(model, params, batch, rng):
        return compute_nll(model, params, batch, n_steps=256, rng=rng, n_projections=32)

    result = train_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-3,
        loss_fn=loss_fn,
        fields=["point_vec"],
        epochs=5,
        test_dataset=test_dset,
        nll_fn=nll_fn,
    )
    eval_params = result.state.get_eval_params()

    # Generate samples and check quality under the true distribution
    n_gen_samples = 100
    samples = generate_samples(
        model,
        eval_params,
        jax.random.PRNGKey(42),
        cond_vecs=jnp.zeros((n_gen_samples, 0)),
        n_steps=100,
    )
    samples_np = np.array(samples)
    sample_log_probs = dist.logpdf(samples_np)

    sample_in_support = np.isfinite(sample_log_probs)
    in_support_frac = np.mean(sample_in_support)
    print(f"Fraction of generated samples in support: {in_support_frac:.4f}")
    if not np.all(sample_in_support):
        assert (
            in_support_frac > 0.8
        ), f"Only {in_support_frac:.1%} of samples are in-support"

    if np.any(sample_in_support):
        in_support_nll = -np.mean(sample_log_probs[sample_in_support])
        print(
            f"In-support sample NLL: {in_support_nll:.6f}, entropy: {dist.entropy:.6f}"
        )
        assert (
            in_support_nll < dist.entropy + 1.0
        ), f"In-support sample NLL {in_support_nll} too high vs entropy {dist.entropy}"

    # Compare model log-likelihoods to true log-likelihoods at uniformly distributed test
    # points. This tests that the model has learned the correct density shape everywhere.
    n_uniform = 256
    uniform_points = np.array(
        sample_sphere(jax.random.PRNGKey(99), n_uniform, domain_dim)
    )
    model_log_probs = -np.array(
        compute_nll(
            model,
            eval_params,
            {"point_vec": jnp.array(uniform_points)},
            n_steps=256,
            rng=jax.random.PRNGKey(123),
            n_projections=32,
        )
    )
    true_log_probs = dist.logpdf(uniform_points)

    # Check density accuracy well inside the support (away from any boundary).
    interior = dist.interior_mask(uniform_points)
    n_interior = np.sum(interior)
    print(f"Interior points: {n_interior}/{n_uniform}")
    assert (
        n_interior >= 10
    ), f"Too few interior points ({n_interior}) for meaningful check"
    mae = np.mean(np.abs(model_log_probs[interior] - true_log_probs[interior]))
    print(f"Interior MAE of log-probs: {mae:.4f}")
    assert mae < 0.5, f"Interior MAE {mae:.4f} too high"

    # Check that the model assigns low density well outside the support.
    exterior = dist.exterior_mask(uniform_points)
    n_exterior = np.sum(exterior)
    if n_exterior > 0:
        mean_interior_lp = np.mean(true_log_probs[interior])
        mean_exterior_lp = np.mean(model_log_probs[exterior])
        print(f"Mean interior true log-prob: {mean_interior_lp:.4f}")
        print(f"Mean exterior model log-prob: {mean_exterior_lp:.4f}")
        # Average exterior density should be <5% of interior density (3 nats gap)
        assert mean_exterior_lp < mean_interior_lp - 3, (
            f"Mean exterior log-prob {mean_exterior_lp:.4f} too close to "
            f"interior mean {mean_interior_lp:.4f} (gap {mean_interior_lp - mean_exterior_lp:.2f}, need >3)"
        )


@pytest.mark.parametrize("domain_dim", [3, 16])
def test_zero_init_uniform_nll(domain_dim):
    """Verify that a freshly initialized model (zero-init output) gives correct uniform NLL.

    At initialization, w_θ ≈ 0, so the ODE velocity ≈ 0, and the model
    should produce uniform density on the sphere.
    """
    model = _make_model(domain_dim, d_model=32)

    params_rng, data_rng = jax.random.split(jax.random.PRNGKey(12345))
    params = model.init(params_rng, *model.dummy_inputs())

    points = sample_sphere(data_rng, 256, domain_dim)
    batch = {"point_vec": points}

    test_nlls = compute_nll(
        model,
        params,
        batch,
        n_steps=256,
        rng=jax.random.PRNGKey(99),
        n_projections=32,
    )
    model_nll = float(jnp.mean(test_nlls))
    uniform_entropy = float(-sphere_log_inverse_surface_area(domain_dim))
    print(f"Model NLL: {model_nll:.4f}, uniform entropy: {uniform_entropy:.4f}")
    np.testing.assert_allclose(model_nll, uniform_entropy, atol=1e-2, rtol=0)


def test_vlb_loss_components():
    """Verify VLB loss components are reasonable at initialization."""
    dim = 3
    model = _make_model(dim, d_model=32)

    params_rng, data_rng, noise_rng = jax.random.split(jax.random.PRNGKey(42), 3)
    params = model.init(params_rng, *model.dummy_inputs())

    x_1 = sample_sphere(data_rng, 64, dim)
    t = jnp.linspace(0.01, 0.99, 64)
    cond_vecs = jnp.zeros((64, 0))

    loss = model.apply(
        params,
        x_1,
        t,
        cond_vecs,
        rngs={"noise": noise_rng},
        method=model.compute_vlb_loss,
    )

    assert jnp.isfinite(loss), f"VLB loss is not finite: {loss}"
    assert float(loss) > 0, f"VLB loss should be positive, got {loss}"

    # Prior loss should be small for the default small κ_min
    kappa_min = jnp.exp(jnp.array(model.init_log_kappa_min))
    prior = float(kl_vmf_uniform(kappa_min, dim))
    print(f"Prior loss: {prior:.6f}")
    assert (
        prior < 1.0
    ), f"Prior loss {prior} too large for κ_min = {float(kappa_min):.3f}"

    # Recon loss should be finite
    kappa_max = jnp.exp(jnp.array(model.init_log_kappa_max))
    recon = float(vmf_recon_loss(kappa_max, dim))
    print(f"Recon loss: {recon:.6f}")
    assert jnp.isfinite(jnp.array(recon)), f"Recon loss is not finite: {recon}"


def test_vlb_loss_gradients_flow():
    """Verify that gradients flow through the VLB loss to both vector field and schedule params."""
    dim = 3
    model = _make_model(dim, d_model=32)

    params_rng, data_rng, noise_rng = jax.random.split(jax.random.PRNGKey(7), 3)
    params = model.init(params_rng, *model.dummy_inputs())

    x_1 = sample_sphere(data_rng, 32, dim)
    t = jax.random.uniform(jax.random.PRNGKey(0), (32,))
    cond_vecs = jnp.zeros((32, 0))

    def loss_fn(p):
        return model.apply(
            p,
            x_1,
            t,
            cond_vecs,
            rngs={"noise": noise_rng},
            method=model.compute_vlb_loss,
        )

    grads = jax.grad(loss_fn)(params)

    # Check vector field grads exist
    vf_grads = grads["params"]["vector_field"]
    flat_vf = jax.tree_util.tree_leaves(vf_grads)
    assert any(jnp.any(g != 0) for g in flat_vf), "No gradient flow to vector field"

    # Check schedule grads exist
    sched_grads = grads["params"]["schedule"]
    flat_sched = jax.tree_util.tree_leaves(sched_grads)
    assert any(jnp.any(g != 0) for g in flat_sched), "No gradient flow to schedule"


def test_vmf_recon_loss_gradient_flows():
    """Test that gradients flow through vmf_recon_loss."""
    grad_fn = jax.grad(lambda k: vmf_recon_loss(k, 768))
    g = grad_fn(jnp.array(100.0))
    assert jnp.isfinite(g)


def _exact_vmf_recon_loss_dim_3(kappa: float) -> float:
    """Closed-form vMF entropy in d=3."""
    log_sinh = kappa - np.log(2.0) + np.log1p(-np.exp(-2.0 * kappa))
    return np.log(4.0 * np.pi) + log_sinh - np.log(kappa) - kappa / np.tanh(kappa) + 1.0


@pytest.mark.parametrize("kappa", [1e-3, 0.1, 1.0, 100.0, 10000.0])
def test_vmf_recon_loss_matches_closed_form_dim_3(kappa):
    """Test vmf_recon_loss against the exact d=3 entropy formula."""
    result = float(vmf_recon_loss(jnp.array(kappa), 3))
    expected = _exact_vmf_recon_loss_dim_3(kappa)
    np.testing.assert_allclose(result, expected, atol=1e-6, err_msg=f"κ={kappa}")


@pytest.mark.parametrize("kappa", [1e-6, 1e-3, 0.1])
def test_vmf_recon_loss_gradient_small_kappa_high_dim(kappa):
    """Test the small-κ entropy gradient in high dimension."""
    dim = 768
    grad_fn = jax.grad(lambda concentration: vmf_recon_loss(concentration, dim))
    gradient = float(grad_fn(jnp.array(kappa)))
    expected = -(kappa / dim + kappa**3 / (2.0 * dim * (dim + 2)))
    np.testing.assert_allclose(
        gradient,
        expected,
        rtol=5e-4,
        atol=1e-12,
        err_msg=f"κ={kappa}",
    )
