"""
Euclidean Variational Diffusion Model (VDM) for spherical data.

Operates in R^d with standard Gaussian noise (Kingma et al. 2021), giving a provably
valid ELBO. Data on S^{d-1} is augmented with small radial noise so it fills a thin
shell in R^d. For sampling, the output is normalized to the unit sphere.

## Theory

### Forward Process
Given data x on S^{d-1}, we augment with radial noise: x_data = r·x where r ~ N(1, σ_r²).
Then add Gaussian noise: z_t = α_t·x_data + σ_t·ε, ε ~ N(0, I).
The schedule γ(t) = log(α²/σ²) = log SNR(t) increases from γ_min (noise) to γ_max (data),
with α² = sigmoid(γ) and σ² = sigmoid(-γ).

### VLB (provably valid)
The ELBO decomposes into:
  L_prior = ½[α_T²·‖x‖² + d·σ_T² − d − d·log σ_T²]
  L_diff  = ½ E_t[ γ'(t) · ‖ε̂_θ − ε‖² ]
  L_recon = ½ d [1 + log 2π − γ_max]
VLB ≥ NLL_ℝd holds by the standard Gaussian VDM construction.

### Spherical NLL
The model's Euclidean density relates to spherical density by:
  NLL_sphere(x) = NLL_ℝd(x) − log σ_radial − ½ log(2π)
This holds because the model learns a radial distribution ≈ N(1, σ_r²), contributing
a constant factor 1/(σ_r√2π) at r=1. Since VLB_ℝd ≥ NLL_ℝd, we also get
VLB_sphere ≥ NLL_sphere.

### Probability Flow ODE
For sampling, integrate from t=0 (noise) to t=1 (data):
    dz/dt = ½ γ'(t) [σ² z − σ ε̂_θ(z, t)]
Then normalize: x = z / ‖z‖.
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

from txt2img_unsupervised.cap_sampling import (
    LogitsTable,
    cap_conditioning_dim,
    encode_cap_params,
    sample_cap,
    sphere_log_inverse_surface_area,
)
from txt2img_unsupervised.config import CapConditioningMode
from txt2img_unsupervised.flow_matching import (
    VectorField,
    sample_sphere,
    stratified_time_sample,
)
from txt2img_unsupervised.learned_schedule import LearnedNoiseSchedule


# =============================================================================
# Model
# =============================================================================


class EuclideanDiffusionModel(nn.Module):
    """Euclidean VDM for data on S^{d-1}, with radial augmentation.

    The neural network predicts the noise ε̂ added in the forward process, and
    __call__ returns the probability flow ODE velocity ½γ'(t)[σ²z − σε̂].
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
    init_log_snr_min: float = -10.0
    init_log_snr_max: float = 10.0

    # Radial augmentation noise std
    sigma_radial: float = 0.01

    # Hard ceiling on the learned log-SNR maximum. Prevents the schedule from pushing
    # γ_max to extreme values where float32 precision degrades and the recon term
    # dominates the VLB without improving actual sample quality.
    log_snr_max_cap: Optional[float] = None

    cap_conditioning: CapConditioningMode = CapConditioningMode.UNCONDITIONED
    d_max_dist: Optional[tuple] = None
    vlb_variance_loss_weight: Optional[float] = None

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
        """Create a VectorField with Euclidean output (no tangent projection)."""
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
            project_to_tangent=False,
        )

    def setup(self):
        if self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            self.logits_table = LogitsTable(self.domain_dim - 1, 8192)
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            raise NotImplementedError("Classifier guidance is not yet implemented")
        self.vector_field = self.mk_vector_field()
        self.schedule = LearnedNoiseSchedule(
            hidden_dim=self.schedule_hidden_dim,
            n_quadrature_points=self.schedule_n_quadrature_points,
            init_log_kappa_min=self.init_log_snr_min,
            init_log_kappa_max=self.init_log_snr_max,
            log_kappa_max_cap=self.log_snr_max_cap,
        )

    @nn.nowrap
    def dummy_inputs(self):
        """Create dummy inputs for model initialization."""
        z = jnp.ones((1, self.domain_dim))
        t = jnp.ones((1,))
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            cap_params = None
        elif self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            cap_params = (jnp.ones((1, self.domain_dim)), jnp.ones((1,)))
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            cap_params = None
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")
        return z, t, cap_params

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

    def gamma_range(self):
        """Return the effective (γ_min, γ_max) schedule endpoints."""
        return self.schedule.log_kappa_min, self.schedule.effective_log_kappa_max

    def prepare_training_conditioning(self, batch):
        """Sample cap parameters for each point in the batch."""
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return None
        elif self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            x1_batch = batch["point_vec"]
            batch_size = x1_batch.shape[0]
            rng = self.make_rng("sample_cap_params")
            rngs = jax.random.split(rng, batch_size)
            return jax.vmap(
                lambda rng, x: sample_cap(self.logits_table, rng, x, self.d_max_dist)
            )(rngs, x1_batch)
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            return None
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")

    def _cap_params_to_cond_vecs(self, cap_params, batch_size):
        """Convert cap_params to conditioning vectors for the vector field."""
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return jnp.zeros((batch_size, 0))
        elif self.cap_conditioning == CapConditioningMode.CONDITIONED_SCORE:
            cap_centers, d_maxes = cap_params
            return encode_cap_params(
                cap_center=cap_centers,
                d_max=d_maxes,
                d_max_dist=self.d_max_dist,
                domain_dim=self.domain_dim,
            )
        elif self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            raise NotImplementedError("Classifier guidance is not yet implemented")
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")

    def _normalize_log_snr(self, log_snr):
        """Normalize log_snr to [0, 1] at initialization using init endpoints. This intentionally
        uses the *initial* endpoints rather than the learned ones, so that changes to the endpoints
        don't change the objective."""
        return (log_snr - self.init_log_snr_min) / (
            self.init_log_snr_max - self.init_log_snr_min
        )

    def predict_eps(self, z, log_snr, cap_params):
        """Predict the noise ε̂ at the given log-SNR level.

        Args:
            z: Points in R^d [batch_size, domain_dim]
            log_snr: Log SNR values [batch_size]
            cap_params: None for UNCONDITIONED, (cap_centers, d_maxes) for CONDITIONED_SCORE

        Returns:
            Predicted noise [batch_size, domain_dim].
        """
        cond_vecs = self._cap_params_to_cond_vecs(cap_params, z.shape[0])
        t_normalized = self._normalize_log_snr(log_snr)
        return self.vector_field(z, t_normalized, cond_vecs)

    def gamma_space_velocity(self, z, log_snr, cap_params):
        """Compute the ODE velocity in log-SNR space: ½[σ²z − σε̂].

        This is dz/dγ where γ = log SNR. It equals (dz/dt)/γ'(t), cancelling
        the schedule derivative so the velocity depends only on the noise level,
        not on how the schedule maps time to noise level. Used for sampling and
        NLL computation where uniform steps in γ-space avoid the pathology of
        schedules that concentrate all denoising into a tiny time interval.

        Args:
            z: Points in R^d [batch_size, domain_dim]
            log_snr: Log SNR values [batch_size]
            cap_params: None for UNCONDITIONED, (cap_centers, d_maxes) for CONDITIONED_SCORE

        Returns:
            dz/dγ vectors [batch_size, domain_dim].
        """
        _, sigma = _snr_to_alpha_sigma(log_snr)
        eps_hat = self.predict_eps(z, log_snr, cap_params)
        return 0.5 * (sigma[:, None] ** 2 * z - sigma[:, None] * eps_hat)

    def __call__(self, z, t, cap_params):
        """Run the model, returning the ODE velocity in t-space: ½γ'(t)[σ²z − σε̂].

        Used for training (VLB loss). For sampling and NLL, use gamma_space_velocity
        which integrates in log-SNR space and avoids schedule-dependent velocity scaling.

        Args:
            z: Points in R^d [batch_size, domain_dim]
            t: Time values [batch_size]
            cap_params: None for UNCONDITIONED, (cap_centers, d_maxes) for CONDITIONED_SCORE

        Returns:
            ODE velocity vectors [batch_size, domain_dim].
        """
        log_snr = self.schedule(t)
        gamma_prime = self.schedule.log_kappa_derivative(t)
        v_gamma = self.gamma_space_velocity(z, log_snr, cap_params)
        return gamma_prime[:, None] * v_gamma

    def compute_vlb_loss(self, x_1, t, cap_params):
        """Compute the VLB loss with proper Gaussian ELBO.

        Args:
            x_1: Data points on S^{d-1} [batch_size, dim]
            t: Uniformly sampled time values [batch_size]
            cap_params: None for UNCONDITIONED, (cap_centers, d_maxes) for CONDITIONED_SCORE

        Returns:
            Tuple of (total_loss, components_dict).
        """
        batch_size = x_1.shape[0]
        dim = self.domain_dim

        # Radial augmentation: perturb data off the sphere
        radial_rng = self.make_rng("noise")
        radial_rng, noise_rng = jax.random.split(radial_rng)
        r = 1.0 + self.sigma_radial * jax.random.normal(radial_rng, (batch_size, 1))
        x_data = x_1 * r

        # Get schedule values
        log_snr = self.schedule(t)
        alpha, sigma = _snr_to_alpha_sigma(log_snr)
        gamma_prime = self.schedule.log_kappa_derivative(t)

        # Forward process: z_t = α·x_data + σ·ε
        eps = jax.random.normal(noise_rng, x_data.shape)
        z_t = alpha[:, None] * x_data + sigma[:, None] * eps

        # Predict noise
        cond_vecs = self._cap_params_to_cond_vecs(cap_params, batch_size)
        t_normalized = self._normalize_log_snr(log_snr)
        eps_hat = self.vector_field(z_t, t_normalized, cond_vecs)

        # Diffusion loss: ½ γ'(t) · ‖ε̂ − ε‖²
        per_sample_sq_err = jnp.sum((eps_hat - eps) ** 2, axis=1)
        diffusion_loss = 0.5 * jnp.mean(gamma_prime * per_sample_sq_err)

        # Prior loss: KL(q(z_T|x) || N(0,I)) at t=0 (noise end)
        log_snr_min = self.schedule.log_kappa_min
        alpha_T, sigma_T = _snr_to_alpha_sigma(log_snr_min)
        x_data_sq_norm = jnp.mean(jnp.sum(x_data**2, axis=1))
        prior_loss = 0.5 * (
            alpha_T**2 * x_data_sq_norm
            + dim * sigma_T**2
            - dim
            - dim * jnp.log(sigma_T**2)
        )

        # Reconstruction loss: ½ d [1 + log 2π − γ_max]
        log_snr_max = self.schedule.effective_log_kappa_max
        recon_loss = 0.5 * dim * (1.0 + jnp.log(2.0 * jnp.pi) - log_snr_max)

        vlb_total = diffusion_loss + prior_loss + recon_loss

        # Spherical VLB: subtract constant offset for comparability
        spherical_offset = jnp.log(self.sigma_radial) + 0.5 * jnp.log(2.0 * jnp.pi)
        vlb_spherical = vlb_total - spherical_offset

        # VLB variance minimization
        if self.vlb_variance_loss_weight is not None:
            f_i = 0.5 * gamma_prime * jax.lax.stop_gradient(per_sample_sq_err)
            variance_loss = jnp.mean(f_i**2) * self.vlb_variance_loss_weight
        else:
            variance_loss = jnp.array(0.0)

        components = {
            "vlb_total": vlb_total,
            "vlb_spherical": vlb_spherical,
            "diffusion": diffusion_loss,
            "prior": prior_loss,
            "recon": recon_loss,
            "variance": variance_loss,
        }
        return vlb_total + variance_loss, components


# =============================================================================
# Training
# =============================================================================


@partial(jax.jit, static_argnames=("model",))
def compute_batch_loss(
    model: EuclideanDiffusionModel,
    params,
    batch: dict,
    rng: Array,
) -> tuple[Array, dict[str, Array]]:
    """Extract data from batch, sample t, compute VLB loss.

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        batch: Batch of data containing "point_vec"
        rng: JAX random key

    Returns:
        Tuple of (total_loss, components_dict).
    """
    x_1 = batch["point_vec"]
    batch_size = x_1.shape[0]

    noise_rng, time_rng, dropout_rng, cap_rng = jax.random.split(rng, 4)

    cap_params = model.apply(
        params,
        batch,
        method=model.prepare_training_conditioning,
        rngs={"sample_cap_params": cap_rng},
    )

    t = stratified_time_sample(time_rng, batch_size)

    return model.apply(
        params,
        x_1,
        t,
        cap_params,
        rngs={"dropout": dropout_rng, "noise": noise_rng},
        method=model.compute_vlb_loss,
    )


# =============================================================================
# Sampling
# =============================================================================


@partial(jax.jit, static_argnames=("model", "n_steps", "batch_size"))
def generate_samples_sde(
    model: EuclideanDiffusionModel,
    params,
    rng: Array,
    cap_params,
    n_steps: int = 200,
    batch_size: Optional[int] = None,
    eta: float = 1.0,
) -> Array:
    """Generate samples using the DDPM ancestral sampler in log-SNR space.

    Uses exact Gaussian reverse transitions (DDPM ancestral step) with the eta
    parameter scaling the noise. eta=1 gives the full reverse SDE; eta<1 retains
    the SDE drift (which provides twice the denoising of the ODE) while reducing
    stochastic spread. This is particularly useful in high dimensions where the
    full SDE noise can push samples away from concentrated targets.

    At each step from γ_curr to γ_next (toward data):
        μ = (α_next/α_curr)·z − (α_next² − α_curr²)/(α_next·σ_curr·α_curr)·ε̂
        σ_post² = (α_next² − α_curr²)·σ_next² / (α_next²·σ_curr²)
        z_next = μ + eta·σ_post·noise

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        rng: JAX random key
        cap_params: None for unconditioned, (cap_centers, d_maxes) for CONDITIONED_SCORE
        n_steps: Number of sampling steps in γ-space
        batch_size: Required if cap_params is None
        eta: Noise scaling in [0, 1]. 1=full DDPM/SDE, 0=deterministic (SDE drift only).

    Returns:
        Generated samples on S^{d-1} [batch_size, domain_dim]
    """
    if cap_params is not None:
        leading_dims = jax.tree.leaves(jax.tree.map(lambda x: x.shape[0], cap_params))
        inferred = leading_dims[0]
        if batch_size is None:
            batch_size = inferred
        elif batch_size != inferred:
            raise ValueError(
                f"batch_size {batch_size} doesn't match cap_params leading dim {inferred}"
            )
    elif batch_size is None:
        raise ValueError("batch_size must be specified when cap_params is None")

    gamma_min, gamma_max = model.apply(params, method=model.gamma_range)
    d_gamma = (gamma_max - gamma_min) / n_steps

    init_rng, loop_rng = jax.random.split(rng)
    z = jax.random.normal(init_rng, (batch_size, model.domain_dim))

    def body_fn(i, state):
        z, rng = state
        rng, step_rng = jax.random.split(rng)

        gamma_curr = gamma_min + i * d_gamma
        gamma_next = gamma_curr + d_gamma

        eps_hat = _predict_eps(model, params, cap_params, z, gamma_curr)

        # Ancestral (DDPM) reverse step with noise scaling by eta.
        # eta=1 is the full DDPM ancestral step (reverse SDE).
        # eta<1 reduces stochastic variation while keeping the SDE drift,
        # which provides some error correction without as much added noise.
        alpha_ratio, eps_coeff, sigma_post = _reverse_step_coefficients(
            gamma_curr, gamma_next
        )
        mu = alpha_ratio * z - eps_coeff * eps_hat

        noise = jax.random.normal(step_rng, z.shape)
        z = mu + jnp.where(i == n_steps - 1, 0.0, eta * sigma_post) * noise
        return z, rng

    z, _ = jax.lax.fori_loop(0, n_steps, body_fn, (z, loop_rng))

    # Normalize to sphere
    return z / jnp.linalg.norm(z, axis=1, keepdims=True)


@partial(jax.jit, static_argnames=("model", "n_steps", "batch_size"))
def generate_samples_ode(
    model: EuclideanDiffusionModel,
    params,
    rng: Array,
    cap_params,
    n_steps: int = 100,
    batch_size: Optional[int] = None,
) -> Array:
    """Generate samples by integrating the probability flow ODE in log-SNR space.

    Integrates dz/dγ = ½[σ²z − σε̂] from γ_min (noise) to γ_max (data) with
    RK4 and uniform steps in γ.

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        rng: JAX random key
        cap_params: None for unconditioned, (cap_centers, d_maxes) for CONDITIONED_SCORE
        n_steps: Number of RK4 integration steps in γ-space
        batch_size: Required if cap_params is None

    Returns:
        Generated samples on S^{d-1} [batch_size, domain_dim]
    """
    if cap_params is not None:
        leading_dims = jax.tree.leaves(jax.tree.map(lambda x: x.shape[0], cap_params))
        inferred = leading_dims[0]
        if batch_size is None:
            batch_size = inferred
        elif batch_size != inferred:
            raise ValueError(
                f"batch_size {batch_size} doesn't match cap_params leading dim {inferred}"
            )
    elif batch_size is None:
        raise ValueError("batch_size must be specified when cap_params is None")

    gamma_min, gamma_max = model.apply(params, method=model.gamma_range)
    d_gamma = (gamma_max - gamma_min) / n_steps

    # Start from standard Gaussian
    z = jax.random.normal(rng, (batch_size, model.domain_dim))

    def body_fn(i, z):
        gamma = gamma_min + i * d_gamma
        return _gamma_rk4_step(model, params, cap_params, z, gamma, d_gamma)

    z = jax.lax.fori_loop(0, n_steps, body_fn, z)

    # Normalize to sphere
    return z / jnp.linalg.norm(z, axis=1, keepdims=True)


# =============================================================================
# NLL evaluation (probability flow ODE)
# =============================================================================


def compute_nll(
    model: EuclideanDiffusionModel,
    params,
    batch: dict,
    n_steps: int = 100,
    rng=None,
    n_projections: int = 10,
    cap_params=None,
) -> Array:
    """Compute spherical NLL via the probability flow ODE.

    Uses the change-of-variables formula with Hutchinson divergence estimation to
    compute the Euclidean log-density, then converts to the spherical NLL:
        NLL_sphere = NLL_ℝd − log(σ_radial) − ½log(2π)

    Note: this is the ODE NLL, not the SDE NLL. The VLB bounds the SDE NLL, not
    this quantity. The two differ when the score function is approximate.

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        batch: Dict with "point_vec" key containing data [batch_size, dim]
        n_steps: Number of integration steps
        rng: JAX random key
        n_projections: Number of projections for divergence estimation
        cap_params: Conditioning parameters

    Returns:
        Spherical NLL per example [batch_size]
    """
    samples = batch["point_vec"]
    nll_euclidean = -_compute_log_probability(
        model,
        params,
        samples,
        cap_params,
        n_steps=n_steps,
        rng=rng,
        n_projections=n_projections,
    )
    offset = jnp.log(model.sigma_radial) + 0.5 * jnp.log(2.0 * jnp.pi)
    return nll_euclidean - offset


# =============================================================================
# Internal helpers
# =============================================================================


def _snr_to_alpha_sigma(log_snr: Array):
    """Convert log SNR to (α, σ) for the variance-preserving forward process.

    α² = sigmoid(γ) = SNR / (1 + SNR)
    σ² = sigmoid(-γ) = 1 / (1 + SNR)

    Args:
        log_snr: log SNR values, any shape.

    Returns:
        (alpha, sigma) tuple, same shapes as log_snr.
    """
    log_alpha_sq, log_sigma_sq = _log_alpha_sigma_sq(log_snr)
    return jnp.exp(0.5 * log_alpha_sq), jnp.exp(0.5 * log_sigma_sq)


def _log_alpha_sigma_sq(log_snr: Array) -> tuple[Array, Array]:
    """Convert log SNR to (log α², log σ²) for the VP forward process."""
    return -jax.nn.softplus(-log_snr), -jax.nn.softplus(log_snr)


def _reverse_step_coefficients(log_snr_curr: Array, log_snr_next: Array):
    """Compute numerically stable ancestral-step coefficients in log-SNR space.

    The naive expression α_next² - α_curr² underflows to zero in float32 once α rounds
    to exactly 1.0, which happens around log-SNR 20. Rewriting the same quantity as
    σ_curr² - σ_next² preserves the denoising and posterior-variance terms deep into
    the high-SNR regime where the schedule still expects them to matter.
    """
    log_alpha_sq_curr, log_sigma_sq_curr = _log_alpha_sigma_sq(log_snr_curr)
    log_alpha_sq_next, log_sigma_sq_next = _log_alpha_sigma_sq(log_snr_next)

    sigma_sq_curr = jnp.exp(log_sigma_sq_curr)
    sigma_sq_next = jnp.exp(log_sigma_sq_next)
    alpha_sq_next = jnp.exp(log_alpha_sq_next)

    alpha_ratio = jnp.exp(0.5 * (log_alpha_sq_next - log_alpha_sq_curr))
    alpha_sq_diff = jnp.maximum(sigma_sq_curr - sigma_sq_next, 0.0)

    safe_sigma_sq_curr = jnp.where(alpha_sq_diff > 0, sigma_sq_curr, 1.0)
    safe_alpha_next = jnp.where(alpha_sq_diff > 0, jnp.sqrt(alpha_sq_next), 1.0)
    safe_alpha_curr = jnp.where(
        alpha_sq_diff > 0, jnp.exp(0.5 * log_alpha_sq_curr), 1.0
    )
    safe_sigma_curr = jnp.where(alpha_sq_diff > 0, jnp.sqrt(safe_sigma_sq_curr), 1.0)

    eps_coeff = alpha_sq_diff / (safe_alpha_next * safe_sigma_curr * safe_alpha_curr)
    sigma_post_sq = alpha_sq_diff * sigma_sq_next / (alpha_sq_next * safe_sigma_sq_curr)
    sigma_post = jnp.sqrt(jnp.maximum(sigma_post_sq, 0.0))
    return alpha_ratio, eps_coeff, sigma_post


def _predict_eps(model, params, cap_params, z, log_snr):
    """Predict ε̂ at the given log-SNR level."""
    gamma_vec = jnp.full((z.shape[0],), log_snr) if jnp.ndim(log_snr) == 0 else log_snr
    return model.apply(params, z, gamma_vec, cap_params, method=model.predict_eps)


def _gamma_velocity(model, params, cap_params, z, log_snr):
    """Compute dz/dγ = ½[σ²z − σε̂] at the given log-SNR.

    This is the schedule-independent velocity used for ODE sampling and NLL.
    """
    gamma_vec = jnp.full((z.shape[0],), log_snr) if jnp.ndim(log_snr) == 0 else log_snr
    return model.apply(
        params, z, gamma_vec, cap_params, method=model.gamma_space_velocity
    )


def _gamma_rk4_step(model, params, cap_params, z, gamma, d_gamma):
    """Single RK4 step for the ODE in log-SNR space: dz/dγ = ½[σ²z − σε̂]."""
    k1 = _gamma_velocity(model, params, cap_params, z, gamma)
    k2 = _gamma_velocity(
        model, params, cap_params, z + 0.5 * d_gamma * k1, gamma + 0.5 * d_gamma
    )
    k3 = _gamma_velocity(
        model, params, cap_params, z + 0.5 * d_gamma * k2, gamma + 0.5 * d_gamma
    )
    k4 = _gamma_velocity(model, params, cap_params, z + d_gamma * k3, gamma + d_gamma)
    return z + (d_gamma / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@partial(jax.jit, static_argnames=("model", "n_projections"))
def _gamma_hutchinson_divergence(
    model: EuclideanDiffusionModel,
    params,
    cap_params,
    z: Array,
    log_snr: float,
    rng: Array,
    n_projections: int,
) -> Array:
    """Estimate div_z(dz/dγ) using Hutchinson's trace estimator.

    Standard Euclidean divergence of the γ-space velocity, no curvature correction.

    Args:
        model: The diffusion model
        params: Model parameters
        cap_params: Conditioning parameters
        z: Current points [batch_size, dim]
        log_snr: Current log SNR (scalar)
        rng: Random key
        n_projections: Number of random projections

    Returns:
        Divergence estimate [batch_size]
    """
    batch_size, dim = z.shape
    effective_projections = min(n_projections, dim)
    gamma_vec = jnp.full((1,), log_snr)
    cap_in_axes = 0 if cap_params is not None else None

    def single_div(z_i, rng_i, cap_params_i):
        dropout_rng, proj_rng = jax.random.split(rng_i)

        def f(zi):
            # Add batch dimension back for single-element forward pass
            if cap_params_i is not None:
                cp = jax.tree.map(lambda x: x[None, ...], cap_params_i)
            else:
                cp = None
            return model.apply(
                params,
                zi[None, :],
                gamma_vec,
                cp,
                method=model.gamma_space_velocity,
                rngs={"dropout": dropout_rng},
            )[0]

        # Orthogonal random projections via QR
        gaussian = jax.random.normal(
            proj_rng, (dim, effective_projections), dtype=z_i.dtype
        )
        q, _ = jnp.linalg.qr(gaussian)
        v_samples = q.T * jnp.sqrt(jnp.asarray(dim, dtype=z_i.dtype))

        def vjp_fn(v):
            _, vjp = jax.vjp(f, z_i)
            return jnp.dot(v, vjp(v)[0])

        trace_estimates = jax.vmap(vjp_fn)(v_samples)
        return jnp.mean(trace_estimates)

    batch_keys = jax.random.split(rng, batch_size)
    return jax.vmap(single_div, in_axes=(0, 0, cap_in_axes))(z, batch_keys, cap_params)


@partial(jax.jit, static_argnames=("model", "n_steps", "n_projections"))
def _compute_log_probability(
    model: EuclideanDiffusionModel,
    params,
    samples: Array,
    cap_params,
    n_steps: int = 100,
    rng=None,
    n_projections: int = 10,
) -> Array:
    """Compute log probability by integrating the ODE backward in log-SNR space.

    Integrates dz/dγ = ½[σ²z − σε̂] from γ_max (data) to γ_min (noise) while
    accumulating div_z(dz/dγ), then evaluates the Gaussian base density.

    The change of variables gives: log p(z_data) = log p_0(z_noise) − ∫ div(dz/dγ) dγ

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        samples: Points to evaluate [batch_size, dim]
        cap_params: Conditioning parameters
        n_steps: Number of integration steps in γ-space
        rng: JAX random key for Hutchinson estimator
        n_projections: Number of random projections for divergence estimation

    Returns:
        Log probabilities [batch_size]
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    batch_size, dim = samples.shape

    gamma_min, gamma_max = model.apply(params, method=model.gamma_range)
    d_gamma = (gamma_max - gamma_min) / n_steps

    def body_fn(i, state):
        z, div_sum, rng = state
        gamma = gamma_max - i * d_gamma  # backward from γ_max

        # Accumulate divergence of dz/dγ
        step_rng, rng = jax.random.split(rng)
        div = _gamma_hutchinson_divergence(
            model, params, cap_params, z, gamma, step_rng, n_projections
        )
        div_sum = div_sum + div * d_gamma

        # RK4 step backward in γ
        z = _gamma_rk4_step(model, params, cap_params, z, gamma, -d_gamma)
        return z, div_sum, rng

    z0, div_sum, _ = jax.lax.fori_loop(
        0, n_steps, body_fn, (samples, jnp.zeros(batch_size), rng)
    )

    # Base density: N(0, I)
    log_p0 = -0.5 * (jnp.sum(z0**2, axis=1) + dim * jnp.log(2.0 * jnp.pi))

    return log_p0 - div_sum


# =============================================================================
# Tests
# =============================================================================


def _make_model(domain_dim, **kwargs):
    """Create an EuclideanDiffusionModel with test-friendly defaults."""
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
    return EuclideanDiffusionModel(**defaults)


def test_vlb_loss_components():
    """Verify VLB loss components are finite and have correct signs at initialization."""
    dim = 3
    model = _make_model(dim, d_model=32)

    params_rng, data_rng, noise_rng = jax.random.split(jax.random.PRNGKey(42), 3)
    params = model.init(params_rng, *model.dummy_inputs())

    x_1 = sample_sphere(data_rng, 64, dim)
    t = jnp.linspace(0.01, 0.99, 64)

    loss, components = model.apply(
        params,
        x_1,
        t,
        None,
        rngs={"noise": noise_rng},
        method=model.compute_vlb_loss,
    )

    assert jnp.isfinite(loss), f"VLB loss is not finite: {loss}"

    for key in ("diffusion", "prior", "recon", "vlb_total", "vlb_spherical"):
        assert key in components, f"Missing component '{key}' in VLB loss"
        assert jnp.isfinite(components[key]), f"VLB component '{key}' is not finite"

    # Diffusion loss should be non-negative
    assert (
        float(components["diffusion"]) >= 0
    ), f"Diffusion loss should be non-negative, got {components['diffusion']}"

    # Prior loss should be small for very low SNR_min
    print(f"Prior: {float(components['prior']):.6f}")
    print(f"Recon: {float(components['recon']):.6f}")
    print(f"Diffusion: {float(components['diffusion']):.6f}")
    print(f"VLB total: {float(components['vlb_total']):.6f}")
    print(f"VLB spherical: {float(components['vlb_spherical']):.6f}")


def test_vlb_loss_gradients_flow():
    """Verify that gradients flow through the VLB loss to both vector field and schedule params."""
    dim = 3
    model = _make_model(dim, d_model=32)

    params_rng, data_rng, noise_rng = jax.random.split(jax.random.PRNGKey(7), 3)
    params = model.init(params_rng, *model.dummy_inputs())

    x_1 = sample_sphere(data_rng, 32, dim)
    t = jax.random.uniform(jax.random.PRNGKey(0), (32,))

    def loss_fn(p):
        loss, _ = model.apply(
            p,
            x_1,
            t,
            None,
            rngs={"noise": noise_rng},
            method=model.compute_vlb_loss,
        )
        return loss

    grads = jax.grad(loss_fn)(params)
    grad_leaves = jax.tree.leaves(grads)
    assert all(jnp.isfinite(g).all() for g in grad_leaves), "Non-finite gradients found"
    assert any(jnp.any(g != 0) for g in grad_leaves), "All gradients are zero"


def test_reverse_step_coefficients_match_naive_formula_away_from_saturation():
    """Stable ancestral coefficients should agree with the naive formula at moderate log-SNR."""
    gamma_curr = jnp.array(1.0, dtype=jnp.float32)
    gamma_next = jnp.array(1.2, dtype=jnp.float32)

    alpha_curr, sigma_curr = _snr_to_alpha_sigma(gamma_curr)
    alpha_next, sigma_next = _snr_to_alpha_sigma(gamma_next)
    naive_alpha_ratio = alpha_next / alpha_curr
    naive_alpha_sq_diff = alpha_next**2 - alpha_curr**2
    naive_eps_coeff = naive_alpha_sq_diff / (alpha_next * sigma_curr * alpha_curr)
    naive_sigma_post = jnp.sqrt(
        naive_alpha_sq_diff * sigma_next**2 / (alpha_next**2 * sigma_curr**2)
    )

    alpha_ratio, eps_coeff, sigma_post = _reverse_step_coefficients(
        gamma_curr, gamma_next
    )
    np.testing.assert_allclose(alpha_ratio, naive_alpha_ratio, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(eps_coeff, naive_eps_coeff, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(sigma_post, naive_sigma_post, rtol=1e-6, atol=1e-7)


def test_reverse_step_coefficients_do_not_freeze_when_alpha_rounds_to_one():
    """The ancestral denoising term must remain non-zero after α saturates in float32."""
    gamma_curr = jnp.array(20.0, dtype=jnp.float32)
    gamma_next = jnp.array(20.02, dtype=jnp.float32)

    alpha_curr, _ = _snr_to_alpha_sigma(gamma_curr)
    alpha_next, _ = _snr_to_alpha_sigma(gamma_next)
    naive_alpha_sq_diff = alpha_next**2 - alpha_curr**2
    assert float(naive_alpha_sq_diff) == 0.0

    alpha_ratio, eps_coeff, sigma_post = _reverse_step_coefficients(
        gamma_curr, gamma_next
    )
    assert float(alpha_ratio) == 1.0
    assert float(eps_coeff) > 0.0
    assert float(sigma_post) > 0.0


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_trivial(domain_dim):
    """Train a model where all data is a single fixed point."""
    from txt2img_unsupervised.training_infra import train_for_tests

    model = _make_model(domain_dim, vlb_variance_loss_weight=1e-3, log_snr_max_cap=10.0)

    batch_size = 256
    first_dim_vec = jnp.zeros(domain_dim)
    first_dim_vec = first_dim_vec.at[0].set(1.0)
    points = repeat(first_dim_vec, "v -> b v", b=batch_size * 100)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    test_dset = Dataset.from_dict({"point_vec": points[:batch_size]}).with_format("np")

    loss_fn = partial(compute_batch_loss, model)

    result = train_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-3,
        loss_fn=loss_fn,
        fields=["point_vec"],
        epochs=6,
        test_dataset=test_dset,
    )

    eval_params = result.state.get_eval_params()

    # Generate samples with the SDE sampler (the distribution the VLB bounds)
    n_gen_samples = 100
    target = np.zeros(domain_dim)
    target[0] = 1.0
    samples = generate_samples_sde(
        model,
        eval_params,
        jax.random.PRNGKey(20260314),
        cap_params=None,
        batch_size=n_gen_samples,
        n_steps=500,
        eta=1.0,
    )
    samples_np = np.array(samples)
    cosine_sims = samples_np @ target

    n_close_99 = np.sum(cosine_sims > 0.99)
    n_close_95 = np.sum(cosine_sims > 0.95)
    n_close_90 = np.sum(cosine_sims > 0.9)
    print(
        f"Samples near target: {n_close_99}/{n_gen_samples} at cosine>0.99, "
        f"{n_close_95}/{n_gen_samples} at cosine>0.95, "
        f"{n_close_90}/{n_gen_samples} at cosine>0.9 (mean {cosine_sims.mean():.4f})"
    )
    assert (
        n_close_99 >= 0.9 * n_gen_samples
    ), f"Only {n_close_99}/{n_gen_samples} samples have cosine > 0.99"

    # Print schedule endpoints and VLB diagnostics
    gamma_min, gamma_max = model.apply(eval_params, method=model.gamma_range)
    print(
        f"Learned schedule: γ_min={float(gamma_min):.2f}, γ_max={float(gamma_max):.2f}"
    )
    vlb_total = result.test_aux["vlb_total"]
    spherical_offset = float(jnp.log(model.sigma_radial) + 0.5 * jnp.log(2.0 * jnp.pi))
    vlb_spherical = vlb_total - spherical_offset
    print(f"VLB spherical: {vlb_spherical:.4f}")
    assert vlb_spherical < -5, (
        f"VLB spherical {vlb_spherical:.4f} should be < -5 "
        f"for single-point distribution"
    )


def _sample_sphere_np(n, dim, rng):
    """Sample uniformly from the unit sphere using numpy."""
    points = rng.standard_normal((n, dim))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points


@dataclass
class _TargetDistribution:
    """A target distribution on the sphere for training tests.

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
    """Train on a distribution and verify SDE samples match the target.

    Uses the SDE sampler (whose distribution the VLB bounds) rather than ODE NLL,
    because the euclidean VDM's ODE NLL converges much more slowly than its sample
    quality in higher dimensions.
    """
    from txt2img_unsupervised.training_infra import train_for_tests

    dist = _make_target_distribution(dist_name, domain_dim)

    model = _make_model(
        domain_dim,
        d_model=256,
        vlb_variance_loss_weight=1e-3,
        log_snr_max_cap=10.0,
    )

    batch_size = 1024
    n_samples = 32768

    data_rng = np.random.default_rng(42)
    points = dist.rvs(n_samples, data_rng)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")

    test_dset = Dataset.from_dict(
        {"point_vec": dist.rvs(batch_size * 8, data_rng)}
    ).with_format("np")

    print(f"Distribution entropy: {dist.entropy:.6f}")

    loss_fn = partial(compute_batch_loss, model)

    match (dist_name, domain_dim):
        case ("hemisphere", 16):
            epochs = 40
        case ("vmf", 16):
            epochs = 20
        case _:
            epochs = 15
    result = train_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-3,
        loss_fn=loss_fn,
        fields=["point_vec"],
        epochs=epochs,
        test_dataset=test_dset,
    )
    eval_params = result.state.get_eval_params()

    # Generate SDE samples — these come from the distribution the VLB bounds
    n_gen = 5000
    samples_np = np.array(
        generate_samples_sde(
            model,
            eval_params,
            jax.random.PRNGKey(42),
            cap_params=None,
            batch_size=n_gen,
            n_steps=500,
            eta=1.0,
        )
    )
    ref_np = dist.rvs(n_gen, np.random.default_rng(123))

    # 1. Most samples should lie in the support
    log_probs = dist.logpdf(samples_np)
    in_support = np.isfinite(log_probs)
    in_support_frac = np.mean(in_support)
    print(f"In-support fraction: {in_support_frac:.4f}")
    if not np.all(in_support):
        assert (
            in_support_frac > 0.98
        ), f"Only {in_support_frac:.1%} of samples in support"

    # 2. Cross-entropy under the true density should approximate entropy.
    # E_model[-log p_true(x)] ≈ H(p_true) when model ≈ true distribution.
    if np.any(in_support):
        cross_ent = -np.mean(log_probs[in_support])
        print(f"Cross-entropy: {cross_ent:.4f}, entropy: {dist.entropy:.4f}")
        assert (
            cross_ent < dist.entropy + 1.0
        ), f"Cross-entropy {cross_ent:.4f} too far from entropy {dist.entropy:.4f}"

    # 3. Very few samples should land in the exterior region
    exterior_frac = np.mean(dist.exterior_mask(samples_np))
    if exterior_frac > 0:
        print(f"Exterior fraction: {exterior_frac:.4f}")
    assert exterior_frac < 0.05, f"Too many samples in exterior: {exterior_frac:.1%}"

    # 4. KS test on many random 1D projections. Each projection collapses the
    #    d-dimensional comparison to a scalar, so we test many directions to cover
    #    the full distribution shape.
    n_projections = 20
    proj_rng = np.random.default_rng(456)
    directions = proj_rng.standard_normal((n_projections, domain_dim))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    ks_stats = []
    for direction in directions:
        stat, _ = stats.ks_2samp(samples_np @ direction, ref_np @ direction)
        ks_stats.append(stat)

    max_ks = max(ks_stats)
    mean_ks = np.mean(ks_stats)
    print(
        f"KS stats over {n_projections} projections: max={max_ks:.4f}, mean={mean_ks:.4f}"
    )
    assert (
        max_ks < 0.05
    ), f"Max KS statistic {max_ks:.4f} too large across {n_projections} projections"
