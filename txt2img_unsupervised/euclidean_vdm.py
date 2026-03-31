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
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import diffrax
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

from txt2img_unsupervised.cap_sampling import (
    DEFAULT_CAP_FEATURES,
    LogitsTable,
    cap_conditioning_dim,
    encode_cap_params,
    sample_cap,
    sample_from_cap_v,
    sample_negative_cap_center,
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


def classifier_logit_correction(table, d_max):
    """Compute the logit correction for classifier guidance sampling bias.

    During training, positive caps (containing x1) are sampled uniformly from
    the d_max-cap around x1 (area A), while negative caps are sampled from the
    complement (area S-A). This makes the Bayes-optimal classifier learn a logit
    shifted by log((S-A)/A) relative to the true log-odds of P(x1 in cap | z_t, t).
    Adding log(A/(S-A)) = log(F/(1-F)) to the raw logit recovers the correct value,
    where F = A/S is the fractional cap area.

    Uses LogitsTable.log_cap_size to compute log(F) in log space, avoiding underflow
    for small caps in high dimensions where the fractional area is far below float
    range.

    Args:
        table: LogitsTable for the sphere dimension.
        d_max: Cosine distance threshold defining the cap boundary.

    Returns:
        Scalar logit correction to add to raw classifier logits.
    """
    assert jnp.ndim(d_max) == 0, f"d_max must be scalar, got shape {jnp.shape(d_max)}"
    log_F = table.log_cap_size(d_max)
    log_one_minus_F = table.log_cap_size(2.0 - d_max)
    return log_F - log_one_minus_F


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
    mlp_dropout_rate: float | None
    input_dropout_rate: float | None
    mlp_always_inject: frozenset[Literal["x", "t", "cond"]] = field(
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
    log_snr_max_cap: float | None = None

    cap_conditioning: CapConditioningMode = CapConditioningMode.UNCONDITIONED
    d_max_dist: tuple | None = None
    vlb_variance_loss_weight: float | None = None
    classifier_loss_weight: float = 1.0
    cap_features: frozenset[
        Literal["cap_center", "d_max", "log_cap_odds", "cos_sim", "z_norm"]
    ] = field(default_factory=lambda: DEFAULT_CAP_FEATURES)

    @property
    def conditioning_dim(self) -> int:
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return 0
        elif self.cap_conditioning in (
            CapConditioningMode.CONDITIONED_SCORE,
            CapConditioningMode.CLASSIFIER_GUIDANCE,
        ):
            return cap_conditioning_dim(self.domain_dim, self.cap_features)
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
            self.logits_table = LogitsTable(self.domain_dim - 1, 8192)
            self.classifier_head = nn.Dense(1, name="classifier_head")
        self.vector_field = self.mk_vector_field()
        self.schedule = LearnedNoiseSchedule(
            hidden_dim=self.schedule_hidden_dim,
            n_quadrature_points=self.schedule_n_quadrature_points,
            init_gamma_min=self.init_log_snr_min,
            init_gamma_max=self.init_log_snr_max,
            gamma_max_cap=self.log_snr_max_cap,
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
            cap_params = (jnp.ones((1, self.domain_dim)), jnp.ones((1,)))
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")
        return z, t, cap_params

    @nn.nowrap
    def mk_partition_map(self, use_muon: bool):
        """Create a partition map for optimizer configuration with muP scaling."""
        params_map = {
            "vector_field": self.mk_vector_field().mk_partition_map(use_muon)["params"],
            "schedule": "schedule",
        }
        if self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            if use_muon:
                params_map["classifier_head"] = {
                    "kernel": "adam_scaled",
                    "bias": "adam_fixed",
                }
            else:
                params_map["classifier_head"] = {
                    "kernel": "scaled_lr",
                    "bias": "fixed_lr",
                }
        return {"params": params_map}

    @nn.nowrap
    def scale_lr(self, lr: float) -> float:
        """Scaled learning rate for hidden layers."""
        return self.mk_vector_field().scale_lr(lr)

    def gamma_range(self):
        """Return the effective (γ_min, γ_max) schedule endpoints."""
        return self.schedule.gamma_min, self.schedule.effective_gamma_max

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
            x1_batch = batch["point_vec"]
            batch_size = x1_batch.shape[0]
            rng = self.make_rng("sample_cap_params")

            def sample_one(rng, x1):
                label_rng, cap_rng = jax.random.split(rng)
                is_positive = jax.random.bernoulli(label_rng, 0.5)

                # Sample d_max from d_max_dist (same for positive and negative)
                pos_center, d_max = sample_cap(
                    self.logits_table, cap_rng, x1, self.d_max_dist
                )
                neg_center = sample_negative_cap_center(
                    cap_rng, self.logits_table, x1, d_max
                )

                center = jnp.where(is_positive, pos_center, neg_center)
                return center, d_max, is_positive.astype(jnp.float32)

            rngs = jax.random.split(rng, batch_size)
            cap_centers, d_maxes, labels = jax.vmap(sample_one)(rngs, x1_batch)
            return {"cap_centers": cap_centers, "d_maxes": d_maxes, "labels": labels}
        else:
            raise ValueError(f"Unknown cap conditioning mode: {self.cap_conditioning}")

    def _cap_params_to_cond_vecs(self, cap_params, batch_size, z=None):
        """Convert cap_params to conditioning vectors for the vector field.

        Args:
            cap_params: None or (cap_centers, d_maxes).
            batch_size: Number of samples.
            z: Points in R^d, required if cap_features includes cos_sim or z_norm.
        """
        if self.cap_conditioning == CapConditioningMode.UNCONDITIONED:
            return jnp.zeros((batch_size, 0))
        elif self.cap_conditioning in (
            CapConditioningMode.CONDITIONED_SCORE,
            CapConditioningMode.CLASSIFIER_GUIDANCE,
        ):
            cap_centers, d_maxes = cap_params
            return encode_cap_params(
                cap_center=cap_centers,
                d_max=d_maxes,
                d_max_dist=self.d_max_dist,
                domain_dim=self.domain_dim,
                features=self.cap_features,
                z=z,
                table=self.logits_table,
            )
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

        For CLASSIFIER_GUIDANCE, always uses zeros conditioning (unconditioned noise
        prediction). Cap information only enters through the classifier pass.

        Args:
            z: Points in R^d [batch_size, domain_dim]
            log_snr: Log SNR values [batch_size]
            cap_params: None for UNCONDITIONED/CLASSIFIER_GUIDANCE,
                (cap_centers, d_maxes) for CONDITIONED_SCORE

        Returns:
            Predicted noise [batch_size, domain_dim].
        """
        if self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            cond_vecs = jnp.zeros((z.shape[0], self.conditioning_dim))
        else:
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
        gamma_prime = self.schedule.gamma_derivative(t)
        v_gamma = self.gamma_space_velocity(z, log_snr, cap_params)
        # Trigger parameter creation for classifier head (needed for model.init)
        if self.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
            self.classifier_head(jnp.zeros((z.shape[0], self.d_model))).squeeze(-1)
        return gamma_prime[:, None] * v_gamma

    def classifier_forward(self, z, log_snr, cap_params):
        """Forward pass for the classifier: backbone with cap conditioning → linear head.

        Runs the backbone with cap information in the conditioning vector, then
        projects the post-norm MLP activations to a scalar logit per sample.

        Args:
            z: Points in R^d [batch_size, domain_dim]
            log_snr: Log SNR values [batch_size]
            cap_params: (cap_centers, d_maxes) specifying the cap to classify against

        Returns:
            Corrected logits [batch_size] — positive means "x1 likely in cap".
            Includes a logit correction for the training sampling bias: positive
            caps are sampled from a region of area A while negative caps come
            from area S-A, shifting the raw logit by log((S-A)/A). The
            correction adds log(A/(S-A)) to recover calibrated log-odds.
        """
        cond_vecs = self._cap_params_to_cond_vecs(cap_params, z.shape[0], z=z)
        t_normalized = self._normalize_log_snr(log_snr)
        _, mlp_out = self.vector_field.call_with_intermediates(
            z, t_normalized, cond_vecs
        )
        _, d_maxes = cap_params
        assert d_maxes.ndim == 1, (
            f"d_maxes must be [batch_size], got shape {d_maxes.shape}"
        )
        raw_logits = self.classifier_head(mlp_out).squeeze(-1)
        correction = jax.vmap(
            lambda d: classifier_logit_correction(self.logits_table, d)
        )(d_maxes)
        return raw_logits + correction

    def _forward_diffusion(self, x_1, t):
        """Shared forward diffusion setup for VLB and classifier losses.

        Augments data with radial noise, computes schedule values, and adds
        Gaussian noise to produce z_t.

        Returns:
            (x_data, z_t, eps, log_snr, alpha, sigma, gamma_prime)
        """
        batch_size = x_1.shape[0]

        radial_rng = self.make_rng("noise")
        radial_rng, noise_rng = jax.random.split(radial_rng)
        r = 1.0 + self.sigma_radial * jax.random.normal(radial_rng, (batch_size, 1))
        x_data = x_1 * r

        log_snr = self.schedule(t)
        alpha, sigma = _snr_to_alpha_sigma(log_snr)
        gamma_prime = self.schedule.gamma_derivative(t)

        eps = jax.random.normal(noise_rng, x_data.shape)
        z_t = alpha[:, None] * x_data + sigma[:, None] * eps

        return x_data, z_t, eps, log_snr, alpha, sigma, gamma_prime

    def _compute_vlb_components(
        self, x_data, z_t, eps, log_snr, gamma_prime, cond_vecs
    ):
        """Compute VLB loss components from pre-computed forward diffusion state.

        Args:
            x_data: Radially augmented data [batch_size, dim]
            z_t: Noisy samples [batch_size, dim]
            eps: True noise [batch_size, dim]
            log_snr: Log SNR values [batch_size]
            gamma_prime: Schedule derivative [batch_size]
            cond_vecs: Conditioning vectors for the backbone [batch_size, conditioning_dim]

        Returns:
            Dict of VLB loss components.
        """
        dim = self.domain_dim
        t_normalized = self._normalize_log_snr(log_snr)
        eps_hat = self.vector_field(z_t, t_normalized, cond_vecs)

        per_sample_sq_err = jnp.sum((eps_hat - eps) ** 2, axis=1)
        diffusion_loss = 0.5 * jnp.mean(gamma_prime * per_sample_sq_err)

        log_snr_min = self.schedule.gamma_min
        alpha_T, sigma_T = _snr_to_alpha_sigma(log_snr_min)
        x_data_sq_norm = jnp.mean(jnp.sum(x_data**2, axis=1))
        prior_loss = 0.5 * (
            alpha_T**2 * x_data_sq_norm
            + dim * sigma_T**2
            - dim
            - dim * jnp.log(sigma_T**2)
        )

        log_snr_max = self.schedule.effective_gamma_max
        recon_loss = 0.5 * dim * (1.0 + jnp.log(2.0 * jnp.pi) - log_snr_max)

        vlb_total = diffusion_loss + prior_loss + recon_loss
        spherical_offset = jnp.log(self.sigma_radial) + 0.5 * jnp.log(2.0 * jnp.pi)
        vlb_spherical = vlb_total - spherical_offset

        if self.vlb_variance_loss_weight is not None:
            f_i = 0.5 * gamma_prime * jax.lax.stop_gradient(per_sample_sq_err)
            variance_loss = jnp.mean(f_i**2) * self.vlb_variance_loss_weight
        else:
            variance_loss = jnp.array(0.0)

        return {
            "vlb_total": vlb_total,
            "vlb_spherical": vlb_spherical,
            "diffusion": diffusion_loss,
            "prior": prior_loss,
            "recon": recon_loss,
            "variance": variance_loss,
        }

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
        x_data, z_t, eps, log_snr, alpha, sigma, gamma_prime = self._forward_diffusion(
            x_1, t
        )
        cond_vecs = self._cap_params_to_cond_vecs(cap_params, batch_size)
        components = self._compute_vlb_components(
            x_data, z_t, eps, log_snr, gamma_prime, cond_vecs
        )
        total_loss = components["vlb_total"] + components["variance"]
        return total_loss, components

    def compute_vlb_and_classifier_loss(self, x_1, t, classifier_cap_params):
        """Compute VLB loss + classifier cross-entropy loss for CLASSIFIER_GUIDANCE.

        Two forward passes through the backbone:
        1. VLB pass: zeros conditioning → eps_hat → VLB loss
        2. Classifier pass: cap conditioning → mlp_out → linear → logit → cross-entropy

        Args:
            x_1: Data points on S^{d-1} [batch_size, dim]
            t: Uniformly sampled time values [batch_size]
            classifier_cap_params: Dict with 'cap_centers', 'd_maxes', 'labels'

        Returns:
            Tuple of (total_loss, components_dict).
        """
        batch_size = x_1.shape[0]
        x_data, z_t, eps, log_snr, alpha, sigma, gamma_prime = self._forward_diffusion(
            x_1, t
        )

        # Pass 1: VLB with zeros conditioning (no cap info)
        zero_cond = jnp.zeros((batch_size, self.conditioning_dim))
        vlb_components = self._compute_vlb_components(
            x_data, z_t, eps, log_snr, gamma_prime, zero_cond
        )

        # Pass 2: classifier with cap conditioning
        cap_centers = classifier_cap_params["cap_centers"]
        d_maxes = classifier_cap_params["d_maxes"]
        labels = classifier_cap_params["labels"]
        cap_params = (cap_centers, d_maxes)

        cond_vecs = self._cap_params_to_cond_vecs(cap_params, batch_size, z=z_t)
        t_normalized = self._normalize_log_snr(log_snr)
        _, mlp_out = self.vector_field.call_with_intermediates(
            z_t, t_normalized, cond_vecs
        )
        logits = self.classifier_head(mlp_out).squeeze(-1)

        classifier_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

        total_loss = (
            vlb_components["vlb_total"]
            + vlb_components["variance"]
            + self.classifier_loss_weight * classifier_loss
        )

        components = {**vlb_components, "classifier": classifier_loss}
        return total_loss, components


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
    """Extract data from batch, sample t, compute loss.

    For UNCONDITIONED/CONDITIONED_SCORE: computes VLB loss.
    For CLASSIFIER_GUIDANCE: computes VLB + classifier cross-entropy loss.

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

    conditioning = model.apply(
        params,
        batch,
        method=model.prepare_training_conditioning,
        rngs={"sample_cap_params": cap_rng},
    )

    t = stratified_time_sample(time_rng, batch_size)

    if model.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
        return model.apply(
            params,
            x_1,
            t,
            conditioning,
            rngs={"dropout": dropout_rng, "noise": noise_rng},
            method=model.compute_vlb_and_classifier_loss,
        )
    else:
        return model.apply(
            params,
            x_1,
            t,
            conditioning,
            rngs={"dropout": dropout_rng, "noise": noise_rng},
            method=model.compute_vlb_loss,
        )


# =============================================================================
# Sampling
# =============================================================================


def _predict_eps_guided(model, params, cap_params, z, log_snr):
    """Predict ε̂ with classifier guidance applied.

    Two forward passes:
    1. Unconditioned noise prediction (zeros conditioning)
    2. Per-sample classifier gradient (cap conditioning) via backprop

    The guided prediction is: ε̂_guided = ε̂ − σ · ∇_z log P(x1 in cap | z, t)
    """
    eps_hat = _predict_eps(model, params, None, z, log_snr)

    cap_centers, d_maxes = cap_params
    gamma_vec = jnp.full((1,), log_snr)

    def single_log_prob(z_i, cap_center_i, d_max_i):
        cp = (cap_center_i[None], d_max_i[None])
        logit = model.apply(
            params, z_i[None], gamma_vec, cp, method=model.classifier_forward
        )[0]
        return jax.nn.log_sigmoid(logit)

    grad_log_p = jax.vmap(jax.grad(single_log_prob))(z, cap_centers, d_maxes)
    _, sigma = _snr_to_alpha_sigma(log_snr)
    return eps_hat - sigma * grad_log_p


@partial(jax.jit, static_argnames=("model", "n_steps", "batch_size"))
def generate_samples_sde(
    model: EuclideanDiffusionModel,
    params,
    rng: Array,
    cap_params,
    n_steps: int = 200,
    batch_size: int | None = None,
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

    For CLASSIFIER_GUIDANCE with cap_params not None, applies classifier guidance:
        ε̂_guided = ε̂ − σ · ∇_z log P(x1 in cap | z, t)

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        rng: JAX random key
        cap_params: None for unconditioned, (cap_centers, d_maxes) for conditioned.
            For CLASSIFIER_GUIDANCE: None = no guidance, tuple = guide toward cap.
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

    use_guidance = (
        model.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE
        and cap_params is not None
    )

    gamma_min, gamma_max = model.apply(params, method=model.gamma_range)
    d_gamma = (gamma_max - gamma_min) / n_steps

    init_rng, loop_rng = jax.random.split(rng)
    z = jax.random.normal(init_rng, (batch_size, model.domain_dim))

    def body_fn(i, state):
        z, rng = state
        rng, step_rng = jax.random.split(rng)

        gamma_curr = gamma_min + i * d_gamma
        gamma_next = gamma_curr + d_gamma

        if use_guidance:
            eps_hat = _predict_eps_guided(model, params, cap_params, z, gamma_curr)
        else:
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


@partial(
    jax.jit, static_argnames=("model", "n_steps", "batch_size", "method", "max_steps")
)
def generate_samples_ode(
    model: EuclideanDiffusionModel,
    params,
    rng: Array,
    cap_params,
    n_steps: int | None = None,
    batch_size: int | None = None,
    method: str = "diffrax",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 4096,
) -> Array:
    """Generate samples by integrating the probability flow ODE in log-SNR space. Note this samples
    from a *different* distribution than generate_samples_sde, and *not* the one that's constrained
    by the VLB. They're only the same when the score is perfect i.e. they're never the same. The
    only reason to use this is potentially better performance, especially when using tsit5.

    Integrates dz/dγ = ½[σ²z − σε̂] from γ_min (noise) to γ_max (data).

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        rng: JAX random key
        cap_params: None for unconditioned, (cap_centers, d_maxes) for conditioned.
            For CLASSIFIER_GUIDANCE: None = no guidance, tuple = guide toward cap.
        n_steps: Number of RK4 steps (required for method="rk4", forbidden for "diffrax")
        batch_size: Required if cap_params is None
        method: "diffrax" (adaptive Tsit5) or "rk4" (fixed-step)
        atol: Absolute tolerance for diffrax adaptive solver
        rtol: Relative tolerance for diffrax adaptive solver
        max_steps: Maximum number of solver steps for diffrax

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

    # Start from standard Gaussian
    z = jax.random.normal(rng, (batch_size, model.domain_dim))

    if method == "diffrax":
        if n_steps is not None:
            raise ValueError(
                "n_steps must not be specified when using diffrax adaptive solver"
            )
        term = diffrax.ODETerm(_make_sampling_velocity(model))
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=gamma_min,
            t1=gamma_max,
            dt0=(gamma_max - gamma_min) / 100,
            y0=z,
            args=(params, cap_params),
            stepsize_controller=controller,
            max_steps=max_steps,
        )
        z = sol.ys[0]  # SaveAt default: final value only, leading dim 1
    elif method == "rk4":
        if n_steps is None:
            raise ValueError("n_steps is required for rk4 method")
        d_gamma = (gamma_max - gamma_min) / n_steps

        def body_fn(i, z):
            gamma = gamma_min + i * d_gamma
            return _gamma_rk4_step(model, params, cap_params, z, gamma, d_gamma)

        z = jax.lax.fori_loop(0, n_steps, body_fn, z)
    else:
        raise ValueError(f"Unknown ODE solver method: {method}")

    # Normalize to sphere
    return z / jnp.linalg.norm(z, axis=1, keepdims=True)


# =============================================================================
# NLL evaluation (probability flow ODE)
# =============================================================================


def compute_nll(
    model: EuclideanDiffusionModel,
    params,
    batch: dict,
    n_steps: int | None = None,
    rng=None,
    n_projections: int = 10,
    cap_params=None,
    method: str = "diffrax",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 4096,
) -> Array:
    """Compute spherical NLL via the probability flow ODE.

    Uses the change-of-variables formula with Hutchinson divergence estimation to compute the
    Euclidean log-density, then converts to the spherical NLL:
        NLL_sphere = NLL_ℝd − log(σ_radial) − ½log(2π)

    Note: this is the ODE NLL, not the SDE NLL. The VLB bounds the SDE NLL, not this quantity. The
    two differ when the score function is approximate (i.e. always). In general, this is not a great
    measure for model quality.

    Args:
        model: Euclidean diffusion model
        params: Model parameters
        batch: Dict with "point_vec" key containing data [batch_size, dim]
        n_steps: Number of integration steps (required for "rk4", forbidden for "diffrax")
        rng: JAX random key
        n_projections: Number of projections for divergence estimation
        cap_params: Conditioning parameters
        method: "diffrax" (adaptive Tsit5) or "rk4" (fixed-step)
        atol: Absolute tolerance for diffrax adaptive solver
        rtol: Relative tolerance for diffrax adaptive solver
        max_steps: Maximum number of solver steps for diffrax

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
        method=method,
        atol=atol,
        rtol=rtol,
        max_steps=max_steps,
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
    For CLASSIFIER_GUIDANCE with cap_params, uses guided eps prediction.
    """
    use_guidance = (
        model.cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE
        and cap_params is not None
    )
    if use_guidance:
        eps_hat = _predict_eps_guided(model, params, cap_params, z, log_snr)
    else:
        eps_hat = _predict_eps(model, params, cap_params, z, log_snr)
    _, sigma = _snr_to_alpha_sigma(log_snr)
    return 0.5 * (sigma**2 * z - sigma * eps_hat)


def _make_sampling_velocity(model):
    """Create a diffrax-compatible velocity function for ODE sampling.

    Captures ``model`` (a compile-time constant) in the closure so it doesn't
    go through diffrax's ``args``.

    Returns:
        ``f(gamma, z, args)`` where ``args = (params, cap_params)``.
    """

    def velocity(gamma, z, args):
        params, cap_params = args
        return _gamma_velocity(model, params, cap_params, z, gamma)

    return velocity


def _z_only_norm(state):
    """RMS norm over only the z component of an augmented ``(z, div_sum)`` state.

    PIDController applies its norm to the scaled error estimate (same pytree
    structure as the state).  By ignoring the divergence leaf, step-size
    decisions are driven entirely by the ODE trajectory accuracy.
    """
    z, _div_sum = state
    return jnp.sqrt(jnp.mean(z**2))


def _make_augmented_velocity(model, n_projections):
    """Create a diffrax-compatible augmented velocity for NLL computation.

    Integrates both the ODE trajectory and the Hutchinson divergence
    accumulator simultaneously.  ``model`` and ``n_projections`` are captured
    in the closure because they must be compile-time constants.

    NLL computation integrates backward (gamma_max to gamma_min), but the
    change-of-variables formula requires ``integral_{gamma_min}^{gamma_max} div dgamma``,
    so the divergence is negated here to compensate.

    Returns:
        ``f(gamma, (z, div_sum), args)`` where
        ``args = (params, cap_params, base_rng)``.
    """

    def velocity(gamma, state, args):
        z, _div_sum = state
        params, cap_params, base_rng = args

        dz_dgamma = _gamma_velocity(model, params, cap_params, z, gamma)

        # Derive a per-evaluation RNG from the base key and the current gamma
        # value so the Hutchinson estimator gets varying projections without
        # needing RNG state in the ODE.
        gamma_bits = jax.lax.bitcast_convert_type(jnp.float32(gamma), jnp.int32)
        step_rng = jax.random.fold_in(base_rng, gamma_bits)
        div = _gamma_hutchinson_divergence(
            model, params, cap_params, z, gamma, step_rng, n_projections
        )

        # Negate: NLL integrates backward (gamma_max to gamma_min), but
        # the formula needs integral_{gamma_min}^{gamma_max} div dgamma.
        return dz_dgamma, -div

    return velocity


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
    cap_in_axes = 0 if cap_params is not None else None

    def single_div(z_i, rng_i, cap_params_i):
        def f(zi):
            if cap_params_i is not None:
                cp = jax.tree.map(lambda x: x[None, ...], cap_params_i)
            else:
                cp = None
            return _gamma_velocity(model, params, cp, zi[None, :], log_snr)[0]

        # Orthogonal random projections via QR
        gaussian = jax.random.normal(
            rng_i, (dim, effective_projections), dtype=z_i.dtype
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


@partial(
    jax.jit,
    static_argnames=("model", "n_steps", "n_projections", "method", "max_steps"),
)
def _compute_log_probability(
    model: EuclideanDiffusionModel,
    params,
    samples: Array,
    cap_params,
    n_steps: int | None = None,
    rng=None,
    n_projections: int = 10,
    method: str = "diffrax",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    max_steps: int = 4096,
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
        n_steps: Number of integration steps (required for "rk4", forbidden for "diffrax")
        rng: JAX random key for Hutchinson estimator
        n_projections: Number of random projections for divergence estimation
        method: "diffrax" (adaptive Tsit5) or "rk4" (fixed-step)
        atol: Absolute tolerance for diffrax adaptive solver
        rtol: Relative tolerance for diffrax adaptive solver
        max_steps: Maximum number of solver steps for diffrax

    Returns:
        Log probabilities [batch_size]
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    batch_size, dim = samples.shape

    gamma_min, gamma_max = model.apply(params, method=model.gamma_range)

    if method == "diffrax":
        if n_steps is not None:
            raise ValueError(
                "n_steps must not be specified when using diffrax adaptive solver"
            )
        term = diffrax.ODETerm(_make_augmented_velocity(model, n_projections))
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=rtol, atol=atol, norm=_z_only_norm)
        y0 = (samples, jnp.zeros(batch_size))
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=gamma_max,
            t1=gamma_min,
            dt0=-(gamma_max - gamma_min) / 100,
            y0=y0,
            args=(params, cap_params, rng),
            stepsize_controller=controller,
            max_steps=max_steps,
        )
        z0 = sol.ys[0][0]  # [batch_size, dim]
        div_sum = sol.ys[1][0]  # [batch_size]
    elif method == "rk4":
        if n_steps is None:
            raise ValueError("n_steps is required for rk4 method")
        d_gamma = (gamma_max - gamma_min) / n_steps

        def body_fn(i, state):
            z, div_sum, step_rng = state
            gamma = gamma_max - i * d_gamma  # backward from γ_max

            # Accumulate divergence of dz/dγ
            step_rng, rng_i = jax.random.split(step_rng)
            div = _gamma_hutchinson_divergence(
                model, params, cap_params, z, gamma, rng_i, n_projections
            )
            div_sum = div_sum + div * d_gamma

            # RK4 step backward in γ
            z = _gamma_rk4_step(model, params, cap_params, z, gamma, -d_gamma)
            return z, div_sum, step_rng

        z0, div_sum, _ = jax.lax.fori_loop(
            0, n_steps, body_fn, (samples, jnp.zeros(batch_size), rng)
        )
    else:
        raise ValueError(f"Unknown ODE solver method: {method}")

    # Base density: N(0, I)
    log_p0 = -0.5 * (jnp.sum(z0**2, axis=1) + dim * jnp.log(2.0 * jnp.pi))

    return log_p0 - div_sum


# =============================================================================
# Tests
# =============================================================================


def _make_model(domain_dim, **kwargs):
    """Create a EuclideanDiffusionModel with test-friendly defaults."""
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
    assert float(components["diffusion"]) >= 0, (
        f"Diffusion loss should be non-negative, got {components['diffusion']}"
    )

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
    assert n_close_99 >= 0.9 * n_gen_samples, (
        f"Only {n_close_99}/{n_gen_samples} samples have cosine > 0.99"
    )

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
    elif name == "cap":
        d_max = 0.5
        cap_center = np.zeros(domain_dim)
        cap_center[0] = 1.0
        table = LogitsTable(d=domain_dim - 1, n=8192)
        log_cap_frac = float(table.log_cap_size(d_max))
        log_inv_sa = float(sphere_log_inverse_surface_area(domain_dim))
        cap_logp = log_inv_sa - log_cap_frac
        margin = 0.05

        def cap_rvs(n, rng):
            jax_key = jax.random.PRNGKey(rng.integers(0, 2**31))
            return np.asarray(
                sample_from_cap_v(jax_key, table, jnp.array(cap_center), d_max, n)
            )

        def cap_logpdf(x):
            cos_dists = 1.0 - x @ cap_center
            result = np.full(x.shape[0], cap_logp)
            result[cos_dists > d_max] = -np.inf
            return result

        return _TargetDistribution(
            rvs=cap_rvs,
            logpdf=cap_logpdf,
            entropy=-cap_logp,
            interior_mask=lambda x: (1.0 - x @ cap_center) < d_max - margin,
            exterior_mask=lambda x: (1.0 - x @ cap_center) > d_max + margin,
        )
    else:
        raise ValueError(f"Unknown distribution: {name}")


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
@pytest.mark.parametrize("dist_name", ["vmf", "uniform", "hemisphere", "cap"])
def test_train_distribution(domain_dim, dist_name):
    """Train on a distribution and verify SDE samples match the target.

    Uses the SDE sampler (whose distribution the VLB bounds) rather than ODE NLL,
    because the euclidean VDM's ODE NLL converges much more slowly than its sample
    quality in higher dimensions.
    """
    from txt2img_unsupervised.training_infra import train_for_tests

    dist = _make_target_distribution(dist_name, domain_dim)

    if dist_name == "cap":
        model = _make_model(
            domain_dim,
            n_layers=6,
            d_model=512,
            vlb_variance_loss_weight=1e-3,
            log_snr_max_cap=10.0,
        )
    else:
        model = _make_model(
            domain_dim,
            d_model=256,
            vlb_variance_loss_weight=1e-3,
            log_snr_max_cap=10.0,
        )

    batch_size = 1024
    n_samples = 32768 if dist_name != "cap" else 500_000

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
        case ("cap", 16):
            epochs = 40
        case ("vmf", 16):
            epochs = 20
        case ("cap", _):
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
        # The "cap" distribution has a sharp boundary where SDE boundary
        # leakage is worse in high dimensions (concentration of measure puts
        # most mass near the boundary). Use a softer sanity check for it;
        # the KS test below is the main distributional assertion.
        in_support_threshold = 0.95 if dist_name == "cap" else 0.98
        assert in_support_frac > in_support_threshold, (
            f"Only {in_support_frac:.1%} of samples in support"
        )

    # 2. Cross-entropy under the true density should approximate entropy.
    # E_model[-log p_true(x)] ≈ H(p_true) when model ≈ true distribution.
    if np.any(in_support):
        cross_ent = -np.mean(log_probs[in_support])
        print(f"Cross-entropy: {cross_ent:.4f}, entropy: {dist.entropy:.4f}")
        assert cross_ent < dist.entropy + 1.0, (
            f"Cross-entropy {cross_ent:.4f} too far from entropy {dist.entropy:.4f}"
        )

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
    assert max_ks < 0.05, (
        f"Max KS statistic {max_ks:.4f} too large across {n_projections} projections"
    )

    # 5. For distributions with a natural center (cap, hemisphere), compare
    #    cosine distance from that center — tests the radial profile.
    if dist_name in ("cap", "hemisphere"):
        center_np = np.zeros(domain_dim)
        center_np[0] = 1.0
        model_cos_dists = 1.0 - samples_np[in_support] @ center_np
        ref_cos_dists = 1.0 - ref_np @ center_np
        cos_dist_ks, _ = stats.ks_2samp(model_cos_dists, ref_cos_dists)
        print(f"KS on cosine distance from center: {cos_dist_ks:.4f}")
        assert cos_dist_ks < 0.05, (
            f"Cosine distance KS statistic {cos_dist_ks:.4f} too large"
        )


# Per-config test parameters for test_train_cap_conditioned.
# All configs use the same model architecture (4 layers, 256 d_model) and training set (2M samples).
# Step counts target ~4 minutes of training time per config. CG gets fewer steps than CS because
# each CG training step does 2 forward + 2 backward passes (~1.3x slower per step on GPU).
# CG thresholds are tighter than CS: classifier guidance should produce higher quality despite fewer
# training steps, because it separates distribution learning from cap conditioning.
# Values: (steps, in_cap_threshold, ks_threshold)
# fmt: off
_CAP_CONDITIONED_TEST_CONFIGS: dict[
    tuple[int, str, CapConditioningMode], tuple[int, float, float]
] = {
    #                                                           steps   in_cap   KS
    (3,  "uniform", CapConditioningMode.CONDITIONED_SCORE):    (8500,  0.93,    0.06),
    (3,  "uniform", CapConditioningMode.CLASSIFIER_GUIDANCE):  (6500,  0.96,    0.06),
    (3,  "vmf",     CapConditioningMode.CONDITIONED_SCORE):    (8500,  0.88,    0.06),
    (3,  "vmf",     CapConditioningMode.CLASSIFIER_GUIDANCE):  (6500,  0.93,    0.06),
    (16, "uniform", CapConditioningMode.CONDITIONED_SCORE):    (8500,  0.90,    0.08),
    (16, "uniform", CapConditioningMode.CLASSIFIER_GUIDANCE):  (6500,  0.95,    0.06),
    (16, "vmf",     CapConditioningMode.CONDITIONED_SCORE):    (8500,  0.90,    0.08),
    (16, "vmf",     CapConditioningMode.CLASSIFIER_GUIDANCE):  (6500,  0.95,    0.06),
}
# fmt: on


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
@pytest.mark.parametrize("data_distribution", ["uniform", "vmf"])
@pytest.mark.parametrize(
    "cap_conditioning",
    [CapConditioningMode.CONDITIONED_SCORE, CapConditioningMode.CLASSIFIER_GUIDANCE],
)
def test_train_cap_conditioned(domain_dim, data_distribution, cap_conditioning):
    """Train a cap-conditioned model and verify conditioned generation and density.

    All configurations use the same model architecture and training set size. Step counts
    are calibrated so each config trains for approximately 4 minutes. Classifier guidance
    gets fewer steps (each step is slower due to 2 forward + 2 backward passes) but is
    expected to hit tighter accuracy thresholds than conditioned score.
    """
    jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")
    from txt2img_unsupervised.training_infra import train_for_tests

    train_steps, in_cap_threshold, ks_threshold = _CAP_CONDITIONED_TEST_CONFIGS[
        (domain_dim, data_distribution, cap_conditioning)
    ]

    d_max_dist = ((0.8, 1.0), (0.2, 2.0))
    model_kwargs = dict(
        n_layers=4,
        d_model=256,
        cap_conditioning=cap_conditioning,
        d_max_dist=d_max_dist,
        vlb_variance_loss_weight=1e-3,
    )
    if cap_conditioning == CapConditioningMode.CLASSIFIER_GUIDANCE:
        model_kwargs["classifier_loss_weight"] = 10.0
        model_kwargs["cap_features"] = frozenset(
            {"cap_center", "d_max", "log_cap_odds", "cos_sim", "z_norm"}
        )
    model = _make_model(domain_dim, **model_kwargs)

    batch_size = 2048
    n_train_samples = 2_000_000
    n_test_samples = 4096

    # Generate training data
    rng = jax.random.PRNGKey(42)
    if data_distribution == "uniform":
        points = np.asarray(
            sample_sphere(rng, n_train_samples + n_test_samples, domain_dim)
        )
    elif data_distribution == "vmf":
        mean_direction = np.zeros(domain_dim)
        # Axis 1 so vmf center differs from cap center (axis 0)
        mean_direction[1] = 1.0
        vmf_dist = stats.vonmises_fisher(mean_direction, 5.0)
        np_seed = int(jax.random.randint(rng, (), 0, 2**31 - 1))
        points = vmf_dist.rvs(n_train_samples + n_test_samples, random_state=np_seed)
    else:
        raise ValueError(f"Unknown data_distribution: {data_distribution}")

    train_points = points[:n_train_samples]
    test_points = points[n_train_samples:]
    train_dataset = Dataset.from_dict({"point_vec": train_points}).with_format("np")
    test_dataset = Dataset.from_dict({"point_vec": test_points}).with_format("np")

    loss_fn = partial(compute_batch_loss, model)

    result = train_for_tests(
        model,
        train_dataset,
        batch_size,
        learning_rate=1e-3,
        schedule_learning_rate=3e-4,
        use_muon=True,
        loss_fn=loss_fn,
        fields=["point_vec"],
        steps=train_steps,
        test_dataset=test_dataset,
    )
    eval_params = result.state.get_eval_params()

    gamma_min, gamma_max = model.apply(eval_params, method=model.gamma_range)
    print(
        f"Learned schedule: γ_min={float(gamma_min):.2f}, γ_max={float(gamma_max):.2f}"
    )

    # --- Distribution verification via SDE samples and KS tests ---
    # The key property is that the conditioned density inside the cap is proportional to the
    # unconditioned density — the constraint should not distort the distribution. Boundary sharpness
    # is secondary, for embedding generation we don't care if the generated embedding is actually
    # 0.81 away from the prompt embedding instead of <= 0.8, and if we decide we do we can rejection
    # sample.
    cap_center = jnp.zeros(domain_dim).at[0].set(1.0)
    table = LogitsTable(d=domain_dim - 1, n=8192)
    d_max_values = [1.0, 0.5]
    n_gen = 5000
    n_projections = 20

    for d_max_test in d_max_values:
        print(f"\n--- Distribution check for d_max={d_max_test} ---")

        # Generate conditioned SDE samples
        gen_cap_centers = repeat(cap_center, "d -> n d", n=n_gen)
        gen_d_maxes = jnp.full((n_gen,), d_max_test)
        gen_rng = jax.random.PRNGKey(int(d_max_test * 1000))

        samples_np = np.asarray(
            generate_samples_sde(
                model,
                eval_params,
                gen_rng,
                cap_params=(gen_cap_centers, gen_d_maxes),
                n_steps=500,
                eta=1.0,
            )
        )

        sample_cos_dists = 1.0 - samples_np @ np.asarray(cap_center)

        # Sanity check: the model is actually conditioning on the cap (most samples are in or very
        # near it), not ignoring the constraint.
        in_cap_frac = float(np.mean(sample_cos_dists <= d_max_test))
        in_cap_margin = in_cap_frac - in_cap_threshold
        print(
            f"  In-cap fraction: {in_cap_frac:.4f} "
            f"(threshold: {in_cap_threshold:.4f}, margin: {in_cap_margin:+.4f})"
        )
        assert in_cap_frac >= in_cap_threshold, (
            f"In-cap fraction {in_cap_frac:.4f} below threshold {in_cap_threshold:.4f} "
            f"(d_max={d_max_test}, {cap_conditioning.value})"
        )

        outside_mask = sample_cos_dists > d_max_test
        if np.any(outside_mask):
            max_overshoot = float(np.max(sample_cos_dists[outside_mask]) - d_max_test)
            print(f"  Max overshoot past boundary: {max_overshoot:.4f}")

        # Main assertion: the distribution inside the cap should match the true cap-restricted
        # distribution. This verifies that conditioning doesn't distort the density.
        if data_distribution == "uniform":
            ref_np = np.asarray(
                sample_from_cap_v(
                    jax.random.PRNGKey(789), table, cap_center, d_max_test, n_gen
                )
            )
        elif data_distribution == "vmf":
            # Rejection sampling: generate vMF samples and keep those in the cap
            cap_area_frac = float(jnp.exp(table.log_cap_size(d_max_test)))
            # vMF with center orthogonal to cap center — acceptance rate ≈ cap area fraction
            oversample = max(int(5.0 / cap_area_frac), 10)
            vmf_pool = vmf_dist.rvs(
                n_gen * oversample,
                random_state=np.random.default_rng(789),
            )
            pool_cos_dists = 1.0 - vmf_pool @ np.asarray(cap_center)
            in_cap = pool_cos_dists <= d_max_test
            ref_np = vmf_pool[in_cap][:n_gen]
            print(
                f"  vMF rejection sampling: {np.sum(in_cap)}/{len(vmf_pool)} accepted"
            )
            assert len(ref_np) >= n_gen // 2, (
                f"Too few reference samples from rejection sampling: "
                f"{len(ref_np)} (need {n_gen // 2})"
            )
        else:
            raise ValueError(f"Unknown data_distribution: {data_distribution}")

        # Filter model samples to those inside the cap for fair KS comparison
        samples_in_cap = samples_np[~outside_mask]

        # KS test on cosine distance from cap center — checks the radial distribution within the cap
        # matches the expected distribution.
        model_cap_cos_dists = sample_cos_dists[~outside_mask]
        ref_cap_cos_dists = 1.0 - ref_np @ np.asarray(cap_center)
        cos_dist_ks, _ = stats.ks_2samp(model_cap_cos_dists, ref_cap_cos_dists)
        cos_dist_ks_margin = ks_threshold - cos_dist_ks
        print(
            f"  KS on cosine distance from cap center: {cos_dist_ks:.4f} "
            f"(threshold: {ks_threshold:.4f}, margin: {cos_dist_ks_margin:+.4f})"
        )
        assert cos_dist_ks < ks_threshold, (
            f"Cosine distance KS statistic {cos_dist_ks:.4f} >= {ks_threshold:.4f} for d_max={d_max_test}"
        )

        # KS test on random 1D projections
        proj_rng = np.random.default_rng(456 + int(d_max_test * 1000))
        directions = proj_rng.standard_normal((n_projections, domain_dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        ks_stats = []
        for direction in directions:
            stat, _ = stats.ks_2samp(samples_in_cap @ direction, ref_np @ direction)
            ks_stats.append(stat)

        max_ks = max(ks_stats)
        mean_ks = np.mean(ks_stats)
        proj_ks_margin = ks_threshold - max_ks
        print(
            f"  KS stats over {n_projections} projections: "
            f"max={max_ks:.4f}, mean={mean_ks:.4f} "
            f"(threshold: {ks_threshold:.4f}, margin: {proj_ks_margin:+.4f})"
        )
        assert max_ks < ks_threshold, (
            f"Max KS statistic {max_ks:.4f} >= {ks_threshold:.4f} for d_max={d_max_test}"
        )


def _init_small_model(dim=3):
    """Initialize a small model and params for diffrax tests."""
    model = _make_model(dim, d_model=32)
    params_rng = jax.random.PRNGKey(42)
    params = model.init(params_rng, *model.dummy_inputs())
    return model, params


def test_diffrax_ode_samples_on_sphere():
    """Diffrax ODE sampler produces finite outputs on the unit sphere."""
    model, params = _init_small_model(dim=3)
    samples = generate_samples_ode(
        model,
        params,
        jax.random.PRNGKey(0),
        cap_params=None,
        batch_size=16,
    )
    assert samples.shape == (16, 3)
    assert jnp.all(jnp.isfinite(samples)), "Non-finite samples"
    norms = jnp.linalg.norm(samples, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_diffrax_nll_finite():
    """NLL via diffrax produces finite values."""
    model, params = _init_small_model(dim=3)
    x = sample_sphere(jax.random.PRNGKey(1), 8, 3)
    batch = {"point_vec": x}
    nll = compute_nll(model, params, batch, rng=jax.random.PRNGKey(2), n_projections=3)
    assert nll.shape == (8,)
    assert jnp.all(jnp.isfinite(nll)), f"Non-finite NLL values: {nll}"


def _train_vmf_mixture_model():
    """Train a small model on a two-component vMF mixture for diffrax tests.

    Returns a model trained on a bimodal distribution so that sample comparisons
    are meaningful (unlike a point mass where all samples collapse together).
    """
    from txt2img_unsupervised.training_infra import train_for_tests

    dim = 3
    model = _make_model(dim, vlb_variance_loss_weight=1e-3, log_snr_max_cap=10.0)

    mu1 = np.array([1.0, 0.0, 0.0])
    mu2 = np.array([0.0, 1.0, 0.0])
    kappa = 5.0
    vmf1 = stats.vonmises_fisher(mu1, kappa)
    vmf2 = stats.vonmises_fisher(mu2, kappa)

    n_per_component = 12_800
    points = np.vstack([vmf1.rvs(n_per_component), vmf2.rvs(n_per_component)])

    batch_size = 256
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    test_dset = Dataset.from_dict({"point_vec": points[:batch_size]}).with_format("np")

    result = train_for_tests(
        model,
        dset,
        batch_size,
        learning_rate=1e-3,
        loss_fn=partial(compute_batch_loss, model),
        fields=["point_vec"],
        epochs=4,
        test_dataset=test_dset,
    )
    return model, result.state.get_eval_params()


@pytest.mark.usefixtures("starts_with_progressbar")
def test_diffrax_ode_matches_rk4():
    """Diffrax adaptive and fixed-step RK4 produce similar samples on a bimodal distribution."""
    model, eval_params = _train_vmf_mixture_model()

    rng = jax.random.PRNGKey(99)
    n_samples = 128

    samples_diffrax = generate_samples_ode(
        model, eval_params, rng, cap_params=None, batch_size=n_samples
    )
    samples_rk4 = generate_samples_ode(
        model,
        eval_params,
        rng,
        cap_params=None,
        batch_size=n_samples,
        method="rk4",
        n_steps=200,
    )

    # Same initial noise + same ODE => nearly identical outputs
    pairwise_cos = np.sum(np.array(samples_diffrax) * np.array(samples_rk4), axis=1)
    print(f"Pairwise cosine sim (diffrax vs rk4): mean={pairwise_cos.mean():.4f}")
    assert pairwise_cos.mean() > 0.99, (
        f"Diffrax and RK4 samples should be very similar, "
        f"got mean pairwise cosine {pairwise_cos.mean():.4f}"
    )


@pytest.mark.usefixtures("starts_with_progressbar")
def test_diffrax_nll_matches_rk4():
    """Diffrax adaptive and fixed-step RK4 produce similar NLL values."""
    model, eval_params = _train_vmf_mixture_model()

    # In 3D with n_projections >= dim, the Hutchinson estimator computes the
    # exact trace (3 orthogonal projections span R^3). So the only source of
    # difference between the two methods is integration accuracy, and both
    # should agree closely.
    test_points = sample_sphere(jax.random.PRNGKey(5), 16, 3)
    batch = {"point_vec": test_points}
    rng = jax.random.PRNGKey(7)

    nll_diffrax = compute_nll(model, eval_params, batch, rng=rng, n_projections=10)
    nll_rk4 = compute_nll(
        model,
        eval_params,
        batch,
        rng=rng,
        n_projections=10,
        method="rk4",
        n_steps=200,
    )

    mean_diffrax = float(np.array(nll_diffrax).mean())
    mean_rk4 = float(np.array(nll_rk4).mean())
    print(f"NLL diffrax: {mean_diffrax:.4f}")
    print(f"NLL rk4: {mean_rk4:.4f}")
    print(f"Mean abs diff: {abs(mean_diffrax - mean_rk4):.4f}")
    assert abs(mean_diffrax - mean_rk4) < 0.1, (
        f"NLL estimates too far apart: diffrax={mean_diffrax:.4f}, rk4={mean_rk4:.4f}"
    )


# =============================================================================
# Classifier guidance tests
# =============================================================================


def test_classifier_loss_gradients_flow():
    """Gradients from both VLB and classifier losses flow to backbone parameters."""
    dim = 3
    d_max_dist = ((1.0, 2.0),)
    model = _make_model(
        dim,
        d_model=32,
        cap_conditioning=CapConditioningMode.CLASSIFIER_GUIDANCE,
        d_max_dist=d_max_dist,
    )

    rng = jax.random.PRNGKey(42)
    params_rng, data_rng, loss_rng = jax.random.split(rng, 3)
    params = model.init(params_rng, *model.dummy_inputs())

    x_1 = sample_sphere(data_rng, 16, dim)
    batch = {"point_vec": x_1}

    def loss_fn(params):
        return compute_batch_loss(model, params, batch, loss_rng)

    (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

    # Loss should be finite
    assert jnp.isfinite(loss), f"Loss is not finite: {loss}"
    assert jnp.isfinite(components["classifier"]), (
        f"Classifier loss is not finite: {components['classifier']}"
    )
    assert components["classifier"] > 0, "Classifier loss should be positive"

    # Backbone should receive gradients
    vf_grads = grads["params"]["vector_field"]
    vf_grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(vf_grads)))
    assert vf_grad_norm > 0, "No gradients flowing to vector_field"

    # Classifier head should receive gradients
    ch_grads = grads["params"]["classifier_head"]
    ch_grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(ch_grads)))
    assert ch_grad_norm > 0, "No gradients flowing to classifier_head"

    # Schedule should receive gradients
    sched_grads = grads["params"]["schedule"]
    sched_grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in jax.tree.leaves(sched_grads)))
    assert sched_grad_norm > 0, "No gradients flowing to schedule"


@pytest.mark.usefixtures("starts_with_progressbar")
def test_classifier_calibration_uniform():
    """Train classifier on uniform S^2 data, verify outputs match true P(x1 in cap | z_t, t).

    For uniform data the posterior q(x1 | z_t, t) is a von Mises-Fisher distribution
    concentrated around z_t's direction, so P(x1 in cap | z_t, t) depends on the
    angle between z_t and the cap center (it is NOT simply cap_area/sphere_area).
    We estimate the true P via importance-weighted Monte Carlo and compare to the
    classifier's output.
    """
    jax.config.update("jax_compilation_cache_dir", "/tmp/t2i-u-jax-cache")
    from txt2img_unsupervised.training_infra import train_for_tests

    dim = 3
    d_max_dist = ((1.0, 2.0),)
    model = _make_model(
        dim,
        n_layers=2,
        d_model=64,
        cap_conditioning=CapConditioningMode.CLASSIFIER_GUIDANCE,
        d_max_dist=d_max_dist,
        vlb_variance_loss_weight=1e-3,
        classifier_loss_weight=10.0,
    )

    rng = jax.random.PRNGKey(42)
    n_train = 32_768
    n_test = 4096
    all_points = np.asarray(sample_sphere(rng, n_train + n_test, dim))
    train_dataset = Dataset.from_dict({"point_vec": all_points[:n_train]}).with_format(
        "np"
    )
    test_dataset = Dataset.from_dict({"point_vec": all_points[n_train:]}).with_format(
        "np"
    )

    result = train_for_tests(
        model,
        train_dataset,
        batch_size=512,
        learning_rate=1e-3,
        schedule_learning_rate=3e-4,
        use_muon=True,
        loss_fn=partial(compute_batch_loss, model),
        fields=["point_vec"],
        epochs=40,
        test_dataset=test_dataset,
    )
    eval_params = result.state.get_eval_params()

    gamma_min, gamma_max = model.apply(eval_params, method=model.gamma_range)
    print(
        f"Learned schedule: γ_min={float(gamma_min):.2f}, γ_max={float(gamma_max):.2f}"
    )

    # MC reference pool for importance-weighted estimation of true P(x1 in cap | z_t, t)
    n_mc = 2_000_000
    mc_points = np.asarray(sample_sphere(jax.random.PRNGKey(999), n_mc, dim))

    # Test configurations: (z_t direction, cap_center, d_max)
    test_configs = [
        # z_t near cap → high P
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]), 0.5),
        # z_t far from cap → low P
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([-1.0, 0.0, 0.0]), 0.5),
        # z_t at 90° from cap center
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]), 0.5),
        # hemisphere (d_max=1.0)
        (jnp.array([1.0, 0.0, 0.0]), jnp.array([1.0, 0.0, 0.0]), 1.0),
        # large cap (d_max=1.5)
        (jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]), 1.5),
    ]

    # Test at several noise levels across the learned schedule
    gamma_range = float(gamma_max) - float(gamma_min)
    gammas = [
        float(gamma_min) + 0.15 * gamma_range,  # high noise
        float(gamma_min) + 0.5 * gamma_range,  # medium noise
        float(gamma_min) + 0.85 * gamma_range,  # low noise
    ]

    errors = []
    for gamma in gammas:
        alpha, sigma = _snr_to_alpha_sigma(gamma)
        alpha_f, sigma_f = float(alpha), float(sigma)

        for z_dir, cap_center, d_max in test_configs:
            # Construct a z_t: signal along z_dir + a fixed noise offset
            z_t = alpha_f * z_dir + sigma_f * jnp.array([0.1, -0.3, 0.2])

            # Classifier prediction (with logit correction already applied)
            logit = model.apply(
                eval_params,
                z_t[None],
                jnp.array([gamma]),
                (cap_center[None], jnp.array([d_max])),
                method=model.classifier_forward,
            )
            p_pred = float(jax.nn.sigmoid(logit[0]))

            # MC estimate of true P(x1 in cap | z_t, t)
            # q(z_t | x1, t) ∝ exp(-||z_t - α·x1||² / (2σ²))
            # Ignoring radial augmentation (σ_radial=0.01 ≈ 0)
            dots = mc_points @ np.asarray(z_t)
            log_w = alpha_f * dots / (sigma_f**2)
            log_w -= np.max(log_w)
            w = np.exp(log_w)

            cos_dists = 1.0 - mc_points @ np.asarray(cap_center)
            in_cap = cos_dists <= d_max
            p_true = float(np.sum(w * in_cap) / np.sum(w))

            err = abs(p_pred - p_true)
            errors.append(err)
            print(
                f"  γ={gamma:.1f} z→cap_cos={float(z_dir @ cap_center):.2f} "
                f"d_max={d_max:.1f}: p_pred={p_pred:.4f} p_true={p_true:.4f} "
                f"err={err:.4f}"
            )
            assert err < 0.1, (
                f"Classifier prediction {p_pred:.4f} too far from true "
                f"probability {p_true:.4f} (err={err:.4f})"
            )

    print(f"Mean absolute error: {np.mean(errors):.4f}")
    print(f"Max absolute error: {np.max(errors):.4f}")


def test_classifier_guidance_768d():
    """Verify classifier guidance mechanism works at CLIP embedding dimensionality (768D).

    Full distributional convergence at 768D requires very long training (the existing
    tests also only go up to dim=16 for convergence). This test verifies that the
    mechanism works at 768D: initialization, forward pass, classifier gradient, and
    guided sampling all produce finite results with correct shapes.
    """
    domain_dim = 768
    d_max_dist = ((1.0, 2.0),)
    model = _make_model(
        domain_dim,
        d_model=64,
        n_layers=2,
        cap_conditioning=CapConditioningMode.CLASSIFIER_GUIDANCE,
        d_max_dist=d_max_dist,
    )

    rng = jax.random.PRNGKey(0)
    params = model.init(rng, *model.dummy_inputs())

    # Verify forward pass produces finite results
    z = jax.random.normal(jax.random.PRNGKey(1), (4, domain_dim))
    log_snr = jnp.full((4,), 0.0)
    eps_hat = model.apply(params, z, log_snr, None, method=model.predict_eps)
    assert eps_hat.shape == (4, domain_dim)
    assert jnp.all(jnp.isfinite(eps_hat))

    # Verify classifier forward produces finite logits
    cap_center = jnp.zeros(domain_dim).at[0].set(1.0)
    cap_params = (repeat(cap_center, "d -> n d", n=4), jnp.full((4,), 0.5))
    logits = model.apply(
        params, z, log_snr, cap_params, method=model.classifier_forward
    )
    assert logits.shape == (4,)
    assert jnp.all(jnp.isfinite(logits))

    # Verify guided eps prediction produces finite results
    guided_eps = _predict_eps_guided(model, params, cap_params, z, 0.0)
    assert guided_eps.shape == (4, domain_dim)
    assert jnp.all(jnp.isfinite(guided_eps))

    # Guided prediction should differ from unguided (classifier gradient is non-zero)
    unguided_eps = _predict_eps(model, params, None, z, 0.0)
    assert not jnp.allclose(guided_eps, unguided_eps, atol=1e-6), (
        "Guided and unguided predictions are identical — classifier gradient may be zero"
    )

    # Verify SDE sampling with guidance produces finite on-sphere outputs
    samples = generate_samples_sde(
        model, params, jax.random.PRNGKey(2), cap_params, n_steps=10
    )
    assert samples.shape == (4, domain_dim)
    assert jnp.all(jnp.isfinite(samples))
    norms = jnp.linalg.norm(samples, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_classifier_logit_correction_hemisphere():
    """d_max=1.0 is a hemisphere (F=0.5), so the correction should be zero."""
    for d in [2, 15, 767]:
        table = LogitsTable(d, 8192)
        correction = classifier_logit_correction(table, 1.0)
        np.testing.assert_allclose(float(correction), 0.0, atol=1e-4)


@pytest.mark.parametrize("d_max", [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8])
def test_classifier_logit_correction_3d(d_max):
    """On S^2 (domain_dim=3), cap fractional area is d_max/2, giving an
    analytically known correction of log(d_max / (2 - d_max))."""
    table = LogitsTable(2, 8192)
    expected = float(jnp.log(d_max / (2.0 - d_max)))
    actual = float(classifier_logit_correction(table, d_max))
    np.testing.assert_allclose(actual, expected, atol=5e-4)


def test_classifier_logit_correction_sign():
    """Correction is negative for small caps (d_max < 1) and positive for large caps."""
    for d in [2, 15, 767]:
        table = LogitsTable(d, 8192)
        assert classifier_logit_correction(table, 0.5) < 0
        assert classifier_logit_correction(table, 1.5) > 0
