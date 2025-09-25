"""Mixtures of von Mises-Fisher distributions in JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
from jax import Array, random
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from txt2img_unsupervised import vmf

_NEAR_UNIFORM_THRESHOLD = 1e-8
_NEAR_DEGENERATE_THRESHOLD = 1e-5
_WEIGHT_FLOOR = 1e-8


@dataclass(frozen=True)
class VmfMixture:
    """Parameters for a mixture of von Mises-Fisher distributions."""

    weights: Array
    mus: Array
    kappas: Array

    def __post_init__(self) -> None:
        mus = jnp.asarray(self.mus)
        if mus.ndim != 2:
            raise ValueError("mus must have shape (k, d)")

        dtype = mus.dtype
        weights = jnp.asarray(self.weights, dtype=dtype)
        kappas = jnp.asarray(self.kappas, dtype=dtype)

        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if kappas.ndim != 1:
            raise ValueError("kappas must be one-dimensional")
        if weights.shape[0] != mus.shape[0] or kappas.shape[0] != mus.shape[0]:
            raise ValueError("weights, mus, and kappas must agree on component count")
        if bool(jnp.any(weights < 0)):
            raise ValueError("weights must be non-negative")

        total_weight = jnp.sum(weights)
        if not np.isfinite(float(total_weight)) or float(total_weight) <= 0.0:
            raise ValueError("weights must sum to a positive finite value")

        normalized_weights = weights / total_weight

        norms = jnp.linalg.norm(mus, axis=1, keepdims=True)
        mus_unit = mus / jnp.maximum(norms, 1e-8)

        object.__setattr__(self, "weights", normalized_weights)
        object.__setattr__(self, "mus", mus_unit)
        object.__setattr__(self, "kappas", kappas)

    @property
    def num_components(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.mus.shape[1])


def log_prob(x: Array, mixture: VmfMixture) -> Array:
    """Compute log-density under a von Mises-Fisher mixture."""

    x = jnp.asarray(x)
    component_terms = []
    for idx in range(mixture.num_components):
        component_terms.append(
            vmf.log_prob(x, mixture.mus[idx], float(mixture.kappas[idx]))
        )
    stacked = jnp.stack(component_terms, axis=0)

    log_weights = jnp.log(mixture.weights + _WEIGHT_FLOOR)

    # Handle both single point (1D) and batch (2D) cases
    if x.ndim == 1:
        # For single point: stacked is (n_components,), log_weights is (n_components,)
        return jsp.logsumexp(stacked + log_weights, axis=0)
    else:
        # For batch: stacked is (n_components, n_points), log_weights needs broadcasting
        return jsp.logsumexp(stacked + log_weights[:, None], axis=0)


def sample(key: random.PRNGKey, mixture: VmfMixture, n_samples: int) -> Array:
    """Draw samples from a von Mises-Fisher mixture."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    key_comp, key_sample = random.split(key)
    component_indices = random.categorical(
        key_comp, jnp.log(mixture.weights), shape=(n_samples,)
    )

    component_indices_np = np.asarray(component_indices)
    component_keys = random.split(key_sample, mixture.num_components)
    samples_np = np.empty((n_samples, mixture.dim), dtype=np.asarray(mixture.mus).dtype)

    for component_idx in range(mixture.num_components):
        positions = np.where(component_indices_np == component_idx)[0]
        if positions.size == 0:
            continue
        draw = vmf.sample(
            component_keys[component_idx],
            mixture.mus[component_idx],
            float(mixture.kappas[component_idx]),
            int(positions.size),
        )
        samples_np[positions] = np.asarray(draw)

    return jnp.asarray(samples_np)


def fit(
    x: Array,
    n_components: int,
    key: random.PRNGKey,
    max_iters: int = 100,
    tol: float = 1e-5,
    max_kappa: Optional[float] = vmf._DEFAULT_MAX_KAPPA,
    weights: Optional[Array] = None,
) -> Tuple[VmfMixture, Array]:
    """Fit a von Mises-Fisher mixture using expectation-maximization.

    Args:
        x: Observations on the unit sphere with shape (n, d).
        n_components: Number of mixture components to fit.
        key: PRNG key for initialization.
        max_iters: Maximum number of EM iterations.
        tol: Convergence tolerance on log-likelihood improvements.
        max_kappa: Optional upper bound for component concentration parameters passed through to
            the underlying vMF fits. ``None`` disables clamping.
        weights: Optional log-weights for each example, shape (n,). If provided, examples
            are weighted according to softmax(weights) in the EM algorithm.

    Returns:
        A tuple containing the fitted mixture parameters and posterior responsibilities
        with shape (n, n_components).
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive")

    x = jnp.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must have shape (n, d)")

    n, dim = x.shape
    if n_components > n:
        raise ValueError("n_components cannot exceed the number of samples")

    # Input validation for weights
    if weights is not None:
        weights = jnp.asarray(weights, dtype=x.dtype)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        if not jnp.all(jnp.isfinite(weights)):
            raise ValueError("weights must be finite")
        # Convert log-weights to probabilities via softmax
        example_probs = jax.nn.softmax(weights)
    else:
        example_probs = None

    dtype = x.dtype
    permutation = random.permutation(key, n)
    init_indices = permutation[:n_components]
    mus = x[init_indices]
    norms = jnp.linalg.norm(mus, axis=1, keepdims=True)
    mus = mus / jnp.maximum(norms, 1e-8)
    kappas = jnp.ones(n_components, dtype=dtype) * 10.0
    weights = jnp.ones(n_components, dtype=dtype) / n_components

    prev_log_likelihood = -np.inf
    responsibilities = jnp.full((n, n_components), 1.0 / n_components, dtype=dtype)

    for _ in range(max_iters):
        component_terms = []
        for idx in range(n_components):
            component_terms.append(vmf.log_prob(x, mus[idx], float(kappas[idx])))
        stacked = jnp.stack(component_terms, axis=1)

        log_weights = jnp.log(weights + _WEIGHT_FLOOR)
        log_joint = stacked + log_weights
        log_norm = jsp.logsumexp(log_joint, axis=1)
        responsibilities = jnp.exp(log_joint - log_norm[:, None])

        # Include example weights in log-likelihood and responsibilities if provided
        if example_probs is not None:
            log_likelihood = float(jnp.sum(example_probs * log_norm))
            # Weight responsibilities by example probabilities
            responsibilities = responsibilities * example_probs[:, None]
            effective_n = jnp.sum(example_probs)
        else:
            log_likelihood = float(jnp.sum(log_norm))
            effective_n = float(n)

        component_masses = jnp.sum(responsibilities, axis=0)
        weights_new = component_masses / effective_n
        weights_new = jnp.maximum(weights_new, _WEIGHT_FLOOR)
        weights_new = weights_new / jnp.sum(weights_new)

        new_mus = []
        new_kappas = []
        for idx in range(n_components):
            resp = responsibilities[:, idx]
            mass = float(component_masses[idx])
            if mass <= _NEAR_UNIFORM_THRESHOLD:
                new_mus.append(mus[idx])
                new_kappas.append(float(kappas[idx]))
                continue

            weighted_sum = jnp.sum(resp[:, None] * x, axis=0)
            weighted_norm = jnp.linalg.norm(weighted_sum)
            mu_hat = weighted_sum / jnp.maximum(weighted_norm, 1e-8)
            r_bar = float(weighted_norm / mass)

            if r_bar < _NEAR_UNIFORM_THRESHOLD:
                kappa_hat = 0.0
            elif r_bar >= 1.0 - _NEAR_DEGENERATE_THRESHOLD:
                effective_r = min(r_bar, 1.0 - 1e-8)
                kappa_hat = vmf._solve_kappa_mle(effective_r, dim, max_kappa=max_kappa)
            else:
                kappa_hat = vmf._solve_kappa_mle(r_bar, dim, max_kappa=max_kappa)

            new_mus.append(mu_hat)
            new_kappas.append(kappa_hat)

        mus = jnp.stack(new_mus)
        kappas = jnp.asarray(new_kappas, dtype=dtype)
        weights = weights_new

        improvement = abs(log_likelihood - prev_log_likelihood)
        if improvement < tol:
            prev_log_likelihood = log_likelihood
            break

        prev_log_likelihood = log_likelihood

    mixture = VmfMixture(weights=weights, mus=mus, kappas=kappas)
    return mixture, responsibilities
