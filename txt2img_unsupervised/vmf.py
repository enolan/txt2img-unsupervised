"""von Mises-Fisher distributions in JAX."""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import Array, random
import jax.scipy.special as jsp
import numpy as np
import scipy.special as sps

_WOOD_MAX_ITERS = 1000
_BESSEL_SERIES_TOL = 1e-12
_BESSEL_SERIES_MAX_TERMS = 1000
_DEFAULT_MAX_KAPPA = 1e8


def _log_modified_bessel_small_x(nu: float, x: float) -> float:
    """Series expansion for ``log I_ν(x)`` when ``x`` is tiny."""

    if x == 0.0:
        return 0.0 if nu == 0.0 else -np.inf

    log_x_over_2 = np.log(x / 2.0)
    log_gamma = sps.gammaln(nu + 1.0)

    term = 1.0
    total = 1.0
    for k in range(1, _BESSEL_SERIES_MAX_TERMS):
        term *= (x * x / 4.0) / (k * (k + nu))
        total += term
        if term < _BESSEL_SERIES_TOL * total:
            break

    return nu * log_x_over_2 - log_gamma + np.log(total)


def _log_modified_bessel_i(nu: float, x: float) -> Array:
    """Compute ``log I_ν(x)`` via SciPy with overflow-safe scaling."""

    x_np = np.asarray(x, dtype=np.float64)
    nu_np = np.asarray(nu, dtype=np.float64)

    if x_np == 0.0:
        if nu_np == 0.0:
            return jnp.asarray(0.0, dtype=jnp.asarray(x).dtype)
        return jnp.asarray(-jnp.inf, dtype=jnp.asarray(x).dtype)

    scaled = sps.ive(nu_np, x_np)
    if scaled > 0.0 and np.isfinite(scaled):
        log_val = np.log(scaled) + np.abs(x_np)
    else:
        log_val = _log_modified_bessel_small_x(nu_np, x_np)

    return jnp.asarray(log_val, dtype=jnp.asarray(x).dtype)


def log_normalization_constant(kappa: float, dim: int) -> float:
    """Compute log normalization constant for vMF distribution.

    Args:
        kappa: Concentration parameter (>= 0)
        dim: Dimension of the sphere (ambient space dimension)

    Returns:
        Log normalization constant C_d(kappa)
    """
    if kappa == 0.0:
        # Uniform distribution on sphere: normalization constant = 1/surface_area
        # Surface area = 2*pi^(d/2) / Gamma(d/2)
        # So log normalization = -log(2) - (d/2)*log(pi) + log(Gamma(d/2))
        return -jnp.log(2.0) - (dim / 2.0) * jnp.log(jnp.pi) + jsp.gammaln(dim / 2.0)

    nu = dim / 2.0 - 1.0  # Bessel function order

    # Compute log I_nu(kappa) using our helper function
    log_bessel = _log_modified_bessel_i(nu, kappa)

    # C_d(kappa) = kappa^(d/2-1) / ((2*pi)^(d/2) * I_(d/2-1)(kappa))
    log_c = (
        (dim / 2.0 - 1.0) * jnp.log(kappa)
        - (dim / 2.0) * jnp.log(2 * jnp.pi)
        - log_bessel
    )

    return log_c


def log_prob(x: Array, mu: Array, kappa: float) -> Array:
    """Compute log probability density of vMF distribution.

    Args:
        x: Points on unit sphere, shape (..., d)
        mu: Mean direction on unit sphere, shape (d,)
        kappa: Concentration parameter (>= 0)

    Returns:
        Log probability densities, shape (...)
    """
    # Validate inputs are on unit sphere
    x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    mu_norm = jnp.linalg.norm(mu)

    # Ensure unit vectors (with small tolerance)
    x = x / jnp.maximum(x_norm, 1e-8)
    mu = mu / jnp.maximum(mu_norm, 1e-8)

    dim = mu.shape[-1]

    # Compute dot product
    dot_product = jnp.sum(x * mu, axis=-1)

    # Log probability = log C_d(kappa) + kappa * mu^T x
    log_c = log_normalization_constant(kappa, dim)
    return log_c + kappa * dot_product


def sample(key: random.PRNGKey, mu: Array, kappa: float, n_samples: int) -> Array:
    """Sample from von Mises-Fisher distribution using Wood's algorithm.

    Args:
        key: JAX random key
        mu: Mean direction on unit sphere, shape (d,)
        kappa: Concentration parameter (>= 0)
        n_samples: Number of samples to generate

    Returns:
        Samples from vMF distribution, shape (n_samples, d)
    """
    dim = mu.shape[-1]

    # Ensure mu is unit vector
    mu = mu / jnp.linalg.norm(mu)

    if kappa == 0.0:
        # Uniform distribution on sphere
        return _sample_uniform_sphere(key, dim, n_samples)

    return _sample_wood(key, mu, kappa, n_samples)


def _sample_uniform_sphere(key: random.PRNGKey, dim: int, n_samples: int) -> Array:
    """Sample uniformly from unit sphere."""
    # Sample from standard normal and normalize
    samples = random.normal(key, (n_samples, dim))
    norms = jnp.linalg.norm(samples, axis=-1, keepdims=True)
    return samples / norms


def _sample_wood(key: random.PRNGKey, mu: Array, kappa: float, n_samples: int) -> Array:
    """Sample using Wood's algorithm for vMF distribution."""
    dim = mu.shape[-1]

    key1, key2 = random.split(key)

    # Sample w from the marginal distribution on [-1, 1]
    w = _sample_w_wood(key1, kappa, dim, n_samples)

    # Sample v uniformly from S^(d-2) and embed in orthogonal subspace to mu
    # Generate random vector in full space
    v_full = random.normal(key2, (n_samples, dim))

    # Project out mu component and normalize
    mu_component = jnp.sum(v_full * mu[None, :], axis=1, keepdims=True)
    v_orthogonal = v_full - mu_component * mu[None, :]
    v_orthogonal_norms = jnp.linalg.norm(v_orthogonal, axis=1, keepdims=True)
    v_orthogonal = v_orthogonal / jnp.maximum(v_orthogonal_norms, 1e-8)

    # Construct final samples: x = w * mu + sqrt(1 - w^2) * v_orthogonal
    sqrt_term = jnp.sqrt(jnp.maximum(1 - w**2, 0.0))
    samples = w[..., None] * mu[None, :] + sqrt_term[..., None] * v_orthogonal

    # Normalize to ensure unit vectors (should already be close)
    norms = jnp.linalg.norm(samples, axis=-1, keepdims=True)
    return samples / jnp.maximum(norms, 1e-8)


def _sample_w_wood(
    key: random.PRNGKey, kappa: float, dim: int, n_samples: int
) -> Array:
    """Sample the longitudinal component using Wood's rejection sampler."""

    keys = random.split(key, n_samples)

    def sample_single(single_key: random.PRNGKey) -> Array:
        kappa_arr = jnp.asarray(kappa)
        dim_arr = jnp.asarray(dim, dtype=kappa_arr.dtype)
        alpha = 0.5 * (dim_arr - 1.0)
        b = (
            -2.0 * kappa_arr + jnp.sqrt(4.0 * kappa_arr**2 + (dim_arr - 1.0) ** 2)
        ) / (dim_arr - 1.0)
        x0 = (1.0 - b) / (1.0 + b)
        c = kappa_arr * x0 + (dim_arr - 1.0) * jnp.log(1.0 - x0 * x0)

        def cond(state):
            _, _, accepted, iterations, _ = state
            not_finished = jnp.logical_not(accepted)
            below_cap = iterations < _WOOD_MAX_ITERS
            return jnp.logical_and(not_finished, below_cap)

        def body(state):
            key_inner, w_current, accepted, iterations, last_candidate = state
            key_inner, key_z, key_u = random.split(key_inner, 3)
            z = random.beta(key_z, alpha, alpha)
            w_candidate = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
            log_accept = (
                kappa_arr * w_candidate
                + (dim_arr - 1.0) * jnp.log(1.0 - x0 * w_candidate)
                - c
            )
            log_u = jnp.log(random.uniform(key_u))
            accept = log_u <= log_accept
            w_next = jnp.where(accept, w_candidate, w_current)
            accepted_next = jnp.logical_or(accepted, accept)
            return (
                key_inner,
                w_next,
                accepted_next,
                iterations + 1,
                w_candidate,
            )

        dtype = jnp.asarray(kappa).dtype
        init_state = (
            single_key,
            jnp.array(1.0, dtype=dtype),
            jnp.array(False),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(1.0, dtype=dtype),
        )

        _, w_final, accepted_final, _, last_candidate = jax.lax.while_loop(
            cond, body, init_state
        )

        return jnp.where(accepted_final, w_final, last_candidate)

    return jax.vmap(sample_single)(keys)


def fit(
    x: Array,
    max_kappa: Optional[float] = _DEFAULT_MAX_KAPPA,
    weights: Optional[Array] = None,
) -> Tuple[Array, float]:
    """Fit vMF distribution parameters via maximum likelihood.

    Args:
        x: Data points on unit sphere, shape (n, d)
        max_kappa: Optional upper bound for the concentration estimate. ``None`` disables
            clamping.
        weights: Optional log-weights for each example, shape (n,). If provided, examples
            are weighted according to softmax(weights) in the MLE computation.

    Returns:
        Tuple of (mu_hat, kappa_hat)
    """
    n, dim = x.shape

    # Input validation for weights
    if weights is not None:
        weights = jnp.asarray(weights)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        if not jnp.all(jnp.isfinite(weights)):
            raise ValueError("weights must be finite")

    # Compute empirical mean direction (weighted or unweighted)
    if weights is None:
        R = jnp.mean(x, axis=0)
    else:
        # Convert log-weights to probabilities via softmax
        probs = jax.nn.softmax(weights)
        R = jnp.sum(probs[:, None] * x, axis=0)

    R_norm = jnp.linalg.norm(R)

    mu_hat = R / jnp.maximum(R_norm, 1e-8)

    # Solve for kappa using MLE equation: A_d(kappa) = ||R||
    # where A_d(kappa) = I_{d/2}(kappa) / I_{d/2-1}(kappa)

    if R_norm < 1e-8:
        # Nearly uniform distribution
        kappa_hat = 0.0
    else:
        if R_norm >= 1.0 - 1e-5:
            effective_r = float(min(R_norm, 1.0 - 1e-8))
            kappa_hat = _solve_kappa_mle(effective_r, dim, max_kappa=max_kappa)
        else:
            kappa_hat = _solve_kappa_mle(float(R_norm), dim, max_kappa=max_kappa)

    return mu_hat, kappa_hat


def _solve_kappa_mle(
    r_norm: float, dim: int, max_kappa: Optional[float] = _DEFAULT_MAX_KAPPA
) -> float:
    """Solve for kappa in MLE equation ``A_d(kappa) = r_norm``."""

    nu_upper = dim / 2.0
    nu_lower = dim / 2.0 - 1.0

    def bessel_ratio(kappa: float) -> float:
        kappa_val = float(kappa)
        if kappa_val <= 0.0:
            return 0.0
        log_upper = _log_modified_bessel_i(nu_upper, kappa_val)
        log_lower = _log_modified_bessel_i(nu_lower, kappa_val)
        return float(jnp.exp(log_upper - log_lower))

    def objective(kappa: float) -> float:
        return bessel_ratio(kappa) - r_norm

    # Initial guess using asymptotic inverse
    if r_norm > 0.9:
        # Use asymptotic formula: kappa ≈ (d-1)/(2(1-r_norm))
        kappa_init = (dim - 1.0) / (2.0 * (1.0 - r_norm + 1e-8))
    else:
        # Use small kappa approximation: A_d(kappa) ≈ kappa / (d-1)
        kappa_init = r_norm * (dim - 1.0)

    # Simple bisection method for root finding
    kappa_low = 0.0
    kappa_high = max(1000.0, 2.0 * kappa_init)
    if max_kappa is not None:
        kappa_high = min(kappa_high, max_kappa)

    for _ in range(32):
        if objective(kappa_high) > 0:
            break
        if max_kappa is not None and kappa_high >= max_kappa:
            return float(max_kappa)
        new_high = kappa_high * 2.0
        if max_kappa is not None:
            new_high = min(new_high, max_kappa)
        kappa_high = new_high
    else:
        if max_kappa is not None:
            return float(max_kappa)
        raise RuntimeError("Failed to bracket kappa in _solve_kappa_mle")

    for _ in range(100):
        kappa_mid = 0.5 * (kappa_low + kappa_high)
        obj_mid = objective(kappa_mid)

        if abs(obj_mid) < 1e-8:
            kappa_high = kappa_mid
            break

        if obj_mid > 0:
            kappa_high = kappa_mid
        else:
            kappa_low = kappa_mid

    estimate = 0.5 * (kappa_low + kappa_high)
    if max_kappa is not None:
        estimate = min(estimate, max_kappa)
    return float(estimate)
