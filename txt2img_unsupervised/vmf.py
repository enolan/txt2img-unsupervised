"""von Mises-Fisher distributions in JAX."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, random
import numpy as np
import pytest
import scipy.special as sps

from txt2img_unsupervised.cap_sampling import sphere_log_inverse_surface_area

_WOOD_MAX_ITERS = 1000
# Gauss-Legendre quadrature nodes and weights for implicit reparameterization gradient.
# Precomputed once at import time in float64 (via numpy/scipy), then cast to float32.
_N_QUAD = 128
_GL_NODES_NP, _GL_WEIGHTS_NP = np.polynomial.legendre.leggauss(_N_QUAD)
_GL_NODES = jnp.array(_GL_NODES_NP, dtype=jnp.float32)
_GL_WEIGHTS = jnp.array(_GL_WEIGHTS_NP, dtype=jnp.float32)


def _prepare_sample_inputs(
    mu: Array, kappa, n_samples: int
) -> Tuple[Array, Array, int]:
    """Normalize and broadcast inputs for vMF sampling.

    Args:
        mu: Mean direction(s), shape (d,) or (n_samples, d)
        kappa: Concentration parameter(s), scalar or shape (n_samples,)
        n_samples: Number of samples

    Returns:
        (mu, kappa, dim) where mu is (n_samples, d) normalized, kappa is (n_samples,)
    """
    assert jnp.issubdtype(
        mu.dtype, jnp.floating
    ), f"mu must have float dtype, got {mu.dtype}"

    if mu.ndim == 1:
        dim = mu.shape[0]
        mu = mu / jnp.linalg.norm(mu)
        mu = jnp.broadcast_to(mu, (n_samples, dim))
        kappa = jnp.full((n_samples,), kappa)
    elif mu.ndim == 2:
        batch_size, dim = mu.shape
        kappa = jnp.asarray(kappa)
        assert (
            batch_size == n_samples
        ), f"mu batch size ({batch_size}) must match n_samples ({n_samples})"
        assert kappa.shape == (
            n_samples,
        ), f"kappa must have shape ({n_samples},), got {kappa.shape}"
        mu = mu / jnp.linalg.norm(mu, axis=1, keepdims=True)
    else:
        raise ValueError(f"mu must be 1D or 2D, got {mu.ndim}D")

    return mu, kappa, dim


def _combine_w_with_tangent(key_v: random.PRNGKey, mu: Array, w: Array) -> Array:
    """Sample a tangent direction and combine with w to produce vMF samples.

    Args:
        key_v: JAX random key for the tangent direction
        mu: Mean directions, shape (n, d), assumed already normalized
        w: Cosine component values, shape (n,)

    Returns:
        Samples on the unit sphere, shape (n, d)
    """
    n, dim = mu.shape
    v_full = random.normal(key_v, (n, dim))
    mu_component = jnp.sum(v_full * mu, axis=1, keepdims=True)
    v_orthogonal = v_full - mu_component * mu
    v_orthogonal = v_orthogonal / jnp.maximum(
        jnp.linalg.norm(v_orthogonal, axis=1, keepdims=True), 1e-8
    )

    # important to enforce input to sqrt is positive because d/dw sqrt(1 - w^2) is -inf when
    # w = 1.0
    sqrt_term = jnp.sqrt(jnp.maximum(1 - w**2, 1e-12))
    samples = w[:, None] * mu + sqrt_term[:, None] * v_orthogonal

    return samples / jnp.maximum(jnp.linalg.norm(samples, axis=1, keepdims=True), 1e-8)


@partial(jax.jit, static_argnames=("n_samples",), inline=True)
def sample(key: random.PRNGKey, mu: Array, kappa, n_samples: int) -> Array:
    """Sample from von Mises-Fisher distribution using Wood's algorithm.

    Differentiable w.r.t. both mu and kappa. Gradients through kappa use
    implicit reparameterization (custom_vjp) since Wood's rejection loop
    is not directly differentiable.

    Supports two modes:
    - Unbatched: mu shape (d,), kappa scalar -> (n_samples, d) samples from single distribution
    - Batched: mu shape (n_samples, d), kappa shape (n_samples,) -> (n_samples, d) samples,
      one from each distribution

    Args:
        key: JAX random key
        mu: Mean direction(s) on unit sphere, shape (d,) or (n_samples, d)
        kappa: Concentration parameter(s) (>= 0), scalar or shape (n_samples,)
        n_samples: Number of samples to generate

    Returns:
        Samples from vMF distribution, shape (n_samples, d)
    """
    mu, kappa, dim = _prepare_sample_inputs(mu, kappa, n_samples)

    key_w, key_v = random.split(key)
    keys_w = random.split(key_w, n_samples)
    w = jax.vmap(lambda k, kap: _sample_w_reparam(k, kap, dim))(keys_w, kappa)

    return _combine_w_with_tangent(key_v, mu, w)


def _sample_w_single(key: random.PRNGKey, kappa: Array, dim: int) -> Array:
    """Sample a single w value using Wood's rejection sampler.

    Args:
        key: JAX random key
        kappa: Concentration parameter (scalar array)
        dim: Dimension of the sphere

    Returns:
        Single w value in [-1, 1]
    """
    dim_arr = jnp.asarray(dim, dtype=kappa.dtype)
    alpha = 0.5 * (dim_arr - 1.0)
    b = (dim_arr - 1.0) / (
        jnp.sqrt(4.0 * kappa**2 + (dim_arr - 1.0) ** 2) + 2.0 * kappa
    )
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (dim_arr - 1.0) * jnp.log(1.0 - x0 * x0)

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
            kappa * w_candidate + (dim_arr - 1.0) * jnp.log(1.0 - x0 * w_candidate) - c
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

    init_state = (
        key,
        jnp.array(1.0, dtype=kappa.dtype),
        jnp.array(False),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1.0, dtype=kappa.dtype),
    )

    _, w_final, accepted_final, _, last_candidate = jax.lax.while_loop(
        cond, body, init_state
    )

    return jnp.where(accepted_final, w_final, last_candidate)


def _compute_dw_dkappa(w: Array, kappa: Array, dim: int) -> Array:
    """Compute dw/dkappa via the implicit function theorem.

    For w drawn from the vMF marginal CDF F(w; kappa), the implicit
    reparameterization gradient is:

        dw/dkappa = -dF(w; kappa)/dkappa / f(w; kappa)

    which simplifies (after cancelling the normalization constant) to:

        dw/dkappa = -integral_{-1}^{w} exp(kappa*(t-w))
                     * ((1-t^2)/(1-w^2))^{(d-3)/2}
                     * (t - A_d(kappa)) dt

    The unsigned envelope of the integrand (without the (t - A_d) factor) peaks
    at t_peak = kappa*(1-t^2)/(d-3), approximately kappa/(d-3) for small t.
    The width around the peak scales as 1/sqrt(d-3). We choose the integration
    range to cover the envelope's support, capped at [-1, w].

    Args:
        w: Sample from the vMF marginal, scalar in [-1, 1]
        kappa: Concentration parameter, scalar
        dim: Ambient dimension

    Returns:
        Scalar dw/dkappa
    """
    d = jnp.float32(dim)
    half_exp = (d - 3.0) / 2.0
    a_d = mean_resultant_length(kappa, dim)
    one_minus_w2 = jnp.maximum(1.0 - w * w, 1e-30)

    # The unsigned envelope exp(kappa*t + half_exp*log(1-t^2)) peaks where
    # kappa = 2*half_exp*t / (1-t^2). Solving the quadratic
    # kappa*t^2 + 2*half_exp*t - kappa = 0 gives:
    # t_peak = (-half_exp + sqrt(half_exp^2 + kappa^2)) / kappa
    he_safe = jnp.maximum(half_exp, 0.5)
    kappa_safe = jnp.maximum(kappa, 1e-10)
    t_peak = (-he_safe + jnp.sqrt(he_safe * he_safe + kappa_safe * kappa_safe)) / (
        kappa_safe
    )

    # Width from second derivative of log-envelope at peak:
    # h''(t) = -2*half_exp*(1+t^2)/(1-t^2)^2 ≈ -2*half_exp for small t
    # => sigma ≈ 1/sqrt(2*half_exp)
    sigma = 1.0 / jnp.sqrt(jnp.maximum(2.0 * he_safe, 1.0))

    # Cover the full support: from well below t_peak to w
    lower = jnp.maximum(-1.0, t_peak - 8.0 * sigma)
    # Ensure the range isn't degenerate when w is near or below t_peak
    lower = jnp.minimum(lower, w - sigma)
    lower = jnp.maximum(lower, -1.0)

    # Map GL nodes from [-1, 1] to [lower, w]
    half_range = (w - lower) / 2.0
    midpoint = (w + lower) / 2.0
    t = midpoint + half_range * _GL_NODES

    # Evaluate integrand: exp(kappa*(t-w)) * ratio^half_exp * (t - A_d)
    log_exp_part = kappa * (t - w)
    t_clamped = jnp.clip(t, -1.0 + 1e-7, 1.0 - 1e-7)
    log_ratio = half_exp * (
        jnp.log(jnp.maximum(1.0 - t_clamped * t_clamped, 1e-30)) - jnp.log(one_minus_w2)
    )

    log_abs_integrand = log_exp_part + log_ratio
    integrand = jnp.exp(log_abs_integrand) * (t - a_d)

    integral = half_range * jnp.dot(_GL_WEIGHTS, integrand)
    return -integral


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def _sample_w_reparam(key: random.PRNGKey, kappa: Array, dim: int) -> Array:
    """Sample w from the vMF marginal with a custom backward pass.

    Forward: exact samples via Wood's rejection sampler.
    Backward: exact dw/dkappa via implicit reparameterization.
    """
    return _sample_w_single(key, kappa, dim)


def _sample_w_reparam_fwd(key, kappa, dim):
    """Forward pass: sample w and save residuals for backward."""
    w = _sample_w_single(key, kappa, dim)
    return w, (w, kappa)


def _sample_w_reparam_bwd(dim, res, g):
    """Backward pass: compute gradient dw/dkappa via implicit reparameterization."""
    w, kappa = res
    dw_dk = _compute_dw_dkappa(w, kappa, dim)
    return (None, g * dw_dk)


_sample_w_reparam.defvjp(_sample_w_reparam_fwd, _sample_w_reparam_bwd)


def _hankel_log_bessel(nu: Array, x: Array) -> Array:
    """Compute log I_ν(x) via the Hankel large-x expansion.

    I_ν(x) ~ e^x/√(2πx) · H(ν, x) where H = 1 + Σ correction terms in 1/x.
    Accurate for large x with any ν ≥ 0. Uses 4 correction terms giving
    ~(1/x)^5 relative accuracy.
    """
    mu = 4.0 * nu * nu
    inv_8x = 1.0 / (8.0 * jnp.maximum(x, 1.0))
    t = jnp.ones_like(x)
    h = jnp.ones_like(x)
    for k in range(1, 5):
        t = t * ((2 * k - 1) ** 2 - mu) * inv_8x / k
        h = h + t
    return (
        x
        - 0.5 * jnp.log(2.0 * jnp.pi * jnp.maximum(x, 1.0))
        + jnp.log(jnp.maximum(h, 1e-30))
    )


def log_bessel_iv(nu: Array, x: Array) -> Array:
    """Compute log I_ν(x) in a JAX-differentiable way.

    Uses three regimes selected via jnp.where (all computed for tracing):
    - Series: convergent power series, used when x is not too large relative to ν.
    - Olver: uniform asymptotic expansion (NIST 10.41.3), accurate for ν ≥ 1.
    - Hankel: large-x expansion, used for small ν (e.g. d=2 where ν=0) beyond
      the series range.

    Args:
        nu: Order of the Bessel function (scalar or array, >= 0).
        x: Argument (scalar or array, must be > 0 for meaningful results).

    Returns:
        log I_ν(x), same shape as broadcast(nu, x).
    """
    nu = jnp.asarray(nu, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)

    # --- Series expansion ---
    # I_ν(x) = (x/2)^ν / Γ(ν+1) · Σ_{k=0}^∞ (x²/4)^k / (k! · (ν+1)···(ν+k))
    # The kth term ratio is x²/(4·k·(k+ν)), so with N terms the series converges
    # when the largest ratio (at k=1) x²/(4·(1+ν)) is moderate.
    log_half_x = jnp.log(jnp.maximum(x, 1e-30) / 2.0)
    leading = nu * log_half_x - jax.lax.lgamma(nu + 1.0)

    x_sq_over_4 = x * x / 4.0
    n_terms = 60
    log_x_sq_over_4 = jnp.log(x_sq_over_4)

    def _series_step(carry, k):
        log_ratio_cumsum, log_sum_correction = carry
        log_ratio_cumsum = log_ratio_cumsum + log_x_sq_over_4 - jnp.log(k * (k + nu))
        log_sum_correction = jnp.logaddexp(log_sum_correction, log_ratio_cumsum)
        return (log_ratio_cumsum, log_sum_correction), None

    (_, log_sum_correction), _ = jax.lax.scan(
        _series_step,
        (jnp.zeros_like(x), jnp.full_like(x, -jnp.inf)),
        jnp.arange(1, n_terms + 1, dtype=jnp.float32),
    )

    # log(1 + exp(log_sum_correction)) via logaddexp to avoid overflow
    small_x_result = leading + jnp.logaddexp(0.0, log_sum_correction)

    # --- Olver's uniform asymptotic expansion (NIST 10.41.3) ---
    # I_ν(νz) ~ e^{ν·η(z)} / √(2πν) · (1+z²)^{-1/4} · Σ u_k(t)/ν^k
    # z = x/ν, η(z) = √(1+z²) + ln(z/(1+√(1+z²))), t = 1/√(1+z²)
    # Requires ν ≥ 1 for accuracy; clamping is safe since this branch is only
    # selected when nu >= 0.5 (see jnp.where below).
    nu_safe = jnp.maximum(nu, 0.5)
    z = x / nu_safe
    z_safe = jnp.maximum(z, 1e-10)

    sqrt_1_z2 = jnp.sqrt(1.0 + z_safe * z_safe)
    eta = sqrt_1_z2 + jnp.log(z_safe / (1.0 + sqrt_1_z2))

    t = 1.0 / sqrt_1_z2

    # Olver polynomials u_k(t)
    u0 = 1.0
    u1 = (3.0 * t - 5.0 * t**3) / 24.0
    u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
    u3 = (
        30375.0 * t**3 - 369603.0 * t**5 + 765765.0 * t**7 - 425425.0 * t**9
    ) / 414720.0

    inv_nu = 1.0 / nu_safe
    correction = u0 + u1 * inv_nu + u2 * inv_nu**2 + u3 * inv_nu**3

    olver_result = (
        nu_safe * eta
        - 0.5 * jnp.log(2.0 * jnp.pi * nu_safe)
        - 0.25 * jnp.log(1.0 + z_safe * z_safe)
        + jnp.log(jnp.maximum(correction, 1e-30))
    )

    # --- Hankel expansion for small ν ---
    hankel_result = _hankel_log_bessel(nu, x)

    # Use series when x² < 2·N·(N+ν), i.e. the Nth term ratio x²/(4N(N+ν)) < 0.5
    # and the remaining tail converges geometrically.
    series_threshold = jnp.sqrt(2.0 * n_terms * (n_terms + nu))
    use_series = x < series_threshold
    # Olver needs ν ≥ 1; use Hankel for small ν (d=2 where ν=0).
    use_hankel = jnp.logical_and(~use_series, nu < 0.5)
    return jnp.where(
        use_series, small_x_result, jnp.where(use_hankel, hankel_result, olver_result)
    )


def log_normalization_constant(kappa: Array, dim: int) -> Array:
    """Compute log normalization constant for vMF distribution, JAX-differentiable.

    C_d(κ) = κ^{d/2-1} / ((2π)^{d/2} · I_{d/2-1}(κ))

    At κ=0, the vMF is uniform and C_d(0) = 1/|S^{d-1}|. For small κ, the general
    formula involves cancellation of O(d·log d) intermediates, so we return the
    uniform-density formula directly.

    Args:
        kappa: Concentration parameter (JAX array, >= 0).
        dim: Ambient space dimension (int).

    Returns:
        log C_d(κ), same shape as kappa.
    """
    nu = dim / 2.0 - 1.0
    kappa_safe = jnp.maximum(kappa, 1e-20)
    log_bessel = log_bessel_iv(nu, kappa_safe)
    log_c = nu * jnp.log(kappa_safe) - (dim / 2.0) * jnp.log(2.0 * jnp.pi) - log_bessel
    # At small κ the formula subtracts O(d·log d) intermediates, losing precision.
    # The uniform-density formula avoids this entirely.
    log_c_uniform = sphere_log_inverse_surface_area(dim)
    return jnp.where(kappa < 1e-10, log_c_uniform, log_c)


def _log_bessel_ratio(nu_upper: Array, nu_lower: Array, x: Array) -> Array:
    """Compute log(I_{nu_upper}(x) / I_{nu_lower}(x)) stably in float32.

    Requires nu_upper = nu_lower + 1. Unlike subtracting two separate log-Bessel
    values (which suffers catastrophic cancellation at large x where both are ~x),
    this computes the log-ratio directly so all intermediate quantities are O(1).

    Uses three regimes (all computed for JAX tracing, selected via jnp.where):
    - Series: ratio of power series, with the O(x) leading terms cancelled analytically.
    - Olver: stable log-ratio of Olver's uniform asymptotic expansion (needs ν ≥ 1).
    - Hankel: ratio of Hankel large-x expansions (works for any ν ≥ 0, needs large x).
    """
    nu_upper = jnp.asarray(nu_upper, dtype=jnp.float32)
    nu_lower = jnp.asarray(nu_lower, dtype=jnp.float32)
    x = jnp.asarray(x, dtype=jnp.float32)

    # --- Series regime ---
    # I_{ν₁}(x)/I_{ν₂}(x) = (x/(2ν₁)) · S(ν₁,x)/S(ν₂,x)
    # where S(ν,x) = 1 + Σ_{k=1}^N (x²/4)^k / (k!·(ν+1)···(ν+k))
    # The O(x) leading terms (x/2)^ν / Γ(ν+1) cancel in the ratio, leaving O(1).
    n_terms = 60
    log_x_sq_4 = jnp.log(jnp.maximum(x * x / 4.0, 1e-38))

    def _series_step(carry, k):
        lr_u, lsc_u, lr_l, lsc_l = carry
        lr_u = lr_u + log_x_sq_4 - jnp.log(k * (k + nu_upper))
        lsc_u = jnp.logaddexp(lsc_u, lr_u)
        lr_l = lr_l + log_x_sq_4 - jnp.log(k * (k + nu_lower))
        lsc_l = jnp.logaddexp(lsc_l, lr_l)
        return (lr_u, lsc_u, lr_l, lsc_l), None

    init = (
        jnp.zeros_like(x),
        jnp.full_like(x, -jnp.inf),
        jnp.zeros_like(x),
        jnp.full_like(x, -jnp.inf),
    )
    (_, lsc_u, _, lsc_l), _ = jax.lax.scan(
        _series_step, init, jnp.arange(1, n_terms + 1, dtype=jnp.float32)
    )

    log_S_ratio = jnp.logaddexp(0.0, lsc_u) - jnp.logaddexp(0.0, lsc_l)
    series_result = jnp.log(jnp.maximum(x, 1e-38) / (2.0 * nu_upper)) + log_S_ratio

    # --- Olver regime: stable log-ratio ---
    # Instead of log(I_{ν₁}) - log(I_{ν₂}) (which subtracts two ~x values),
    # compute each piece of the Olver expansion's difference analytically.
    # Requires ν ≥ 1 for accuracy; clamping is safe since this branch is only
    # selected when nu_lower >= 0.5 (see jnp.where below).
    nu1 = jnp.maximum(nu_upper, 1.0)
    nu2 = jnp.maximum(nu_lower, 0.5)

    s1 = jnp.sqrt(nu1**2 + x**2)
    s2 = jnp.sqrt(nu2**2 + x**2)

    # Δf: difference of leading Olver terms ν·η(x/ν)
    # = (ν₁+ν₂)/(s₁+s₂) + ln(x) - ν₂·log1p((1+α)/(ν₂+s₂)) - ln(ν₁+s₁)
    # where α = (ν₁+ν₂)/(s₁+s₂) = s₁-s₂ (cancellation-free sqrt difference)
    alpha = (nu1 + nu2) / (s1 + s2)
    delta_f = (
        alpha
        + jnp.log(jnp.maximum(x, 1e-38))
        - nu2 * jnp.log1p((1.0 + alpha) / (nu2 + s2))
        - jnp.log(nu1 + s1)
    )

    # Δg: -0.5·ln(2πν₁) + 0.5·ln(2πν₂) = -0.5·log1p(1/ν₂)
    delta_g = -0.5 * jnp.log1p(1.0 / nu2)

    # Δh: from the (1+z²)^{-1/4} prefactors
    # = -0.25·log1p((ν₁+ν₂)/(ν₂²+x²)) + 0.5·log1p(1/ν₂)
    delta_h = -0.25 * jnp.log1p((nu1 + nu2) / (nu2**2 + x**2)) + 0.5 * jnp.log1p(
        1.0 / nu2
    )

    # Δj: log(correction₁/correction₂), both polynomials near 1 for large ν
    t1 = nu1 / s1
    t2 = nu2 / s2

    def _olver_corr(t, nu):
        inv = 1.0 / nu
        u1 = (3.0 * t - 5.0 * t**3) / 24.0
        u2 = (81.0 * t**2 - 462.0 * t**4 + 385.0 * t**6) / 1152.0
        u3 = (
            30375.0 * t**3 - 369603.0 * t**5 + 765765.0 * t**7 - 425425.0 * t**9
        ) / 414720.0
        return 1.0 + u1 * inv + u2 * inv**2 + u3 * inv**3

    delta_j = jnp.log(_olver_corr(t1, nu1) / _olver_corr(t2, nu2))

    olver_result = delta_f + delta_g + delta_h + delta_j

    # --- Hankel regime: large-x expansion for any ν ---
    # I_ν(x) ~ e^x/√(2πx) · H(ν,x) where H = 1 + Σ correction terms in 1/x.
    # The e^x/√(2πx) cancels in the ratio, leaving log(H(ν₁,x)/H(ν₂,x)).
    # Since H ≈ 1 + O(1/x), this subtraction is numerically stable.
    def _hankel_sum(nu, x_h):
        """Hankel asymptotic sum H(ν,x) for I_ν(x), 4 correction terms."""
        mu = 4.0 * nu * nu
        inv_8x = 1.0 / (8.0 * jnp.maximum(x_h, 1.0))
        t = jnp.ones_like(x_h)
        h = jnp.ones_like(x_h)
        for k in range(1, 5):
            t = t * ((2 * k - 1) ** 2 - mu) * inv_8x / k
            h = h + t
        return h

    hankel_result = jnp.log(jnp.maximum(_hankel_sum(nu_upper, x), 1e-30)) - jnp.log(
        jnp.maximum(_hankel_sum(nu_lower, x), 1e-30)
    )

    # Switch to asymptotic expansion when the Nth series term ratio
    # x²/(4N(N+ν)) < 0.25. The Olver/Hankel branches are already more accurate
    # than the series at this point (series accumulates float32 rounding over
    # the N-step scan), so switching earlier avoids a precision dip at the boundary.
    threshold = jnp.sqrt(n_terms * (n_terms + nu_lower))
    use_series = x < threshold
    # Olver needs ν ≥ 1; use Hankel for small ν (d=2 where nu_lower=0).
    use_hankel = jnp.logical_and(~use_series, nu_lower < 0.5)
    return jnp.where(
        use_series, series_result, jnp.where(use_hankel, hankel_result, olver_result)
    )


def mean_resultant_length(kappa: Array, dim: int) -> Array:
    """Compute the mean resultant length A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ).

    Uses a stable log-ratio computation that avoids catastrophic cancellation
    from subtracting two large, nearly equal log-Bessel values.

    Args:
        kappa: Concentration parameter (JAX array).
        dim: Ambient space dimension (int).

    Returns:
        A_d(κ), same shape as kappa.
    """
    return jnp.exp(_log_bessel_ratio(dim / 2.0, dim / 2.0 - 1.0, kappa))


def _neg_log_bessel_series_sum(kappa: Array, dim: int) -> Array:
    """Compute -log(S_ν(κ)) where S_ν = I_ν(κ) · Γ(ν+1) · (2/κ)^ν.

    S_ν(κ) = 1 + Σ_{k=1}^∞ (κ²/4)^k / (k! · (ν+1)···(ν+k))

    The identity log C_d(κ) + log|S^{d-1}| = -log S_ν(κ) holds exactly. This
    avoids the catastrophic cancellation between log C_d and log|S| (both
    ~O(d·log d) in magnitude) that destroys precision in float32 for small κ.
    For small κ, S_ν ≈ 1 + κ²/(2d), so -log S_ν ≈ -κ²/(2d).

    Computed in log space to handle the full range of κ without overflow.
    """
    nu = dim / 2.0 - 1.0
    # Compute log(κ²/4) as 2·log(κ) - log(4) instead of log(κ²/4) to avoid
    # κ² underflowing to denormal in float32 (happens below κ ≈ 1e-19).
    # Caller guarantees kappa ≥ 1e-20, so log(kappa) is finite.
    log_x_sq_4 = 2.0 * jnp.log(kappa) - jnp.log(4.0)
    n_terms = 60

    def step(carry, k):
        log_ratio_cumsum, log_sum = carry
        log_ratio_cumsum = log_ratio_cumsum + log_x_sq_4 - jnp.log(k * (k + nu))
        log_sum = jnp.logaddexp(log_sum, log_ratio_cumsum)
        return (log_ratio_cumsum, log_sum), None

    (_, log_sum), _ = jax.lax.scan(
        step,
        (jnp.zeros_like(kappa), jnp.full_like(kappa, -jnp.inf)),
        jnp.arange(1, n_terms + 1, dtype=jnp.float32),
    )

    # -log(S_ν) = -log(1 + exp(log_sum)) = -logaddexp(0, log_sum)
    return -jnp.logaddexp(0.0, log_sum)


def _kl_hankel(kappa: Array, dim: int) -> Array:
    """Compute KL(vMF ‖ Uniform) via the Hankel asymptotic expansion.

    The standard formula log C_d(κ) + κ·A_d(κ) + log|S| subtracts two O(κ)
    intermediates (log C_d ≈ -κ, κ·A_d ≈ κ) to get an O(log κ) result, losing
    precision in float32. This rearrangement avoids the cancellation.

    Starting from the Hankel expansion I_ν(κ) ~ e^κ/√(2πκ) · H_ν(κ):
        log C_d + κ·A_d = ((d-1)/2)·log(κ/(2π))
                        + κ·(H_{ν+1}(κ)/H_ν(κ) - 1) - log(H_ν(κ))
    where H_ν is the Hankel correction sum (≈ 1 for large κ), so all terms
    are at most O(log κ). The O(κ) pieces cancel algebraically.

    The term κ·(H_{ν+1}/H_ν - 1) is computed as κ·(H_{ν+1} - H_ν)/H_ν with
    the numerator accumulated term-by-term to avoid subtracting numbers near 1.

    Requires κ >> ν for the Hankel expansion to be accurate.
    """
    nu = dim / 2.0 - 1.0
    nu_upper = dim / 2.0
    mu_lower = 4.0 * nu * nu
    mu_upper = 4.0 * nu_upper * nu_upper
    inv_8k = 1.0 / (8.0 * jnp.maximum(kappa, 1.0))

    # Accumulate H_ν and κ·(H_{ν+1} - H_ν) term by term.
    # For the kth Hankel term a_k^(ν) = Π_{j=1}^k ((2j-1)²-4ν²)/(j·8κ),
    # the κ in the k=1 factor cancels the outer κ exactly, so
    # κ·(a_1^upper - a_1^lower) = (4ν² - 4(ν+1)²)/8 = -(ν+0.5), independent
    # of κ — no floating-point cancellation at all.
    t_lower = jnp.ones_like(kappa)
    t_upper = jnp.ones_like(kappa)
    h_lower = jnp.ones_like(kappa)
    kappa_delta_sum = jnp.zeros_like(kappa)

    for k in range(1, 5):
        factor = ((2 * k - 1) ** 2 - mu_lower) * inv_8k / k
        factor_u = ((2 * k - 1) ** 2 - mu_upper) * inv_8k / k
        t_lower = t_lower * factor
        t_upper = t_upper * factor_u
        h_lower = h_lower + t_lower
        kappa_delta_sum = kappa_delta_sum + kappa * (t_upper - t_lower)

    log_surface_area = -sphere_log_inverse_surface_area(dim)
    kappa_safe = jnp.maximum(kappa, 1e-20)

    return (
        (nu + 0.5) * jnp.log(kappa_safe / (2.0 * jnp.pi))
        + kappa_delta_sum / h_lower
        - jnp.log(jnp.maximum(h_lower, 1e-30))
        + log_surface_area
    )


def kl_vmf_uniform(kappa: Array, dim: int) -> Array:
    """Compute KL(vMF(μ, κ) ‖ Uniform(S^{d-1})).

    KL = log C_d(κ) + κ·A_d(κ) + log|S^{d-1}|

    where |S^{d-1}| is the surface area. All terms are JAX-differentiable.

    Uses three computational paths to maintain precision across all κ and d:
    - Small κ: cancellation-free series KL = -log(S_ν(κ)) + κ·A_d(κ), avoiding
      the ~O(d·log d) cancellation between log C_d and log|S|.
    - Large κ, small ν: Hankel-based formula that algebraically cancels the O(κ)
      terms in log C_d and κ·A_d, avoiding float32 precision loss.
    - Large κ, large ν: direct formula, where KL ∝ ν·log(κ) is large enough
      that the O(κ·ε) float32 error is negligible.

    Args:
        kappa: Concentration parameter (JAX array).
        dim: Ambient space dimension (int).

    Returns:
        KL divergence, same shape as kappa.
    """
    nu = dim / 2.0 - 1.0
    # Floor at 1e-20 to avoid log(0). At κ=0, kappa_safe=1e-20 and the
    # small-κ path naturally returns ~0 (S_ν ≈ 1 so -log(S_ν) ≈ 0, and A_d ≈ 0).
    kappa_safe = jnp.maximum(kappa, 1e-20)

    a_d = mean_resultant_length(kappa_safe, dim)

    # Small-κ path: KL = -log(S_ν) + κ·A_d
    kl_small = _neg_log_bessel_series_sum(kappa_safe, dim) + kappa_safe * a_d

    # Hankel path: cancellation-free for large κ (requires κ >> ν)
    kl_hankel = _kl_hankel(kappa_safe, dim)

    # Direct path: log C_d + κ·A_d + log|S|
    log_c = log_normalization_constant(kappa_safe, dim)
    log_surface_area = -sphere_log_inverse_surface_area(dim)
    kl_direct = log_c + kappa_safe * a_d + log_surface_area

    # Series is accurate when the 60-term sum converges: κ² < 4·N·(N+ν).
    n_terms = 60.0
    series_limit = jnp.sqrt(jnp.float32(n_terms * (n_terms + nu)))
    use_series = kappa_safe < series_limit

    # Hankel is accurate when |a_1| = |4ν²-1|/(8κ) < 0.3, i.e., the leading
    # correction term is small enough for the 4-term expansion to converge.
    hankel_min_kappa = jnp.float32((4.0 * nu * nu + 1.0) / 2.4)
    use_hankel = jnp.logical_and(~use_series, kappa_safe > hankel_min_kappa)

    return jnp.where(use_series, kl_small, jnp.where(use_hankel, kl_hankel, kl_direct))


def log_prob(x: Array, mu: Array, kappa: float) -> Array:
    """Compute log probability density of vMF distribution.

    Args:
        x: Points on unit sphere, shape (..., d)
        mu: Mean direction on unit sphere, shape (d,)
        kappa: Concentration parameter (>= 0)

    Returns:
        Log probability densities, shape (...)
    """
    x_norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    mu_norm = jnp.linalg.norm(mu)
    x = x / jnp.maximum(x_norm, 1e-8)
    mu = mu / jnp.maximum(mu_norm, 1e-8)

    dim = mu.shape[-1]
    dot_product = jnp.sum(x * mu, axis=-1)
    log_c = log_normalization_constant(kappa, dim)
    return log_c + kappa * dot_product


# =============================================================================
# Tests for JAX-differentiable vMF utilities
# =============================================================================


@pytest.mark.parametrize(
    "nu,x",
    [
        (0.0, 0.5),
        (0.0, 5.0),
        (0.0, 50.0),
        (0.0, 100.0),  # d=2, Hankel branch
        (0.0, 500.0),  # d=2, Hankel branch, large κ
        (1.0, 0.1),
        (1.0, 10.0),
        (1.0, 100.0),
        (5.0, 0.5),
        (5.0, 10.0),
        (5.0, 50.0),
        (10.0, 1.0),
        (10.0, 20.0),
        (10.0, 100.0),
        (383.0, 100.0),
        (383.0, 300.0),  # near old series/Olver boundary
        (383.0, 500.0),
        (383.0, 1000.0),
        (383.0, 9000.0),  # large κ, tests float precision
    ],
)
def test_log_bessel_iv_matches_scipy(nu, x):
    """Test that log_bessel_iv matches scipy's ive across a range of ν and x values."""
    jax_result = float(log_bessel_iv(nu, x))
    scipy_ive = float(sps.ive(nu, x))
    scipy_result = np.log(scipy_ive) + x
    np.testing.assert_allclose(jax_result, scipy_result, rtol=2e-4)


@pytest.mark.parametrize("dim", [2, 3, 4, 16, 100, 768])
def test_log_normalization_constant_at_kappa_zero(dim):
    """At κ=0 the vMF is uniform, so log C_d(0) = log(1/|S^{d-1}|)."""
    result = float(log_normalization_constant(jnp.array(0.0), dim))
    # Float64 reference: -log(2) - (d/2)*log(π) + lgamma(d/2)
    expected = float(
        -np.log(2.0) - (dim / 2.0) * np.log(np.pi) + sps.gammaln(dim / 2.0)
    )
    np.testing.assert_allclose(result, expected, atol=1e-3, err_msg=f"dim={dim}")


def test_kl_vmf_uniform_zero_at_kappa_zero():
    """Test that KL(vMF(κ=0) ‖ Uniform) = 0, including exact κ=0."""
    for dim in [3, 16, 768]:
        for kappa_val in [0.0, 1e-10, 1e-20]:
            kl = float(kl_vmf_uniform(jnp.array(kappa_val), dim))
            np.testing.assert_allclose(
                kl, 0.0, atol=1e-3, err_msg=f"dim={dim}, κ={kappa_val}"
            )


def test_kl_vmf_uniform_monotonically_increasing():
    """Test that KL divergence increases with κ."""
    kappas = jnp.array([0.1, 1.0, 5.0, 10.0, 50.0, 100.0])
    for dim in [3, 16, 768]:
        kls = jax.vmap(lambda k: kl_vmf_uniform(k, dim))(kappas)
        diffs = jnp.diff(kls)
        assert jnp.all(diffs > 0), f"KL not monotonically increasing for dim={dim}"


def test_log_bessel_iv_gradient_flows():
    """Test that gradients flow through log_bessel_iv."""
    grad_fn = jax.grad(lambda x: log_bessel_iv(5.0, x))
    g = grad_fn(jnp.array(10.0))
    assert jnp.isfinite(g)
    assert g != 0.0


def test_mean_resultant_length_bounded_and_accurate():
    """Test that A_d(κ) ∈ (0,1) and matches the asymptotic A_d ≈ 1 - (d-1)/(2κ) when κ >> d."""
    for dim in [3, 16, 768]:
        for kappa_val in [100.0, 1000.0, 9000.0]:
            a_d = float(mean_resultant_length(jnp.array(kappa_val), dim))
            assert 0 < a_d < 1, f"A_d={a_d} out of (0,1) for dim={dim}, κ={kappa_val}"
            # First-order asymptotic is only accurate when κ >> d
            if kappa_val > 10 * dim:
                asymptotic = 1.0 - (dim - 1) / (2 * kappa_val)
                np.testing.assert_allclose(
                    a_d,
                    asymptotic,
                    rtol=1e-3,
                    err_msg=f"dim={dim}, κ={kappa_val}",
                )


def test_kl_vmf_uniform_gradient_flows():
    """Test that gradients flow through kl_vmf_uniform."""
    grad_fn = jax.grad(lambda k: kl_vmf_uniform(k, 768))
    g = grad_fn(jnp.array(10.0))
    assert jnp.isfinite(g)
    assert g > 0.0  # KL is increasing in κ


@pytest.mark.parametrize(
    "kappa,dim",
    [
        (1e-3, 768),
        (0.01, 768),
        (0.1, 768),
        (0.13533528, 768),
        (0.5, 768),
        (1.0, 768),
        (5.0, 768),
        (20.0, 768),
        (1e-3, 3),
        (0.1, 3),
        (1.0, 3),
        (1e-3, 16),
        (0.1, 16),
        (1.0, 16),
    ],
)
def test_kl_small_kappa_high_dim(kappa, dim):
    """Test KL accuracy for small κ where float32 cancellation is severe.

    The leading-order series KL ≈ κ²/(2d) is accurate to ~κ²/d² relative error,
    so we use it as reference for small enough κ.
    """
    kl = float(kl_vmf_uniform(jnp.array(kappa), dim))
    # Two-term series: KL ≈ κ²/(2d) + κ⁴/(8d(d+2))
    kl_series = kappa**2 / (2 * dim) + kappa**4 / (8 * dim * (dim + 2))
    # The series is accurate to ~(κ/sqrt(d))^6 relative error; use generous tolerance
    rel_tol = max(3 * (kappa / np.sqrt(dim)) ** 2, 1e-4)
    np.testing.assert_allclose(
        kl,
        kl_series,
        rtol=rel_tol,
        err_msg=f"dim={dim}, κ={kappa}",
    )


def test_kl_gradient_nonzero_small_kappa():
    """Test that gradients are non-zero at small κ in high dimensions.

    This directly tests the fix for the bug where kl_vmf_uniform(0.135, 768)
    returned 0.0 with zero gradient, blocking the learned schedule's prior loss.
    """
    for kappa_val in [0.01, 0.1, 0.13533528, 0.5, 1.0]:
        grad_fn = jax.grad(lambda k: kl_vmf_uniform(k, 768))
        g = float(grad_fn(jnp.array(kappa_val)))
        assert jnp.isfinite(g), f"gradient not finite at κ={kappa_val}"
        # dKL/dκ ≈ κ/d for the leading term
        expected_grad = kappa_val / 768.0
        np.testing.assert_allclose(g, expected_grad, rtol=0.1, err_msg=f"κ={kappa_val}")


@pytest.mark.parametrize("kappa", [85.0, 100.0, 200.0, 500.0, 1000.0])
def test_mean_resultant_length_d2(kappa):
    """Test A_2(κ) = I_1(κ)/I_0(κ) for large κ where d=2 uses the Hankel branch."""
    a_d = float(mean_resultant_length(jnp.array(kappa), 2))
    ref = float(sps.ive(1, kappa) / sps.ive(0, kappa))
    np.testing.assert_allclose(a_d, ref, rtol=1e-4, err_msg=f"κ={kappa}")


def test_kl_crossover_continuity():
    """Test that KL is smooth across regime boundaries.

    Evaluates at the series/asymptotic crossover (κ = series_limit) and checks
    that the two code paths agree.
    """
    for dim in [3, 16, 768]:
        nu = dim / 2.0 - 1.0
        limit = np.sqrt(60 * (60 + nu))
        eps = 0.5
        kl_below = float(kl_vmf_uniform(jnp.array(limit - eps), dim))
        kl_above = float(kl_vmf_uniform(jnp.array(limit + eps), dim))
        # A small step across the boundary should produce a small change in KL
        np.testing.assert_allclose(
            kl_below, kl_above, rtol=0.02, err_msg=f"dim={dim}, κ≈{limit:.1f}"
        )


@pytest.mark.parametrize("kappa", [85.0, 100.0, 200.0, 500.0])
def test_log_normalization_constant_d2_large_kappa(kappa):
    """Test log C_2(κ) for large κ where d=2 uses the Hankel branch."""
    result = float(log_normalization_constant(jnp.array(kappa), 2))
    # Reference: log C_2(κ) = -log(2π) - log(I_0(κ))
    log_bessel_ref = float(np.log(sps.ive(0, kappa)) + kappa)
    expected = -np.log(2 * np.pi) - log_bessel_ref
    np.testing.assert_allclose(result, expected, atol=1e-3, err_msg=f"κ={kappa}")


@pytest.mark.parametrize(
    "dim,kappa",
    [
        # Near the series/Olver boundary for each dimension
        (3, 55.0),
        (3, 60.0),
        (3, 65.0),
        (3, 85.0),
        (768, 160.0),
        (768, 165.0),
        (768, 230.0),
    ],
)
def test_mean_resultant_length_near_boundary(dim, kappa):
    """Test A_d(κ) accuracy near the series/Olver transition threshold.

    The series branch accumulates float32 rounding over the scan, so switching
    to the Olver branch earlier avoids a precision dip near the old boundary.
    """
    a_d = float(mean_resultant_length(jnp.array(kappa), dim))
    nu_upper = dim / 2.0
    nu_lower = dim / 2.0 - 1.0
    ref = float(sps.ive(nu_upper, kappa) / sps.ive(nu_lower, kappa))
    np.testing.assert_allclose(a_d, ref, rtol=2e-5, err_msg=f"dim={dim}, κ={kappa}")


@pytest.mark.parametrize(
    "dim,kappas",
    [
        (2, [100.0, 1000.0, 10000.0, 100000.0]),
        (3, [100.0, 1000.0, 10000.0, 71000.0, 100000.0]),
        (4, [100.0, 1000.0, 10000.0, 100000.0]),
    ],
)
def test_kl_large_kappa_low_dim(dim, kappas):
    """Test KL accuracy and monotonicity at large κ for low dimensions.

    The Hankel-based formula avoids the O(κ) cancellation that causes the
    direct formula to lose precision in float32 when d is small and κ is large.
    """
    nu = dim / 2.0 - 1.0
    kl_prev = -1.0
    for k in kappas:
        kl = float(kl_vmf_uniform(jnp.array(k), dim))
        # Float64 reference
        log_iv = float(np.log(sps.ive(nu, k)) + k)
        log_iv1 = float(np.log(sps.ive(nu + 1, k)) + k)
        a_d_ref = np.exp(log_iv1 - log_iv)
        log_c_ref = nu * np.log(k) - (dim / 2.0) * np.log(2 * np.pi) - log_iv
        log_surf = float(dim / 2 * np.log(np.pi) + np.log(2) - sps.gammaln(dim / 2))
        kl_ref = log_c_ref + k * a_d_ref + log_surf
        np.testing.assert_allclose(kl, kl_ref, rtol=1e-4, err_msg=f"dim={dim}, κ={k}")
        assert kl > kl_prev, f"KL not monotone: dim={dim}, κ={k}, kl={kl} ≤ {kl_prev}"
        kl_prev = kl


def test_kl_no_dead_zone_near_zero():
    """Test that kl_vmf_uniform has non-zero value and gradient for tiny κ > 0.

    Verifies removal of the hard-zero guard that previously killed gradients
    for κ < 1e-10.
    """
    for kappa_val in [1e-11, 1e-9, 1e-7]:
        kl = float(kl_vmf_uniform(jnp.array(kappa_val), 768))
        # KL ≈ κ²/(2d) — in float32 this is tiny but should not be exactly 0
        # for κ ≥ ~1e-9 (κ²/(2d) ≈ 6.5e-22 which is above float32 min subnormal)
        if kappa_val >= 1e-9:
            assert kl > 0, f"KL should be positive at κ={kappa_val}, got {kl}"
        grad = float(jax.grad(lambda k: kl_vmf_uniform(k, 768))(jnp.array(kappa_val)))
        if kappa_val >= 1e-9:
            assert grad > 0, f"gradient should be positive at κ={kappa_val}, got {grad}"
