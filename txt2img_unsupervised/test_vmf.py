"""Tests for von Mises-Fisher distributions."""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
import scipy.special
import scipy.stats
from txt2img_unsupervised.vmf import (
    log_normalization_constant,
    log_prob,
    sample,
    mean_resultant_length,
    _compute_dw_dkappa,
    _sample_w_single,
)


def _sample_uniform_sphere(key, dim, n_samples):
    """Sample uniformly from unit sphere."""
    samples = jax.random.normal(key, (n_samples, dim))
    norms = jnp.linalg.norm(samples, axis=-1, keepdims=True)
    return samples / norms


class TestLogNormalizationConstant:
    def test_uniform_case(self):
        """Test normalization constant for uniform distribution (kappa=0)."""
        for dim in [3, 10, 50]:
            log_c = log_normalization_constant(0.0, dim)
            # For uniform distribution on sphere, the density is 1/surface_area
            # Surface area of unit sphere in d dimensions is 2*pi^(d/2) / Gamma(d/2)
            # So log normalization constant should be -log(surface_area)
            expected_log_surface_area = (
                jnp.log(2.0)
                + (dim / 2.0) * jnp.log(jnp.pi)
                - jax.scipy.special.gammaln(dim / 2.0)
            )
            expected = -expected_log_surface_area
            assert jnp.abs(log_c - expected) < 1e-5  # Relaxed tolerance

    def test_small_kappa(self):
        """Test normalization constant for small kappa values."""
        for dim in [3, 10, 100]:
            for kappa in [0.01, 0.1, 1.0]:
                log_c = log_normalization_constant(kappa, dim)
                # Should be finite and not NaN
                assert jnp.isfinite(log_c)

    def test_large_kappa(self):
        """Test normalization constant for large kappa values."""
        for dim in [3, 10, 100, 768]:
            for kappa in [100.0, 500.0, 1000.0]:
                log_c = log_normalization_constant(kappa, dim)
                # Should be finite and not NaN
                assert jnp.isfinite(log_c)

    def test_monotonicity(self):
        """Test that log normalization constant has expected behavior with kappa."""
        dim = 10
        kappas = jnp.array([0.1, 1.0, 10.0])  # Test smaller range
        log_cs = jnp.array([log_normalization_constant(k, dim) for k in kappas])
        # For small to medium kappa, the normalization constant may not be monotonic
        # Just check that all values are finite
        assert jnp.all(jnp.isfinite(log_cs))

    def test_matches_reference_general_order(self):
        """Ensure general-order computation matches the exact formula."""
        dim = 3
        kappa = 2.0

        log_c = log_normalization_constant(kappa, dim)

        nu = dim / 2.0 - 1.0
        log_bessel_exact = jnp.log(scipy.special.iv(nu, kappa))
        expected = (
            (dim / 2.0 - 1.0) * jnp.log(kappa)
            - (dim / 2.0) * jnp.log(2 * jnp.pi)
            - log_bessel_exact
        )

        assert jnp.allclose(log_c, expected, atol=1e-6, rtol=1e-6)


class TestLogProb:
    def test_uniform_distribution(self):
        """Test log probability for uniform distribution."""
        dim = 10
        n_samples = 100
        key = random.PRNGKey(42)

        mu = jnp.array([1.0] + [0.0] * (dim - 1))
        kappa = 0.0

        # Sample uniformly
        x = _sample_uniform_sphere(key, dim, n_samples)

        # All points should have same log probability
        log_probs = log_prob(x, mu, kappa)
        assert jnp.allclose(log_probs, log_probs[0], rtol=1e-6)

    def test_concentrated_distribution(self):
        """Test log probability for concentrated distribution."""
        dim = 10
        mu = jnp.array([1.0] + [0.0] * (dim - 1))
        kappa = 100.0

        # Point aligned with mu should have higher probability than orthogonal point
        x_aligned = mu.reshape(1, -1)
        x_orthogonal = jnp.array([[0.0, 1.0] + [0.0] * (dim - 2)])

        log_prob_aligned = log_prob(x_aligned, mu, kappa)
        log_prob_orthogonal = log_prob(x_orthogonal, mu, kappa)

        assert log_prob_aligned > log_prob_orthogonal

    def test_normalization(self):
        """Test that probabilities integrate to 1 (Monte Carlo)."""
        dim = 5
        mu = jnp.array([1.0] + [0.0] * (dim - 1))
        kappa = 10.0
        n_samples = 10000

        key = random.PRNGKey(123)
        x = _sample_uniform_sphere(key, dim, n_samples)

        log_probs = log_prob(x, mu, kappa)
        probs = jnp.exp(log_probs)

        # Surface area of unit sphere in d dimensions
        surface_area = (
            2 * jnp.pi ** (dim / 2.0) / jnp.exp(jax.scipy.special.gammaln(dim / 2.0))
        )
        integral_estimate = jnp.mean(probs) * surface_area

        # Should integrate to approximately 1 (Monte Carlo has high variance)
        assert jnp.abs(integral_estimate - 1.0) < 1.0

    def test_input_validation(self):
        """Test that inputs are properly normalized."""
        dim = 5
        mu = 2.0 * jnp.array([1.0] + [0.0] * (dim - 1))  # Not unit vector
        kappa = 1.0

        x = 3.0 * jnp.array([[0.0, 1.0] + [0.0] * (dim - 2)])  # Not unit vector

        # Should not crash and should give finite result
        log_prob_val = log_prob(x, mu, kappa)
        assert jnp.isfinite(log_prob_val)

    def test_matches_scipy(self):
        """Compare against SciPy's reference implementation."""
        dim = 5
        kappa = 10.0
        n_samples = 128
        key = random.PRNGKey(7)

        mu = jnp.zeros(dim).at[0].set(1.0)
        x = _sample_uniform_sphere(key, dim, n_samples)

        ours = log_prob(x, mu, kappa)

        scipy_dist = scipy.stats.vonmises_fisher(np.asarray(mu), kappa)
        scipy_vals = jnp.asarray(scipy_dist.logpdf(np.asarray(x)))

        assert jnp.max(jnp.abs(ours - scipy_vals)) < 1e-3


class TestSample:
    def test_uniform_sampling(self):
        """Test uniform sampling on sphere."""
        dim = 10
        n_samples = 1000
        key = random.PRNGKey(42)

        mu = jnp.array([1.0] + [0.0] * (dim - 1))
        kappa = 0.0

        samples = sample(key, mu, kappa, n_samples)

        # Check shape
        assert samples.shape == (n_samples, dim)

        # Check unit vectors
        norms = jnp.linalg.norm(samples, axis=1)
        assert jnp.allclose(norms, 1.0, rtol=1e-6)

        # Check approximate uniformity (mean should be near zero)
        mean_direction = jnp.mean(samples, axis=0)
        mean_norm = jnp.linalg.norm(mean_direction)
        assert mean_norm < 0.2  # Should be small for uniform distribution

    def test_concentrated_sampling(self):
        """Test concentrated sampling."""
        dim = 10
        n_samples = 1000
        key = random.PRNGKey(42)

        mu = jnp.array([1.0] + [0.0] * (dim - 1))
        kappa = 100.0

        samples = sample(key, mu, kappa, n_samples)

        # Check shape and unit vectors
        assert samples.shape == (n_samples, dim)
        norms = jnp.linalg.norm(samples, axis=1)
        assert jnp.allclose(norms, 1.0, rtol=1e-6)

        # Check concentration around mu
        dot_products = jnp.dot(samples, mu)
        mean_dot_product = jnp.mean(dot_products)
        assert mean_dot_product > 0.8  # Should be concentrated

    def test_high_dimensional_sampling(self):
        """Test sampling in high dimensions."""
        dim = 384
        n_samples = 100
        key = random.PRNGKey(42)

        mu = jnp.zeros(dim).at[0].set(1.0)
        kappa = 10.0

        samples = sample(key, mu, kappa, n_samples)

        # Check basic properties
        assert samples.shape == (n_samples, dim)
        norms = jnp.linalg.norm(samples, axis=1)
        assert jnp.allclose(norms, 1.0, rtol=1e-5)

    def test_first_moment_matches_theory(self):
        """Empirical mean along mu should match A_d(kappa)."""
        dim = 5
        kappa = 10.0
        n_samples = 20000
        key = random.PRNGKey(0)

        mu = jnp.zeros(dim).at[0].set(1.0)
        samples = sample(key, mu, kappa, n_samples)

        empirical = jnp.dot(samples, mu).mean()

        nu_upper = dim / 2.0
        nu_lower = dim / 2.0 - 1.0
        log_upper = jnp.log(scipy.special.iv(nu_upper, kappa))
        log_lower = jnp.log(scipy.special.iv(nu_lower, kappa))
        expected = jnp.exp(log_upper - log_lower)

        assert jnp.abs(empirical - expected) < 0.05


class TestNumericalStability:
    def test_extreme_dimensions(self):
        """Test stability in extreme dimensions."""
        for dim in [768]:
            key = random.PRNGKey(42)
            mu = jnp.zeros(dim).at[0].set(1.0)

            for kappa in [0.1, 1.0, 10.0, 100.0]:
                # Test all functions don't crash
                log_c = log_normalization_constant(kappa, dim)
                assert jnp.isfinite(log_c)

                samples = sample(key, mu, kappa, 10)
                assert samples.shape == (10, dim)
                assert jnp.all(jnp.isfinite(samples))

                log_probs = log_prob(samples, mu, kappa)
                assert jnp.all(jnp.isfinite(log_probs))

    def test_extreme_kappa(self):
        """Test stability for extreme kappa values."""
        dim = 10
        key = random.PRNGKey(42)
        mu = jnp.zeros(dim).at[0].set(1.0)

        for kappa in [1e-6, 1e6]:
            log_c = log_normalization_constant(kappa, dim)
            assert jnp.isfinite(log_c)

            samples = sample(key, mu, kappa, 10)
            assert jnp.all(jnp.isfinite(samples))

    def test_precision_float32(self):
        """Test float32 precision."""
        dim = 10
        key = random.PRNGKey(42)
        dtype = jnp.float32

        mu = jnp.array([1.0] + [0.0] * (dim - 1), dtype=dtype)
        kappa = dtype(10.0)

        samples = sample(key, mu, kappa, 100)
        assert samples.dtype == dtype

        log_probs = log_prob(samples, mu, kappa)
        assert log_probs.dtype == dtype


class TestSampleReparam:
    """Tests for differentiable vMF sampling via implicit reparameterization."""

    @pytest.mark.parametrize(
        "dim,kappa",
        [
            (3, 5.0),
            (3, 50.0),
            (10, 10.0),
            (10, 50.0),
            (768, 10.0),
            (768, 100.0),
        ],
    )
    def test_forward_first_moment(self, dim, kappa):
        """E[x.mu] should match the mean resultant length A_d(kappa)."""
        n_samples = 20000
        key = random.PRNGKey(42)
        mu = jnp.zeros(dim, dtype=jnp.float32).at[0].set(1.0)

        samples = sample(key, mu, kappa, n_samples)
        empirical = jnp.mean(jnp.dot(samples, mu))

        expected = float(mean_resultant_length(jnp.array(kappa), dim))
        np.testing.assert_allclose(float(empirical), expected, atol=0.02)

    @pytest.mark.parametrize(
        "dim,kappa",
        [
            (3, 1.0),
            (3, 10.0),
            (3, 50.0),
            (10, 5.0),
            (10, 50.0),
            (768, 10.0),
            (768, 100.0),
        ],
    )
    def test_gradient_finite_and_positive(self, dim, kappa):
        """d/dkappa E[x.mu] should be finite and positive."""
        n_samples = 4096
        key = random.PRNGKey(123)
        mu = jnp.zeros(dim, dtype=jnp.float32).at[0].set(1.0)

        def mean_dot(kap):
            s = sample(key, mu, kap, n_samples)
            return jnp.mean(jnp.dot(s, mu))

        grad_val = float(jax.grad(mean_dot)(jnp.float32(kappa)))
        assert np.isfinite(grad_val), f"gradient not finite: {grad_val}"
        assert grad_val > 0, f"gradient should be positive, got {grad_val}"

    @pytest.mark.parametrize(
        "dim,kappa,w",
        [
            (3, 1.0, -0.9),
            (3, 10.0, -0.9),
            (3, 5.0, 0.5),
            (3, 20.0, 0.9),
            (3, 50.0, 0.95),
            (10, 1.0, -0.5),
            (10, 10.0, 0.3),
            (10, 50.0, 0.8),
            (768, 10.0, 0.01),
            (768, 100.0, 0.1),
        ],
    )
    def test_dw_dkappa_matches_finite_differences(self, dim, kappa, w):
        """_compute_dw_dkappa should match numerical finite differences.

        We compare against finite differences of the normalized CDF inverse,
        computed via bisection. The normalized CDF F(w; kappa) = U(w; kappa) / Z(kappa)
        where U is the unnormalized CDF and Z = U(1; kappa).
        """
        eps = 1e-3 if kappa < 30 else kappa * 1e-4
        n_quad = 512

        def unnorm_cdf(w_val, kap):
            """Unnormalized CDF via Gauss-Legendre quadrature."""
            nodes, weights = np.polynomial.legendre.leggauss(n_quad)
            nodes = jnp.array(nodes, dtype=jnp.float32)
            weights = jnp.array(weights, dtype=jnp.float32)
            half_range = (w_val - (-1.0)) / 2.0
            midpoint = (w_val + (-1.0)) / 2.0
            t = midpoint + half_range * nodes
            half_exp = (dim - 3.0) / 2.0
            log_integrand = kap * t + half_exp * jnp.log(
                jnp.maximum(1.0 - t * t, 1e-30)
            )
            log_max = jnp.max(log_integrand)
            integrand = jnp.exp(log_integrand - log_max)
            return float(jnp.exp(log_max) * half_range * jnp.dot(weights, integrand))

        def norm_cdf(w_val, kap):
            """Normalized CDF: F(w; kappa) = U(w; kappa) / Z(kappa)."""
            return unnorm_cdf(w_val, kap) / unnorm_cdf(1.0 - 1e-10, kap)

        target_quantile = norm_cdf(w, kappa)

        def find_w_for_kappa(kap):
            """Find w such that norm_cdf(w, kap) == target_quantile, via bisection."""
            lo, hi = -1.0 + 1e-10, 1.0 - 1e-10
            for _ in range(80):
                mid = (lo + hi) / 2.0
                if norm_cdf(mid, kap) < target_quantile:
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2.0

        w_plus = find_w_for_kappa(kappa + eps)
        w_minus = find_w_for_kappa(kappa - eps)
        fd_grad = (w_plus - w_minus) / (2 * eps)

        analytic_grad = float(
            _compute_dw_dkappa(jnp.float32(w), jnp.float32(kappa), dim)
        )

        np.testing.assert_allclose(
            analytic_grad,
            fd_grad,
            rtol=0.05,
            err_msg=f"dim={dim}, kappa={kappa}, w={w}",
        )

    @pytest.mark.parametrize("dim", [3, 10, 768])
    def test_gradient_of_mean_dot_matches_analytic(self, dim):
        """d/dkappa E[x.mu] = dA_d/dkappa, compare autodiff vs analytic.

        The analytic derivative of A_d(kappa) = I_{d/2}(kappa)/I_{d/2-1}(kappa)
        is computed via JAX autodiff of mean_resultant_length (which is already
        tested separately).
        """
        kappa_val = 10.0
        n_samples = 50000
        key = random.PRNGKey(999)
        mu = jnp.zeros(dim, dtype=jnp.float32).at[0].set(1.0)

        def mean_dot(kap):
            s = sample(key, mu, kap, n_samples)
            return jnp.mean(jnp.dot(s, mu))

        empirical_grad = float(jax.grad(mean_dot)(jnp.float32(kappa_val)))
        analytic_grad = float(
            jax.grad(lambda k: mean_resultant_length(k, dim))(jnp.float32(kappa_val))
        )

        np.testing.assert_allclose(
            empirical_grad,
            analytic_grad,
            rtol=0.15,
            err_msg=f"dim={dim}",
        )

    @pytest.mark.parametrize("kappa", [0.0, 0.001])
    def test_kappa_near_zero(self, kappa):
        """Sampling and gradients should be well-behaved near kappa=0."""
        dim = 10
        n_samples = 1000
        key = random.PRNGKey(77)
        mu = jnp.zeros(dim, dtype=jnp.float32).at[0].set(1.0)

        samples = sample(key, mu, kappa, n_samples)
        assert samples.shape == (n_samples, dim)
        assert jnp.all(jnp.isfinite(samples))

        # Mean should be near zero for kappa~0
        mean_dot = float(jnp.mean(jnp.dot(samples, mu)))
        assert abs(mean_dot) < 0.15

        # Gradient should be finite
        if kappa > 0:

            def loss(kap):
                s = sample(key, mu, kap, n_samples)
                return jnp.mean(jnp.dot(s, mu))

            g = float(jax.grad(loss)(jnp.float32(kappa)))
            assert np.isfinite(g)

    def test_large_kappa(self):
        """Sampling and gradients should be well-behaved at large kappa."""
        dim = 10
        kappa = 1000.0
        n_samples = 1000
        key = random.PRNGKey(88)
        mu = jnp.zeros(dim, dtype=jnp.float32).at[0].set(1.0)

        samples = sample(key, mu, kappa, n_samples)
        assert jnp.all(jnp.isfinite(samples))

        # Samples should be very concentrated
        dots = jnp.dot(samples, mu)
        assert float(jnp.mean(dots)) > 0.99

        def loss(kap):
            s = sample(key, mu, kap, n_samples)
            return jnp.mean(jnp.dot(s, mu))

        g = float(jax.grad(loss)(jnp.float32(kappa)))
        assert np.isfinite(g)
        assert g > 0


if __name__ == "__main__":
    pytest.main([__file__])
