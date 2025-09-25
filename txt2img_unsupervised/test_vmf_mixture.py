"""Tests for mixtures of von Mises-Fisher distributions."""

import itertools

import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy.special as jsp
import numpy as np
import pytest

from txt2img_unsupervised import vmf
from txt2img_unsupervised import vmf_mixture


def _manual_log_prob(x: jnp.ndarray, mixture: vmf_mixture.VmfMixture) -> jnp.ndarray:
    component_terms = []
    for idx in range(mixture.num_components):
        component_terms.append(
            vmf.log_prob(x, mixture.mus[idx], float(mixture.kappas[idx]))
        )
    stacked = jnp.stack(component_terms, axis=0)
    log_weights = jnp.log(mixture.weights)
    return jsp.logsumexp(stacked + log_weights[:, None], axis=0)


class TestLogProb:
    def test_log_prob_matches_manual(self):
        weights = jnp.array([0.7, 0.3], dtype=jnp.float32)
        mus = jnp.stack(
            [
                jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            ]
        )
        kappas = jnp.array([5.0, 10.0], dtype=jnp.float32)
        mixture = vmf_mixture.VmfMixture(weights=weights, mus=mus, kappas=kappas)

        x = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        )
        x = x / jnp.linalg.norm(x, axis=1, keepdims=True)

        expected = _manual_log_prob(x, mixture)
        actual = vmf_mixture.log_prob(x, mixture)

        assert jnp.allclose(actual, expected, atol=1e-6)


class TestSample:
    def test_sample_component_proportions(self):
        weights = jnp.array([0.8, 0.2], dtype=jnp.float32)
        mus = jnp.stack(
            [
                jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            ]
        )
        kappas = jnp.array([50.0, 50.0], dtype=jnp.float32)
        mixture = vmf_mixture.VmfMixture(weights=weights, mus=mus, kappas=kappas)

        key = random.PRNGKey(0)
        samples = vmf_mixture.sample(key, mixture, 1000)

        similarities = samples @ mixture.mus.T
        predicted_components = jnp.argmax(similarities, axis=1)
        counts = jnp.bincount(predicted_components, length=mixture.num_components)
        empirical = counts / samples.shape[0]

        assert jnp.allclose(empirical, mixture.weights, atol=0.1)


class TestFit:
    def test_fit_recovers_simple_mixture(self):
        true_weights = jnp.array([0.6, 0.4], dtype=jnp.float32)
        true_mus = jnp.stack(
            [
                jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32),
            ]
        )
        true_kappas = jnp.array([15.0, 25.0], dtype=jnp.float32)
        true_mixture = vmf_mixture.VmfMixture(
            weights=true_weights, mus=true_mus, kappas=true_kappas
        )

        key_data, key_fit = random.split(random.PRNGKey(1))
        samples = vmf_mixture.sample(key_data, true_mixture, 1200)

        fitted, responsibilities = vmf_mixture.fit(
            samples, n_components=2, key=key_fit, max_iters=120, tol=1e-3
        )

        assert responsibilities.shape == (samples.shape[0], 2)

        best_weight_error = np.inf
        best_alignment = None
        for perm in itertools.permutations(range(2)):
            permuted_weights = fitted.weights[jnp.array(perm)]
            permuted_mus = fitted.mus[jnp.array(perm)]
            permuted_kappas = fitted.kappas[jnp.array(perm)]

            weight_error = float(jnp.max(jnp.abs(permuted_weights - true_weights)))
            if weight_error < best_weight_error:
                best_weight_error = weight_error
                best_alignment = (permuted_weights, permuted_mus, permuted_kappas)

        assert best_alignment is not None
        weights_aligned, mus_aligned, kappas_aligned = best_alignment

        assert jnp.max(jnp.abs(weights_aligned - true_weights)) < 0.1
        dot_products = jnp.sum(mus_aligned * true_mus, axis=1)
        assert jnp.all(dot_products > 0.8)
        relative_kappa_error = jnp.abs(kappas_aligned - true_kappas) / true_kappas
        assert jnp.max(relative_kappa_error) < 0.5

    def test_fit_handles_small_component_mass(self):
        data = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        fitted, _ = vmf_mixture.fit(
            data, n_components=3, key=random.PRNGKey(2), max_iters=50, tol=1e-4
        )

        assert fitted.weights.shape == (3,)
        assert fitted.mus.shape == (3, data.shape[1])
        assert fitted.kappas.shape == (3,)

    def test_fit_respects_max_kappa(self):
        data = jnp.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        )

        mixture, _ = vmf_mixture.fit(
            data,
            n_components=2,
            key=random.PRNGKey(3),
            max_iters=50,
            tol=1e-4,
            max_kappa=250.0,
        )

        assert jnp.all(mixture.kappas <= 250.0 + 1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
