"""
Implementation of the Muon optimizer.
Adapted from https://kellerjordan.github.io/posts/muon/
"""

from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optax.tree_utils as otu
import pytest


def newton_schulz(G, steps=5, eps=1e-7):
    "Orthogonalize G using the Newton-Schulz method"
    assert G.ndim == 2, "Cannot orthogonalize a non-matrix"
    G = jnp.asarray(G)
    a, b, c = (3.4445, -4.7750, 2.0315)
    G = G / (jnp.linalg.norm(G) + eps)
    if G.shape[0] > G.shape[1]:
        G = G.T
        transpose = True
    else:
        transpose = False
    for _ in range(steps):
        A = G @ G.T
        B = b * A + c * A @ A
        G = a * G + B @ G
    if transpose:
        G = G.T
    return G


def _compute_orthogonality_error(matrix):
    """Helper function to compute how far a matrix is from being orthogonal."""
    is_tall = matrix.shape[0] >= matrix.shape[1]

    if is_tall:
        orthogonality_check = matrix.T @ matrix
        expected_size = matrix.shape[1]
    else:
        orthogonality_check = matrix @ matrix.T
        expected_size = matrix.shape[0]

    expected = jnp.eye(expected_size)
    return jnp.linalg.norm(orthogonality_check - expected)


@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (5, 3), (3, 5), (10, 8)])
@pytest.mark.parametrize("steps", [1, 3, 5])
def test_newton_schulz_approximate_orthogonality(shape, steps):
    """Test that newton_schulz significantly improves orthogonality.
    Note: Uses the 'cursed quintic' iteration which oscillates rather than converges perfectly.
    """
    rng = jax.random.PRNGKey(42)
    G = jax.random.normal(rng, shape)

    # Add small regularization to avoid singular matrices
    if shape[0] >= shape[1]:
        G = G + 0.1 * jnp.eye(shape[0], shape[1])
    else:
        G = G + 0.1 * jnp.eye(shape[0], shape[1])[: shape[0], :]

    # Measure input orthogonality (after normalization like the algorithm does)
    G_normalized = G / jnp.linalg.norm(G)
    input_error = _compute_orthogonality_error(G_normalized)

    # Apply Newton-Schulz and measure output orthogonality
    result = newton_schulz(G, steps=steps)
    result_error = _compute_orthogonality_error(result)

    assert result.shape == shape

    # Set step-dependent expectations (cursed quintic oscillates, so be realistic)
    if steps == 1:
        min_improvement, max_error = 1.2, 1.8  # Gentle expectations for single step
    elif steps <= 3:
        min_improvement, max_error = 1.5, 1.2  # Moderate expectations
    else:
        min_improvement, max_error = (
            1.6,
            1.0,
        )  # Higher expectations (but not too high due to oscillation)

    # Verify orthogonalization actually happens
    improvement = input_error / (result_error + 1e-8)
    assert (
        improvement > min_improvement
    ), f"Expected {min_improvement}x improvement with {steps} steps, got {improvement:.3f}x (input: {input_error:.3f}, result: {result_error:.3f})"

    # Verify result is reasonably orthogonal
    assert (
        result_error < max_error
    ), f"Result should be reasonably orthogonal with {steps} steps, got error {result_error:.3f}"


def test_newton_schulz_assertion_non_matrix():
    """Test that newtonSchulz raises assertion error for non-2D arrays."""
    rng = jax.random.PRNGKey(42)

    # Test with 1D array
    G_1d = jax.random.normal(rng, (5,))
    with pytest.raises(AssertionError, match="Cannot orthogonalize a non-matrix"):
        newton_schulz(G_1d)

    # Test with 3D array
    G_3d = jax.random.normal(rng, (3, 3, 3))
    with pytest.raises(AssertionError, match="Cannot orthogonalize a non-matrix"):
        newton_schulz(G_3d)


def test_newton_schulz_identity_matrix():
    """Test that newton_schulz handles identity matrix reasonably."""
    I = jnp.eye(4)
    result = newton_schulz(I)

    assert result.shape == (4, 4)
    assert jnp.all(jnp.isfinite(result))

    # Identity should remain very close to orthogonal (it's already orthogonal)
    orthogonality_error = _compute_orthogonality_error(result)
    assert (
        orthogonality_error < 1.0
    ), f"Identity matrix result should stay orthogonal, got error {orthogonality_error:.3f}"


def test_newton_schulz_small_matrix():
    """Test newton_schulz with small matrices."""
    G = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = newton_schulz(G)

    assert result.shape == (2, 2)
    assert jnp.all(jnp.isfinite(result))

    # Should achieve reasonable orthogonality even for small matrices
    orthogonality_error = _compute_orthogonality_error(result)
    assert (
        orthogonality_error < 1.5
    ), f"Small matrix should be reasonably orthogonal, got error {orthogonality_error:.3f}"


def test_newton_schulz_rectangular_matrices():
    """Test newton_schulz with various rectangular matrices."""
    rng = jax.random.PRNGKey(123)

    # Test tall matrix (more rows than columns)
    G_tall = jax.random.normal(rng, (6, 4)) + 0.1 * jnp.eye(6, 4)
    result_tall = newton_schulz(G_tall)

    assert result_tall.shape == (6, 4)
    assert jnp.all(jnp.isfinite(result_tall))

    # Should orthogonalize columns for tall matrix
    orthogonality_error_tall = _compute_orthogonality_error(result_tall)
    assert (
        orthogonality_error_tall < 1.2
    ), f"Tall matrix should be reasonably orthogonal, got error {orthogonality_error_tall:.3f}"

    # Test wide matrix (more columns than rows)
    G_wide = jax.random.normal(rng, (3, 7)) + 0.1 * jnp.eye(3, 7)[:3, :]
    result_wide = newton_schulz(G_wide)

    assert result_wide.shape == (3, 7)
    assert jnp.all(jnp.isfinite(result_wide))

    # Should orthogonalize rows for wide matrix
    orthogonality_error_wide = _compute_orthogonality_error(result_wide)
    assert (
        orthogonality_error_wide < 1.2
    ), f"Wide matrix should be reasonably orthogonal, got error {orthogonality_error_wide:.3f}"


def test_newton_schulz_oscillation_property():
    """Test that the cursed quintic exhibits oscillation as expected."""
    rng = jax.random.PRNGKey(456)
    G = jax.random.normal(rng, (4, 4)) + 0.1 * jnp.eye(4)

    # Test different step counts - the cursed quintic oscillates
    result_5_steps = newton_schulz(G, steps=5)
    result_10_steps = newton_schulz(G, steps=10)
    result_15_steps = newton_schulz(G, steps=15)

    # All should be finite and reasonable
    for result in [result_5_steps, result_10_steps, result_15_steps]:
        assert jnp.all(jnp.isfinite(result))
        assert result.shape == (4, 4)

    # Due to oscillation, more steps don't necessarily mean better orthogonality
    # Just check that we get different results (showing the oscillation)
    assert not jnp.allclose(result_5_steps, result_10_steps, atol=1e-6)
    assert not jnp.allclose(result_10_steps, result_15_steps, atol=1e-6)


def test_newton_schulz_numerical_stability():
    """Test newton_schulz with matrices that might cause numerical issues."""
    rng = jax.random.PRNGKey(789)

    # Test with very small matrix elements
    G_small = 1e-6 * jax.random.normal(rng, (3, 3))
    result_small = newton_schulz(G_small)

    assert jnp.all(jnp.isfinite(result_small))
    assert result_small.shape == (3, 3)

    # Should still orthogonalize despite small input values
    orthogonality_error = _compute_orthogonality_error(result_small)
    assert (
        orthogonality_error < 1.5
    ), f"Should handle small values and orthogonalize, got error {orthogonality_error:.3f}"

    # Test with matrix close to singular (but not singular due to eps parameter)
    singular_like = jnp.array([[1.0, 1.0], [1.0, 1.0001]])
    result_singular = newton_schulz(singular_like)

    # Should handle gracefully and still produce orthogonal result
    assert jnp.all(jnp.isfinite(result_singular))
    orthogonality_error_sing = _compute_orthogonality_error(result_singular)
    assert (
        orthogonality_error_sing < 2.0
    ), f"Should handle near-singular matrices, got error {orthogonality_error_sing:.3f}"


@pytest.mark.parametrize("eps", [1e-12, 1e-7, 1e-3])
def test_newton_schulz_eps_parameter(eps):
    """Test the effect of different eps values."""
    rng = jax.random.PRNGKey(999)
    G = jax.random.normal(rng, (4, 3))

    result = newton_schulz(G, eps=eps)

    assert jnp.all(jnp.isfinite(result))
    assert result.shape == (4, 3)

    # Should orthogonalize regardless of eps value
    orthogonality_error = _compute_orthogonality_error(result)
    assert (
        orthogonality_error < 1.2
    ), f"Should orthogonalize with eps={eps}, got error {orthogonality_error:.3f}"


def test_newton_schulz_normalization():
    """Test that the input normalization works correctly."""
    rng = jax.random.PRNGKey(111)

    # Test with different scales
    for scale in [0.1, 1.0, 10.0, 100.0]:
        G = scale * jax.random.normal(rng, (3, 3))
        result = newton_schulz(G)

        assert jnp.all(jnp.isfinite(result))
        assert result.shape == (3, 3)

        # Should orthogonalize regardless of input scale
        orthogonality_error = _compute_orthogonality_error(result)
        assert (
            orthogonality_error < 1.2
        ), f"Should orthogonalize with scale={scale}, got error {orthogonality_error:.3f}"


def test_newton_schulz_transpose_logic():
    """Test the transpose logic for tall vs wide matrices."""
    rng = jax.random.PRNGKey(222)

    # Test that tall and wide matrices are handled correctly
    G_tall = jax.random.normal(rng, (5, 3))
    G_wide = jax.random.normal(rng, (3, 5))

    result_tall = newton_schulz(G_tall)
    result_wide = newton_schulz(G_wide)

    assert result_tall.shape == (5, 3)
    assert result_wide.shape == (3, 5)
    assert jnp.all(jnp.isfinite(result_tall))
    assert jnp.all(jnp.isfinite(result_wide))

    # Both should be reasonably orthogonal
    error_tall = _compute_orthogonality_error(result_tall)
    error_wide = _compute_orthogonality_error(result_wide)
    assert (
        error_tall < 1.2
    ), f"Tall matrix should be orthogonal, got error {error_tall:.3f}"
    assert (
        error_wide < 1.2
    ), f"Wide matrix should be orthogonal, got error {error_wide:.3f}"


def test_newton_schulz_jit_compatibility():
    """Test that newtonSchulz works with JAX JIT compilation."""
    rng = jax.random.PRNGKey(333)
    G = jax.random.normal(rng, (4, 4))

    # Compile the function
    jit_newtonSchulz = jax.jit(newton_schulz)

    # Both versions should produce the same result
    result_regular = newton_schulz(G)
    result_jit = jit_newtonSchulz(G)

    np.testing.assert_allclose(result_regular, result_jit, atol=1e-10, rtol=1e-10)


class MuonState(NamedTuple):
    mu: optax.Updates


def muonize(beta: float) -> optax.GradientTransformation:
    "Gradient transformation that applies the Muon optimizer."

    def init_fn(params: optax.Params) -> MuonState:
        return MuonState(mu=otu.tree_zeros_like(params))

    def update_fn(
        updates: optax.Updates, state: MuonState, params: Optional[optax.Params] = None
    ) -> Tuple[optax.Updates, MuonState]:
        del params
        new_state = MuonState(mu=optax.update_moment(state.mu, updates, beta, 1))

        def orthogonalize_param(param: jax.Array) -> jax.Array:
            assert param.ndim >= 2, (
                "Muon optimizer requires parameters to be at least 2D. Use another optimizer for 1 "
                "or 0 dimensional parameters."
            )
            if param.ndim == 2:
                return newton_schulz(param)
            else:
                # In cases with higher dimensional arrays, e.g. a query parameter for an attention
                # layer inside a scan with shape (n_layers, n_query_heads, d_model, head_dim), we
                # orthogonalize the last two dimensions.
                num_leading_dims = param.ndim - 2
                vmapped_newton_schulz = newton_schulz
                for _ in range(num_leading_dims):
                    vmapped_newton_schulz = jax.vmap(vmapped_newton_schulz)
                return vmapped_newton_schulz(param)

        output = jax.tree_util.tree_map(orthogonalize_param, new_state.mu)
        return output, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def test_muonize_2d_parameters():
    """Test muonize with simple 2D parameters."""
    rng = jax.random.PRNGKey(42)

    # Create a simple 2D parameter
    params = {"weight": jax.random.normal(rng, (4, 3))}

    # Create the optimizer
    optimizer = muonize(beta=0.9)
    opt_state = optimizer.init(params)

    # Create some dummy gradients
    grads = {"weight": jax.random.normal(rng, (4, 3))}

    # Apply one update
    updates, new_opt_state = optimizer.update(grads, opt_state)

    # Check shapes are preserved
    assert updates["weight"].shape == (4, 3)
    assert jnp.all(jnp.isfinite(updates["weight"]))

    # Check that the result is reasonably orthogonal
    orthogonality_error = _compute_orthogonality_error(updates["weight"])
    assert (
        orthogonality_error < 1.5
    ), f"2D parameter should be orthogonalized, got error {orthogonality_error:.3f}"


def test_muonize_higher_dimensional_parameters():
    """Test muonize with 3D and 4D parameters to verify vmap implementation."""
    rng = jax.random.PRNGKey(123)
    param_3d_rng, param_4d_rng, grad_3d_rng, grad_4d_rng = jax.random.split(rng, 4)

    # Create parameters with different dimensions
    params = {
        "weight_3d": jax.random.normal(param_3d_rng, (2, 4, 3)),  # 3D
        "weight_4d": jax.random.normal(param_4d_rng, (2, 3, 4, 3)),  # 4D
    }

    # Create the optimizer
    optimizer = muonize(beta=0.9)
    opt_state = optimizer.init(params)

    # Create some dummy gradients
    grads = {
        "weight_3d": jax.random.normal(grad_3d_rng, (2, 4, 3)),
        "weight_4d": jax.random.normal(grad_4d_rng, (2, 3, 4, 3)),
    }

    # Apply one update
    updates, new_opt_state = optimizer.update(grads, opt_state)

    # Check shapes are preserved
    assert updates["weight_3d"].shape == (2, 4, 3)
    assert updates["weight_4d"].shape == (2, 3, 4, 3)
    assert jnp.all(jnp.isfinite(updates["weight_3d"]))
    assert jnp.all(jnp.isfinite(updates["weight_4d"]))

    # For 3D parameter, check each 2D slice is orthogonalized
    for i in range(2):
        slice_2d = updates["weight_3d"][i]
        orthogonality_error = _compute_orthogonality_error(slice_2d)
        assert (
            orthogonality_error < 1.5
        ), f"3D parameter slice {i} should be orthogonalized, got error {orthogonality_error:.3f}"

    # For 4D parameter, check each 2D slice is orthogonalized
    for i in range(2):
        for j in range(3):
            slice_2d = updates["weight_4d"][i, j]
            orthogonality_error = _compute_orthogonality_error(slice_2d)
            assert (
                orthogonality_error < 1.5
            ), f"4D parameter slice [{i},{j}] should be orthogonalized, got error {orthogonality_error:.3f}"


def test_muonize_momentum_accumulation():
    """Test that momentum is properly accumulated across multiple steps."""
    params_rng, grad_rng = jax.random.split(jax.random.PRNGKey(456))

    # Create a simple parameter
    params = {"weight": jax.random.normal(params_rng, (3, 3))}

    # Create the optimizer with high momentum
    optimizer = muonize(beta=0.95)
    opt_state = optimizer.init(params)

    # Apply multiple updates with the same gradient
    consistent_grad = {"weight": jax.random.normal(grad_rng, (3, 3)) * 0.1}

    # All updates should be finite and properly shaped
    for i in range(10):
        updates, opt_state = optimizer.update(consistent_grad, opt_state)
        assert updates["weight"].shape == (3, 3)
        assert jnp.all(jnp.isfinite(updates["weight"]))

        # Each should be reasonably orthogonal
        orthogonality_error = _compute_orthogonality_error(updates["weight"])
        assert (
            orthogonality_error < 1.5
        ), f"Updates should be orthogonalized, got error {orthogonality_error:.3f}"


def test_muonize_assertion_1d_parameter():
    """Test that muonize raises assertion error for 1D parameters."""
    rng = jax.random.PRNGKey(789)

    # Create a 1D parameter (should fail)
    params = {"bias": jax.random.normal(rng, (5,))}

    optimizer = muonize(beta=0.9)
    opt_state = optimizer.init(params)

    grads = {"bias": jax.random.normal(rng, (5,))}

    # Should raise assertion error when trying to orthogonalize 1D parameter
    with pytest.raises(
        AssertionError, match="Muon optimizer requires parameters to be at least 2D"
    ):
        optimizer.update(grads, opt_state)


def test_muonize_toy_neural_network():
    """Test muonize on a toy neural network with 2D and 3D parameters."""
    rng = jax.random.PRNGKey(42)
    param_rng, data_rng = jax.random.split(rng, 2)

    # Network dimensions
    input_dim, hidden_dim, output_dim = 8, 16, 4
    batch_size, num_mlp_layers = 32, 3

    # Initialize parameters (all 2D+, no biases to keep test simple)
    w1_rng, w2_rng, mh_rng = jax.random.split(param_rng, 3)

    params = {
        "in_proj": jax.random.normal(w1_rng, (input_dim, hidden_dim)) * 0.1,  # 2D
        "out_proj": jax.random.normal(w2_rng, (hidden_dim, output_dim)) * 0.1,  # 2D
        "mlp_weights": jax.random.normal(
            mh_rng, (num_mlp_layers, hidden_dim, hidden_dim)
        )
        * 0.1,  # 3D
    }

    # Set up Muon optimizer with learning rate scaling
    optimizer = optax.chain(muonize(beta=0.95), optax.scale_by_learning_rate(0.03))

    opt_state = optimizer.init(params)

    # Define forward pass
    def forward(params, x):
        # First layer: x @ w1
        h = x @ params["in_proj"]

        for layer in range(num_mlp_layers):
            h = h + jax.nn.gelu(h @ params["mlp_weights"][layer])
        # Final layer
        output = h @ params["out_proj"]
        return output

    # Generate synthetic training data
    x_rng, y_rng = jax.random.split(data_rng, 2)
    X = jax.random.normal(x_rng, (batch_size, input_dim))
    y = jax.random.normal(y_rng, (batch_size, output_dim))

    # Loss function
    def loss_fn(params, x, y):
        pred = forward(params, x)
        return jnp.mean((pred - y) ** 2)

    # Training step
    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss, updates

    # Record initial loss
    initial_loss = loss_fn(params, X, y)
    assert jnp.isfinite(initial_loss), "Initial loss should be finite"

    # Train for several steps
    current_params = params
    current_opt_state = opt_state
    losses = []

    for step in range(100):
        current_params, current_opt_state, loss, updates = train_step(
            current_params, current_opt_state, X, y
        )
        if step % 10 == 0:
            print(f"Step {step}, loss: {loss}")
        losses.append(loss)

    final_loss = losses[-1]

    # Verify training worked
    assert jnp.isfinite(final_loss), f"Final loss should be finite, got {final_loss}"
    assert final_loss < initial_loss * 0.5, (
        f"Loss should decrease significantly: initial={initial_loss:.4f}, "
        f"final={final_loss:.4f}, reduction factor={final_loss/initial_loss:.3f}"
    )
    assert (
        final_loss < 0.05
    ), f"Final loss should be reasonably small, got {final_loss:.4f}"

    # Verify loss generally decreased over time (allowing for some oscillation)
    mid_point = len(losses) // 2
    early_avg = jnp.mean(jnp.array(losses[:mid_point]))
    late_avg = jnp.mean(jnp.array(losses[mid_point:]))
    assert late_avg < early_avg, (
        f"Loss should generally decrease over training: "
        f"early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
    )
