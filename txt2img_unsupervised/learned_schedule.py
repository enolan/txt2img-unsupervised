"""Learned noise schedule for VLB-based score matching.

Implements a monotonic function that maps t ∈ [0, 1] to log κ(t), the log-concentration
of the vMF noise distribution. Monotonicity is guaranteed by construction: we define a
positive rate function g(t) = softplus(NN(t)) with an unconstrained neural network, then
integrate to get the normalized CDF F(t) = ∫₀ᵗ g(s)ds / ∫₀¹ g(s)ds. Since g > 0, F is
strictly increasing. The integral is computed via trapezoidal quadrature on a uniform grid.

Unlike positive-weight MLPs (which can only represent convex monotonic functions), this
construction can represent arbitrary monotonic functions — both convex and concave — since
the unconstrained NN can make g increase or decrease freely.

The output exactly hits learned endpoints:
    log_kappa(0) = log_kappa_min   (noisy end, near uniform)
    log_kappa(1) = log_kappa_max   (clean end, near data)

This is used in the VLB loss as:
    L_diff = (1/2) E_t [ γ'(t) · ||w_target - w_θ||² ]
where γ(t) = log κ(t) and γ'(t) is computed via autodiff through this network.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax import Array


class LearnedNoiseSchedule(nn.Module):
    """Monotonic network mapping t ∈ [0, 1] → log κ ∈ [log_kappa_min, log_kappa_max].

    The schedule is parameterized as:
        log_kappa(t) = log_kappa_min + (log_kappa_max - log_kappa_min) · F(t)
    where F(t) = ∫₀ᵗ g(s)ds / ∫₀¹ g(s)ds is the normalized CDF of a positive rate
    function g(t) = softplus(NN(t)).

    Attributes:
        hidden_dim: Width of the hidden layer in the rate network.
        n_quadrature_points: Number of grid points for numerical integration.
        init_log_kappa_min: Initial value for the learned log_kappa_min endpoint.
        init_log_kappa_max: Initial value for the learned log_kappa_max endpoint.
    """

    hidden_dim: int = 32
    n_quadrature_points: int = 1024
    init_log_kappa_min: float = -0.693  # log(0.5)
    init_log_kappa_max: float = 9.210  # log(10000)

    def setup(self):
        assert (
            self.n_quadrature_points >= 2
        ), f"n_quadrature_points must be >= 2, got {self.n_quadrature_points}"
        self.log_kappa_min = self.param(
            "log_kappa_min",
            lambda _: jnp.array(self.init_log_kappa_min),
        )
        self.log_kappa_max = self.param(
            "log_kappa_max",
            lambda _: jnp.array(self.init_log_kappa_max),
        )
        self.dense0 = nn.Dense(self.hidden_dim)
        self.dense1 = nn.Dense(1)

    def _rate(self, t: Array) -> Array:
        """Compute the positive rate function g(t) = softplus(NN(t)).

        Using an unconstrained NN allows g to increase or decrease freely, enabling
        both convex and concave schedules.
        """
        x = t[..., None]  # (..., 1)
        x = jax.nn.tanh(self.dense0(x))
        x = self.dense1(x).squeeze(-1)
        return jax.nn.softplus(x)

    def _normalized_cdf(self, t: Array) -> Array:
        """Compute F(t) = ∫₀ᵗ g(s)ds / ∫₀¹ g(s)ds via trapezoidal quadrature."""
        n = self.n_quadrature_points
        grid = jnp.linspace(0.0, 1.0, n)
        rates = self._rate(grid)

        dt = 1.0 / (n - 1)
        segment_areas = (rates[:-1] + rates[1:]) / 2 * dt
        cumulative = jnp.concatenate([jnp.array([0.0]), jnp.cumsum(segment_areas)])
        f_grid = cumulative / cumulative[-1]

        # Linearly interpolate F at the requested t values
        idx_float = t * (n - 1)
        idx_lo = jnp.clip(jnp.floor(idx_float).astype(jnp.int32), 0, n - 2)
        frac = idx_float - idx_lo
        return f_grid[idx_lo] * (1 - frac) + f_grid[idx_lo + 1] * frac

    def __call__(self, t: Array) -> Array:
        """Compute log κ(t) for the given time values.

        Args:
            t: Time values, any shape. Each element should be in [0, 1].

        Returns:
            log κ(t) values, same shape as t.
        """
        f = self._normalized_cdf(t)
        return self.log_kappa_min + (self.log_kappa_max - self.log_kappa_min) * f

    def log_kappa_derivative(self, t: Array) -> Array:
        """Compute γ'(t) = d(log κ)/dt for the given time values.

        Call via schedule.apply(params, t, method=schedule.log_kappa_derivative).

        Args:
            t: Time values, shape (batch,).

        Returns:
            γ'(t) values, same shape as t.
        """
        scalar_fn = lambda ti: self(ti)
        return jax.vmap(jax.grad(scalar_fn))(t)


def visualize_schedule(
    schedule: LearnedNoiseSchedule,
    params,
    width: int = 60,
    height: int = 20,
):
    """Print a terminal chart of the learned schedule using unicode block characters.

    Shows log κ(t) on the y-axis vs t on the x-axis, with axis labels and tick marks.

    Args:
        schedule: The LearnedNoiseSchedule module.
        params: Model parameters.
        width: Number of columns for the plot area.
        height: Number of rows for the plot area.
    """
    t_values = jnp.linspace(0.0, 1.0, width)
    log_kappa_values = np.array(schedule.apply(params, t_values))

    y_min, y_max = float(log_kappa_values.min()), float(log_kappa_values.max())
    y_pad = (y_max - y_min) * 0.02
    y_min -= y_pad
    y_max += y_pad

    # The 8 unicode block elements ▁▂▃▄▅▆▇█ divide each character cell into
    # 8 vertical sub-rows, giving 8x the effective vertical resolution.
    blocks = " ▁▂▃▄▅▆▇█"
    sub_rows = height * 8

    # Map each value to a sub-row index
    sub_indices = np.clip(
        ((log_kappa_values - y_min) / (y_max - y_min) * sub_rows).astype(int),
        0,
        sub_rows - 1,
    )

    # Build the grid top-to-bottom. Each column is filled from the bottom up
    # to the curve height: full blocks below, a partial block at the curve.
    label_width = 10
    lines = []
    for row in range(height - 1, -1, -1):
        row_bottom = row * 8
        row_top = row_bottom + 8
        chars = []
        for col in range(width):
            si = sub_indices[col]
            if si >= row_top:
                chars.append("█")
            elif si > row_bottom:
                chars.append(blocks[si - row_bottom])
            else:
                chars.append(" ")

        # Y-axis label for top, middle, and bottom rows
        if row == height - 1:
            label = f"{y_max:>{label_width}.2f}"
        elif row == height // 2:
            label = f"{(y_min + y_max) / 2:>{label_width}.2f}"
        elif row == 0:
            label = f"{y_min:>{label_width}.2f}"
        else:
            label = " " * label_width

        lines.append(f"{label} │{''.join(chars)}│")

    # X-axis
    border = " " * label_width + " └" + "─" * width + "┘"
    x_labels = " " * (label_width + 2) + "0" + " " * (width - 5) + "t=1"

    header = " " * label_width + "  log κ(t)"
    print(header)
    for line in lines:
        print(line)
    print(border)
    print(x_labels)


# =============================================================================
# Tests
# =============================================================================


def test_output_shape():
    """Test that the schedule produces outputs with the correct shape."""
    schedule = LearnedNoiseSchedule()
    params = schedule.init(jax.random.PRNGKey(0), jnp.array(0.5))

    # Scalar input
    out = schedule.apply(params, jnp.array(0.5))
    assert out.shape == ()

    # Batched input
    t_batch = jnp.linspace(0.0, 1.0, 10)
    out_batch = schedule.apply(params, t_batch)
    assert out_batch.shape == (10,)


def test_endpoints():
    """Test that the schedule exactly hits the learned endpoints."""
    for log_kappa_min, log_kappa_max in [(-0.693, 9.210), (-2.0, 5.0), (0.0, 15.0)]:
        schedule = LearnedNoiseSchedule(
            init_log_kappa_min=log_kappa_min,
            init_log_kappa_max=log_kappa_max,
        )
        params = schedule.init(jax.random.PRNGKey(42), jnp.array(0.5))

        val_at_0 = schedule.apply(params, jnp.array(0.0))
        val_at_1 = schedule.apply(params, jnp.array(1.0))

        np.testing.assert_allclose(float(val_at_0), log_kappa_min, atol=1e-5)
        np.testing.assert_allclose(float(val_at_1), log_kappa_max, atol=1e-5)


def test_monotonicity():
    """Test that log κ(t) is strictly increasing across many random initializations."""
    t_values = jnp.linspace(0.0, 1.0, 200)

    for seed in range(20):
        schedule = LearnedNoiseSchedule()
        params = schedule.init(jax.random.PRNGKey(seed), jnp.array(0.5))
        log_kappa_values = schedule.apply(params, t_values)

        diffs = jnp.diff(log_kappa_values)
        assert jnp.all(diffs > 0), (
            f"Monotonicity violated at seed {seed}: "
            f"min diff = {float(jnp.min(diffs)):.6e}"
        )


def test_derivative_positive():
    """Test that γ'(t) > 0 everywhere (consequence of monotonicity)."""
    schedule = LearnedNoiseSchedule()
    params = schedule.init(jax.random.PRNGKey(7), jnp.array(0.5))

    # Avoid exact endpoints where interpolation gradient could be numerically tricky
    t_values = jnp.linspace(0.01, 0.99, 100)
    derivs = schedule.apply(params, t_values, method=schedule.log_kappa_derivative)

    assert jnp.all(
        derivs > 0
    ), f"Derivative not positive everywhere: min = {float(jnp.min(derivs)):.6e}"


def test_derivative_matches_finite_differences():
    """Test that autodiff derivative matches finite difference approximation."""
    schedule = LearnedNoiseSchedule()
    params = schedule.init(jax.random.PRNGKey(3), jnp.array(0.5))

    t_values = jnp.linspace(0.1, 0.9, 50)
    analytic_derivs = schedule.apply(
        params, t_values, method=schedule.log_kappa_derivative
    )

    # float32 limits FD precision; eps=1e-3 balances truncation vs rounding error
    eps = 1e-3
    fd_derivs = (
        schedule.apply(params, t_values + eps) - schedule.apply(params, t_values - eps)
    ) / (2 * eps)

    # The autodiff derivative is the slope of the piecewise-linear interpolation,
    # which differs slightly from the FD estimate due to grid discretization.
    np.testing.assert_allclose(
        np.array(analytic_derivs), np.array(fd_derivs), rtol=1e-2
    )


def test_jit_compatible():
    """Test that the schedule and its derivative work under JIT."""
    schedule = LearnedNoiseSchedule()
    params = schedule.init(jax.random.PRNGKey(0), jnp.array(0.5))

    jitted_apply = jax.jit(schedule.apply)
    jitted_deriv = jax.jit(
        lambda p, t: schedule.apply(p, t, method=schedule.log_kappa_derivative)
    )

    t = jnp.array([0.2, 0.5, 0.8])
    log_kappa = jitted_apply(params, t)
    derivs = jitted_deriv(params, t)

    assert log_kappa.shape == (3,)
    assert derivs.shape == (3,)
    assert jnp.all(jnp.isfinite(log_kappa))
    assert jnp.all(jnp.isfinite(derivs))


def test_gradients_flow_to_all_params():
    """Test that gradients from the output flow to all parameters including endpoints."""
    schedule = LearnedNoiseSchedule()
    params = schedule.init(jax.random.PRNGKey(0), jnp.array(0.5))

    def loss_fn(p):
        t = jnp.linspace(0.0, 1.0, 10)
        return jnp.sum(schedule.apply(p, t) ** 2)

    grads = jax.grad(loss_fn)(params)

    flat_grads, tree_def = jax.tree_util.tree_flatten_with_path(grads)
    for path, grad_leaf in flat_grads:
        path_str = "/".join(str(k) for k in path)
        assert jnp.any(grad_leaf != 0), f"Zero gradient for {path_str}"


def test_custom_hidden_dim():
    """Test that different hidden dimensions work correctly."""
    for hidden_dim in [8, 64, 128]:
        schedule = LearnedNoiseSchedule(hidden_dim=hidden_dim)
        params = schedule.init(jax.random.PRNGKey(0), jnp.array(0.5))

        t = jnp.linspace(0.0, 1.0, 50)
        log_kappa = schedule.apply(params, t)
        assert log_kappa.shape == (50,)

        diffs = jnp.diff(log_kappa)
        assert jnp.all(diffs > 0), f"Monotonicity violated with hidden_dim={hidden_dim}"


def _isotonic_regression(y):
    """Find the non-decreasing function that minimizes MSE to y.

    Uses the Pool Adjacent Violators Algorithm (PAVA). Maintains a stack of blocks,
    each storing (sum, count). When a new value would violate monotonicity (its value
    is less than the previous block's mean), merge the two blocks — replacing both
    with their combined mean. This propagates backward until monotonicity is restored.
    """
    # Each block is [sum, count]; the block's value is sum/count.
    blocks = []
    for val in y:
        blocks.append([float(val), 1])
        # Merge backward while the last block's mean is less than the previous block's
        while len(blocks) > 1:
            if blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                blocks[-2][0] += blocks[-1][0]
                blocks[-2][1] += blocks[-1][1]
                blocks.pop()
            else:
                break
    # Expand blocks back to per-element values
    result = []
    for s, c in blocks:
        result.extend([s / c] * c)
    return np.array(result)


def _fit_schedule_to_target(target_values, n_steps=8000, hidden_dim=64, peak_lr=1e-2):
    """Fit a LearnedNoiseSchedule to target values on a uniform grid over [0, 1]."""
    n_points = len(target_values)
    t_jnp = jnp.linspace(0.0, 1.0, n_points)
    fit_target = jnp.array(target_values)

    schedule = LearnedNoiseSchedule(hidden_dim=hidden_dim)
    params = schedule.init(jax.random.PRNGKey(0), jnp.array(0.5))

    optimizer = optax.adam(
        optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=peak_lr,
            warmup_steps=500,
            decay_steps=n_steps,
        )
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state):
        def loss_fn(p):
            return jnp.mean((schedule.apply(p, t_jnp) - fit_target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    for _ in range(n_steps):
        params, opt_state, loss = train_step(params, opt_state)

    final_pred = np.array(schedule.apply(params, t_jnp))
    return final_pred, params, schedule


@pytest.mark.parametrize(
    "target_name,target_fn",
    [
        ("10*ln(x+1)", lambda x: 10 * np.log(x + 1)),
        ("10*x^2", lambda x: 10 * x**2),
        ("10*(exp(x)-1)", lambda x: 10 * (np.exp(x) - 1)),
        ("20*sin(x/2)", lambda x: 20 * np.sin(x / 2.0)),
        ("20*|x-0.25|", lambda x: 20 * np.abs(x - 0.25)),
    ],
)
def test_learn_function(target_name, target_fn):
    """Test that the schedule can learn arbitrary target functions.

    For monotonic targets, the schedule should achieve a very close fit. For non-monotonic
    targets, the schedule (being monotonic by construction) should converge to approximately
    the isotonic regression — the closest monotonically non-decreasing function in L2.
    """
    n_points = 200
    t_np = np.linspace(0.0, 1.0, n_points)
    target = target_fn(t_np)

    final_pred, params, schedule = _fit_schedule_to_target(target)

    # The best a monotonic function can do is the isotonic regression
    isotonic_target = _isotonic_regression(target)
    isotonic_mse = np.mean((isotonic_target - target) ** 2)

    schedule_mse = np.mean((final_pred - target) ** 2)
    isotonic_range = isotonic_target[-1] - isotonic_target[0]
    residual_mse = np.mean((final_pred - isotonic_target) ** 2)
    relative_rmse = np.sqrt(residual_mse) / isotonic_range

    print(
        f"{target_name}: schedule MSE={schedule_mse:.6f}, isotonic MSE={isotonic_mse:.6f}"
    )
    print(
        f"  residual RMSE vs isotonic: {np.sqrt(residual_mse):.6f}, relative: {relative_rmse:.6f}"
    )
    visualize_schedule(schedule, params, width=20, height=20)

    # Schedule's MSE to the target should be close to the optimal monotonic MSE
    assert schedule_mse < isotonic_mse + isotonic_range**2 * 0.02**2, (
        f"{target_name}: schedule MSE {schedule_mse:.4f} too far above "
        f"isotonic MSE {isotonic_mse:.4f}"
    )

    # The schedule should closely match the isotonic regression
    assert (
        relative_rmse < 0.02
    ), f"{target_name}: relative RMSE vs isotonic {relative_rmse:.4f} too high"

    # Check endpoints converged (tolerance scales with range)
    endpoint_atol = max(0.1, isotonic_range * 0.02)
    np.testing.assert_allclose(final_pred[0], isotonic_target[0], atol=endpoint_atol)
    np.testing.assert_allclose(final_pred[-1], isotonic_target[-1], atol=endpoint_atol)
