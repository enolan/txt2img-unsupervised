"""
Optax optimizer wrapper that skips batches where gradient norm exceeds a threshold based on recent
history. The intent is to skip outlier batches that occur very occasionally, which have huge norms
and break or set back learning. Won't help if your training is unstable for reasons that wouldn't be
fixed with a larger batch size.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from typing import NamedTuple


class AdaptiveGradientSkipState(NamedTuple):
    skip_count: int
    """The number of updates that have been skipped."""
    skipped_last: bool
    """Whether the last update was skipped."""
    inner_state: optax.OptState
    """The state of the inner optimizer."""
    total_steps: int
    """The total number of steps taken."""
    historical_norms: jax.Array
    last_idx: int


def adaptive_gradient_skip(
    inner_optimizer: optax.GradientTransformation,
    history_len: int = 100,
    threshold_factor: float = 1.5,
) -> optax.GradientTransformation:
    """Wraps an optimizer to skip updates that exceed the largest gradient norm among the past
    history_len updates by at least a factor of threshold_factor. No skipping occurs until
    history_len steps have been taken.

    Args:
        inner_optimizer: The optimizer to wrap.
        history_len: The number of past gradient norms to keep track of.
        threshold_factor: The factor by which the maximum historical gradient norm is scaled to
            determine the threshold for skipping updates.

    Returns:
        A `GradientTransformation` object.
    """
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    if threshold_factor < 1:
        raise ValueError("threshold_factor must be >= 1")

    def init_fn(params):
        return AdaptiveGradientSkipState(
            historical_norms=jnp.zeros(history_len),
            skip_count=0,
            skipped_last=False,
            last_idx=-1,
            inner_state=inner_optimizer.init(params),
            total_steps=0,
        )

    def update_fn(updates, state, params=None):
        skip_threshold = jnp.max(state.historical_norms) * threshold_factor
        this_norm = optax.global_norm(updates)

        next_idx = (state.last_idx + 1) % history_len
        new_norms = state.historical_norms.at[next_idx].set(this_norm)
        state = state._replace(
            historical_norms=new_norms,
            last_idx=next_idx,
            total_steps=state.total_steps + 1,
        )

        def skip_updates():
            new_updates = jax.tree.map(jnp.zeros_like, updates)
            new_state = state._replace(
                skipped_last=True, skip_count=state.skip_count + 1
            )
            return new_updates, new_state

        def do_updates():
            new_state = state._replace(skipped_last=False)
            new_updates, new_inner_state = inner_optimizer.update(
                updates, new_state.inner_state, params
            )
            new_state = new_state._replace(inner_state=new_inner_state)
            return new_updates, new_state

        return jax.lax.cond(
            jnp.logical_and(
                this_norm > skip_threshold, state.total_steps > history_len
            ),
            skip_updates,
            do_updates,
        )

    return optax.GradientTransformation(init_fn, update_fn)


@pytest.mark.parametrize("jit", [True, False])
def test_doesnt_skip_below_threshold(jit):
    """Test that the optimizer doesn't skip updates when the gradient norm is below the threshold."""
    inner_opt = optax.sgd(learning_rate=0.1)
    opt = adaptive_gradient_skip(inner_opt, history_len=10, threshold_factor=2.0)
    update_fn = jax.jit(opt.update, donate_argnums=(1, 2)) if jit else opt.update
    params = jnp.ones(10)
    state = opt.init(params)

    for _ in range(20):
        grads = jnp.full(10, 0.1)
        updates, state = update_fn(grads, state, params)
        np.testing.assert_allclose(updates, grads * -0.1)
        params = optax.apply_updates(params, updates)


@pytest.mark.parametrize("jit", [True, False])
def test_skips_above_threshold(jit):
    """Test that the optimizer skips updates when the gradient norm is above the threshold."""
    inner_opt = optax.adam(learning_rate=0.1)
    opt = adaptive_gradient_skip(inner_opt, history_len=3, threshold_factor=1.5)
    update_fn = jax.jit(opt.update, donate_argnums=(1, 2)) if jit else opt.update
    params = jnp.ones(10)
    state = opt.init(params)
    adam_state = inner_opt.init(params)
    adam_params = params

    # Do a bunch of steps that shouldn't be skipped, checking that regular adam and the wrapped adam
    # do the same thing.
    normal_grads = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 1.1]]
    for grad in normal_grads:
        updates, state = update_fn(grad, state, params)
        adam_updates, adam_state = inner_opt.update(grad, adam_state, adam_params)
        np.testing.assert_allclose(updates, adam_updates)
        params = optax.apply_updates(params, updates)
        adam_params = optax.apply_updates(adam_params, adam_updates)
        assert not state.skipped_last

    assert state.skip_count == 0
    assert state.total_steps == 4

    # Do a couple of steps that should be skipped
    skip_grads = [jnp.full_like(params, v) for v in [2.1, 3.5]]
    for grad in skip_grads:
        updates, state = update_fn(grad, state, params)
        adam_updates, adam_state = inner_opt.update(grad, adam_state, adam_params)
        params = optax.apply_updates(params, updates)
        adam_params = optax.apply_updates(adam_params, adam_updates)
        np.testing.assert_array_equal(updates, jnp.zeros_like(params))
        assert not np.array_equal(adam_updates, jnp.zeros_like(adam_params))
        assert state.skipped_last

    assert state.skip_count == 2
    assert state.total_steps == 6

    # one more step that shouldn't be skipped
    updates, state = update_fn(jnp.full_like(params, 0.1), state, params)
    assert not np.array_equal(updates, jnp.zeros_like(params))
    assert not state.skipped_last
    assert state.skip_count == 2
    assert state.total_steps == 7
