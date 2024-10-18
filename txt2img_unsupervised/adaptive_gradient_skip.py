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
    quantile: float = 1.0,
) -> optax.GradientTransformation:
    """Wraps an optimizer to skip updates that exceed the specified quantile value of the past
    gradient norms by at least a factor of threshold_factor. For example if history_len is 100,
    quantile is 0.95 and threshold_factor is 1.5, any updates with norm greater than 1.5 times the
    95th percentile of the past 100 gradient norms will be skipped. No skipping occurs until
    history_len steps have been taken.

    Args:
        inner_optimizer: The optimizer to wrap.
        history_len: The number of past gradient norms to keep track of.
        threshold_factor: The factor by which the quantile of historical gradient norms is scaled to
            determine the threshold for skipping updates.
        quantile: The quantile to use for determining the threshold (default: 1.0, which is the maximum).

    Returns:
        A `GradientTransformation` object.
    """
    if history_len <= 0:
        raise ValueError("history_len must be positive")
    if threshold_factor < 1:
        raise ValueError("threshold_factor must be >= 1")
    if not 0 < quantile <= 1:
        raise ValueError("quantile must be in the range (0, 1]")

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
        quantile_norm = jnp.quantile(state.historical_norms, quantile)
        skip_threshold = quantile_norm * threshold_factor
        this_norm = optax.global_norm(updates)

        next_idx = (state.last_idx + 1) % history_len
        new_norms = state.historical_norms.at[next_idx].set(this_norm)
        state = state._replace(
            historical_norms=new_norms,
            last_idx=next_idx,
            total_steps=state.total_steps + 1,
        )

        def skip_updates():
            clipped_updates = jax.tree.map(
                lambda p: p / this_norm * skip_threshold, updates
            )
            new_updates, new_inner_state = inner_optimizer.update(
                clipped_updates, state.inner_state, params
            )
            new_state = state._replace(
                skipped_last=True,
                skip_count=state.skip_count + 1,
                inner_state=new_inner_state,
            )
            return new_updates, new_state

        def do_updates():
            new_updates, new_inner_state = inner_optimizer.update(
                updates, state.inner_state, params
            )
            new_state = state._replace(skipped_last=False, inner_state=new_inner_state)
            return new_updates, new_state

        return jax.lax.cond(
            jnp.logical_and(
                this_norm > skip_threshold, state.total_steps > history_len
            ),
            skip_updates,
            do_updates,
        )

    return optax.GradientTransformation(init_fn, update_fn)


@pytest.mark.parametrize("quantile", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("jit", [True, False])
def test_doesnt_skip_below_threshold(quantile, jit):
    """Test that the optimizer doesn't skip updates when the gradient norm is below the threshold."""
    inner_opt = optax.sgd(learning_rate=0.1)
    opt = adaptive_gradient_skip(
        inner_opt, history_len=10, threshold_factor=2.0, quantile=quantile
    )
    update_fn = jax.jit(opt.update, donate_argnums=(1, 2)) if jit else opt.update
    params = jnp.ones(10)
    state = opt.init(params)

    for _ in range(20):
        grads = jnp.full(10, 0.1)
        updates, state = update_fn(grads, state, params)
        np.testing.assert_allclose(updates, grads * -0.1)
        params = optax.apply_updates(params, updates)


@pytest.mark.parametrize("quantile", [1.0, 0.95, 0.5])
@pytest.mark.parametrize("jit", [True, False])
def test_clips_above_threshold(jit, quantile):
    """Test that the optimizer clips updates when the gradient norm is above the threshold."""
    inner_opt = optax.sgd(learning_rate=0.1)
    opt = adaptive_gradient_skip(
        inner_opt, history_len=3, threshold_factor=1.5, quantile=quantile
    )
    update_fn = jax.jit(opt.update, donate_argnums=(1, 2)) if jit else opt.update
    params = jnp.ones(10)
    state = opt.init(params)
    sgd_state = inner_opt.init(params)
    sgd_params = params

    # Do a bunch of steps that shouldn't be clipped, checking that regular sgd and the wrapped sgd
    # do the same thing.
    normal_grads = {}
    normal_grads[1.0] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 1.1]]
    normal_grads[0.95] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 1.1]]
    normal_grads[0.5] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 0.14]]
    for grad in normal_grads[quantile]:
        updates, state = update_fn(grad, state, params)
        sgd_updates, sgd_state = inner_opt.update(grad, sgd_state, sgd_params)
        np.testing.assert_allclose(updates, sgd_updates)
        params = optax.apply_updates(params, updates)
        sgd_params = optax.apply_updates(sgd_params, sgd_updates)
        assert not state.skipped_last

    assert state.skip_count == 0
    assert state.total_steps == 4

    # Do a couple of steps that should be clipped
    clip_grads = {}
    clip_grads[1.0] = [jnp.full_like(params, v) for v in [1.67, 3.5]]
    clip_grads[0.95] = [jnp.full_like(params, v) for v in [1.67, 3.5]]
    clip_grads[0.5] = [jnp.full_like(params, v) for v in [0.25, 0.22]]
    for grad in clip_grads[quantile]:
        updates, state = update_fn(grad, state, params)
        sgd_updates, sgd_state = inner_opt.update(grad, sgd_state, sgd_params)
        params = optax.apply_updates(params, updates)
        sgd_params = optax.apply_updates(sgd_params, sgd_updates)

        # Updates should be nonzero, but their norm should be clipped
        assert not np.array_equal(updates, jnp.zeros_like(params))
        assert optax.global_norm(updates) < optax.global_norm(sgd_updates)

        assert state.skipped_last

    assert state.skip_count == 2
    assert state.total_steps == 6

    # one more step that shouldn't be clipped
    updates, state = update_fn(jnp.full_like(params, 0.1), state, params)
    assert not np.array_equal(updates, jnp.zeros_like(params))
    assert not state.skipped_last
    assert state.skip_count == 2
    assert state.total_steps == 7
