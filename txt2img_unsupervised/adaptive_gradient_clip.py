"""
Optax optimizer wrapper that clips batches where gradient norm exceeds a threshold based on recent
history. The intent is to clip outlier batches that occur very occasionally, which have huge norms
and break or set back learning. Won't help if your training is unstable for reasons that wouldn't be
fixed with a larger batch size.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from typing import NamedTuple


class AdaptiveGradientClipState(NamedTuple):
    clip_count: int
    """The number of updates that have been clipped."""
    clipped_last: bool
    """Whether the last update was clipped."""
    inner_state: optax.OptState
    """The state of the inner optimizer."""
    total_steps: int
    """The total number of steps taken."""
    historical_norms: jax.Array
    last_idx: int


def adaptive_gradient_clip(
    inner_optimizer: optax.GradientTransformation,
    history_len: int = 100,
    threshold_factor: float = 1.5,
    quantile: float = 1.0,
) -> optax.GradientTransformation:
    """Wraps an optimizer to clip updates that exceed the specified quantile value of the past
    gradient norms by at least a factor of threshold_factor. For example if history_len is 100,
    quantile is 0.95 and threshold_factor is 1.5, any updates with norm greater than 1.5 times the
    95th percentile of the past 100 gradient norms will be clipped to that value. No clipping occurs
    until history_len steps have been taken.

    Args:
        inner_optimizer: The optimizer to wrap.
        history_len: The number of past gradient norms to keep track of.
        threshold_factor: The factor by which the quantile of historical gradient norms is scaled to
            determine the threshold for clipping updates.
        quantile: The quantile to use for determining the threshold (default: 1.0, which is the
            maximum).

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
        return AdaptiveGradientClipState(
            historical_norms=jnp.zeros(history_len),
            clip_count=0,
            clipped_last=False,
            last_idx=-1,
            inner_state=inner_optimizer.init(params),
            total_steps=0,
        )

    def update_fn(updates, state, params=None):
        quantile_norm = jnp.quantile(state.historical_norms, quantile)
        clip_threshold = quantile_norm * threshold_factor
        this_norm = optax.global_norm(updates)

        next_idx = (state.last_idx + 1) % history_len
        new_norms = state.historical_norms.at[next_idx].set(this_norm)
        state = state._replace(
            historical_norms=new_norms,
            last_idx=next_idx,
            total_steps=state.total_steps + 1,
        )

        def clip_grads():
            scale = clip_threshold / this_norm
            new_state = state._replace(
                clipped_last=True, clip_count=state.clip_count + 1
            )
            return jax.tree.map(lambda x: x * scale, updates), new_state

        clip = jnp.logical_and(
            this_norm > clip_threshold, state.total_steps > history_len
        )
        updates, state = jax.lax.cond(
            clip,
            clip_grads,
            lambda: (updates, state._replace(clipped_last=False)),
        )

        new_updates, new_inner_state = inner_optimizer.update(
            updates, state.inner_state, params
        )
        state = state._replace(inner_state=new_inner_state)

        return new_updates, state

    return optax.GradientTransformation(init_fn, update_fn)


@pytest.mark.parametrize("quantile", [0.5, 0.9, 1.0])
@pytest.mark.parametrize("jit", [True, False])
def test_doesnt_clip_below_threshold(quantile, jit):
    """Test that the optimizer doesn't clip updates when the gradient norm is below the threshold."""
    inner_opt = optax.sgd(learning_rate=0.1)
    opt = adaptive_gradient_clip(
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
    inner_opt = optax.adam(learning_rate=0.1)
    opt = adaptive_gradient_clip(
        inner_opt, history_len=3, threshold_factor=1.5, quantile=quantile
    )
    update_fn = jax.jit(opt.update, donate_argnums=(1, 2)) if jit else opt.update
    params = jnp.ones(10)
    state = opt.init(params)
    adam_state = inner_opt.init(params)
    adam_params = params

    # Do a bunch of steps that shouldn't be clipped, checking that regular adam and the wrapped adam
    # do the same thing.
    normal_grads = {}
    normal_grads[1.0] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 1.1]]
    normal_grads[0.95] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 1.1]]
    normal_grads[0.5] = [jnp.full_like(params, v) for v in [0.1, 1.1, 0.1, 0.14]]
    for grad in normal_grads[quantile]:
        updates, state = update_fn(grad, state, params)
        adam_updates, adam_state = inner_opt.update(grad, adam_state, adam_params)
        np.testing.assert_allclose(updates, adam_updates)
        params = optax.apply_updates(params, updates)
        adam_params = optax.apply_updates(adam_params, adam_updates)
        assert not state.clipped_last

    assert state.clip_count == 0
    assert state.total_steps == 4

    # Do a couple of steps that should be clipped
    clip_grads = {}
    clip_grads[1.0] = [jnp.full_like(params, v) for v in [1.67, 3.5]]
    clip_grads[0.95] = [jnp.full_like(params, v) for v in [1.67, 3.5]]
    clip_grads[0.5] = [jnp.full_like(params, v) for v in [0.25, 0.22]]
    for grad in clip_grads[quantile]:
        updates, state = update_fn(grad, state, params)
        adam_updates, adam_state = inner_opt.update(grad, adam_state, adam_params)
        params = optax.apply_updates(params, updates)
        adam_params = optax.apply_updates(adam_params, adam_updates)
        np.testing.assert_array_less(jnp.abs(updates), jnp.abs(params))
        assert state.clipped_last

    assert state.clip_count == 2
    assert state.total_steps == 6
