"""
A version of Optax's scale_by_rms with the tweak from "ADOPT: Modified Adam Can Converge with Any β2
with the Optimal Rate", and a schedule-free version of ADOPT with decoupled weight decay. There's no
implementation of the regular scheduleful ADOPT.
Hopefully this leads to more stable training, especially at smaller batch sizes.

https://arxiv.org/abs/2411.02853
"""

import jax
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from optax.contrib import schedule_free
from typing import NamedTuple, Optional


class ScaleByAdoptRMSState(NamedTuple):
    is_first_step: bool
    nu: jax.Array


def scale_by_adopt_rms(
    decay: float = 0.9999, eps: float = 1e-6
) -> optax.GradientTransformation:
    def init_fn(params):
        return ScaleByAdoptRMSState(is_first_step=True, nu=otu.tree_zeros_like(params))

    def update_fn(updates, state, params=None):
        # See Algorithm 2, on page 16 in appendix C of the ADOPT paper.
        # The key difference from regular RMSProp is that we use the previous step's nu to scale the
        # updates, *before* updating the moving average. We also use max(sqrt(var), eps) instead of
        # sqrt(var) + eps when scaling.
        del params

        def first_step():
            # On the first step we only initialize the variance average, and don't do an update.
            nu = jax.tree.map(lambda arr: arr * arr, updates)
            return otu.tree_zeros_like(updates), ScaleByAdoptRMSState(
                is_first_step=False, nu=nu
            )

        def not_first_step():
            scaled_updates = jax.tree.map(
                lambda g, n: g / jnp.maximum(jnp.sqrt(n), eps), updates, state.nu
            )
            nu = optax.update_moment_per_elem_norm(updates, state.nu, decay, 2)
            return scaled_updates, ScaleByAdoptRMSState(is_first_step=False, nu=nu)

        return jax.lax.cond(state.is_first_step, first_step, not_first_step)

    return optax.GradientTransformation(init=init_fn, update=update_fn)


def schedule_free_adoptw(
    learning_rate: float,
    warmup_steps: Optional[int] = None,
    b1: float = 0.9,
    b2: float = 0.9999,
    eps: float = 1e-6,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """A schedule-free version of ADOPT with decoupled weight decay."""
    if warmup_steps is not None:
        learning_rate = optax.schedules.warmup_constant_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
        )
    # Momentumless ADOPT-W
    optimizer = optax.chain(
        scale_by_adopt_rms(b2, eps),
        optax.add_decayed_weights(weight_decay),
        optax.scale_by_learning_rate(learning_rate),
    )
    return schedule_free(
        optimizer, learning_rate=learning_rate, b1=b1, weight_lr_power=2
    )
