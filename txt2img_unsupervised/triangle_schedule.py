import optax


def triangle_schedule(max_lr: float, total_steps: int) -> optax.Schedule:
    """Simple linear trianguar learning rate schedule. Best schedule found in Cramming paper.
    https://arxiv.org/abs/2212.14034"""
    sched_up = optax.linear_schedule(
        init_value=0, end_value=max_lr, transition_steps=total_steps // 2
    )
    sched_down = optax.linear_schedule(
        init_value=max_lr, end_value=0, transition_steps=total_steps // 2
    )
    return optax.join_schedules([sched_up, sched_down], [total_steps // 2])
