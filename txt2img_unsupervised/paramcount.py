import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import Any, Dict, Union
from math import prod
from dataclasses import replace


@partial(jax.jit, static_argnames=["module"])
def count_parameters(module: nn.Module, *dummy_args, **dummy_kwargs) -> int:
    """Count total parameters in a Flax module using lazy initialization.

    Args:
        module: The Flax module to count parameters for
        *dummy_args: Positional arguments for the module (as ShapeDtypeStruct)
        **dummy_kwargs: Keyword arguments for the module (as ShapeDtypeStruct)

    Returns:
        Total number of parameters as an integer
    """
    # Use lazy_init to get parameter structure without allocating arrays
    variables = module.lazy_init(jax.random.PRNGKey(0), *dummy_args, **dummy_kwargs)

    # Extract only the parameters (ignore other variable collections like batch stats)
    params = variables.get("params", variables)

    # Count parameters using concrete shapes with regular Python operations
    def count_single_param(x):
        return prod(x.shape)

    param_counts = jax.tree.map(
        count_single_param, params, is_leaf=lambda x: hasattr(x, "shape")
    )
    total_params = jax.tree.reduce(lambda x, y: x + y, param_counts, 0)
    return total_params


def find_optimal_attribute_value(
    module: nn.Module,
    target_param_count: int,
    attribute_name: str,
    *dummy_args,
    **dummy_kwargs,
) -> Union[int, float]:
    """Find the optimal value for a module attribute to get close to target parameter count.

    Uses binary search to efficiently find the attribute value that results in a module
    with parameter count closest to the target.

    Args:
        module: The Flax module to optimize
        target_param_count: Target number of parameters
        attribute_name: Name of the attribute to optimize (e.g., 'd_model', 'n_layers')
        *dummy_args: Arguments to pass to count_parameters
        **dummy_kwargs: Keyword arguments to pass to count_parameters

    Returns:
        The optimal value for the specified attribute

    Raises:
        ValueError: If the attribute doesn't exist or has invalid value
    """
    # Get current value and determine type
    if not hasattr(module, attribute_name):
        raise ValueError(f"Module does not have attribute '{attribute_name}'")

    current_value = getattr(module, attribute_name)
    is_integer = isinstance(current_value, int)

    if current_value <= 0:
        raise ValueError(
            f"Attribute {attribute_name} must be positive, got {current_value}"
        )

    # Get current parameter count to determine search direction
    current_params = count_parameters(module, *dummy_args, **dummy_kwargs)

    # Find search bounds using exponential search
    if current_params < target_param_count:
        # Need to increase the attribute value
        lower_bound = current_value
        upper_bound = current_value

        # Exponentially increase upper bound until we overshoot target
        while True:
            upper_bound = upper_bound * 2
            test_module = replace(module, **{attribute_name: upper_bound})
            param_count = count_parameters(test_module, *dummy_args, **dummy_kwargs)
            if param_count >= target_param_count:
                break
    else:
        # Need to decrease the attribute value
        upper_bound = current_value
        lower_bound = current_value

        # Exponentially decrease lower bound until we undershoot target
        while lower_bound > 1:
            lower_bound = max(1, lower_bound // 2 if is_integer else lower_bound / 2)
            test_module = replace(module, **{attribute_name: lower_bound})
            param_count = count_parameters(test_module, *dummy_args, **dummy_kwargs)
            if param_count <= target_param_count:
                break

    # Binary search between bounds
    best_value = current_value
    best_diff = abs(current_params - target_param_count)

    max_iterations = 50  # Prevent infinite loops
    for _ in range(max_iterations):
        if is_integer:
            if upper_bound - lower_bound <= 1:
                break
            mid = (lower_bound + upper_bound) // 2
        else:
            if abs(upper_bound - lower_bound) < 1e-6:
                break
            mid = (lower_bound + upper_bound) / 2

        # Test this value
        test_module = replace(module, **{attribute_name: mid})
        param_count = count_parameters(test_module, *dummy_args, **dummy_kwargs)
        diff = abs(param_count - target_param_count)

        # Update best if this is closer
        if diff < best_diff:
            best_diff = diff
            best_value = mid

        # Adjust search bounds
        if param_count < target_param_count:
            lower_bound = mid + (1 if is_integer else 1e-9)
        else:
            upper_bound = mid - (1 if is_integer else 1e-9)

    return int(best_value) if is_integer else best_value
