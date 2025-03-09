"""
Infrastructure code for training models.
"""

from typing import Optional, Tuple
import pytest


def plan_steps(
    train_set_size: int,
    batch_size: int,
    epochs: int = 0,
    examples: int = 0,
    steps: int = 0,
) -> Tuple[int, int, int, int, Optional[int]]:
    """
    Plan the number of epochs and steps to train for. Given a requested number of epochs, examples,
    and steps, this function calculates the steps and epochs needed to train for the sum of all
    three. So for example you could pass epochs=0, examples=20_000, and steps=0 and if the train set
    size was 15_000, and the batch size was 10, this function would return (1500, 2000, 1, 2, 500)
    meaning you'd train for 1 full epoch and a partial epoch of 500 steps for a total of 2000 steps.

    Args:
        train_set_size: The number of training examples.
        batch_size: The number of examples per batch.
        epochs: The number of epochs to train for.
        examples: The number of examples to train for.
        steps: The number of steps to train for.

    Returns:
        A tuple of (steps_per_epoch, total_steps, complete_epochs, total_epochs, steps_in_partial_epoch)
    """
    steps_per_epoch = train_set_size // batch_size

    extra_examples = examples + steps * batch_size
    extra_steps = extra_examples // batch_size
    extra_full_epochs = extra_steps // steps_per_epoch
    extra_steps_in_partial_epoch = extra_steps % steps_per_epoch

    complete_epochs = epochs + extra_full_epochs
    total_epochs = complete_epochs + (1 if extra_steps_in_partial_epoch > 0 else 0)

    return (
        steps_per_epoch,
        complete_epochs * steps_per_epoch + extra_steps_in_partial_epoch,
        complete_epochs,
        total_epochs,
        extra_steps_in_partial_epoch if extra_steps_in_partial_epoch > 0 else None,
    )


@pytest.mark.parametrize(
    "train_set_size, batch_size, epochs, examples, steps, expected_steps_per_epoch, expected_total_steps, expected_complete_epochs, expected_total_epochs, expected_steps_in_partial_epoch",
    [
        # Example from docstring
        (15_000, 10, 0, 20_000, 0, 1500, 2000, 1, 2, 500),
        # Training for exactly one epoch
        (10_000, 32, 1, 0, 0, 312, 312, 1, 1, None),
        # Training for a specific number of steps
        (5_000, 16, 0, 0, 500, 312, 500, 1, 2, 188),
        # Combination of epochs and examples
        (8_000, 64, 2, 3_200, 0, 125, 300, 2, 3, 50),
        # Multiple epochs with no partial epoch
        (6_000, 50, 3, 0, 0, 120, 360, 3, 3, None),
        # Using all three parameters
        (4_000, 25, 1, 1_000, 20, 160, 220, 1, 2, 60),
    ],
)
def test_plan_steps(
    train_set_size,
    batch_size,
    epochs,
    examples,
    steps,
    expected_steps_per_epoch,
    expected_total_steps,
    expected_complete_epochs,
    expected_total_epochs,
    expected_steps_in_partial_epoch,
):
    """Test that plan_steps works correctly with various inputs."""
    (
        steps_per_epoch,
        total_steps,
        complete_epochs,
        total_epochs,
        steps_in_partial_epoch,
    ) = plan_steps(
        train_set_size=train_set_size,
        batch_size=batch_size,
        epochs=epochs,
        examples=examples,
        steps=steps,
    )

    assert steps_per_epoch == expected_steps_per_epoch
    assert total_steps == expected_total_steps
    assert complete_epochs == expected_complete_epochs
    assert total_epochs == expected_total_epochs
    assert steps_in_partial_epoch == expected_steps_in_partial_epoch
