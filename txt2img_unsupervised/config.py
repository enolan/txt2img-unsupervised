"""Serializable configuration for the model and training parameters."""
import argparse
import dacite
import jax
import jax.numpy as jnp
import json
import pytest
from copy import copy
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer models."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    image_tokens: int
    use_biases: bool
    activation_function: Callable[[jax.Array], jax.Array]
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    pre_norm: bool = False
    clip_conditioning: bool = False
    clip_caps: bool = False
    clip_cap_count: int = None
    # Should always be true, defaults to false for backwards compatability
    corrected_cap_projections: bool = False
    do_clip_feedforward: bool = False
    norm_clip_embeddings: bool = False
    image_dropout: Optional[float] = None
    clip_dropout: Optional[float] = None

    @staticmethod
    def from_json_dict(dict: dict[str, Any]) -> "ModelConfig":
        """Convert a dictionary parsed from JSON to a ModelConfig object."""
        out = copy(dict)
        if "activation_function" not in dict:
            out["activation_function"] = jax.nn.relu  # make old checkpoints work
        else:
            out["activation_function"] = str_to_x_or_valueerror(
                dict["activation_function"], str_to_activation, "activation function"
            )
        out["activations_dtype"] = str_to_x_or_valueerror(
            dict["activations_dtype"], str_to_dtype, "activations dtype"
        )
        if "weights_dtype" not in dict:
            out["weights_dtype"] = jnp.float32
        else:
            out["weights_dtype"] = str_to_x_or_valueerror(
                dict["weights_dtype"], str_to_dtype, "weights dtype"
            )
        if "image_tokens" not in dict and "seq_len" in dict:
            out["image_tokens"] = dict["seq_len"]
        out = dacite.from_dict(
            data_class=ModelConfig, data=out, config=dacite.Config(check_types=False)
        )
        out.validate()
        return out

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a ModelConfig object to a dictionary that can be serialized to JSON."""
        out = copy(self.__dict__)
        out["activation_function"] = x_to_str_or_valueerror(
            self.activation_function, activation_to_str, "activation function"
        )
        out["activations_dtype"] = x_to_str_or_valueerror(
            self.activations_dtype, dtype_to_str, "dtype"
        )
        out["weights_dtype"] = x_to_str_or_valueerror(
            self.weights_dtype, dtype_to_str, "dtype"
        )
        return out

    def validate(self):
        """Validate the configuration."""
        dtypes_error = (
            "float16 and bfloat16 activations must be used with weights of the same dtype or "
            f"float32 weights, got activations in {self.activations_dtype} and weights in "
            f"{self.weights_dtype}"
        )
        if self.activations_dtype != self.weights_dtype:
            if (
                self.activations_dtype == jnp.float16
                or self.activations_dtype == jnp.bfloat16
            ):
                if self.weights_dtype != jnp.float32:
                    raise ValueError(dtypes_error)
            elif self.activations_dtype == jnp.float32:
                raise ValueError(dtypes_error)
            else:
                raise ValueError(f"Unknown activations_dtype {self.activations_dtype}")


def invert_dict(d: dict[Any, Any]) -> dict[Any, Any]:
    """Invert a dictionary."""
    return {v: k for k, v in d.items()}


str_to_dtype: dict[str, jnp.dtype] = {
    "float32": jnp.float32,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
}

dtype_to_str: dict[jnp.dtype, str] = invert_dict(str_to_dtype)

str_to_activation: dict[str, Callable[[jax.Array], jax.Array]] = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
}

activation_to_str: dict[Callable[[jax.Array], jax.Array], str] = invert_dict(
    str_to_activation
)


def str_to_x_or_valueerror(x_str: str, d: dict[str, Any], tyname: str) -> Any:
    """Convert a string to a value in a dictionary or raise a ValueError."""
    if x_str in d:
        return d[x_str]
    else:
        raise ValueError(f"Unknown {tyname} {x_str}")


def x_to_str_or_valueerror(x: Any, d: dict[Any, str], tyname: str) -> str:
    """Convert a value to a string in a dictionary or raise a ValueError."""
    if x in d:
        return d[x]
    else:
        raise ValueError(f"Unknown {tyname} {x}")


def test_modelconfig_roundtrip_from_json() -> None:
    """Test that converting from json and back is the identity."""
    json_str = """{
        "d_model": 512,
        "num_heads": 8,
        "ff_dim": 2048,
        "dropout": 0.1,
        "image_dropout": 0.5,
        "clip_dropout": 0.5,
        "n_layers": 6,
        "image_tokens": 2048,
        "use_biases": true,
        "activations_dtype": "float32",
        "activation_function": "relu",
        "weights_dtype": "float32",
        "pre_norm": false,
        "clip_cap_count": null,
        "clip_caps": false,
        "clip_conditioning": false,
        "corrected_cap_projections": true,
        "do_clip_feedforward": false,
        "norm_clip_embeddings": false
        }"""
    cfg = ModelConfig.from_json_dict(json.loads(json_str))
    assert ModelConfig.to_json_dict(cfg) == json.loads(json_str)


def test_modelconfig_roundtrip_from_object() -> None:
    """Test that converting from a ModelConfig object and back is the identity."""
    cfg = ModelConfig(
        d_model=420_69,
        num_heads=8,
        ff_dim=2048,
        dropout=0.1,
        n_layers=6,
        image_tokens=42,
        use_biases=True,
        activations_dtype=jnp.bfloat16,
        activation_function=jax.nn.gelu,
    )
    assert ModelConfig.from_json_dict(ModelConfig.to_json_dict(cfg)) == cfg


class LearningRateSchedule(Enum):
    CONSTANT = "constant"  # Constant learning rate
    TRIANGLE = (
        "triangle"  # Linear warmup to peak at halfway through, then linear decay to 0
    )
    WARMUP_PLUS_COSINE = (
        "warmup_plus_cosine"  # Linear warmup for a fixed # of steps, then cosine decay.
    )
    WARMUP_PLUS_SCHEDULE_FREE = "warmup_plus_schedule_free"  # Linear warmup for a fixed # of steps, then schedule-free Adam.


str_to_learning_rate_schedule = {
    "constant": LearningRateSchedule.CONSTANT,
    "triangle": LearningRateSchedule.TRIANGLE,
    "warmup_plus_cosine": LearningRateSchedule.WARMUP_PLUS_COSINE,
    "warmup_plus_schedule_free": LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
}


learning_rate_schedule_to_str = invert_dict(str_to_learning_rate_schedule)


def remove_nones_from_dict(d):
    """Remove None values from a dictionary, so dataclasses with optional fields get serialized as
    JSON objects without null values."""
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class TrainingConfig:
    learning_rate: float  # peak learning rate
    batch_size: int
    epochs: int  # How many epochs to train for
    learning_rate_schedule: LearningRateSchedule
    gradient_accumulation_steps: int
    gradient_clipping: Optional[float]
    # How many steps to linearly increase the learning rate when using WARMUP_PLUS_COSINE_LR. With
    # the other schedules this value must be None
    warmup_steps: Optional[int] = None
    schedule_free_beta1: Optional[float] = None
    weight_decay: float = 0.0
    training_images: int = 0  # How many images to train for (in addition to epochs)
    adaptive_gradient_skip: bool = False
    adaptive_gradient_skip_history_len: Optional[int] = None
    adaptive_gradient_skip_threshold_factor: Optional[float] = None
    adaptive_gradient_skip_quantile: Optional[float] = None

    @staticmethod
    def from_json_dict(dict: dict[str, Any]) -> "TrainingConfig":
        """Convert a dictionary parsed from JSON to a TrainingConfig object."""
        dict = copy(dict)
        dict["learning_rate_schedule"] = str_to_x_or_valueerror(
            dict["learning_rate_schedule"],
            str_to_learning_rate_schedule,
            "learning rate schedule",
        )
        config = dacite.from_dict(data_class=TrainingConfig, data=dict)
        config.validate()
        return config

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a TrainingConfig object to a dictionary that can be serialized to JSON."""
        dict = remove_nones_from_dict(copy(self.__dict__))
        dict["learning_rate_schedule"] = self.learning_rate_schedule.value
        return dict

    def validate(self):
        """Validate the configuration."""
        if self.adaptive_gradient_skip:
            if (
                self.adaptive_gradient_skip_history_len is None
                or self.adaptive_gradient_skip_threshold_factor is None
                or self.adaptive_gradient_skip_quantile is None
            ):
                raise ValueError(
                    "adaptive_gradient_skip_history_len, adaptive_gradient_skip_threshold_factor, "
                    "and adaptive_gradient_skip_quantile must be set when adaptive_gradient_skip "
                    "is enabled"
                )
        else:
            if (
                self.adaptive_gradient_skip_history_len is not None
                or self.adaptive_gradient_skip_threshold_factor is not None
                or self.adaptive_gradient_skip_quantile is not None
            ):
                raise ValueError(
                    "adaptive_gradient_skip_history_len, adaptive_gradient_skip_threshold_factor, "
                    "and adaptive_gradient_skip_quantile should not be set when "
                    "adaptive_gradient_skip is disabled"
                )

        def get_schedule_error_message(schedule, warmup_required, beta1_required):
            warmup_state = "set" if warmup_required else "unset"
            beta1_state = "set" if beta1_required else "unset"
            return f"{schedule} schedule requires warmup_steps to be {warmup_state} and schedule_free_beta1 to be {beta1_state}"

        if self.learning_rate_schedule == LearningRateSchedule.CONSTANT:
            if self.warmup_steps is not None or self.schedule_free_beta1 is not None:
                raise ValueError(get_schedule_error_message("constant", False, False))
        elif self.learning_rate_schedule == LearningRateSchedule.TRIANGLE:
            if self.warmup_steps is not None or self.schedule_free_beta1 is not None:
                raise ValueError(get_schedule_error_message("triangle", False, False))
        elif self.learning_rate_schedule == LearningRateSchedule.WARMUP_PLUS_COSINE:
            if self.warmup_steps is None or self.schedule_free_beta1 is not None:
                raise ValueError(
                    get_schedule_error_message("warmup plus cosine", True, False)
                )
        elif (
            self.learning_rate_schedule
            == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
        ):
            if self.warmup_steps is None or self.schedule_free_beta1 is None:
                raise ValueError(
                    get_schedule_error_message("warmup plus schedule-free", True, True)
                )
        else:
            raise ValueError(
                f"Unknown learning rate schedule {self.learning_rate_schedule}"
            )


def merge_attrs(data_class, args: Any) -> None:
    """Merge arguments from argparse/wandb.config into a configuration."""
    for attrname in data_class.__dict__:
        if hasattr(args, attrname) and getattr(args, attrname) is not None:
            setattr(data_class, attrname, getattr(args, attrname))


def test_trainingconfig_merge_argparse() -> None:
    """Test that merging arguments from argparse works."""
    cfg = TrainingConfig(
        learning_rate=1,
        batch_size=4096,
        epochs=1,
        learning_rate_schedule=LearningRateSchedule.TRIANGLE,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
    )
    args = argparse.Namespace(
        learning_rate=2,
        batch_size=None,
        epochs=2,
        triangle_schedule=None,
        gradient_accumulation_steps=None,
        gradient_clipping=None,
    )
    merge_attrs(cfg, args)
    assert cfg == TrainingConfig(
        learning_rate=2,
        batch_size=4096,
        epochs=2,
        learning_rate_schedule=LearningRateSchedule.TRIANGLE,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
    )


_test_json_strs = [
    """{
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 100,
        "training_images": 0,
        "learning_rate_schedule": "triangle",
        "gradient_accumulation_steps": 1,
        "adaptive_gradient_skip": true,
        "adaptive_gradient_skip_history_len": 100,
        "adaptive_gradient_skip_threshold_factor": 1.1,
        "adaptive_gradient_skip_quantile": 0.95,
        "weight_decay": 0.1
        }""",
    """{
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 100,
        "training_images": 0,
        "learning_rate_schedule": "warmup_plus_schedule_free",
        "warmup_steps": 100,
        "schedule_free_beta1": 0.9,
        "gradient_accumulation_steps": 1,
        "adaptive_gradient_skip": false,
        "weight_decay": 0.0
        }""",
]


@pytest.mark.parametrize("json_str", _test_json_strs)
def test_trainingconfig_roundtrip_from_json(json_str: str) -> None:
    """Test that converting from json and back is the identity."""
    cfg = TrainingConfig.from_json_dict(json.loads(json_str))
    assert TrainingConfig.to_json_dict(cfg) == json.loads(json_str)


def test_trainingconfig_roundtrip_from_object() -> None:
    """Test that converting from a TrainingConfig object and back is the identity."""
    cfg = TrainingConfig(
        learning_rate=1,
        batch_size=4096,
        epochs=1,
        learning_rate_schedule=LearningRateSchedule.TRIANGLE,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
    )
    assert TrainingConfig.from_json_dict(TrainingConfig.to_json_dict(cfg)) == cfg
