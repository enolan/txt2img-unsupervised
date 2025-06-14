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
from typing import Any, Callable, ClassVar, Dict, Optional, Type


class BaseModelConfig:
    """Base interface for all model configs."""

    model_type: ClassVar[str] = "base"

    @classmethod
    def from_json_dict(cls, dict: dict[str, Any]) -> "BaseModelConfig":
        """Convert a dictionary parsed from JSON to a ModelConfig object."""
        # Determine which subclass to use based on model_type in the dictionary
        if "model_type" in dict:
            model_type = dict["model_type"]
            subclass_map = {
                "transformer": TransformerModelConfig,
                "flow_matching": FlowMatchingModelConfig,
            }

            if model_type in subclass_map:
                return subclass_map[model_type].from_json_dict(dict)

        # For backward compatibility - if no model_type, assume transformer
        if any(k in dict for k in ["image_tokens", "num_heads", "ff_dim"]):
            return TransformerModelConfig.from_json_dict(dict)

        # If it has flow matching specific fields
        if any(
            k in dict
            for k in ["domain_dim", "reference_directions", "conditioning_dim"]
        ):
            return FlowMatchingModelConfig.from_json_dict(dict)

        # Otherwise, we can't determine the type
        raise ValueError("Could not determine model type from config")

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a ModelConfig object to a dictionary that can be serialized to JSON."""
        raise NotImplementedError("Subclasses must implement this method")

    def validate(self):
        """Validate the configuration."""
        # To be implemented by subclasses
        pass


@dataclass
class TransformerModelConfig(BaseModelConfig):
    """Configuration for transformer models."""

    # Required parameters first
    n_layers: int
    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    image_tokens: int
    use_biases: bool
    activation_function: Callable[[jax.Array], jax.Array]

    # Optional parameters with defaults
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    pre_norm: bool = False
    clip_conditioning: bool = False
    clip_caps: bool = False
    clip_cap_count: Optional[int] = None
    # Should always be true, defaults to false for backwards compatability
    corrected_cap_projections: bool = False
    do_clip_feedforward: bool = False
    norm_clip_embeddings: bool = False
    image_dropout: Optional[float] = None
    clip_dropout: Optional[float] = None

    # Class variable to store the model type
    model_type: ClassVar[str] = "transformer"

    @classmethod
    def from_json_dict(cls, dict: dict[str, Any]) -> "TransformerModelConfig":
        """Convert a dictionary parsed from JSON to a TransformerModelConfig object."""
        out = copy(dict)
        if "activation_function" not in dict:
            out["activation_function"] = jax.nn.relu  # make old checkpoints work
        else:
            out["activation_function"] = str_to_x_or_valueerror(
                dict["activation_function"], str_to_activation, "activation function"
            )

        if "activations_dtype" in dict:
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
            data_class=cls, data=out, config=dacite.Config(check_types=False)
        )
        out.validate()
        return out

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a TransformerModelConfig object to a dictionary that can be serialized to JSON."""
        out = copy(self.__dict__)

        # Add model type to identify this config
        out["model_type"] = self.model_type

        # Convert dtypes to strings
        out["activations_dtype"] = x_to_str_or_valueerror(
            self.activations_dtype, dtype_to_str, "dtype"
        )
        out["weights_dtype"] = x_to_str_or_valueerror(
            self.weights_dtype, dtype_to_str, "dtype"
        )

        # Convert activation function to string
        out["activation_function"] = x_to_str_or_valueerror(
            self.activation_function, activation_to_str, "activation function"
        )

        return out

    def validate(self):
        """Validate the configuration."""
        # Common validation for transformer models
        if self.activations_dtype == jnp.float32 and self.weights_dtype != jnp.float32:
            raise ValueError(
                "It doesn't make sense to use float32 activations with float16 or bfloat16 weights"
            )


@dataclass
class FlowMatchingModelConfig(BaseModelConfig):
    """Configuration for flow matching models."""

    # Required parameters first
    n_layers: int
    domain_dim: int
    reference_directions: Optional[int]
    conditioning_dim: Optional[int]
    time_dim: int
    use_pre_mlp_projection: bool
    d_model: int
    mlp_expansion_factor: int
    mlp_dropout_rate: Optional[float]
    input_dropout_rate: Optional[float]

    # Optional parameters with defaults
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    d_model_base: int = 512
    variance_base: float = 1 / 512
    alpha_input: float = 1.0
    alpha_output: float = 1.0

    # Class variable to store the model type
    model_type: ClassVar[str] = "flow_matching"

    @classmethod
    def from_json_dict(cls, dict: dict[str, Any]) -> "FlowMatchingModelConfig":
        """Convert a dictionary parsed from JSON to a FlowMatchingModelConfig object."""
        out = copy(dict)

        if "activations_dtype" in dict:
            out["activations_dtype"] = str_to_x_or_valueerror(
                dict["activations_dtype"], str_to_dtype, "activations dtype"
            )

        if "weights_dtype" not in dict:
            out["weights_dtype"] = jnp.float32
        else:
            out["weights_dtype"] = str_to_x_or_valueerror(
                dict["weights_dtype"], str_to_dtype, "weights dtype"
            )

        out = dacite.from_dict(
            data_class=cls, data=out, config=dacite.Config(check_types=False)
        )
        out.validate()
        return out

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a FlowMatchingModelConfig object to a dictionary that can be serialized to JSON."""
        out = copy(self.__dict__)

        # Add model type to identify this config
        out["model_type"] = self.model_type

        # Convert dtypes to strings
        out["activations_dtype"] = x_to_str_or_valueerror(
            self.activations_dtype, dtype_to_str, "dtype"
        )
        out["weights_dtype"] = x_to_str_or_valueerror(
            self.weights_dtype, dtype_to_str, "dtype"
        )

        return out

    def validate(self):
        """Validate flow matching specific configuration."""
        # Common validation for flow matching models
        if self.activations_dtype == jnp.float32 and self.weights_dtype != jnp.float32:
            raise ValueError(
                "It doesn't make sense to use float32 activations with float16 or bfloat16 weights"
            )

        # Flow matching specific validation
        if not self.use_pre_mlp_projection:
            total_input_dim = (
                (
                    self.reference_directions
                    if self.reference_directions is not None
                    else self.domain_dim
                )
                + self.conditioning_dim
                + self.time_dim
            )
            if total_input_dim > self.d_model:
                raise ValueError(
                    f"Input dimensions ({total_input_dim}) exceed d_model ({self.d_model}). "
                    f"Increase d_model or reduce input dimensions."
                )

        if self.time_dim % 2 != 0:
            raise ValueError("Time dimension must be even")


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


def test_transformermodelconfig_roundtrip_from_json() -> None:
    """Test that converting TransformerModelConfig from json and back is the identity."""
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
        "norm_clip_embeddings": false,
        "model_type": "transformer"
        }"""
    cfg = TransformerModelConfig.from_json_dict(json.loads(json_str))
    assert TransformerModelConfig.to_json_dict(cfg) == json.loads(json_str)


def test_transformermodelconfig_roundtrip_from_object() -> None:
    """Test that converting from a TransformerModelConfig object and back is the identity."""
    cfg = TransformerModelConfig(
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
    assert (
        TransformerModelConfig.from_json_dict(TransformerModelConfig.to_json_dict(cfg))
        == cfg
    )


class LearningRateSchedule(Enum):
    CONSTANT = "constant"  # Constant learning rate
    TRIANGLE = (
        "triangle"  # Linear warmup to peak at halfway through, then linear decay to 0
    )
    WARMUP_PLUS_COSINE = (
        "warmup_plus_cosine"  # Linear warmup for a fixed # of steps, then cosine decay.
    )
    WARMUP_PLUS_SCHEDULE_FREE = "warmup_plus_schedule_free"  # Linear warmup for a fixed # of steps, then schedule-free Adam.
    CONSTANT_PLUS_LINEAR_DECAY = "constant_plus_linear_decay"  # Constant learning rate for a fixed # of steps, then linear decay to 0.


str_to_learning_rate_schedule = {
    "constant": LearningRateSchedule.CONSTANT,
    "triangle": LearningRateSchedule.TRIANGLE,
    "warmup_plus_cosine": LearningRateSchedule.WARMUP_PLUS_COSINE,
    "warmup_plus_schedule_free": LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
    "constant_plus_linear_decay": LearningRateSchedule.CONSTANT_PLUS_LINEAR_DECAY,
}


learning_rate_schedule_to_str = invert_dict(str_to_learning_rate_schedule)


def remove_nones_from_dict(d):
    """Remove None values from a dictionary, so dataclasses with optional fields get serialized as
    JSON objects without null values."""
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int  # How many epochs to train for
    learning_rate_schedule: LearningRateSchedule
    gradient_accumulation_steps: int
    gradient_clipping: Optional[float]
    learning_rate: Optional[float] = None  # peak learning rate when using single lr
    # How many steps to linearly increase the learning rate when using WARMUP_PLUS_COSINE_LR. With
    # the other schedules this value must be None
    warmup_steps: Optional[int] = None
    # How many steps to decay the learning rate over when using CONSTANT_PLUS_LINEAR_DECAY. With
    # the other schedules this value must be None
    decay_steps: Optional[int] = None
    schedule_free_beta1: Optional[float] = None
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    training_images: int = 0  # How many images to train for (in addition to epochs)
    adaptive_gradient_clip: bool = False
    adaptive_gradient_clip_history_len: Optional[int] = None
    adaptive_gradient_clip_threshold_factor: Optional[float] = None
    adaptive_gradient_clip_quantile: Optional[float] = None
    # Muon optimizer settings
    use_muon: bool = False
    muon_beta: float = 0.95  # Momentum parameter for Muon optimizer
    muon_learning_rate: Optional[
        float
    ] = None  # Learning rate for Muon parameters (if None, uses learning_rate)
    adam_learning_rate: Optional[
        float
    ] = None  # Learning rate for Adam parameters when using Muon (if None, uses learning_rate)

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
        if self.adaptive_gradient_clip:
            if (
                self.adaptive_gradient_clip_history_len is None
                or self.adaptive_gradient_clip_threshold_factor is None
                or self.adaptive_gradient_clip_quantile is None
            ):
                raise ValueError(
                    "adaptive_gradient_clip_history_len, adaptive_gradient_clip_threshold_factor, "
                    "and adaptive_gradient_clip_quantile must be set when adaptive_gradient_clip "
                    "is enabled"
                )
        else:
            if (
                self.adaptive_gradient_clip_history_len is not None
                or self.adaptive_gradient_clip_threshold_factor is not None
                or self.adaptive_gradient_clip_quantile is not None
            ):
                raise ValueError(
                    "adaptive_gradient_clip_history_len, adaptive_gradient_clip_threshold_factor, "
                    "and adaptive_gradient_clip_quantile should not be set when "
                    "adaptive_gradient_clip is disabled"
                )

        def get_schedule_error_message(
            schedule, warmup_required, beta1_required, decay_required=False
        ):
            warmup_state = "set" if warmup_required else "unset"
            beta1_state = "set" if beta1_required else "unset"
            decay_state = "set" if decay_required else "unset"
            return f"{schedule} schedule requires warmup_steps to be {warmup_state}, schedule_free_beta1 to be {beta1_state}, and decay_steps to be {decay_state}"

        if self.learning_rate_schedule == LearningRateSchedule.CONSTANT:
            if (
                self.warmup_steps is not None
                or self.schedule_free_beta1 is not None
                or self.decay_steps is not None
            ):
                raise ValueError(
                    get_schedule_error_message("constant", False, False, False)
                )
        elif self.learning_rate_schedule == LearningRateSchedule.TRIANGLE:
            if (
                self.warmup_steps is not None
                or self.schedule_free_beta1 is not None
                or self.decay_steps is not None
            ):
                raise ValueError(
                    get_schedule_error_message("triangle", False, False, False)
                )
        elif self.learning_rate_schedule == LearningRateSchedule.WARMUP_PLUS_COSINE:
            if (
                self.warmup_steps is None
                or self.schedule_free_beta1 is not None
                or self.decay_steps is not None
            ):
                raise ValueError(
                    get_schedule_error_message("warmup plus cosine", True, False, False)
                )
        elif (
            self.learning_rate_schedule
            == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
        ):
            if (
                self.warmup_steps is None
                or self.schedule_free_beta1 is None
                or self.decay_steps is not None
            ):
                raise ValueError(
                    get_schedule_error_message(
                        "warmup plus schedule-free", True, True, False
                    )
                )
        elif (
            self.learning_rate_schedule
            == LearningRateSchedule.CONSTANT_PLUS_LINEAR_DECAY
        ):
            if (
                self.decay_steps is None
                or self.warmup_steps is not None
                or self.schedule_free_beta1 is not None
            ):
                raise ValueError(
                    get_schedule_error_message(
                        "constant plus linear decay", False, False, True
                    )
                )
        else:
            raise ValueError(
                f"Unknown learning rate schedule {self.learning_rate_schedule}"
            )

        # Muon validation
        if self.use_muon:
            if (
                self.learning_rate_schedule
                == LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE
            ):
                raise ValueError(
                    "Muon optimizer is not compatible with schedule-free optimizers. "
                    "Use a different learning rate schedule when use_muon=True."
                )
            if self.muon_beta <= 0 or self.muon_beta >= 1:
                raise ValueError(
                    f"muon_beta must be between 0 and 1, got {self.muon_beta}"
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
        "adaptive_gradient_clip": true,
        "adaptive_gradient_clip_history_len": 100,
        "adaptive_gradient_clip_threshold_factor": 1.1,
        "adaptive_gradient_clip_quantile": 0.95,
        "weight_decay": 0.1,
        "adam_beta2": 0.999,
        "use_muon": false,
        "muon_beta": 0.95
        }""",
    """{
        "batch_size": 4,
        "epochs": 100,
        "training_images": 0,
        "learning_rate_schedule": "warmup_plus_schedule_free",
        "warmup_steps": 100,
        "schedule_free_beta1": 0.9,
        "gradient_accumulation_steps": 1,
        "adaptive_gradient_clip": false,
        "weight_decay": 0.0,
        "adam_beta2": 0.9,
        "use_muon": false,
        "muon_beta": 0.95
        }""",
    """{
        "batch_size": 4,
        "epochs": 100,
        "training_images": 0,
        "learning_rate_schedule": "constant",
        "gradient_accumulation_steps": 1,
        "adaptive_gradient_clip": false,
        "weight_decay": 0.0,
        "adam_beta2": 0.999,
        "use_muon": true,
        "muon_beta": 0.95,
        "muon_learning_rate": 1e-2,
        "adam_learning_rate": 1e-3
        }""",
    """{
        "batch_size": 8,
        "epochs": 50,
        "training_images": 0,
        "learning_rate_schedule": "constant_plus_linear_decay",
        "decay_steps": 1000,
        "gradient_accumulation_steps": 1,
        "adaptive_gradient_clip": false,
        "weight_decay": 0.0,
        "adam_beta2": 0.999,
        "use_muon": false,
        "muon_beta": 0.95
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


def test_trainingconfig_muon_validation():
    """Test Muon-specific validation in TrainingConfig."""
    # Test valid Muon configuration
    valid_muon_cfg = TrainingConfig(
        learning_rate=1e-3,
        batch_size=32,
        epochs=1,
        learning_rate_schedule=LearningRateSchedule.CONSTANT,
        gradient_accumulation_steps=1,
        gradient_clipping=None,
        use_muon=True,
        muon_beta=0.95,
        muon_learning_rate=2e-3,
        adam_learning_rate=1e-3,
    )
    # Should not raise
    valid_muon_cfg.validate()

    # Test Muon with schedule-free (should fail)
    with pytest.raises(
        ValueError,
        match="Muon optimizer is not compatible with schedule-free optimizers",
    ):
        invalid_cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            epochs=1,
            learning_rate_schedule=LearningRateSchedule.WARMUP_PLUS_SCHEDULE_FREE,
            warmup_steps=100,
            schedule_free_beta1=0.9,
            gradient_accumulation_steps=1,
            gradient_clipping=None,
            use_muon=True,
            muon_beta=0.95,
        )
        invalid_cfg.validate()

    # Test invalid muon_beta values
    with pytest.raises(ValueError, match="muon_beta must be between 0 and 1"):
        invalid_beta_cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            epochs=1,
            learning_rate_schedule=LearningRateSchedule.CONSTANT,
            gradient_accumulation_steps=1,
            gradient_clipping=None,
            use_muon=True,
            muon_beta=1.5,  # Invalid: > 1
        )
        invalid_beta_cfg.validate()

    with pytest.raises(ValueError, match="muon_beta must be between 0 and 1"):
        invalid_beta_cfg = TrainingConfig(
            learning_rate=1e-3,
            batch_size=32,
            epochs=1,
            learning_rate_schedule=LearningRateSchedule.CONSTANT,
            gradient_accumulation_steps=1,
            gradient_clipping=None,
            use_muon=True,
            muon_beta=0.0,  # Invalid: = 0
        )
        invalid_beta_cfg.validate()
