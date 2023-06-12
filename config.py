"""Serializable configuration for the model and training parameters."""
import argparse
import dacite
import jax.numpy as jnp
import json
from copy import copy
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ModelConfig:
    """Configuration for the transformer models."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    seq_len: int
    use_biases: bool
    activations_dtype: jnp.dtype = jnp.float32

    @staticmethod
    def from_json_dict(dict: dict[str, Any]) -> "ModelConfig":
        """Convert a dictionary parsed from JSON to a ModelConfig object."""
        out = copy(dict)
        if dict["activations_dtype"] == "float32":
            out["activations_dtype"] = jnp.float32
        elif dict["activations_dtype"] == "float16":
            out["activations_dtype"] = jnp.float16
        elif dict["activations_dtype"] == "bfloat16":
            out["activations_dtype"] = jnp.bfloat16
        else:
            raise ValueError(f"Unknown dtype {dict['activations_dtype']}")
        return dacite.from_dict(
            data_class=ModelConfig, data=out, config=dacite.Config(check_types=False)
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a ModelConfig object to a dictionary that can be serialized to JSON."""
        out = copy(self.__dict__)
        if self.activations_dtype == jnp.float32:
            out["activations_dtype"] = "float32"
        elif self.activations_dtype == jnp.float16:
            out["activations_dtype"] = "float16"
        elif self.activations_dtype == jnp.bfloat16:
            out["activations_dtype"] = "bfloat16"
        else:
            raise ValueError(f"Unknown dtype {self.activations_dtype}")
        return out


def test_modelconfig_roundtrip_from_json() -> None:
    """Test that converting from json and back is the identity."""
    json_str = """{
        "d_model": 512,
        "num_heads": 8,
        "ff_dim": 2048,
        "dropout": 0.1,
        "n_layers": 6,
        "seq_len": 2048,
        "use_biases": true,
        "activations_dtype": "float32"}"""
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
        seq_len=42,
        use_biases=True,
        activations_dtype=jnp.bfloat16,
    )
    assert ModelConfig.from_json_dict(ModelConfig.to_json_dict(cfg)) == cfg


@dataclass
class TrainingConfig:
    learning_rate: float
    batch_size: int
    epochs: int
    triangle_schedule: bool
    gradient_accumulation_steps: int
    gradient_clipping: Optional[float]

    @staticmethod
    def from_json_dict(dict: dict[str, Any]) -> "TrainingConfig":
        """Convert a dictionary parsed from JSON to a TrainingConfig object."""
        dict = copy(dict)
        if dict["gradient_clipping"] == "None":
            dict["gradient_clipping"] = None
        return dacite.from_dict(data_class=TrainingConfig, data=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Convert a TrainingConfig object to a dictionary that can be serialized to JSON."""
        dict = copy(self.__dict__)
        if dict["gradient_clipping"] is None:
            dict["gradient_clipping"] = "None"
        return dict


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
        triangle_schedule=True,
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
        triangle_schedule=True,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
    )


def test_trainingconfig_roundtrip_from_json() -> None:
    """Test that converting from json and back is the identity."""
    json_str = """{
        "learning_rate": 1e-4,
        "batch_size": 4,
        "epochs": 100,
        "triangle_schedule": true,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": "None"}"""
    cfg = TrainingConfig.from_json_dict(json.loads(json_str))
    assert TrainingConfig.to_json_dict(cfg) == json.loads(json_str)


def test_trainingconfig_roundtrip_from_object() -> None:
    """Test that converting from a TrainingConfig object and back is the identity."""
    cfg = TrainingConfig(
        learning_rate=1,
        batch_size=4096,
        epochs=1,
        triangle_schedule=True,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
    )
    assert TrainingConfig.from_json_dict(TrainingConfig.to_json_dict(cfg)) == cfg
