"""
Flow matching model for spherical data.

Explanation, with help from Claude 3.7 Sonnet and Gemini 2.5 Pro:

This implementation extends Flow Matching with Optimal Transport (FM-OT) from Euclidean spaces to
the unit sphere, preserving its key theoretical properties while respecting the manifold's geometry.

## Theory and Design

In Euclidean spaces, standard FM-OT uses conditional probability paths based on Gaussian
distributions that interpolate linearly between a noise distribution and the target. The resulting
vector field produces straight-line trajectories with constant direction - the most efficient paths
in Euclidean space.

On the sphere, we adapt this paradigm by:

1. **Conditional Probability Paths**: Instead of Gaussian distributions, we use von Mises-Fisher
   (vMF) distributions. At t=0, we begin with a uniform distribution on the sphere (kappa_0=0). As t
   increases, the vMF distribution becomes more concentrated around the target (kappa_t =
   t*kappa_1). Note that kappa_1, the kappa parameter at t=1, is part of the definition of our
   probability paths, but not used in when calculating the target vector field or flow map. We leave
   it out of the definitions of our models.

2. **Geodesic Trajectories**: While Euclidean FM-OT follows straight lines, spherical FM-OT follows 
   geodesics (great circles) - the spherical equivalent of straight lines. These paths represent the 
   optimal transport solution on the sphere, minimizing the distance traveled along the manifold.

3. **Vector Field**: The vector field at point x points toward the target x1 along the geodesic 
   connecting them, and has magnitude equal to the length of the geodesic connecting the origin
   sample x0 from the base distribution to the target x1. Mathematically, this is achieved by
   projecting x1 onto the tangent space at x: u_t(x|x1) = acos(x1·x0) * normalize(x1 - (x1·x)x)
   
   This field has two key properties:
   - It is always tangent to the sphere (orthogonal to x)
   - It points along the geodesic toward x1
   - Integrating it from t=0 to t=1 generates a geodesic connecting x0 to x1

4. **Flow Map**: The flow map represents the position at time t of a particle starting at x0 
   and flowing toward x1. In Euclidean space, this is a linear interpolation; on the sphere, 
   it's the spherical linear interpolation (slerp):
   ψ_t(x0) = sin((1-t)θ)/sin(θ) · x0 + sin(tθ)/sin(θ) · x1
   where θ is the angle between x0 and x1.

Like Euclidean FM-OT, this spherical formulation maintains: (1) paths of constant velocity (now
constant angular velocity), (2) efficient sampling with fewer integration steps, and (3) theoretical
guarantees from continuous normalizing flows. The difference is that all computations respect the
geometry of the sphere, ensuring flows remain on the manifold.

"""
from dataclasses import dataclass, field, replace
from datasets import Dataset
from einops import rearrange, repeat
from flax import linen as nn
from flax.training import train_state
from functools import partial
from math import floor, ceil
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    FrozenSet,
)
from enum import Enum
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import stats
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from .cap_sampling import (
    LogitsTable,
    process_d_max_dist,
    random_pt_with_cosine_similarity,
    sample_cap,
    sample_from_cap,
    sphere_log_inverse_surface_area,
)


def sinusoidal_scalar_encoding(s, dim: int):
    """Sinusoidal encoding for a scalar in [0, 1] with configurable dimension (even)."""
    assert dim % 2 == 0
    half_d = dim // 2

    min_freq = 1.0
    max_freq = 1000.0
    freqs = jnp.exp(jnp.linspace(jnp.log(min_freq), jnp.log(max_freq), half_d))

    sin_components = jnp.sin(2 * jnp.pi * freqs * s)
    cos_components = jnp.cos(2 * jnp.pi * freqs * s)

    sin_means = -(jnp.cos(2 * jnp.pi * freqs) - 1.0) / (2 * jnp.pi * freqs)
    cos_means = jnp.sin(2 * jnp.pi * freqs) / (2 * jnp.pi * freqs)

    sin_components = sin_components - sin_means
    cos_components = cos_components - cos_means

    encoding = jnp.concatenate([sin_components, cos_components])
    encoding = encoding * jnp.sqrt(2)
    assert encoding.shape == (dim,)
    return encoding


class MLPBlock(nn.Module):
    """
    Single block of a residual MLP a la the standard pre-norm transformer with SwiGLU.
    """

    bottleneck_dim: int
    expansion_factor: int
    dropout_rate: Optional[float]

    activations_dtype: jnp.dtype
    weights_dtype: jnp.dtype
    param_variance: float

    # Extra features injected in addition to main bottleneck_dim input.
    inj_features: FrozenSet[str] = field(default_factory=frozenset)

    def setup(self) -> None:
        self.norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=jnp.float32,
        )
        self.gate_proj = nn.Dense(
            features=self.expansion_factor * self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(self.param_variance)),
        )
        self.value_proj = nn.Dense(
            features=self.expansion_factor * self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(self.param_variance)),
        )
        self.out_proj = nn.Dense(
            features=self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(
                stddev=jnp.sqrt(self.param_variance / self.expansion_factor)
            ),
        )
        if self.dropout_rate is not None:
            self.dropout = nn.Dropout(rate=self.dropout_rate, deterministic=False)
        else:
            self.dropout = None

        for key in self.inj_features:
            setattr(
                self,
                f"gate_proj_{key}",
                nn.Dense(
                    features=self.expansion_factor * self.bottleneck_dim,
                    dtype=self.activations_dtype,
                    param_dtype=self.weights_dtype,
                    kernel_init=nn.initializers.variance_scaling(
                        scale=1.0, mode="fan_in", distribution="normal"
                    ),
                    use_bias=False,  # Bias would be redundant with gate_proj
                ),
            )
            setattr(
                self,
                f"value_proj_{key}",
                nn.Dense(
                    features=self.expansion_factor * self.bottleneck_dim,
                    dtype=self.activations_dtype,
                    param_dtype=self.weights_dtype,
                    kernel_init=nn.initializers.variance_scaling(
                        scale=1.0, mode="fan_in", distribution="normal"
                    ),
                    use_bias=False,  # Bias would be redundant with value_proj
                ),
            )

    def __call__(self, x, inj):
        """Run the layer forward.

        inj is a carry for scan containing a dict with keys matching the inj_features set. mapped to
        per-example feature arrays. If None, no injection is performed.
        """
        assert len(x.shape) == 2
        assert x.shape[1] == self.bottleneck_dim
        x_normed = self.norm(x)
        if self.dropout is not None:
            x_normed = self.dropout(x_normed)

        # Scale projections to maintain consistent variance regardless of injection feature count
        scale_factor = 1.0 / jnp.sqrt(len(self.inj_features) + 1)
        gate = self.gate_proj(x_normed) * scale_factor
        value = self.value_proj(x_normed) * scale_factor

        # Project injected inputs and sum with scaled gate and value to maintain variance balance.
        if self.inj_features:
            if inj is None:
                raise ValueError("injection configured but inj carry was None")
            # ensure consistent addition order, regardless of set construction order. can matter
            # since float addition is not commutative.
            for key in sorted(self.inj_features):
                feat = inj.get(key, None)
                if feat is None:
                    raise ValueError(f"inj carry missing '{key}' feature")
                gate = gate + getattr(self, f"gate_proj_{key}")(feat) * scale_factor
                value = value + getattr(self, f"value_proj_{key}")(feat) * scale_factor
        else:
            if inj is not None:
                raise ValueError("inj carry was not None but inj_features is empty")
        gated = jax.nn.silu(gate) * value
        out = self.out_proj(gated)
        assert out.shape == x.shape
        return x + out, None


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block with full attention and SwiGLU MLP.
    """

    d_model: int
    n_heads: int
    mlp_expansion_factor: int
    mlp_dropout_rate: Optional[float]

    activations_dtype: jnp.dtype
    weights_dtype: jnp.dtype
    param_variance: float

    def setup(self) -> None:
        self.norm1 = nn.LayerNorm(dtype=self.activations_dtype, param_dtype=jnp.float32)

        # Use standard Flax MHA, but override attention_fn to set muP scaling (1/d_head)
        def _mup_attention_fn(
            query,
            key,
            value,
            bias=None,
            mask=None,
            *,
            is_causal: bool = False,
            **kwargs,
        ):
            # Enforce muP scaling: (q @ k^T) / d_head
            d_head = query.shape[-1]
            return jax.nn.dot_product_attention(
                query,
                key,
                value,
                bias=bias,
                mask=mask,
                scale=1.0 / float(d_head),
                is_causal=is_causal,
            )

        self.attn = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(self.param_variance)),
            dropout_rate=0.0,
            attention_fn=_mup_attention_fn,
        )
        self.norm2 = nn.LayerNorm(dtype=self.activations_dtype, param_dtype=jnp.float32)

        self.gate_proj = nn.Dense(
            features=self.mlp_expansion_factor * self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(self.param_variance)),
        )
        self.value_proj = nn.Dense(
            features=self.mlp_expansion_factor * self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(self.param_variance)),
        )
        self.out_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(
                stddev=jnp.sqrt(self.param_variance / self.mlp_expansion_factor)
            ),
        )

        self.dropout = (
            nn.Dropout(rate=self.mlp_dropout_rate, deterministic=False)
            if self.mlp_dropout_rate is not None
            else None
        )

    def __call__(self, tokens, _):
        # tokens: [batch, seq_len, d_model]
        assert tokens.ndim == 3
        assert tokens.shape[-1] == self.d_model

        y = self.norm1(tokens)
        if self.dropout is not None:
            y = self.dropout(y)
        attn_out = self.attn(y, y, y, deterministic=True)
        tokens = tokens + attn_out

        y2 = self.norm2(tokens)
        if self.dropout is not None:
            y2 = self.dropout(y2)
        gate = self.gate_proj(y2)
        value = self.value_proj(y2)
        ffn_out = self.out_proj(jax.nn.silu(gate) * value)
        tokens = tokens + ffn_out

        return tokens, None


class TransformerVectorField(nn.Module):
    """
    Drop-in replacement for VectorField using a Transformer backbone.

    Inputs become tokens:
      - x projected to d_model
      - t encoded via sinusoidal encoding, then projected to d_model
      - optional conditioning vector projected to d_model (skipped if conditioning_dim == 0)
      - n_learned_tokens trainable tokens

    Output is taken from the 0th learned token after the transformer stack, projected to domain_dim
    and projected into the tangent space at x.
    """

    # Dimensions
    domain_dim: int
    conditioning_dim: int

    # Architecture
    n_layers: int
    d_model: int
    mlp_expansion_factor: int
    n_heads: int
    n_learned_tokens: int

    # Time encoding
    time_dim: int = 128

    # Dropouts
    mlp_dropout_rate: Optional[float] = None
    input_dropout_rate: Optional[float] = None

    # Dtypes / scaling
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    d_model_base: int = 512
    variance_base: float = 1 / 512
    alpha_input: float = 1.0
    alpha_output: float = 1.0

    @property
    def d_model_scale_factor(self) -> float:
        return self.d_model / self.d_model_base

    def setup(self) -> None:
        if self.n_learned_tokens < 1:
            raise ValueError("n_learned_tokens must be at least 1")
        if self.time_dim % 2 != 0:
            raise ValueError("time_dim must be even for sinusoidal time encoding")

        hidden_init_variance = self.variance_base / self.d_model_scale_factor

        # Input projections
        # We scale x by sqrt(domain_dim) in __call__, so inputs to this layer have per-dim var ≈ 1.
        # Use variance scaling for fan-in so outputs are approximately unit variance at init.
        self.in_proj_x = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            use_bias=True,
        )
        if self.conditioning_dim > 0:
            self.in_proj_cond = nn.Dense(
                features=self.d_model,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
                kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
                use_bias=True,
            )
        else:
            self.in_proj_cond = None

        # Project sinusoidally encoded time (time_dim) to d_model
        self.time_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            use_bias=True,
        )

        # Learnable tokens
        self.learned_tokens = self.param(
            "learned_tokens",
            nn.initializers.normal(stddev=1.0),
            (self.n_learned_tokens, self.d_model),
        )

        # No explicit token-type biases; Dense layers include biases initialized to zero.

        self.input_dropout = (
            nn.Dropout(rate=self.input_dropout_rate, deterministic=False)
            if self.input_dropout_rate is not None
            else None
        )

        # Transformer stack (scan + remat)
        self.transformer_blocks = nn.scan(
            nn.remat(TransformerBlock),
            variable_axes={"params": 0, "intermediates": 0},
            variable_broadcast=False,
            split_rngs={"params": True, "dropout": True},
            length=self.n_layers,
        )(
            d_model=self.d_model,
            n_heads=self.n_heads,
            mlp_expansion_factor=self.mlp_expansion_factor,
            mlp_dropout_rate=self.mlp_dropout_rate,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            param_variance=hidden_init_variance,
        )

        self.final_norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=jnp.float32,
        )

        self.out_proj = nn.Dense(
            features=self.domain_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(hidden_init_variance)),
        )

    def time_encoding(self, t):
        """Encode time to a time_dim-dimensional vector."""
        return sinusoidal_scalar_encoding(t, self.time_dim)

    def dummy_inputs(self):
        x = jnp.ones((1, self.domain_dim))
        t = jnp.ones((1,))
        cond_vec = jnp.zeros((1, self.conditioning_dim))
        return x, t, cond_vec

    def prepare_training_conditioning(self, batch):
        if "cond_vec" in batch:
            return batch["cond_vec"]
        else:
            batch_size = batch["point_vec"].shape[0]
            return jnp.zeros((batch_size, 0))

    def scale_lr(self, lr: float) -> float:
        return lr / self.d_model_scale_factor

    def mk_partition_map(self, use_muon: bool):
        """Partition map for muP and optional Muon for transformer.

        Groups:
        - Without Muon: "fixed_lr" and "scaled_lr".
        - With Muon: "adam_fixed", "adam_scaled", "muon_scaled" (no muon_fixed group here).
        """
        if use_muon:
            muon_dense = {"bias": "adam_fixed", "kernel": "muon_scaled"}
            adam_dense = {"bias": "adam_fixed", "kernel": "adam_scaled"}

            params_map: Dict[str, Any] = {
                # Input/output layers use Adam per Muon guidance
                "in_proj_x": adam_dense,
                "time_proj": adam_dense,
                # Optional conditional projection is added below when present
                "learned_tokens": "adam_fixed",
                "final_norm": "adam_fixed",
                "out_proj": adam_dense,
                "transformer_blocks": {
                    "norm1": "adam_fixed",
                    "attn": {
                        "query": muon_dense,
                        "key": muon_dense,
                        "value": muon_dense,
                        "out": muon_dense,
                    },
                    "norm2": "adam_fixed",
                    "gate_proj": muon_dense,
                    "value_proj": muon_dense,
                    "out_proj": muon_dense,
                },
            }
            if self.conditioning_dim > 0:
                params_map["in_proj_cond"] = adam_dense
            return {"params": params_map}
        else:
            dense = {"bias": "fixed_lr", "kernel": "scaled_lr"}
            params_map = {
                "in_proj_x": dense,
                "time_proj": dense,
                # Optional conditional projection is added below when present
                "learned_tokens": "fixed_lr",
                "final_norm": "fixed_lr",
                "out_proj": dense,
                "transformer_blocks": {
                    "norm1": "fixed_lr",
                    "attn": {
                        "query": dense,
                        "key": dense,
                        "value": dense,
                        "out": dense,
                    },
                    "norm2": "fixed_lr",
                    "gate_proj": dense,
                    "value_proj": dense,
                    "out_proj": dense,
                },
            }
            if self.conditioning_dim > 0:
                params_map["in_proj_cond"] = dense
            return {"params": params_map}

    def __call__(self, x, t, cond_vec):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cond_vec.shape == (batch_size, self.conditioning_dim)

        # Tokenize inputs
        # Scale x by sqrt(domain_dim) so each component has var≈1 for uniform-on-sphere x
        x_tok = self.in_proj_x(x * jnp.sqrt(self.domain_dim))  # [b, d_model]
        # Time token: sinusoidal encoding (time_dim) projected to d_model
        t_enc = jax.vmap(self.time_encoding)(t)  # [b, time_dim]
        t_tok = self.time_proj(t_enc)  # [b, d_model]

        tokens = [x_tok[:, None, :], t_tok[:, None, :]]
        if self.conditioning_dim > 0:
            # Assumes cond_vec has mean 0 and var 1 per component by dataset design
            cond_tok = self.in_proj_cond(cond_vec)
            tokens.append(cond_tok[:, None, :])
        derived_tokens = jnp.concatenate(tokens, axis=1)

        learned = repeat(self.learned_tokens, "t d -> b t d", b=batch_size)

        seq = jnp.concatenate(
            [derived_tokens, learned], axis=1
        )  # [b, seq_len, d_model]
        if self.input_dropout is not None:
            seq = self.input_dropout(seq)
        seq = seq * self.alpha_input

        seq, _ = self.transformer_blocks(seq, None)
        seq = self.final_norm(seq)

        derived_len = 2 + (1 if self.conditioning_dim > 0 else 0)
        learned0 = seq[:, derived_len + 0, :]  # [b, d_model]

        points_out = self.out_proj(learned0)  # [b, domain_dim]

        # Project to tangent space at x
        dot_products = jnp.sum(points_out * x, axis=1, keepdims=True)
        tangent_outputs = points_out - dot_products * x

        # Match VectorField initial magnitude scaling to ~pi/2
        expected_magnitude = jnp.sqrt(2) * jnp.exp(
            jax.lax.lgamma(self.domain_dim / 2)
            - jax.lax.lgamma((self.domain_dim - 1) / 2)
        )
        domain_scale_factor = (jnp.pi / 2) / expected_magnitude
        tangent_outputs = tangent_outputs * domain_scale_factor * self.alpha_output

        return tangent_outputs


class VectorField(nn.Module):
    """
    Model of a vector field conditioned on a time t in [0, 1] and a arbitrary-dimensional vector.
    The vector field is over unit vectors in the domain.

    The inputs are processed as follows:
    * If reference_directions is not None, encode the input unit vectors as a set of cosine
      similarities with n randomly selected reference directions. Note that in order for this
      to capture all the information reference_directions must be at least domain_dim. If
      it is None, pass it through unmodified.
    * Encode the time parameter: sinusoidal encoding if time_dim specified, scalar if None.
    * Pass the conditioning vector through unmodified. It should have mean 0 and variance 1.
    * Concatenate the above into a single vector.
    * If use_pre_mlp_projection is true, apply a linear layer to the concatenated vector to create a
      d_model-dimensional vector.
    * If use_pre_mlp_projection is false:
      * The following must sum to less than or equal to d_model:
        * reference_directions if it is not None, otherwise domain_dim
        * time_dim if it is not None, otherwise 1 (for scalar encoding)
        * conditioning_dim
      * If they're less, pad the vector with zeros.
    * If use_pre_mlp_projection is false, add a learnable bias to the vector.
    * Apply dropout to the vector if input_dropout_rate is not None.
    * Feed that to the MLP.
    """

    # Dimension of the sphere and the vector field
    domain_dim: int
    # If reference_directions is not None, encode the input unit vectors as a set of cosine
    # similarities with n randomly selected reference directions. Note that in order for this
    # encoding to capture all the information reference_directions must be at least domain_dim. If
    # it is None, pass it through unmodified.
    reference_directions: Optional[int]
    # Dimension of the conditioning vector. Can be zero for unconditioned models.
    conditioning_dim: int
    # Dimension of the time encoding. Must be even if specified. If None, uses scalar encoding.
    time_dim: Optional[int]
    # If true, multiply the inputs by a learnable matrix
    use_pre_mlp_projection: bool
    # Number of MLP blocks
    n_layers: int
    # Width of the MLP
    d_model: int
    # MLP expansion factor
    mlp_expansion_factor: int
    # Dropout rate for the MLP
    mlp_dropout_rate: Optional[float]
    # Dropout rate for the input
    input_dropout_rate: Optional[float]

    # Which original inputs to inject at every MLP block. When a feature is injected,
    # it is omitted from the initial MLP input.
    mlp_always_inject: FrozenSet[Literal["x", "t", "cond"]] = field(
        default_factory=frozenset
    )

    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32

    d_model_base: int = 512  # Baseline d_model that muP scaling is relative to
    variance_base: float = 1 / 512  # baseline variance used when d_model = d_model_base
    alpha_input: float = 1.0  # scaling factor for inputs
    alpha_output: float = 1.0  # scaling factor for outputs

    @property
    def input_feature_dim(self) -> int:
        return (
            self.reference_directions
            if self.reference_directions is not None
            else self.domain_dim
        )

    @property
    def total_input_dim(self) -> int:
        time_dim = 1 if self.time_dim is None else self.time_dim
        return self.input_feature_dim + self.conditioning_dim + time_dim

    @property
    def initial_total_input_dim(self) -> int:
        """
        Total dim of features fed to the initial MLP input. We don't send injected features to the
        initial MLP input since that would be redundant.
        """
        dim = 0
        if "x" not in self.mlp_always_inject:
            dim += self.input_feature_dim
        if "t" not in self.mlp_always_inject:
            dim += 1 if self.time_dim is None else self.time_dim
        if "cond" not in self.mlp_always_inject:
            dim += self.conditioning_dim
        return dim

    @property
    def d_model_scale_factor(self) -> float:
        "m_d in muP."
        return self.d_model / self.d_model_base

    def mk_partition_map(self, use_muon: bool):
        """
        Create a partition map for optimizer configuration with muP scaling.

        Args:
            use_muon: If False, creates a simple muP partition map with "fixed_lr" and "scaled_lr" groups.
                     If True, creates a mixed Muon/Adam partition map with four groups:
                     - adam_fixed: Adam parameters with fixed learning rate (biases, norms, pre_mlp_proj, out_proj)
                     - adam_scaled: Adam parameters with scaled learning rate (some parameter groups)
                     - muon_fixed: Muon parameters with fixed learning rate (none in typical flow matching models)
                     - muon_scaled: Muon parameters with scaled learning rate (MLP projection kernels)

        Returns:
            Dictionary suitable for optax.transforms.partition
        """
        if use_muon:
            # Define which parameters use which optimizer
            # https://kellerjordan.github.io/posts/muon/ says "When training a neural network with
            # Muon, scalar and vector parameters of the network, as well as the input and output
            # layers, should be optimized by a standard method such as AdamW" so pre_mlp_proj and
            # out_proj use Adam, while MLP projection kernels use Muon. Which learning rates to
            # scale or not is based on the muP rules.
            muon_dense_layer_partition_map_biasless = {
                "kernel": "muon_scaled",
            }
            muon_dense_layer_partition_map_with_bias = {
                "bias": "adam_fixed",
                "kernel": "muon_scaled",
            }
            params_map = {
                "final_norm": "adam_fixed",
                "out_proj": {
                    "bias": "adam_fixed",
                    "kernel": "adam_scaled",
                },
                "mlp_blocks": {
                    "norm": "adam_fixed",
                    "gate_proj": muon_dense_layer_partition_map_with_bias,
                    "value_proj": muon_dense_layer_partition_map_with_bias,
                    "out_proj": muon_dense_layer_partition_map_with_bias,
                },
            }
            # Optional per-feature injection projections
            for inj_key in self.mlp_always_inject:
                params_map["mlp_blocks"][
                    f"gate_proj_{inj_key}"
                ] = muon_dense_layer_partition_map_biasless
                params_map["mlp_blocks"][
                    f"value_proj_{inj_key}"
                ] = muon_dense_layer_partition_map_biasless

            if self.use_pre_mlp_projection:
                # pre_mlp_proj uses Adam with appropriate muP scaling
                params_map["pre_mlp_proj"] = {
                    "bias": "adam_fixed",
                    "kernel": "adam_scaled",
                }
            else:
                params_map["mlp_in_bias"] = "adam_fixed"
        else:
            # Simple muP partition map
            dense_layer_partition_map_biasless = {"kernel": "scaled_lr"}
            dense_layer_partition_map_with_bias = {
                "bias": "fixed_lr",
                "kernel": "scaled_lr",
            }

            params_map = {
                "final_norm": "fixed_lr",
                "out_proj": dense_layer_partition_map_with_bias,
                "mlp_blocks": {
                    "norm": "fixed_lr",
                    "gate_proj": dense_layer_partition_map_with_bias,
                    "value_proj": dense_layer_partition_map_with_bias,
                    "out_proj": dense_layer_partition_map_with_bias,
                },
            }
            for inj_key in self.mlp_always_inject:
                params_map["mlp_blocks"][
                    f"gate_proj_{inj_key}"
                ] = dense_layer_partition_map_biasless
                params_map["mlp_blocks"][
                    f"value_proj_{inj_key}"
                ] = dense_layer_partition_map_biasless
            if self.use_pre_mlp_projection:
                # pre_mlp_proj should have its learning rate scaled by 1/m_d because it feeds into the
                # MLP and gradients go backwards but its initialization should be determined by
                # total_input_dim because activations go forwards. I think.
                params_map["pre_mlp_proj"] = dense_layer_partition_map_with_bias
            else:
                params_map["mlp_in_bias"] = "fixed_lr"

        return {"params": params_map}

    def scale_lr(self, lr: float) -> float:
        "Scaled learning rate for hidden layers."
        return lr / self.d_model_scale_factor

    def setup(self) -> None:
        hidden_init_variance = self.variance_base / self.d_model_scale_factor

        # Validate mlp_always_inject
        if not isinstance(self.mlp_always_inject, frozenset):
            raise ValueError(
                "mlp_always_inject must be a frozenset of {'x','t','cond'}"
            )
        allowed = {"x", "t", "cond"}
        unknown = set(self.mlp_always_inject) - allowed
        if unknown:
            raise ValueError(f"Unknown mlp_always_inject items: {sorted(unknown)}")
        if "cond" in self.mlp_always_inject and self.conditioning_dim == 0:
            raise ValueError(
                "'cond' specified in mlp_always_inject but conditioning_dim == 0"
            )

        if (
            not self.use_pre_mlp_projection
            and self.initial_total_input_dim > self.d_model
        ):
            raise ValueError(
                f"initial_total_input_dim ({self.initial_total_input_dim}) "
                f"exceeds d_model ({self.d_model}). Reduce one or more of them or increase d_model."
            )
        if self.time_dim is not None and self.time_dim % 2 != 0:
            raise ValueError("Time dimension must be even")

        if self.use_pre_mlp_projection:
            self.pre_mlp_proj = nn.Dense(
                features=self.d_model,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
                # If we have no non-injected inputs this is just a bias, and variance_scaling
                # throws a divide by zero error because of the 0 in the input shape.
                kernel_init=nn.initializers.variance_scaling(
                    scale=1.0, mode="fan_in", distribution="normal"
                )
                if self.initial_total_input_dim > 0
                else nn.initializers.zeros,
                use_bias=True,
            )

        if not self.use_pre_mlp_projection:
            # Set up a bias on the MLP inputs. If we're using pre-mlp-projection, this would be
            # redundant so we don't do it.
            def init_bias(rng, shape, dtype):
                # Zero bias for the components of the input that will actually be set to something,
                # unit normal for the rest to avoid weight tying.
                used_bias = jnp.zeros((self.total_input_dim,), dtype=dtype)
                unused_bias = jax.random.normal(
                    rng, (self.d_model - self.total_input_dim,), dtype=dtype
                )
                return jnp.concatenate([used_bias, unused_bias], axis=0)

            self.mlp_in_bias = self.param(
                "mlp_in_bias", init_bias, (self.d_model,), self.weights_dtype
            )

        if self.input_dropout_rate is not None:
            self.input_dropout = nn.Dropout(
                rate=self.input_dropout_rate, deterministic=False
            )
        else:
            self.input_dropout = None

        self.mlp_blocks = nn.scan(
            nn.remat(MLPBlock),
            variable_axes={"params": 0, "intermediates": 0},
            variable_broadcast=False,
            split_rngs={"params": True, "dropout": True},
            length=self.n_layers,
            in_axes=(nn.broadcast,),
        )(
            bottleneck_dim=self.d_model,
            expansion_factor=self.mlp_expansion_factor,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            param_variance=hidden_init_variance,
            dropout_rate=self.mlp_dropout_rate,
            inj_features=self.mlp_always_inject,
        )

        self.final_norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=jnp.float32,
        )

        self.out_proj = nn.Dense(
            features=self.domain_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(hidden_init_variance)),
        )

        # Used to encode the input unit vectors as a vector of dot products
        if self.reference_directions is not None:
            self.reference_vectors = sample_sphere(
                jax.random.PRNGKey(20250315), self.reference_directions, self.domain_dim
            )

    def time_encoding(self, t):
        """Create encoding for the time parameter: sinusoidal if time_dim specified, scalar if None."""
        if self.time_dim is not None:
            return sinusoidal_scalar_encoding(t, self.time_dim)
        else:
            return jnp.array([(t - 0.5) * jnp.sqrt(12.0)])

    def process_inputs(self, x, t, cond_vec):
        """
        Process the inputs to the model, returning the individual components and combined input.
        This helper method is used for testing the distribution of values.

        Args:
            x: Input unit vectors [batch_size, domain_dim]
            t: Time parameters in [0, 1] [batch_size]
            cond_vec: Conditioning vectors [batch_size, conditioning_dim]

        Returns:
            Dictionary with individual components and combined input:
            - input_features: Input features (either dot products or scaled input vectors) [batch_size, input_feature_dim]
            - time_encoding: Time encoding [batch_size, time_dim]
            - cond_vec: Conditioning vectors [batch_size, conditioning_dim]
            - mlp_input: Combined input to the MLP [batch_size, d_model]
        """
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cond_vec.shape == (batch_size, self.conditioning_dim)

        if self.reference_directions is not None:
            # Encode the input unit vectors as a vector of dot products
            input_features = (x @ self.reference_vectors.T) * jnp.sqrt(self.domain_dim)
            assert input_features.shape == (batch_size, self.reference_directions)
        else:
            # Use the raw input vectors, scaled to have unit variance
            input_features = x * jnp.sqrt(self.domain_dim)
            assert input_features.shape == (batch_size, self.domain_dim)

        time_encoding = jax.vmap(self.time_encoding)(t)
        expected_time_dim = 1 if self.time_dim is None else self.time_dim
        assert time_encoding.shape == (batch_size, expected_time_dim)

        # Build initial input from non-injected components
        components = []
        if "x" not in self.mlp_always_inject:
            components.append(input_features)
        if "t" not in self.mlp_always_inject:
            components.append(time_encoding)
        if "cond" not in self.mlp_always_inject:
            components.append(cond_vec)

        if len(components) > 0:
            concatenated = jnp.concatenate(components, axis=1)
        else:
            concatenated = jnp.zeros((batch_size, 0), dtype=self.activations_dtype)
        assert concatenated.shape[0] == batch_size

        if self.use_pre_mlp_projection:
            mlp_in = self.pre_mlp_proj(concatenated)
        else:
            if self.initial_total_input_dim < self.d_model:
                padding = jnp.zeros(
                    (batch_size, self.d_model - self.initial_total_input_dim)
                )
                mlp_in = jnp.concatenate([concatenated, padding], axis=1)
            else:
                mlp_in = concatenated

        if not self.use_pre_mlp_projection:
            mlp_in = mlp_in + self.mlp_in_bias
        assert mlp_in.shape == (batch_size, self.d_model)

        mlp_in = (
            self.input_dropout(mlp_in)
            if self.input_dropout_rate is not None
            else mlp_in
        )
        mlp_in = mlp_in * self.alpha_input
        assert mlp_in.shape == (batch_size, self.d_model)

        return {
            "input_features": input_features,
            "time_encoding": time_encoding,
            "cond_vec": cond_vec,
            "mlp_input": mlp_in,
        }

    def dummy_inputs(self):
        """Create dummy inputs for model initialization with the correct shapes.

        Returns:
            Tuple of (x, t, cond_vec) with appropriate shapes for initialization.
        """
        x = jnp.ones((1, self.domain_dim))
        t = jnp.ones((1,))
        cond_vec = jnp.ones((1, self.conditioning_dim))
        return x, t, cond_vec

    def prepare_training_conditioning(self, batch):
        """Prepare conditioning data for training - extract from batch.

        Args:
            batch: Training batch, may contain "cond_vec" for conditional models

        Returns:
            Conditioning vectors from the batch [batch_size, conditioning_dim]
        """
        if "cond_vec" in batch:
            return batch["cond_vec"]
        else:
            # For unconditional models, create zero-dimensional conditioning
            batch_size = batch["point_vec"].shape[0]
            return jnp.zeros((batch_size, 0))

    def __call__(self, x, t, cond_vec):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cond_vec.shape == (batch_size, self.conditioning_dim)

        # Process inputs so we can give them to the MLP
        inputs = self.process_inputs(x, t, cond_vec)
        mlp_in = inputs["mlp_input"]

        # Prepare injection carry if configured
        inj = None
        if self.mlp_always_inject:
            inj = {}
            if "x" in self.mlp_always_inject:
                inj["x"] = inputs["input_features"]
            if "t" in self.mlp_always_inject:
                inj["t"] = inputs["time_encoding"]
            if "cond" in self.mlp_always_inject:
                inj["cond"] = inputs["cond_vec"]

        # Run them through our MLP and normalize the output
        mlp_out, _ = self.mlp_blocks(mlp_in, inj)
        mlp_out = self.final_norm(mlp_out)
        assert mlp_out.shape == (batch_size, self.d_model)

        # Project back down to the dimension of the domain
        points_out = self.out_proj(mlp_out)
        assert points_out.shape == (batch_size, self.domain_dim)

        # Project to tangent space of the sphere
        # For a unit vector x, the tangent space projection is: v - (v·x)x
        dot_products = jnp.sum(points_out * x, axis=1, keepdims=True)
        tangent_outputs = points_out - dot_products * x
        assert tangent_outputs.shape == (batch_size, self.domain_dim)

        # At initialization, we want the mean magnitude of our output vectors to be pi/2, since the
        # average geodesic distance between 2 uniform random points on a sphere is pi/2. This should
        # make our initial loss values reasonable. Without scaling, the initial loss values are
        # gigantic - ~766 for 768d model. With it they're ~5.
        # The tangent space projection of a d-dimensional Gaussian has magnitude following a chi
        # distribution with (d-1) degrees of freedom. The expectation is:
        # E[chi(k)] = sqrt(2) * Gamma((k+1)/2) / Gamma(k/2)
        expected_magnitude = jnp.sqrt(2) * jnp.exp(
            jax.lax.lgamma(self.domain_dim / 2)
            - jax.lax.lgamma((self.domain_dim - 1) / 2)
        )

        domain_scale_factor = (jnp.pi / 2) / expected_magnitude

        tangent_outputs = tangent_outputs * domain_scale_factor * self.alpha_output

        return tangent_outputs


@pytest.mark.parametrize("time_dim", [4, 16, 64, None])
def test_vector_field_time_encoding_statistics(time_dim):
    """
    Test that the time encoding function produces vectors with appropriate statistical properties
    when given uniformly distributed time values between 0 and 1.

    The test checks:
    1. That the mean of each component is close to 0
    2. That the standard deviation of each component is close to 1
    """
    rng = jax.random.PRNGKey(42)

    # Create a minimal VectorField model just to test the time_encoding. Parameters other than
    # d_model are irrelevant for this test.
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=0,
        time_dim=time_dim,
        reference_directions=16,
        n_layers=1,
        d_model=256,
        mlp_expansion_factor=1,
        mlp_dropout_rate=None,
        input_dropout_rate=None,
        use_pre_mlp_projection=False,
    )

    params = model.init(rng, jnp.ones((1, 3)), jnp.ones((1,)), jnp.ones((1, 0)))

    n_samples = 10_000
    times = jax.random.uniform(rng, (n_samples,))

    compute_encoding = lambda t: model.apply(params, t, method=model.time_encoding)
    encoded_times = jax.vmap(compute_encoding)(times)

    expected_time_dim = 1 if time_dim is None else time_dim
    assert encoded_times.shape == (n_samples, expected_time_dim)

    # Check statistics
    means = jnp.mean(encoded_times, axis=0)
    stds = jnp.std(encoded_times, axis=0)

    expected_mean = 0.0
    expected_std = 1.0

    np.testing.assert_allclose(means, expected_mean, atol=0.05)
    np.testing.assert_allclose(stds, expected_std, atol=0.1)

    # Check overall vector statistics
    overall_mean = jnp.mean(encoded_times)
    overall_std = jnp.std(encoded_times)

    np.testing.assert_allclose(overall_mean, expected_mean, atol=0.01)
    np.testing.assert_allclose(overall_std, expected_std, atol=0.01)

    print(f"Time encoding test passed for time_dim={time_dim}")
    print(f"  Mean: {overall_mean:.6f} (expected {expected_mean:.6f})")
    print(f"  Std: {overall_std:.6f} (expected {expected_std:.6f})")


@pytest.mark.parametrize("domain_dim", [3, 10])
@pytest.mark.parametrize("conditioning_dim", [0, 4])
def test_vector_field_projections_normalization(domain_dim, conditioning_dim):
    """
    Test that the inputs to the MLP in the VectorField model have the expected distributions.

    Specifically, check that each component has mean 0 and variance 1 and that the combined
    input also has mean 0 and variance 1.

    Assumptions:
    * x is composed of uniformly distributed unit vectors in domain_dim dimensions
    * t is uniformly distributed between 0 and 1
    * cond_vec's components each have mean 0 variance 1
    """
    rng = jax.random.PRNGKey(42)
    time_dim = 16

    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=domain_dim,
        conditioning_dim=conditioning_dim,
        time_dim=time_dim,
        reference_directions=16,
        use_pre_mlp_projection=False,
        n_layers=1,
        d_model=2048,  # Bigger mlp_in vectors reduce the variance of our estimates
        mlp_expansion_factor=1,
        input_dropout_rate=None,
        mlp_dropout_rate=None,
    )

    params_rng, sample_rng = jax.random.split(rng)
    state = create_train_state(params_rng, model, 1e-3)

    # Generate test inputs
    n_samples = 10_000
    keys = jax.random.split(sample_rng, 3)

    x = sample_sphere(keys[0], n_samples, domain_dim)

    t = jax.random.uniform(keys[1], (n_samples,))

    if conditioning_dim > 0:
        cond_vec = jax.random.normal(keys[2], (n_samples, conditioning_dim))
    else:
        cond_vec = jnp.zeros((n_samples, 0))

    inputs = model.apply(state.params, x, t, cond_vec, method=model.process_inputs)

    input_features = inputs["input_features"]
    input_features_mean = jnp.mean(input_features)
    input_features_variance = jnp.mean(jnp.var(input_features, axis=0))

    time_encoding = inputs["time_encoding"]
    time_encoding_mean = jnp.mean(time_encoding)
    time_encoding_variance = jnp.mean(jnp.var(time_encoding, axis=0))

    if conditioning_dim > 0:
        cond_vec_data = inputs["cond_vec"]
        cond_vec_mean = jnp.mean(cond_vec_data)
        cond_vec_variance = jnp.mean(jnp.var(cond_vec_data, axis=0))

    mlp_input = inputs["mlp_input"]
    # The padding components of mlp_input are constant, so they have 0 variance. So we test
    # the statistics of the unpadded part separately from the whole thing.
    unpadded_mlp_input = mlp_input[:, : model.total_input_dim]
    unpadded_mlp_input_mean = jnp.mean(unpadded_mlp_input)
    unpadded_mlp_input_variance = jnp.mean(jnp.var(unpadded_mlp_input, axis=0))

    padded_mlp_input_mean = jnp.mean(mlp_input)
    # note axis: calculating variances inside examples, not features across examples
    padded_mlp_input_variance = jnp.mean(jnp.var(mlp_input, axis=1))
    # Print statistics
    print(
        f"\nTesting with domain_dim={domain_dim}, conditioning_dim={conditioning_dim}, time_dim={time_dim}"
    )
    # print(
    #    f"scaled_x - Mean: {scaled_x_mean:.6f}, Variance: {scaled_x_variance:.6f}"
    # )
    print(
        f"dot_products - Mean: {input_features_mean:.6f}, Variance: {input_features_variance:.6f}"
    )
    print(
        f"time_encoding - Mean: {time_encoding_mean:.6f}, Variance: {time_encoding_variance:.6f}"
    )
    if conditioning_dim > 0:
        print(
            f"cond_vec - Mean: {cond_vec_mean:.6f}, Variance: {cond_vec_variance:.6f}"
        )
    print(
        f"Unpadded MLP input - Mean: {unpadded_mlp_input_mean:.6f}, Variance: {unpadded_mlp_input_variance:.6f}"
    )
    print(
        f"Padded MLP input - Mean: {padded_mlp_input_mean:.6f}, Variance: {padded_mlp_input_variance:.6f}"
    )

    # Check if means are close to 0
    mean_tol = 0.02
    assert (
        abs(input_features_mean) < mean_tol
    ), f"input_features mean should be close to 0, got {input_features_mean}"
    assert (
        abs(input_features_variance - 1.0) < 0.05
    ), f"input_features variance should be close to 1.0, got {input_features_variance}"
    assert (
        abs(time_encoding_mean) < mean_tol
    ), f"time_encoding mean should be close to 0, got {time_encoding_mean}"
    assert (
        abs(time_encoding_variance - 1.0) < 0.05
    ), f"time_encoding variance should be close to 1.0, got {time_encoding_variance}"

    if conditioning_dim > 0:
        assert (
            abs(cond_vec_mean) < mean_tol
        ), f"cond_vec mean should be close to 0, got {cond_vec_mean}"
        assert (
            abs(cond_vec_variance - 1.0) < 0.05
        ), f"cond_vec variance should be close to 1.0, got {cond_vec_variance}"

    assert (
        abs(unpadded_mlp_input_mean) < mean_tol
    ), f"Unpadded MLP input mean should be close to 0, got {unpadded_mlp_input_mean}"
    assert (
        abs(unpadded_mlp_input_variance - 1.0) < 0.05
    ), f"Unpadded MLP input variance should be close to 1.0, got {unpadded_mlp_input_variance}"
    assert (
        abs(padded_mlp_input_mean) < 0.05
    ), f"Padded MLP input mean should be close to 0, got {padded_mlp_input_mean}"
    assert (
        abs(padded_mlp_input_variance - 1.0) < 0.05
    ), f"Padded MLP input variance should be close to 1.0, got {padded_mlp_input_variance}"


@pytest.mark.parametrize("dim", [3, 16])
def test_optimal_transport_field_direction(dim):
    """
    Test that optimal_transport_field produces vectors that are aligned with
    the direct path from x to x1 after tangent space projection.
    """
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(0)

    # Test parameters
    n_samples = 100

    # Generate random points on the sphere
    key1, key2, key3 = jax.random.split(rng, 3)
    x0_samples = sample_sphere(key1, n_samples, dim)
    x1_samples = sample_sphere(key2, n_samples, dim)
    t_values = jax.random.uniform(key3, (n_samples,))

    def compute_and_compare(x0, x1, t):
        # Get the current position and vector field
        x, ot_field = spherical_ot_field(x0, x1, t)

        # Compute direct vector from x to x1 and project to tangent space
        direct = x1 - x
        direct_tangent = direct - jnp.dot(direct, x) * x

        # Compute cosine similarity between expected and actual. If one or both vectors are ~0,
        # treat it as if similarity were 1.
        vectors_are_nonzero = jnp.logical_and(
            jnp.linalg.norm(ot_field) > 1e-8, jnp.linalg.norm(direct_tangent) > 1e-8
        )
        return jax.lax.cond(
            vectors_are_nonzero,
            lambda: jnp.dot(
                ot_field / jnp.linalg.norm(ot_field),
                direct_tangent / jnp.linalg.norm(direct_tangent),
            ),
            lambda: 1.0,
        )

    # Apply to all samples
    similarities = jax.vmap(compute_and_compare)(x0_samples, x1_samples, t_values)

    # Check alignment
    avg_similarity = jnp.mean(similarities)
    min_similarity = jnp.min(similarities)

    # Print results
    print(f"Average alignment (cosine similarity): {avg_similarity:.4f}")
    print(f"Minimum alignment: {min_similarity:.4f}")

    # Assertion to verify alignment
    assert jnp.all(
        similarities > 0.99
    ), "Some vectors are not well aligned with the tangent direction"


def slerp(x, y, t):
    """Spherical linear interpolation."""
    assert len(x.shape) == 1
    assert x.shape == y.shape
    assert t.shape == ()

    cos_angle = jnp.clip(jnp.dot(x, y), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)
    return (
        jnp.sin((1 - t) * angle) / jnp.sin(angle) * x
        + jnp.sin(t * angle) / jnp.sin(angle) * y
    )


def spherical_ot_field(x0, x1, t):
    """
    Special fancy spherical version of the OT field. Compute a tangent vector field on the sphere
    that generates geodesics on the sphere rather than straight lines. Based on vMF distributions
    instead of gaussians.

    Args:
        x0: Starting point on the sphere [dim]
        x1: Target point on the sphere [dim]
        t: Time parameter in [0, 1]

    Returns:
        Tuple of (x, field_value) where:
        - x: Current point on the sphere computed from x0, x1, t
        - field_value: Vector field at x, tangent to the sphere
    """
    assert len(x0.shape) == 1
    assert x0.shape == x1.shape
    assert t.shape == ()

    # Compute the current point by flowing x0 toward x1 for time t
    x = compute_psi_t_spherical(x0, x1, t)

    # Compute the angle between x0 and x1, which determines the speed
    cos_angle = jnp.clip(jnp.dot(x0, x1), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    def handle_general_case():
        # Vector field points towards x1 from x, in the tangent space at x.
        proj_scalar = jnp.dot(x1, x)
        tangent_component = x1 - proj_scalar * x
        tangent_norm = jnp.linalg.norm(tangent_component)
        return jax.lax.cond(
            tangent_norm > 1e-8,
            lambda: angle * tangent_component / tangent_norm,
            lambda: jnp.zeros_like(x),
        )

    field_value = jax.lax.cond(
        cos_angle < -1.0 + 1e-6,
        # If points are opposite/almost opposite, pick an orthogonal vector
        lambda: get_consistent_tangent_direction(x) * angle,
        handle_general_case,
    )

    return x, field_value


def get_consistent_tangent_direction(x):
    """
    Generate a unit vector in the tangent space of x, ensuring that nearby points get similar
    tangent directions, creating smooth paths.

    Args:
        x: Point on the sphere [dim]

    Returns:
        Unit vector in the tangent space of x
    """
    dim = x.shape[0]

    # Instead of picking the "best" basis vector which could lead to discontinuities,
    # we use a fixed reference direction approach:

    # Define two fixed orthogonal reference vectors that will be used for all points
    # These are fixed in the global space (not dependent on x) to ensure consistency
    ref1 = jnp.zeros(dim).at[0].set(1.0)
    ref2 = jnp.zeros(dim).at[1].set(1.0)

    # Project the first reference to the tangent space of x
    ref1_tan = ref1 - jnp.dot(ref1, x) * x
    ref1_norm = jnp.linalg.norm(ref1_tan)

    # For second reference (fallback option)
    ref2_tan = ref2 - jnp.dot(ref2, x) * x
    ref2_norm = jnp.linalg.norm(ref2_tan)

    # Final fallback: permute coordinates
    permuted_x = jnp.roll(x, 1)
    fallback_tan = permuted_x - jnp.dot(permuted_x, x) * x
    # Since x and its rolled version cannot be parallel for dim >= 2,
    # this is safe to normalize
    fallback_norm = jnp.linalg.norm(fallback_tan)

    # Use the first direction with a sufficiently large tangent norm.
    ref1_fallback = jax.lax.cond(
        ref2_norm > 1e-8,
        lambda: ref2_tan / ref2_norm,
        lambda: fallback_tan / fallback_norm,
    )
    return jax.lax.cond(
        ref1_norm > 1e-8, lambda: ref1_tan / ref1_norm, lambda: ref1_fallback
    )


def compute_psi_t_spherical(x0, x1, t):
    """
    Compute the flow map for the spherical OT field.
    In the general case this is just a spherical linear interpolation, but we're careful with the
    antipodal case.

    Args:
        x0: Starting point on the sphere [dim]
        x1: Target point on the sphere [dim]
        t: Time parameter in [0, 1]

    Returns:
        Point on the sphere resulting from flowing along the vector field
    """
    assert len(x0.shape) == 1
    assert x0.shape == x1.shape
    assert t.shape == ()

    cos_angle = jnp.clip(jnp.dot(x0, x1), -1.0, 1.0)

    def handle_close_or_opposite():
        return jax.lax.cond(
            cos_angle > 1.0 - 1e-6,
            lambda: x0,  # If points are very close, just return x0
            lambda: handle_antipodal(),
        )

    def handle_antipodal():
        tangent_dir = get_consistent_tangent_direction(x0)

        # Create a point that's 90 degrees away from x0 in the tangent direction
        # This is a point on the great circle connecting x0 and -x0 via the tangent direction
        midpoint = tangent_dir

        # For antipodal points, we want to go halfway around the sphere in the consistent direction
        # If t <= 0.5, we interpolate from x0 to midpoint
        # If t > 0.5, we interpolate from midpoint to x1 (which is approximately -x0)
        return jax.lax.cond(
            t <= 0.5,
            lambda: slerp(
                x0, midpoint, 2.0 * t
            ),  # Scale t to [0, 1] for the first half
            lambda: slerp(
                midpoint, x1, 2.0 * (t - 0.5)
            ),  # Scale t to [0, 1] for the second half
        )

    return jax.lax.cond(
        jnp.logical_or(cos_angle > 1.0 - 1e-6, cos_angle < -1.0 + 1e-6),
        handle_close_or_opposite,
        lambda: slerp(x0, x1, t),
    )


def conditional_flow_matching_loss(
    model,
    params,
    x0,
    x1,
    t,
    conditioning_data,
    rng=None,
    capture_intermediates=False,
):
    """
    Compute the Conditional Flow Matching loss from eq. 9 in the paper, modified for the spherical
    OT field.

    Args:
        model: Vector field model
        params: Model parameters
        x0: Noise samples from p0 (batched) [batch_size, dim]
        x1: Target samples from p1 (batched) [batch_size, dim]
        t: Time parameters (batched) in [0, 1] [batch_size]
        conditioning_data: Conditioning data for the model (format model-specific)
        rng: JAX random key (required if model uses dropout)

    Returns:
        CFM loss value (scalar)
    """
    assert len(x0.shape) == 2
    batch_size = x0.shape[0]
    assert x0.shape == x1.shape
    assert x0.shape[1] == model.domain_dim
    assert t.shape == (batch_size,)

    # Check if model uses dropout
    if model.input_dropout_rate is not None or model.mlp_dropout_rate is not None:
        assert rng is not None, "rng is required for models with dropout"

    # Compute target vector field (ground truth OT field) and current positions
    psi_ts, target_fields = jax.vmap(spherical_ot_field, in_axes=(0, 0, 0))(x0, x1, t)
    assert psi_ts.shape == x0.shape
    assert target_fields.shape == x0.shape

    # Compute predicted vector field from our model
    apply_res = model.apply(
        params,
        psi_ts,
        t,
        conditioning_data,
        rngs={"dropout": rng},
        capture_intermediates=capture_intermediates,
    )

    if capture_intermediates:
        predicted_field, intermediates = apply_res
    else:
        predicted_field = apply_res
    assert predicted_field.shape == x0.shape

    loss = jnp.mean(jnp.sum((predicted_field - target_fields) ** 2, axis=1))

    if capture_intermediates:
        return loss, intermediates
    else:
        return loss


def create_train_state(rng, model, learning_rate_or_schedule, gradient_clipping=None):
    """Create initial training state for a test. See checkpoint.py for the real one."""
    dummy_inputs = model.dummy_inputs()
    params = model.init(rng, *dummy_inputs)
    if callable(learning_rate_or_schedule):
        scaled_lr_or_schedule = lambda step: model.scale_lr(
            learning_rate_or_schedule(step)
        )
    else:
        scaled_lr_or_schedule = model.scale_lr(learning_rate_or_schedule)

    if gradient_clipping is not None:
        opt_fixed_lr = optax.chain(
            optax.clip_by_global_norm(gradient_clipping),
            optax.adamw(learning_rate_or_schedule, weight_decay=0.001),
        )
        opt_scaled_lr = optax.chain(
            optax.clip_by_global_norm(gradient_clipping),
            optax.adamw(scaled_lr_or_schedule, weight_decay=0.001),
        )
    else:
        opt_fixed_lr = optax.adamw(learning_rate_or_schedule, weight_decay=0.001)
        opt_scaled_lr = optax.adamw(scaled_lr_or_schedule, weight_decay=0.001)

    opt = optax.transforms.partition(
        {"fixed_lr": opt_fixed_lr, "scaled_lr": opt_scaled_lr},
        model.mk_partition_map(use_muon=False),
    )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt)


@partial(jax.jit, static_argnames=("batch_size", "dim"), inline=True)
def sample_sphere(rng, batch_size, dim):
    """
    Sample points uniformly from the unit sphere.

    Args:
        rng: JAX random key
        batch_size: Number of points to sample
        dim: Dimension of the sphere

    Returns:
        Array of points on the unit sphere [batch_size, dim]
    """
    normal_samples = jax.random.normal(rng, (batch_size, dim))
    return normal_samples / jnp.linalg.norm(normal_samples, axis=1, keepdims=True)


@partial(jax.jit, inline=True, static_argnames=("model", "capture_intermediates"))
def compute_batch_loss(model, params, batch, rng, capture_intermediates=False):
    """
    Compute the loss for a batch of data using conditional flow matching.

    Args:
        model: The model (must implement prepare_training_conditioning method)
        params: Model parameters
        batch: Batch of data containing "point_vec" and conditioning data as needed by model
        rng: JAX random key
        capture_intermediates: Whether to capture intermediate values

    Returns:
        loss: The computed loss value
    """
    x1_batch = batch["point_vec"]
    batch_size = x1_batch.shape[0]
    assert x1_batch.shape == (batch_size, model.domain_dim)

    # Split RNG for different purposes
    rng, noise_rng, time_rng, conditioning_rng = jax.random.split(rng, 4)

    # Let model prepare its own conditioning data for training
    conditioning_data = model.apply(
        params,
        batch,
        method=model.prepare_training_conditioning,
        rngs={"sample_weighting_params": conditioning_rng},
    )

    x0_batch = sample_sphere(noise_rng, batch_size, model.domain_dim)
    t = jax.random.uniform(time_rng, (batch_size,))

    return conditional_flow_matching_loss(
        model,
        params,
        x0_batch,
        x1_batch,
        t,
        conditioning_data,
        rng=rng,
        capture_intermediates=capture_intermediates,
    )


@partial(jax.jit, static_argnames=("model"), donate_argnames=("state", "rng"))
def train_step(model, state, batch, rng):
    """
    Train for a single step.

    Args:
        model: The vector field model
        state: Training state
        batch: Batch of data containing "point_vec" and "cond_vec"
        rng: JAX random key

    Returns:
        Updated state, loss value, gradient norm, and updated random key
    """
    rng, next_rng = jax.random.split(rng)

    def loss_fn(params):
        loss = compute_batch_loss(model, params, batch, rng)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)

    state = state.apply_gradients(grads=grads)

    return state, loss, grad_norm, next_rng


def _train_loop_for_tests_generic(
    model,
    dataset,
    batch_size,
    learning_rate,
    epochs,
    test_dataset=None,
    compute_nll_fn=None,
):
    """
    Generic training loop for unit tests.

    Args:
        model: The model to train
        dataset: Training dataset
        batch_size: Batch size for training
        learning_rate: Initial learning rate (will use cosine schedule)
        epochs: Number of epochs to train for
        test_dataset: Optional test dataset for evaluation after each epoch
        compute_nll_fn: Function (model, params, batch, n_steps, rng, n_projections) -> nlls

    Returns:
        state: Final training state
        train_loss: Final training loss
        test_loss: Final test loss (if test_dataset provided)
    """
    params_rng, step_rng = jax.random.split(jax.random.PRNGKey(7357), 2)

    # Calculate total number of steps
    n_samples = len(dataset)
    steps_per_epoch = n_samples // batch_size
    total_steps = steps_per_epoch * epochs

    cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=floor(total_steps * 0.1),
        decay_steps=total_steps,
    )

    n_projections = 10
    nll_steps = 32

    state = create_train_state(params_rng, model, cosine_schedule)
    np_rng = np.random.Generator(np.random.PCG64(seed=42))

    final_test_loss = None
    final_test_nll = None
    first_step = True
    step_count = 0

    print("Train! Loop! Go!")
    with tqdm(range(epochs), desc="Training", unit="epochs") as pbar:
        for epoch in pbar:
            # Training loop
            for i, batch in tenumerate(
                dataset.iter(batch_size, drop_last_batch=True),
                unit="train batches",
                total=len(dataset) // batch_size,
            ):
                norms = jnp.linalg.norm(batch["point_vec"], axis=1, keepdims=True)
                np.testing.assert_allclose(np.asarray(norms), 1.0, rtol=0, atol=1e-6)

                state, train_loss, grad_norm, step_rng = train_step(
                    model, state, batch, step_rng
                )
                jax.tree.map(lambda x: x.copy_to_host_async(), (train_loss, grad_norm))

                step_count += 1
                current_lr = cosine_schedule(step_count)

                if first_step or i % 200 == 0:
                    step_rng, nll_rng = jax.random.split(step_rng)
                    # Use model-specific function to compute NLL
                    train_nlls = -compute_nll_fn(
                        model,
                        state.params,
                        batch,
                        n_steps=nll_steps,
                        rng=nll_rng,
                        n_projections=n_projections,
                    )
                    train_nll = jnp.mean(train_nlls)
                train_loss, grad_norm, train_nll = jax.device_get(
                    (train_loss, grad_norm, train_nll)
                )
                if first_step:
                    tqdm.write(
                        f"First step loss: {train_loss:.6f}, grad norm: {grad_norm:.6f}, lr: {current_lr:.6f}, nll: {train_nll:.6f}"
                    )
                    first_step = False
                pbar.set_postfix(
                    {
                        "loss": float(train_loss),
                        "train_nll": float(train_nll),
                        "grad_norm": float(grad_norm),
                        "lr": float(current_lr),
                    }
                )

            # Evaluate on test dataset if provided
            if test_dataset is not None:
                test_rng, step_rng = jax.random.split(step_rng)
                test_losses = []
                test_nlls = []

                for test_batch in tqdm(
                    test_dataset.iter(batch_size, drop_last_batch=False),
                    unit="test batches",
                    total=len(test_dataset) // batch_size,
                ):
                    test_batch_rng, test_nll_rng, test_rng = jax.random.split(
                        test_rng, 3
                    )

                    batch_test_loss = compute_batch_loss(
                        model, state.params, test_batch, test_batch_rng
                    )

                    test_losses.append(batch_test_loss)

                    # Use model-specific function to compute NLL
                    test_nlls.append(
                        -compute_nll_fn(
                            model,
                            state.params,
                            test_batch,
                            n_steps=nll_steps,
                            rng=test_nll_rng,
                            n_projections=n_projections,
                        )
                    )

                if len(test_losses) > 0:
                    test_losses = jnp.asarray(test_losses)
                    avg_test_loss = jax.device_get(jnp.mean(test_losses))
                    final_test_loss = avg_test_loss

                    test_nlls = jnp.concatenate(test_nlls)
                    avg_test_nll = jax.device_get(jnp.mean(test_nlls))
                    final_test_nll = avg_test_nll

                    tqdm.write(
                        f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Train NLL: {train_nll:.6f}, Grad Norm: {grad_norm:.6f}, Test Loss: {avg_test_loss:.6f}, Test NLL: {avg_test_nll:.6f}"
                    )
                else:
                    tqdm.write(
                        f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Grad Norm: {grad_norm:.6f}, No test batches"
                    )
            else:
                tqdm.write(
                    f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Grad Norm: {grad_norm:.6f}, No test batches"
                )

            # Shuffle dataset for next epoch
            dataset.shuffle(generator=np_rng)

    if test_dataset is not None:
        return state, train_loss, final_test_loss, final_test_nll
    else:
        return state, train_loss


def _compute_nll_vector_field(model, params, batch, n_steps, rng, n_projections):
    """Compute NLL for VectorField - use real conditioning data from batch."""
    if "cond_vec" in batch:
        cond_vecs = batch["cond_vec"]
    else:
        # For unconditional models, create zero-dimensional conditioning
        batch_size = batch["point_vec"].shape[0]
        cond_vecs = jnp.zeros((batch_size, 0))

    return compute_log_probability(
        model=model,
        params=params,
        samples=batch["point_vec"],
        cond_vecs=cond_vecs,
        n_steps=n_steps,
        rng=rng,
        n_projections=n_projections,
    )


# Create VectorField-specific training loop using partial application
_train_loop_for_tests = partial(
    _train_loop_for_tests_generic,
    compute_nll_fn=_compute_nll_vector_field,
)


_baseline_model = VectorField(
    domain_dim=3,
    reference_directions=128,
    n_layers=6,
    d_model=512,
    time_dim=128,
    mlp_expansion_factor=4,
    activations_dtype=jnp.float32,
    weights_dtype=jnp.float32,
    conditioning_dim=0,
    input_dropout_rate=None,
    mlp_dropout_rate=None,
    use_pre_mlp_projection=True,
)


def _mk_model(
    model_kind: str, *, domain_dim: int, conditioning_dim: int = 0, **overrides
):
    """Factory for test models: 'mlp' (VectorField) or 'transformer'."""
    if model_kind == "mlp":
        base = replace(
            _baseline_model,
            domain_dim=domain_dim,
            conditioning_dim=conditioning_dim,
        )
        if overrides:
            base = replace(base, **overrides)
        return base
    elif model_kind == "transformer":
        n_layers = overrides.get("n_layers", 4)
        d_model = overrides.get("d_model", 64)
        return TransformerVectorField(
            domain_dim=domain_dim,
            conditioning_dim=conditioning_dim,
            n_layers=n_layers,
            d_model=d_model,
            mlp_expansion_factor=overrides.get("mlp_expansion_factor", 4),
            n_heads=overrides.get("n_heads", 4),
            # target 8 total tokens for these simple tests
            n_learned_tokens=overrides.get(
                "n_learned_tokens", 6 if conditioning_dim == 0 else 5
            ),
            mlp_dropout_rate=overrides.get("mlp_dropout_rate", None),
            input_dropout_rate=overrides.get("input_dropout_rate", None),
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")


@pytest.mark.parametrize(
    "inject_keys",
    [
        frozenset(),
        frozenset({"x"}),
        frozenset({"t"}),
        frozenset({"x", "t"}),
    ],
)
@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_trivial(model_kind, domain_dim, inject_keys):
    "Train a model with a single example"
    if model_kind == "mlp":
        model = _mk_model("mlp", domain_dim=domain_dim, mlp_always_inject=inject_keys)
        lr = 1e-3
        ep = 2 if domain_dim == 3 else 2
    else:
        model = _mk_model("transformer", domain_dim=domain_dim)
        if inject_keys:
            pytest.skip("Transformer with inject_keys not supported")
        lr = 1e-3
        ep = 3

    batch_size = 256
    # Create a unit vector in the first dimension
    first_dim_vec = jnp.zeros(domain_dim)
    first_dim_vec = first_dim_vec.at[0].set(1.0)
    points = repeat(first_dim_vec, "v -> b v", b=batch_size * 100)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    state, loss = _train_loop_for_tests(model, dset, batch_size, lr, ep)
    print(f"Final loss: {loss:.6f}")
    samples = generate_samples(
        model,
        state.params,
        jax.random.PRNGKey(0),
        cond_vecs=jnp.zeros((20, 0)),
        n_steps=1000,
        method="rk4",
    )
    cos_sims = samples @ points[0]

    print(f"Sample cosine similarities for domain_dim={domain_dim}:")
    for i, (sample, cos_sim) in enumerate(zip(samples, cos_sims)):
        if domain_dim == 3 or i < 3:  # Print all for 3D, just first 3 for 16D
            sample_str = ", ".join([f"{x:9.6f}" for x in sample[: min(3, domain_dim)]])
            if domain_dim > 3:
                sample_str += ", ..."
            print(f"Sample: [{sample_str}]  Cosine similarity: {cos_sim:9.6f}")

    assert cos_sims.shape == (20,)
    # It's very hard for the network to learn the vector field near the antipode of our target
    # point. That region of the sphere is very sparsely covered by paths. So any samples that start
    # there get stuck. Hopefully this isn't an issue with nontrivial distributions.
    high_sims = cos_sims > 0.99
    assert high_sims.mean() >= 0.95


def _vmf_differential_entropy(kappa):
    "Calculate the differential entropy of a von Mises-Fisher distribution. Used for tests."
    sinh_k = np.sinh(kappa)
    coth_k = np.cosh(kappa) / sinh_k
    return np.log(4 * np.pi * sinh_k / kappa) - kappa * coth_k + 1


def vmf_scale_kappa(kappa_src, dim_src, dim_target):
    """
    Scale the von Mises-Fisher concentration parameter from a source dimension to a target
    dimension. The resulting kappa value, when used to specify a vMF distribution in dim_target
    dimensions, will have the same mean similarity to the distribution's mean direction as a
    distribution with the source kappa in dim_src dimensions.
    """
    if dim_src == dim_target:
        return kappa_src

    n_samples = 16_384
    eps = 0.01
    max_iters = 100

    np_rng = np.random.Generator(np.random.PCG64(seed=20250317))

    def get_mean_similarity(kappa, north):
        # There are non-stochastic methods of doing this but they have numerical issues and give bad
        # results. This is slower, but reliable.
        vmf_distribution = stats.vonmises_fisher(north, kappa)
        samples = vmf_distribution.rvs(n_samples, random_state=np_rng)
        return (samples @ north).mean()

    src_north = np.zeros(dim_src)
    src_north[0] = 1.0
    src_mean_similarity = get_mean_similarity(kappa_src, src_north)
    print(f"src mean similarity: {src_mean_similarity:.6f}")

    target_north = np.zeros(dim_target)
    target_north[0] = 1.0

    low = kappa_src if dim_src < dim_target else kappa_src * (dim_src / dim_target)
    incr = 200
    while get_mean_similarity(low, target_north) > src_mean_similarity:
        print(f"find lower bound: {low:.6f}")
        low -= incr
        incr *= 2
    high = kappa_src if dim_src > dim_target else kappa_src * (dim_target / dim_src)
    incr = 200
    while get_mean_similarity(high, target_north) < src_mean_similarity:
        print(f"find upper bound: {high:.6f}")
        high += incr
        incr *= 2

    for i in range(max_iters):
        test_kappa = (low + high) / 2.0
        test_mean_similarity = get_mean_similarity(test_kappa, target_north)
        error = test_mean_similarity - src_mean_similarity
        print(
            f"Iteration {i}, low: {low:.6f}, high: {high:.6f}, test kappa: {test_kappa:.6f}, similarity: {test_mean_similarity:.6f}, error: {error:.6f}"
        )
        if jnp.abs(error) < eps:
            return test_kappa
        if test_mean_similarity > src_mean_similarity:
            high = test_kappa
        else:
            low = test_kappa
    raise ValueError(
        f"Failed to find kappa for dim_src={dim_src} and dim_target={dim_target}, final error: {error:.6f}"
    )


@pytest.mark.parametrize(
    "inject_keys",
    [
        frozenset(),
        frozenset({"x"}),
        frozenset({"t"}),
        frozenset({"x", "t"}),
    ],
)
@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_vmf(model_kind, domain_dim, inject_keys):
    """
    Train a model with data from a von Mises-Fisher distribution and evaluate the samples.
    """
    if model_kind == "mlp":
        model = _mk_model(
            "mlp", domain_dim=domain_dim, n_layers=2, mlp_always_inject=inject_keys
        )
    else:
        model = _mk_model("transformer", domain_dim=domain_dim)
        if inject_keys:
            pytest.skip("Transformer with inject_keys not supported")
    epochs = 1

    batch_size = 512
    n_samples = 32768

    # Create a mean direction that points along the first dimension
    mean_direction = np.zeros(domain_dim)
    mean_direction[0] = 1.0
    kappa = 2

    # Generate samples from von Mises-Fisher distribution
    vmf = stats.vonmises_fisher(mean_direction, kappa)
    points = vmf.rvs(n_samples)

    dset = Dataset.from_dict({"point_vec": points}).with_format("np")

    test_n_samples = 512
    test_points = vmf.rvs(test_n_samples)
    print(f"Test mean log density: {vmf.logpdf(test_points).mean():.6f}")
    test_dset = Dataset.from_dict({"point_vec": test_points}).with_format("np")

    # Compute the differential entropy of the vMF distribution
    differential_entropy = _vmf_differential_entropy(kappa)
    print(f"vMF distribution entropy: {differential_entropy:.6f}")

    state, train_loss, test_loss, test_nll = _train_loop_for_tests(
        model, dset, batch_size, 1e-3, epochs, test_dset
    )
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test nll: {test_nll:.6f}")

    assert test_nll < differential_entropy + 0.5

    n_test_samples = 1_000
    samples = generate_samples(
        model,
        state.params,
        jax.random.PRNGKey(42),
        cond_vecs=jnp.zeros((n_test_samples, 0)),
        n_steps=1000,
        method="rk4",
    )

    # Calculate negative log-likelihood of samples under the VMF distribution
    samples_np = np.array(samples)
    log_probs = vmf.logpdf(samples_np)
    sample_nll = -np.mean(log_probs)

    print(f"Sample NLL: {sample_nll:.6f}")

    # Print some sample points and their log probabilities
    print("\nSample points and their log probabilities:")
    for i in range(min(5, n_test_samples)):
        sample = samples[i]
        log_prob = log_probs[i]
        cos_sim = np.dot(sample, mean_direction)

        if domain_dim == 3:
            sample_str = f"[{sample[0]:9.6f}, {sample[1]:9.6f}, {sample[2]:9.6f}]"
        else:
            # For higher dimensions, just show first few components
            sample_str = f"[{sample[0]:9.6f}, {sample[1]:9.6f}, {sample[2]:9.6f}, ...]"

        print(
            f"Sample: {sample_str}  "
            f"Log prob: {log_prob:9.6f}  Cosine similarity: {cos_sim:9.6f}"
        )

    # Allow some tolerance due to sampling variability
    assert (
        abs(sample_nll - differential_entropy) < 1.0
    ), f"Sample NLL {sample_nll} too far from differential entropy {differential_entropy}"


@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_uniform_zero_field(model_kind, domain_dim):
    """
    Train a model with uniformly distributed data on the sphere and verify that the learned
    vector field is approximately zero everywhere.

    Since the uniform distribution is the stationary distribution (source equals target),
    the optimal flow field should be zero everywhere.
    """
    if model_kind == "mlp":
        model = _mk_model("mlp", domain_dim=domain_dim, n_layers=4, d_model=256)
    else:
        model = _mk_model("transformer", domain_dim=domain_dim)
    epochs = 8

    batch_size = 512
    n_samples = 50000

    # Generate uniformly distributed points on the sphere
    rng = jax.random.PRNGKey(12345)
    data_rng, eval_rng, params_rng = jax.random.split(rng, 3)

    points = sample_sphere(data_rng, n_samples, domain_dim)
    dsets = (
        Dataset.from_dict({"point_vec": points})
        .with_format("np")
        .train_test_split(test_size=512)
    )
    train_dset = dsets["train"]
    test_dset = dsets["test"]

    # Set up test points for evaluation (same for pre- and post-training)
    n_test_points = 1000
    test_rng, eval_rng = jax.random.split(eval_rng)
    test_points = sample_sphere(eval_rng, n_test_points, domain_dim)
    test_times = jax.random.uniform(
        test_rng, (n_test_points,)
    )  # Random times in [0, 1]
    test_cond_vecs = jnp.zeros((n_test_points, 0))

    def compute_and_print_field_stats(params, label):
        field_values = model.apply(params, test_points, test_times, test_cond_vecs)
        field_magnitudes = jnp.linalg.norm(field_values, axis=1)
        mean_magnitude = jnp.mean(field_magnitudes)
        std_magnitude = jnp.std(field_magnitudes)
        max_magnitude = jnp.max(field_magnitudes)

        print(f"{label} vector field statistics:")
        print(f"  Mean magnitude: {mean_magnitude:.6f}")
        print(f"  Std magnitude: {std_magnitude:.6f}")
        print(f"  Max magnitude: {max_magnitude:.6f}")

        return (
            field_values,
            field_magnitudes,
            mean_magnitude,
            std_magnitude,
            max_magnitude,
        )

    # Initialize model and evaluate pre-training vector field
    pre_train_state = create_train_state(params_rng, model, 1e-3)

    # Pre-training evaluation
    (
        pre_field_values,
        pre_field_magnitudes,
        pre_mean_magnitude,
        pre_std_magnitude,
        pre_max_magnitude,
    ) = compute_and_print_field_stats(pre_train_state.params, "Pre-training")

    # Train the model (note: _train_loop_for_tests creates its own training state)
    state, train_loss, test_loss, test_nll = _train_loop_for_tests(
        model, train_dset, batch_size, 1e-3, epochs, test_dset
    )

    expected_nll = -sphere_log_inverse_surface_area(domain_dim)
    print(
        f"Final train loss: {train_loss:.6f}, final test loss: {test_loss:.6f}, final test nll: {test_nll:.6f}, expected nll: {expected_nll:.6f}"
    )
    np.testing.assert_allclose(test_nll, expected_nll, atol=1e-2, rtol=0)

    # Post-training evaluation (same test points as pre-training)
    (
        post_field_values,
        post_field_magnitudes,
        post_mean_magnitude,
        post_std_magnitude,
        post_max_magnitude,
    ) = compute_and_print_field_stats(state.params, "Post-training")
    print(f"  Expected: ~0.0 (uniform distribution should have zero flow)")

    # Compare pre- and post-training
    fmt_delta = (
        lambda label, pre, post: f"  {label} change: {pre:.6f} -> {post:.6f} (Δ = {post - pre:.6f})"
    )
    print(f"Training effect:")
    print(fmt_delta("Mean magnitude", pre_mean_magnitude, post_mean_magnitude))
    print(fmt_delta("Std magnitude", pre_std_magnitude, post_std_magnitude))
    print(fmt_delta("Max magnitude", pre_max_magnitude, post_max_magnitude))

    # Print some example field values
    print(f"\nExample post-training field values at test points:")
    for i in range(min(5, n_test_points)):
        point = test_points[i]
        field = post_field_values[i]
        magnitude = post_field_magnitudes[i]

        if domain_dim == 3:
            point_str = f"[{point[0]:7.4f}, {point[1]:7.4f}, {point[2]:7.4f}]"
            field_str = f"[{field[0]:7.4f}, {field[1]:7.4f}, {field[2]:7.4f}]"
        else:
            point_str = f"[{point[0]:7.4f}, {point[1]:7.4f}, {point[2]:7.4f}, ...]"
            field_str = f"[{field[0]:7.4f}, {field[1]:7.4f}, {field[2]:7.4f}, ...]"

        print(f"  Point: {point_str} -> Field: {field_str}, Magnitude: {magnitude:.6f}")

    assert post_mean_magnitude < 0.1, (
        f"Mean field magnitude {post_mean_magnitude:.6f} too large for uniform distribution "
        f"(should be < 0.1)"
    )

    assert post_std_magnitude < 0.05, (
        f"Std field magnitude {post_std_magnitude:.6f} too large for uniform distribution "
        f"(should be < 0.05)"
    )


@pytest.mark.parametrize(
    "inject_keys",
    [
        frozenset(),
        frozenset({"x"}),
        frozenset({"t"}),
        frozenset({"cond"}),
        frozenset({"x", "t"}),
        frozenset({"x", "cond"}),
        frozenset({"t", "cond"}),
        frozenset({"x", "t", "cond"}),
    ],
)
@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
@pytest.mark.parametrize("domain_dim", [3, 16])
def test_train_conditional_vmf(model_kind, domain_dim, inject_keys):
    """
    Train a model with data from two different von Mises-Fisher distributions
    conditioned on a binary conditioning vector.

    When conditioning vector is 0, samples should come from the first vMF distribution.
    When conditioning vector is 1, samples should come from the second vMF distribution.
    """
    if model_kind == "mlp":
        model = _mk_model(
            "mlp",
            domain_dim=domain_dim,
            conditioning_dim=1,
            n_layers=2,
            mlp_always_inject=inject_keys,
        )
        epochs = 2 if domain_dim == 3 else 4
        batch_size = 256
        lr = 1e-3
    else:
        model = _mk_model("transformer", domain_dim=domain_dim, conditioning_dim=1)
        if inject_keys:
            pytest.skip("Transformer with inject_keys not supported")
        epochs = 2
        batch_size = 256
        lr = 1e-3

    # Define two different vMF distributions
    mean_direction1 = np.zeros(domain_dim)
    mean_direction1[0] = 1.0
    mean_direction2 = np.zeros(domain_dim)
    mean_direction2[1] = 1.0

    vmf_kappa = 2.0 if domain_dim == 3 else 5.0

    vmf1 = stats.vonmises_fisher(mean_direction1, vmf_kappa)
    vmf2 = stats.vonmises_fisher(mean_direction2, vmf_kappa)

    # Sample from our distributions
    np_rng = np.random.Generator(np.random.PCG64(seed=42))
    total_samples = 60_000
    points1 = vmf1.rvs(total_samples // 2)
    points2 = vmf2.rvs(total_samples // 2)

    cond_vec1 = np.zeros((total_samples // 2, 1))
    cond_vec2 = np.ones((total_samples // 2, 1))

    all_points = np.vstack([points1, points2])
    all_cond_vecs = np.vstack([cond_vec1, cond_vec2])

    datasets = (
        Dataset.from_dict({"point_vec": all_points, "cond_vec": all_cond_vecs})
        .with_format("np")
        .train_test_split(test_size=2048)
    )
    train_dataset = datasets["train"]
    test_dataset = datasets["test"]

    state, train_loss, test_loss, test_nll = _train_loop_for_tests(
        model,
        train_dataset,
        batch_size,
        lr,
        epochs,
        test_dataset,
    )
    print(
        f"Final loss - Train: {train_loss:.4f}, Test: {test_loss:.4f}, Test NLL: {test_nll:.4f}"
    )

    # Test the conditioned model
    n_eval_samples = 200
    seed = jax.random.PRNGKey(123)
    seed1, seed2 = jax.random.split(seed)

    # Generate samples with each conditioning
    cond_vec_0 = jnp.zeros((n_eval_samples, 1))
    cond_vec_1 = jnp.ones((n_eval_samples, 1))

    samples_0 = generate_samples(
        model,
        state.params,
        seed1,
        cond_vecs=cond_vec_0,
        n_steps=500,
        method="rk4",
    )
    samples_1 = generate_samples(
        model,
        state.params,
        seed2,
        cond_vecs=cond_vec_1,
        n_steps=500,
        method="rk4",
    )

    # Calculate average cosine similarities
    cos_sim_0_dir1 = np.mean(samples_0 @ mean_direction1)
    cos_sim_0_dir2 = np.mean(samples_0 @ mean_direction2)
    cos_sim_1_dir1 = np.mean(samples_1 @ mean_direction1)
    cos_sim_1_dir2 = np.mean(samples_1 @ mean_direction2)

    # Validate correct conditioning behavior
    print(
        f"Conditioning=0 - Avg similarity with dir1: {cos_sim_0_dir1:.4f}, dir2: {cos_sim_0_dir2:.4f}"
    )
    print(
        f"Conditioning=1 - Avg similarity with dir1: {cos_sim_1_dir1:.4f}, dir2: {cos_sim_1_dir2:.4f}"
    )

    # Samples should align more closely with their respective directions that the other directions
    assert (
        cos_sim_0_dir1 > cos_sim_0_dir2
    ), "Samples with cond=0 don't align more closely with direction1 than direction2"
    # Samples with cond=1 should align better with direction2
    assert (
        cos_sim_1_dir2 > cos_sim_1_dir1
    ), "Samples with cond=1 don't align more closely with direction2 than direction1"

    # Verify that conditioning makes a significant difference
    diff = (cos_sim_0_dir1 - cos_sim_0_dir2) + (cos_sim_1_dir2 - cos_sim_1_dir1)
    assert (
        diff > 0.5
    ), "Conditioning did not produce sufficiently different distributions"

    # Additional check: negative log-likelihood (samples should match their respective distributions)
    samples_0_np = np.array(samples_0)
    samples_1_np = np.array(samples_1)

    nll_0_from_vmf1 = -np.mean(vmf1.logpdf(samples_0_np))
    nll_0_from_vmf2 = -np.mean(vmf2.logpdf(samples_0_np))
    nll_1_from_vmf1 = -np.mean(vmf1.logpdf(samples_1_np))
    nll_1_from_vmf2 = -np.mean(vmf2.logpdf(samples_1_np))

    assert (
        nll_0_from_vmf1 < nll_0_from_vmf2
    ), "Samples with cond=0 don't match distribution 1"
    assert (
        nll_1_from_vmf2 < nll_1_from_vmf1
    ), "Samples with cond=1 don't match distribution 2"

    distribution_nll = vmf1.entropy()
    print(f"Theoretical NLL: {distribution_nll:.4f}")
    print(f"NLL with cond=0 from vmf1: {nll_0_from_vmf1:.4f}")
    print(f"NLL with cond=0 from vmf2: {nll_0_from_vmf2:.4f}")
    print(f"NLL with cond=1 from vmf1: {nll_1_from_vmf1:.4f}")
    print(f"NLL with cond=1 from vmf2: {nll_1_from_vmf2:.4f}")
    assert (
        abs(nll_0_from_vmf1 - distribution_nll) < 0.5
    ), "NLL with cond=0 from vmf1 is not close to the theoretical NLL"
    assert (
        abs(nll_1_from_vmf2 - distribution_nll) < 0.5
    ), "NLL with cond=1 from vmf2 is not close to the theoretical NLL"


@partial(jax.jit, inline=True)
def geodesic_step(x, v, dt):
    """
    Take a step along a geodesic from point x with initial velocity v.

    Args:
        x: Current point on the sphere [batch_size, dim]
        v: Tangent vector at x [batch_size, dim]
        dt: Time step

    Returns:
        New point on the sphere after stepping along geodesic
    """
    assert len(x.shape) == 2
    assert x.shape == v.shape
    if isinstance(dt, jax.Array):
        assert dt.shape == ()
    else:
        assert isinstance(dt, float)

    v_norm = jnp.linalg.norm(v, axis=1, keepdims=True)

    # Angle to rotate = velocity * time
    angle = v_norm * dt

    # Compute the rotation axis (normalized tangent vector)
    axis = jnp.where(
        v_norm > 1e-8,
        v / v_norm,
        # Fallback for zero velocity
        jnp.zeros_like(v),
    )

    # Apply Rodrigues' rotation formula
    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)

    ret = cos_angle * x + sin_angle * axis
    # Normalize for numerical stability
    return ret / jnp.linalg.norm(ret, axis=1, keepdims=True)


@partial(jax.jit, inline=True)
def parallel_transport(v, from_point, to_point):
    """
    Parallel transport a tangent vector v from one point to another on the sphere.

    Args:
        v: Tangent vector at from_point [batch_size, dim]
        from_point: Starting point on sphere [batch_size, dim]
        to_point: Destination point on sphere [batch_size, dim]

    Returns:
        Transported vector in the tangent space of to_point [batch_size, dim]
    """
    # Get unit vector along the geodesic from from_point to to_point
    geodesic_dir = to_point - from_point * jnp.sum(
        to_point * from_point, axis=1, keepdims=True
    )
    geodesic_norm = jnp.linalg.norm(geodesic_dir, axis=1, keepdims=True)

    # If points are very close or antipodal, return the vector as is
    is_valid = geodesic_norm > 1e-8
    geodesic_dir = jnp.where(
        is_valid, geodesic_dir / geodesic_norm, jnp.zeros_like(geodesic_dir)
    )

    # Double cross product gives the parallel transport
    # (I - u⊗u - v⊗v) * w where u=from_point, v=to_point, w=tangent vector
    # This is the same as w - (w·u)u - (w·v)v + (w·u)(u·v)v
    v_dot_from = jnp.sum(v * from_point, axis=1, keepdims=True)
    v_dot_to = jnp.sum(v * to_point, axis=1, keepdims=True)
    from_dot_to = jnp.sum(from_point * to_point, axis=1, keepdims=True)

    result = (
        v
        - v_dot_from * from_point
        - v_dot_to * to_point
        + v_dot_from * from_dot_to * to_point
    )

    # Project to tangent space of to_point to ensure numerical stability
    dot_with_to = jnp.sum(result * to_point, axis=1, keepdims=True)
    return result - dot_with_to * to_point


@partial(jax.jit, inline=True, static_argnames=("f", "f_fixed_static_params"))
def spherical_rk4_step(
    f, f_fixed_static_params, f_fixed_params, f_per_sample_params, x, t, dt, rng=None
):
    """
    Runge-Kutta 4th order step adapted for spherical manifold with correct parallel transport.

    Args:
        f: Function that computes the tangent vector field
        x: Current point on sphere [batch_size, dim]
        t: Current time
        dt: Time step

    Returns:
        Next point on sphere
    """
    # Each result from f is in the tangent space of f's input point, when we combine them we need to
    # parallel transport them all into the tangent space of x.
    k1 = f(f_fixed_static_params, f_fixed_params, f_per_sample_params, x, t, rng)

    x2 = geodesic_step(x, k1, dt / 2)
    k2 = f(
        f_fixed_static_params, f_fixed_params, f_per_sample_params, x2, t + dt / 2, rng
    )
    k2_at_x = parallel_transport(k2, x2, x)

    x3 = geodesic_step(x, k2_at_x, dt / 2)
    k3 = f(
        f_fixed_static_params, f_fixed_params, f_per_sample_params, x3, t + dt / 2, rng
    )
    k3_at_x = parallel_transport(k3, x3, x)

    x4 = geodesic_step(x, k3_at_x, dt)
    k4 = f(f_fixed_static_params, f_fixed_params, f_per_sample_params, x4, t + dt, rng)
    k4_at_x = parallel_transport(k4, x4, x)

    combined_direction = (k1 + 2 * k2_at_x + 2 * k3_at_x + k4_at_x) / 6

    return geodesic_step(x, combined_direction, dt)


@partial(jax.jit, static_argnames=("model",), inline=True)
def _compute_vector_field_for_sampling(
    model,
    params,
    cond_vecs,
    x,
    t,
    rng=None,
):
    rngs_dict = {"dropout": rng} if rng is not None else {}
    return model.apply(
        params,
        x,
        jnp.full((x.shape[0],), t),
        cond_vecs,
        rngs=rngs_dict,
    )


def generate_samples(
    model,
    params,
    rng,
    cond_vecs,
    n_steps=100,
    method="rk4",
):
    """
    Generate samples from a VectorField flow matching model by solving the ODE.

    Args:
        model: Vector field model
        params: Model parameters
        rng: JAX random key
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        method: ODE solver method ('euler', 'midpoint', or 'rk4')

    Returns:
        Generated samples [batch_size, domain_dim]
    """

    assert len(cond_vecs.shape) == 2
    batch_size = cond_vecs.shape[0]
    return generate_samples_inner(
        rng,
        n_steps,
        batch_size,
        method,
        _compute_vector_field_for_sampling,
        model,
        params,
        cond_vecs,
        model.domain_dim,
    )


def generate_samples_inner(
    rng,
    n_steps,
    batch_size,
    method,
    vector_field_fn,
    vector_field_fn_fixed_static_params,
    vector_field_fn_fixed_params,
    vector_field_fn_per_sample_params,
    domain_dim,
):
    """
    Generate samples from a flow matching model by solving the ODE, generic over the method of
    computing the vector field.

    Args:
        rng: JAX random key
        n_steps: Number of integration steps
        batch_size: Number of samples to generate
        method: ODE solver method ('euler', 'midpoint', or 'rk4')
        vector_field_fn: Function that computes the tangent vector field
        vector_field_fn_fixed_static_params: Parameters to pass to vector_field_fn that are constant
            across all samples and may be marked static to jax.jit. (separate parameter so
            static_argnums will work.)
        vector_field_fn_fixed_params: Parameters to pass to vector_field_fn that are constant across
            all samples
        vector_field_fn_per_sample_params: PyTree of per-sample parameters with leading dim
            batch_size.
        domain_dim: Dimension of the domain

    Returns:
        Generated samples [batch_size, dim]
    """
    x0_rng, *dropout_rngs = jax.random.split(rng, num=n_steps + 1)
    # Sample initial points uniformly from the sphere
    x0 = sample_sphere(x0_rng, batch_size, domain_dim)

    # Solve ODE
    dt = 1.0 / n_steps
    x = x0

    # Use tqdm for progress tracking
    step_iter = tqdm(range(n_steps), desc=f"ODE solving ({method})", leave=False)

    if method == "euler":
        # Forward Euler method
        for i in step_iter:
            t = i * dt
            v = vector_field_fn(
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                vector_field_fn_per_sample_params,
                x,
                t,
                dropout_rngs[i],
            )
            x = geodesic_step(x, v, dt)
    elif method == "midpoint":
        # Midpoint method
        for i in step_iter:
            t = i * dt
            # First half-step
            v1 = vector_field_fn(
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                vector_field_fn_per_sample_params,
                x,
                t,
                dropout_rngs[i],
            )
            x_mid = geodesic_step(x, v1, dt / 2)

            # Second half-step using midpoint derivative
            v2 = vector_field_fn(
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                vector_field_fn_per_sample_params,
                x_mid,
                t + 0.5 * dt,
                dropout_rngs[i],
            )
            x = geodesic_step(x, v2, dt)
    elif method == "rk4":
        # 4th order Runge-Kutta method
        for i in step_iter:
            t = i * dt
            x = spherical_rk4_step(
                vector_field_fn,
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                vector_field_fn_per_sample_params,
                x,
                t,
                dt,
                rng=dropout_rngs[i],
            )
    else:
        raise ValueError(f"Unknown ODE solver method: {method}")

    return x


def sample_loop(
    model,
    params,
    n_samples,
    batch_size,
    rng,
    cond_vecs,
    n_steps=100,
    method="rk4",
):
    """
    Generate multiple batches of samples from the flow matching model.
    Handles batching for large sample counts.

    Args:
        model: Vector field model
        params: Model parameters
        n_samples: Total number of samples to generate
        batch_size: Size of each batch for generation
        rng: JAX random key
        cond_vecs: Conditioning vectors [n_samples, cond_dim]
        n_steps: Number of integration steps
        method: ODE solver method ('euler', 'midpoint', or 'rk4')

    Returns:
        Generated samples [n_samples, dim]
    """
    samples = []
    samples_so_far = 0
    from tqdm import trange

    for i in trange(
        ceil(n_samples / batch_size), unit="batch", desc="Generating samples"
    ):
        batch_rng, rng = jax.random.split(rng)
        this_batch_size = min(batch_size, n_samples - samples_so_far)

        # Slice the conditioning vectors
        batch_cond_vecs = cond_vecs[i * batch_size : i * batch_size + this_batch_size]

        samples.append(
            generate_samples(
                model,
                params,
                batch_rng,
                cond_vecs=batch_cond_vecs,
                n_steps=n_steps,
                method=method,
            )
        )
        samples_so_far += this_batch_size

    return jnp.concatenate(samples, axis=0)


@partial(
    jax.jit,
    static_argnames=(
        "vector_field_fn",
        "vector_field_fn_fixed_static_params",
        "n_projections",
    ),
)
def hutchinson_estimator(
    vector_field_fn,
    vector_field_fn_fixed_static_params,
    vector_field_fn_fixed_params,
    vector_field_fn_per_sample_params,
    x,
    t,
    step_rng,
    n_projections,
):
    """
    Estimate the divergence of a vector field on a spherical manifold using Hutchinson's trace
    estimator. Properly handles the (d-1)-dimensional tangent space of the d-dimensional sphere.

    Args:
        vector_field_fn: Vector field function
        vector_field_fn_fixed_static_params: Parameters to pass to vector_field_fn that are constant
            across all samples and may be marked static to jax.jit. (separate parameter so
            static_argnums will work.)
        vector_field_fn_fixed_params: Parameters to pass to vector_field_fn that are constant across
            all samples
        vector_field_fn_per_sample_params: PyTree of per-sample parameters with leading dim
            batch_size.
        x: Current points on the sphere [batch_size, dim]
        t: Current time
        step_rng: JAX random key for this step
        n_projections: Number of random projections to use

    Returns:
        Divergence estimate [batch_size]
    """
    batch_size, dim = x.shape
    assert isinstance(t, float) or (isinstance(t, jnp.ndarray) and t.shape == ())
    assert n_projections > 0, "n_projections must be positive"
    per_sample_leading_dims = jax.tree.map(
        lambda x: x.shape[0], vector_field_fn_per_sample_params
    )
    if per_sample_leading_dims is not None:
        # Handle both single values and tree structures
        if isinstance(per_sample_leading_dims, int):
            # Single array case
            assert (
                per_sample_leading_dims == batch_size
            ), "Per-sample leading dim must equal batch_size"
        else:
            # Tree structure case
            assert all(
                x == batch_size for x in jax.tree.leaves(per_sample_leading_dims)
            ), "All per-sample leading dims must equal batch_size"

    def hutchinson_single(x_i, rng_i, per_sample_i):
        """Compute Hutchinson divergence estimate for a single point."""
        dropout_rng, projection_rng = jax.random.split(rng_i)

        def f(x_single):
            return vector_field_fn(
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                jax.tree.map(
                    lambda x: x[None, :] if x.ndim >= 1 else jnp.array([x]),
                    per_sample_i,
                ),
                x_single[None, :],
                jnp.array([t]),
                dropout_rng,
            )[0]

        # Generate random projection vectors (Rademacher distribution)
        projection_keys = jax.random.split(projection_rng, n_projections)
        v_samples = jax.vmap(
            lambda key: jax.random.rademacher(key, (dim,), dtype=x_i.dtype)
        )(projection_keys)

        # Compute v^T * J * v for each random vector v
        def vjp_fn(v):
            _, vjp = jax.vjp(f, x_i)
            return jnp.dot(v, vjp(v)[0])

        trace_estimates = jax.vmap(vjp_fn)(v_samples)
        trace_estimate = jnp.mean(trace_estimates)

        # Compute the curvature correction term: x^T J x
        # We can compute this exactly using forward-mode autodiff
        _, jvp_result = jax.jvp(f, (x_i,), (x_i,))
        curvature_term = jnp.dot(x_i, jvp_result)

        return trace_estimate - curvature_term

    # Split random keys for each batch element
    batch_keys = jax.random.split(step_rng, batch_size)

    return jax.vmap(hutchinson_single)(x, batch_keys, vector_field_fn_per_sample_params)


@partial(
    jax.jit,
    static_argnames=(
        "vector_field_fn",
        "vector_field_fn_fixed_static_params",
        "n_projections",
    ),
)
def exact_divergence(
    vector_field_fn,
    vector_field_fn_fixed_static_params,
    vector_field_fn_fixed_params,
    vector_field_fn_per_sample_params,
    x,
    t,
    step_rng,
    n_projections,
):
    """
    Compute the exact divergence of a vector field on a spherical manifold.

    Args:
        vector_field_fn: Vector field function
        vector_field_fn_fixed_static_params: Parameters to pass to vector_field_fn that are constant
            across all samples and may be marked static to jax.jit. (separate parameter so
            static_argnums will work.)
        vector_field_fn_fixed_params: Parameters to pass to vector_field_fn that are constant across
            all samples
        vector_field_fn_per_sample_params: PyTree of per-sample parameters with leading dim
            batch_size.
        x: Current points on the sphere [batch_size, dim]
        t: Current time (scalar)
        step_rng: JAX random key (for dropout)
        n_projections: Unused (kept for signature compatibility with hutchinson_estimator)

    Returns:
        Exact divergence [batch_size]
    """
    batch_size, dim = x.shape
    assert isinstance(t, float) or (isinstance(t, jnp.ndarray) and t.shape == ())

    # Helper to compute divergence for regular vector field
    def divergence_single(x_i, per_sample_i):
        def f(x_single):
            return vector_field_fn(
                vector_field_fn_fixed_static_params,
                vector_field_fn_fixed_params,
                jax.tree.map(lambda x: x[None, :], per_sample_i),
                x_single[None, :],
                jnp.array([t]),
                step_rng,
            )[0]

        jac = jax.jacfwd(f)(x_i)
        return jnp.trace(jac) - jnp.dot(x_i, jac @ x_i)

    return jax.vmap(divergence_single)(x, vector_field_fn_per_sample_params)


@partial(
    jax.jit,
    static_argnames=(
        "vector_field_fn",
        "vector_field_fn_fixed_static_params",
        "n_steps",
        "n_projections",
    ),
)
def _reverse_path_and_compute_divergence(
    vector_field_fn,
    vector_field_fn_fixed_static_params,
    vector_field_fn_fixed_params,
    vector_field_fn_per_sample_params,
    samples,
    n_steps,
    rng,
    n_projections=10,
):
    """
    Compute the reverse path and integrate the divergence.

    Args:
        model: Vector field model
        params: Model parameters
        samples: Points on the sphere to evaluate [batch_size, dim]
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        rng: JAX random key for stochastic estimation
        n_projections: Number of random projections to use for divergence estimation

    Returns:
        div_sum: Integrated divergence along the path [batch_size]
    """
    batch_size = samples.shape[0]

    ts = jnp.linspace(1.0, 0.0, n_steps)

    # Initialize state for fori_loop
    # We'll track the current position and the accumulated divergence
    x_t = samples
    div_sum = jnp.zeros(batch_size)

    # Loop going backwards in time while accumulating the divergence
    def body_fun(i, loop_state):
        x_t, div_sum, rng = loop_state
        t = ts[i]

        # Compute divergence at current point
        step_rng_model, step_rng_div, rng = jax.random.split(rng, 3)
        div_t = hutchinson_estimator(
            vector_field_fn,
            vector_field_fn_fixed_static_params,
            vector_field_fn_fixed_params,
            vector_field_fn_per_sample_params,
            x_t,
            t,
            step_rng_div,
            n_projections,
        )
        div_sum = div_sum + div_t * (1.0 / n_steps)

        # Take a step backward along the path
        next_x = spherical_rk4_step(
            vector_field_fn,
            vector_field_fn_fixed_static_params,
            vector_field_fn_fixed_params,
            vector_field_fn_per_sample_params,
            x_t,
            t,
            -1.0 / n_steps,
            step_rng_model,
        )

        return next_x, div_sum, rng

    # Run the loop
    init_state = (x_t, div_sum, rng)
    final_x, final_div_sum, final_rng = jax.lax.fori_loop(
        0, n_steps, body_fun, init_state
    )

    return final_div_sum


def compute_log_probability(
    model,
    params,
    samples,
    cond_vecs,
    n_steps=100,
    rng=None,
    n_projections=10,
):
    """
    Compute the log probability of samples under a flow-matching model.

    Args:
        model: Vector field model
        params: Model parameters
        samples: Points on the sphere to evaluate [batch_size, dim]
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        rng: JAX random key for stochastic estimation (if None, uses deterministic keys)
        n_projections: Number of random projections to use for divergence estimation

    Returns:
        Log probabilities of the samples [batch_size]
    """
    batch_size = samples.shape[0]
    assert samples.shape == (batch_size, model.domain_dim)
    assert cond_vecs.shape == (batch_size, model.conditioning_dim)

    return compute_log_probability_inner(
        _compute_vector_field_for_sampling,
        model,
        params,
        cond_vecs,
        samples,
        n_steps,
        rng,
        n_projections,
    )


def compute_log_probability_inner(
    vector_field_fn,
    vector_field_fn_fixed_static_params,
    vector_field_fn_fixed_params,
    vector_field_fn_per_sample_params,
    samples,
    n_steps=100,
    rng=None,
    n_projections=10,
):
    """
    Compute the log probability of samples under a flow-matching model, generic over the method of
    computing the vector field.

    Args:
        vector_field_fn: Vector field function
        vector_field_fn_fixed_static_params: Parameters to pass to vector_field_fn that are constant
            across all samples and may be marked static to jax.jit. (separate parameter so
            static_argnums will work.)
        vector_field_fn_fixed_params: Parameters to pass to vector_field_fn that are constant across
            all samples
        vector_field_fn_per_sample_params: PyTree of per-sample parameters with leading dim
            batch_size.
        samples: Points on the sphere to evaluate [batch_size, dim]
        n_steps: Number of integration steps
        rng: JAX random key for stochastic estimation (if None, uses deterministic keys)
        n_projections: Number of random projections to use for divergence estimation

    Returns:
        Log probabilities of the samples [batch_size]
    """
    batch_size = samples.shape[0]
    domain_dim = samples.shape[1]
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Normalize samples to ensure they're on the unit sphere
    samples = samples / jnp.linalg.norm(samples, axis=1, keepdims=True)

    div_sum = _reverse_path_and_compute_divergence(
        vector_field_fn,
        vector_field_fn_fixed_static_params,
        vector_field_fn_fixed_params,
        vector_field_fn_per_sample_params,
        samples,
        n_steps=n_steps,
        rng=rng,
        n_projections=n_projections,
    )
    assert div_sum.shape == (batch_size,)

    # Density of the base distribution (uniform on unit sphere)
    log_p0 = sphere_log_inverse_surface_area(domain_dim)
    log_p1 = log_p0 - div_sum
    return log_p1


# Used only for test below. It's important that this hashes to different values depending on the
# field, so the divergence functions get recompiled for the two fields.
@dataclass(frozen=True)
class _DummyModel:
    domain_dim: int
    conditioning_dim: int
    field: str

    def apply(self, params, x, t, cond_vecs, rngs=None):
        if self.field == "zero_divergence":
            return jax.vmap(lambda x_i: jnp.array([-x_i[1], x_i[0], 0]))(x)
        elif self.field == "variable_divergence":
            return jax.vmap(lambda x_i: jnp.array([0, 0, 1]) - x_i * x_i[2])(x)
        else:
            raise ValueError(f"Unknown field: {self.field}")


@pytest.mark.parametrize(
    "divergence_fn,n_projections,field",
    [
        (hutchinson_estimator, 10, "zero_divergence"),
        (hutchinson_estimator, 50, "variable_divergence"),
        (exact_divergence, None, "zero_divergence"),
        (exact_divergence, None, "variable_divergence"),
    ],
)
def test_divergence_estimate(divergence_fn, n_projections, field):
    """
    Test divergence estimators with vector fields that have known divergences.

    Args:
        divergence_fn: Function to compute divergence, either hutchinson_estimator or exact_divergence
        n_projections: Number of random projections to use (only for hutchinson_estimator)
    """

    model = _DummyModel(
        domain_dim=3,
        conditioning_dim=0,
        field=field,
    )

    # Create state with patched apply function
    rng = jax.random.PRNGKey(42)

    # Generate test points
    batch_size = 1000
    x = jax.random.normal(rng, (batch_size, 3))
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)

    if field == "zero_divergence":
        expected_divergence = jnp.full((batch_size,), 0.0)
    elif field == "variable_divergence":
        expected_divergence = jax.vmap(lambda x_i: -2 * x_i[2])(x)
    else:
        raise ValueError(f"Unknown field: {field}")

    # Compute divergence estimate
    t = 0.5
    cond_vecs = jnp.zeros((batch_size, 0))

    # Call the appropriate divergence function
    div_estimates = divergence_fn(
        _compute_vector_field_for_sampling,
        model,
        {},
        None,
        x,
        t,
        step_rng=rng,
        n_projections=n_projections,
    )

    # Calculate error
    error = jnp.abs(div_estimates - expected_divergence)
    mean_error = jnp.mean(error)

    print(f"Mean error: {mean_error}")

    # For exact method, we expect essentially no error
    if divergence_fn == exact_divergence:
        np.testing.assert_allclose(
            div_estimates, expected_divergence, rtol=1e-5, atol=1e-5
        )
    else:
        # Check if the estimate is within 10% of expected value
        assert (
            abs(mean_error) < 0.2
        ), f"Hutchinson estimator not calculating correct divergence: got {mean_error}, expected {expected_divergence}"


def test_vector_field_evaluation():
    """
    Initialize a VectorField, generate random points on the sphere, evaluate the vector field at
    those points/times, and print information about the magnitudes and directions of the vector
    field values. Less a test than a diagnostic tool.
    """
    rng = jax.random.PRNGKey(49)

    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=0,
        time_dim=32,
        reference_directions=16,
        n_layers=2,
        d_model=64,
        mlp_expansion_factor=4,
        use_pre_mlp_projection=False,
        input_dropout_rate=None,
        mlp_dropout_rate=None,
    )
    state = create_train_state(rng, model, 1e-3)

    # Generate points & times
    n_samples = 30
    key1, key2 = jax.random.split(rng)
    points = sample_sphere(key1, n_samples, model.domain_dim)
    times = jax.random.uniform(key2, (n_samples,))

    cond_vecs = jnp.zeros((n_samples, model.conditioning_dim))

    vector_field_values = model.apply(state.params, points, times, cond_vecs)

    dot_products = jnp.sum(points * vector_field_values, axis=1)
    magnitudes = jnp.linalg.norm(vector_field_values, axis=1)

    # Use stereographic projection to represent tangent vectors in (d-1) dimensions so we can easily
    # compare directions
    def stereographic_projection(point, vector):
        """
        Project a tangent vector from an n-dimensional sphere to (n-1) dimensions
        using stereographic projection from a reference point.

        Args:
            point: Point on the n-dimensional sphere [dim]
            vector: Tangent vector at that point [dim]

        Returns:
            (dim-1) coordinates representing the tangent vector
        """
        dim = point.shape[0]

        # Choose a reference point (north pole)
        north_pole = jnp.zeros(dim).at[dim - 1].set(1.0)

        # Check if point is near the north pole
        is_near_pole = jnp.abs(jnp.dot(point, north_pole) - 1.0) < 1e-6

        # If near north pole, use south pole as reference instead
        reference_pole = jnp.where(
            is_near_pole, -north_pole, north_pole  # Use south pole  # Use north pole
        )

        # Determine which coordinate to use for projection
        proj_index = dim - 1

        # Compute the denominator for projection (depends on which pole we're using)
        pole_sign = jnp.where(is_near_pole, -1.0, 1.0)
        denominator = 1.0 - pole_sign * point[proj_index]

        # Ensure vector is tangent to the sphere (orthogonal to point)
        vector_tangent = vector - jnp.dot(vector, point) * point

        # The differential of stereographic projection
        # For a tangent vector v at point p, the formula is:
        # dp_*(v) = (v[:d-1] * (1 - p[d-1]) + v[d-1] * p[:d-1]) / (1 - p[d-1])^2
        # When using south pole, we adjust the formula accordingly

        scaling = 1.0 / denominator

        # Compute the differential based on which pole we're using
        def project_from_south_pole():
            # When projecting from south pole
            proj_vector = jnp.zeros(dim - 1)
            for i in range(dim - 1):
                proj_vector = proj_vector.at[i].set(
                    (
                        vector_tangent[i] * (1.0 + point[proj_index])
                        - vector_tangent[proj_index] * point[i]
                    )
                    * scaling
                    * scaling
                )
            return proj_vector

        def project_from_north_pole():
            # When projecting from north pole
            proj_vector = jnp.zeros(dim - 1)
            for i in range(dim - 1):
                proj_vector = proj_vector.at[i].set(
                    (
                        vector_tangent[i] * (1.0 - point[proj_index])
                        + vector_tangent[proj_index] * point[i]
                    )
                    * scaling
                    * scaling
                )
            return proj_vector

        proj_vector = jax.lax.cond(
            is_near_pole, project_from_south_pole, project_from_north_pole
        )

        return proj_vector

    # Apply to all points and vectors
    projected_vectors = jax.vmap(stereographic_projection)(points, vector_field_values)

    projected_magnitudes = jnp.linalg.norm(projected_vectors, axis=1)
    nonzero_mask = projected_magnitudes > 1e-8
    projected_directions = jnp.zeros_like(projected_vectors)
    if jnp.any(nonzero_mask):
        projected_directions = projected_directions.at[nonzero_mask].set(
            projected_vectors[nonzero_mask] / projected_magnitudes[nonzero_mask, None]
        )

    # Print results
    print("\nVector Field Evaluation Test Results:")
    print("=====================================")
    print(
        f"Model: {model.n_layers} layers, {model.d_model} width, {model.mlp_expansion_factor}x expansion"
    )
    print(f"Number of samples: {n_samples}")
    print(f"Dimension of sphere: {model.domain_dim}")
    print(f"Projection dimension: {model.domain_dim - 1}")
    print(f"Average magnitude: {jnp.mean(magnitudes):.6f}")
    print(f"Min magnitude: {jnp.min(magnitudes):.6f}")
    print(f"Max magnitude: {jnp.max(magnitudes):.6f}")
    print(
        f"Average absolute dot product with point (should be ~0): {jnp.mean(jnp.abs(dot_products)):.6f}"
    )

    # Sort all data by x coordinate (first component of the point)
    sort_indices = jnp.argsort(points[:, 0])
    sorted_points = points[sort_indices]
    sorted_times = times[sort_indices]
    sorted_magnitudes = magnitudes[sort_indices]
    sorted_projected_directions = projected_directions[sort_indices]

    # Print details for all samples in a compact, aligned format
    print(f"\nDetails for all {n_samples} samples (sorted by x coordinate):")
    for i in range(n_samples):
        point_str = ", ".join(
            [f"{sorted_points[i, j]:7.4f}" for j in range(model.domain_dim)]
        )
        dir_str = ", ".join(
            [
                f"{sorted_projected_directions[i, j]:7.4f}"
                for j in range(model.domain_dim - 1)
            ]
        )
        print(
            f"Sample {i+1:2d}: t={sorted_times[i]:.4f} x=[{point_str}] "
            f"Magnitude={sorted_magnitudes[i]:6.4f} Direction{model.domain_dim-1}D=[{dir_str}]"
        )

    # Verify that vector field values are tangent to the sphere
    np.testing.assert_allclose(dot_products, 0.0, atol=1e-6)


def test_vector_field_without_reference_directions():
    """Test that the VectorField works correctly when reference_directions is None."""
    domain_dim = 3
    batch_size = 4
    time_dim = 16
    conditioning_dim = 8

    # Create a model with reference_directions=None
    model = VectorField(
        domain_dim=domain_dim,
        reference_directions=None,  # No reference directions
        conditioning_dim=conditioning_dim,
        time_dim=time_dim,
        use_pre_mlp_projection=False,
        n_layers=2,
        d_model=64,
        mlp_expansion_factor=2,
        mlp_dropout_rate=None,
        input_dropout_rate=None,
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
    )

    # Initialize the model
    rng = jax.random.PRNGKey(0)
    params = model.init(
        rng,
        jnp.ones((batch_size, domain_dim)),
        jnp.ones((batch_size,)),
        jnp.ones((batch_size, conditioning_dim)),
    )

    # Generate some test inputs
    x = sample_sphere(jax.random.PRNGKey(1), batch_size, domain_dim)
    t = jnp.linspace(0, 1, batch_size)
    cond_vec = jnp.ones((batch_size, conditioning_dim))

    # Process the inputs
    inputs = model.apply(params, x, t, cond_vec, method=model.process_inputs)

    # Check that the input features have the correct shape (should be domain_dim)
    input_features = inputs["input_features"]
    assert input_features.shape == (batch_size, domain_dim)

    # Check that the model output has the correct shape
    output = model.apply(params, x, t, cond_vec)
    assert output.shape == (batch_size, domain_dim)

    # Verify that the output vectors are tangent to the sphere
    dot_products = jnp.sum(output * x, axis=1)
    np.testing.assert_allclose(dot_products, 0.0, atol=1e-6)


def create_mollweide_projection_figure(samples, title=None):
    """
    Create a Mollweide projection visualization of 3D points on a unit sphere. Returns a matplotlib
    Figure, which the caller should close.

    Args:
        samples: Array of 3D unit vectors with shape [n_samples, 3]
        title: Optional title for the figure

    Returns:
        matplotlib Figure object
    """
    assert samples.shape[1] == 3, f"Expected 3D samples, got shape {samples.shape}"

    # Create Mollweide projection figure
    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplot(111, projection="mollweide")

    # Convert 3D coordinates to longitude/latitude
    # Mollweide projection expects longitude in [-pi, pi] and latitude in [-pi/2, pi/2]
    longitude = np.arctan2(samples[:, 1], samples[:, 0])  # atan2(y, x) for longitude
    latitude = np.arcsin(samples[:, 2])  # z-coordinate gives latitude (arcsin)

    scatter = ax.scatter(longitude, latitude, s=8, alpha=0.25)

    ax.grid(True, alpha=0.3)

    tick_formatter = ticker.FuncFormatter(lambda x, pos: f"{np.degrees(x):.0f}°")
    # Set up longitude (x) ticks every 15 degrees and latitude (y) ticks every 10 degrees -
    # longitude ranges from -180 to +180 and latitude ranges from -90 to +90.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(np.radians(15)))
    ax.xaxis.set_major_formatter(tick_formatter)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(np.radians(10)))
    ax.yaxis.set_major_formatter(tick_formatter)

    if title is not None:
        ax.set_title(title)
    return fig


@pytest.mark.parametrize("domain_dim", [2, 3, 8, 16, 32, 768])
def test_vector_field_output_magnitude_statistics(domain_dim):
    """
    Test that measures the magnitude statistics of VectorField outputs.

    Initializes a VectorField, generates random input data, computes output vector magnitudes,
    and prints the mean and standard deviation of those magnitudes.
    """
    rng = jax.random.PRNGKey(20250602)

    # Create a VectorField model
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=domain_dim,
        conditioning_dim=0,
        time_dim=32,
        reference_directions=64,
        n_layers=4,
        d_model=256,
        mlp_expansion_factor=4,
        mlp_dropout_rate=None,
        input_dropout_rate=None,
        use_pre_mlp_projection=True,
    )

    # Initialize model parameters
    params_rng, data_rng = jax.random.split(rng)
    params = model.init(params_rng, *model.dummy_inputs())

    # Generate random input data
    n_samples = 1_000_000
    batch_size = 8192
    keys = jax.random.split(data_rng, 3)

    # Generate points uniformly on the sphere
    x = sample_sphere(keys[0], n_samples, domain_dim)

    # Generate random times in [0, 1]
    t = jax.random.uniform(keys[1], (n_samples,))

    # Generate conditioning vectors (empty since conditioning_dim=0)
    cond_vec = jnp.zeros((n_samples, 0))

    # Compute model outputs in batches
    jitted_apply = jax.jit(
        lambda x_batch, t_batch, cond_batch: model.apply(
            params, x_batch, t_batch, cond_batch
        )
    )

    all_magnitudes = []
    all_dot_products = []

    n_batches = (n_samples + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc=f"Processing batches (dim={domain_dim})"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        x_batch = x[start_idx:end_idx]
        t_batch = t[start_idx:end_idx]
        cond_batch = cond_vec[start_idx:end_idx]

        # Compute model outputs for this batch
        output_vectors_batch = jitted_apply(x_batch, t_batch, cond_batch)

        # Compute magnitudes for this batch
        magnitudes_batch = jnp.linalg.norm(output_vectors_batch, axis=1)
        all_magnitudes.append(magnitudes_batch)

        # Compute dot products for this batch
        dot_products_batch = jnp.sum(output_vectors_batch * x_batch, axis=1)
        all_dot_products.append(dot_products_batch)

    # Concatenate all results
    magnitudes = jnp.concatenate(all_magnitudes, axis=0)
    dot_products = jnp.concatenate(all_dot_products, axis=0)
    assert magnitudes.shape == dot_products.shape == (n_samples,)

    # Calculate statistics
    mean_magnitude = jnp.mean(magnitudes)
    std_magnitude = jnp.std(magnitudes)
    min_magnitude = jnp.min(magnitudes)
    max_magnitude = jnp.max(magnitudes)

    # Print results
    print(f"\nVectorField Output Magnitude Statistics (domain_dim={domain_dim}):")
    print(f"Number of samples: {n_samples}")
    print(f"Mean magnitude: {mean_magnitude:.6f}")
    print(f"Std magnitude: {std_magnitude:.6f}")
    print(f"Min magnitude: {min_magnitude:.6f}")
    print(f"Max magnitude: {max_magnitude:.6f}")

    # Verify outputs are tangent to the sphere (should have zero dot product with input points)
    max_dot_product = jnp.max(jnp.abs(dot_products))
    print(f"Max |dot product| with input points: {max_dot_product:.8f} (should be ~0)")

    # For low dimensionality, there's more variance in magnitudes due to the smaller number of
    # weights. I think that's why anyway.
    np.testing.assert_allclose(
        mean_magnitude, jnp.pi / 2, atol=0.1 if domain_dim > 3 else 0.5
    )
    np.testing.assert_allclose(max_dot_product, 0.0, atol=1e-5)

    # Basic sanity check that magnitudes are non-negative and finite
    assert jnp.all(magnitudes >= 0), "All magnitudes should be non-negative"
    assert jnp.all(jnp.isfinite(magnitudes)), "All magnitudes should be finite"


class SamplingAlgorithm(Enum):
    """Enum for selecting cap-constrained sampling algorithms."""

    SIR = "sir"
    REJECTION = "rejection"
    MCMC = "mcmc"


@dataclass
class SamplingDebugInfo:
    """Debug information about sampling performance."""

    n_model_samples_drawn: int
    n_model_probabilities_calculated: int


@dataclass
class RejectionParams:
    """Parameters specific to rejection sampling algorithm."""

    proposal_batch_size: int = 256


def _generate_cap_constrained_samples_rejection(
    model,
    params,
    rng,
    cap_center,
    cap_d_max,
    table,
    cond_vec,
    n_output_samples,
    rejection_params: RejectionParams,
    flow_n_steps: int,
):
    """
    Generate samples from a trained model constrained to lie within a spherical cap using
    rejection sampling.

    Args:
        model: Trained VectorField model
        params: Model parameters
        rng: JAX random key
        cap_center: Center of the spherical cap [domain_dim]
        cap_d_max: Maximum cosine distance defining the cap size
        table: LogitsTable for cap sampling (unused in rejection sampling)
        cond_vec: Conditioning vector shared across all samples [cond_dim]
        n_output_samples: Number of final samples to return
        rejection_params: Rejection sampling specific parameters

    Returns:
        samples: Cap-constrained samples [n_output_samples, domain_dim]
        log_probs: Model log densities for returned samples [n_output_samples]
        ess: Effective sample size (always equals n_output_samples for rejection sampling)
        debug_info: Performance debugging information
    """
    assert cap_center.shape == (model.domain_dim,)
    assert cond_vec.shape == (model.conditioning_dim,)

    threshold = 1.0 - cap_d_max
    samples_chunks = []
    total_proposals = 0

    with tqdm(
        desc="rejection sampling", unit="accepted samples", total=n_output_samples
    ) as pbar:
        while sum(chunk.shape[0] for chunk in samples_chunks) < n_output_samples:
            rng, sub_rng = jax.random.split(rng)

            # Generate batch of conditioning vectors
            batch_cond_vecs = jnp.broadcast_to(
                cond_vec[None, :],
                (rejection_params.proposal_batch_size, model.conditioning_dim),
            )

            # Generate samples from the trained model
            batch_samples = generate_samples(
                model,
                params,
                sub_rng,
                cond_vecs=batch_cond_vecs,
                n_steps=flow_n_steps,
                method="rk4",
            )

            # Check which samples are in the cap
            cos_similarities = batch_samples @ cap_center
            mask = cos_similarities >= threshold
            accepted = batch_samples[mask]

            if accepted.shape[0] > 0:
                samples_chunks.append(accepted)
                pbar.update(accepted.shape[0])

            total_proposals += rejection_params.proposal_batch_size

    # Concatenate and trim to exact number requested
    all_samples = jnp.concatenate(samples_chunks, axis=0)
    final_samples = all_samples[:n_output_samples]

    # Compute probabilities for the returned samples
    batch_cond_vecs = jnp.broadcast_to(
        cond_vec[None, :], (final_samples.shape[0], model.conditioning_dim)
    )
    log_probs = compute_log_probability(
        model,
        params,
        final_samples,
        batch_cond_vecs,
        n_steps=flow_n_steps,
        rng=None,
        n_projections=10,
    )

    # For rejection sampling, ESS equals the number of output samples
    ess = float(n_output_samples)

    debug_info = SamplingDebugInfo(
        n_model_samples_drawn=total_proposals, n_model_probabilities_calculated=0
    )

    return final_samples, log_probs, ess, debug_info


def calculate_autocorrelation(chain_samples, max_lag=None):
    """
    Calculate autocorrelation function for a single MCMC chain using FFT for efficiency.

    Args:
        chain_samples: Array of shape [n_steps, dim] for a single chain
        max_lag: Maximum lag to compute autocorrelation for. If None, uses n_steps // 4

    Returns:
        autocorrelations: Array of autocorrelation values [max_lag + 1]
    """
    n_steps, dim = chain_samples.shape

    if max_lag is None:
        max_lag = n_steps // 4
    max_lag = min(max_lag, n_steps - 1)

    # Center the data by subtracting the mean
    centered = chain_samples - jnp.mean(chain_samples, axis=0, keepdims=True)

    # For multivariate case, we'll compute autocorrelation of the scalar quantity
    # representing the "position" - we'll use the norm of the centered samples
    # This gives us a single autocorrelation function that captures the overall
    # chain mixing behavior
    scalar_series = jnp.linalg.norm(centered, axis=1)

    # Compute autocorrelation using FFT for efficiency
    n = len(scalar_series)
    # Pad to next power of 2 for efficient FFT
    n_fft = 1 << (n - 1).bit_length() + 1

    # Mean-center the scalar series
    scalar_centered = scalar_series - jnp.mean(scalar_series)

    # Compute autocorrelation via FFT
    f_transform = jnp.fft.fft(scalar_centered, n_fft)
    autocorr_fft = jnp.fft.ifft(f_transform * jnp.conj(f_transform)).real

    # Extract the positive lags and normalize
    autocorr = autocorr_fft[: max_lag + 1]
    autocorr = autocorr / autocorr[0]  # Normalize so lag 0 = 1

    return autocorr


def calculate_integrated_autocorr_time(autocorrelations, c=5):
    """
    Calculate integrated autocorrelation time using automatic windowing.

    Args:
        autocorrelations: Array of autocorrelation values
        c: Windowing constant (typically 5-10)

    Returns:
        tau_int: Integrated autocorrelation time
    """
    # Find the cutoff where autocorrelation becomes small
    # We use automatic windowing: stop when W >= c * tau_int
    max_len = len(autocorrelations)

    # Start with a reasonable initial window
    tau_int = 1.0
    for W in range(1, max_len):
        # Calculate integrated autocorrelation time up to window W
        if W < len(autocorrelations):
            tau_int = 1.0 + 2.0 * jnp.sum(autocorrelations[1 : W + 1])
        else:
            tau_int = 1.0 + 2.0 * jnp.sum(autocorrelations[1:])

        # Check windowing condition
        if W >= c * tau_int:
            break

    return tau_int


def calculate_effective_sample_size_single_chain(chain_samples):
    """
    Calculate effective sample size for a single MCMC chain.

    Args:
        chain_samples: Array of shape [n_steps, dim] for a single chain

    Returns:
        ess: Effective sample size for this chain
    """
    n_steps = chain_samples.shape[0]

    if n_steps < 4:
        # Too few samples to calculate meaningful autocorrelation
        return float(n_steps)

    # Calculate autocorrelation
    autocorr = calculate_autocorrelation(chain_samples)

    # Calculate integrated autocorrelation time
    tau_int = calculate_integrated_autocorr_time(autocorr)

    # ESS = N / (2 * tau_int)
    # The factor of 2 comes from the relationship between integrated autocorr time
    # and the variance of the sample mean
    ess = n_steps / tau_int

    # Ensure ESS doesn't exceed the number of samples
    ess = jnp.minimum(ess, float(n_steps))

    return float(ess)


def calculate_total_effective_sample_size(all_chain_samples):
    """
    Calculate total effective sample size across multiple MCMC chains.

    Args:
        all_chain_samples: Array of shape [n_steps, n_chains, dim]

    Returns:
        total_ess: Total effective sample size across all chains
    """
    n_steps, n_chains, dim = all_chain_samples.shape

    # Calculate ESS for each chain individually
    ess_per_chain = []
    for chain_idx in range(n_chains):
        chain_samples = all_chain_samples[:, chain_idx, :]
        chain_ess = calculate_effective_sample_size_single_chain(chain_samples)
        ess_per_chain.append(chain_ess)

    # Total ESS is the sum of individual chain ESS values
    total_ess = sum(ess_per_chain)

    return total_ess, ess_per_chain


@dataclass
class MCMCParams:
    """Parameters specific to Markov Chain Monte Carlo (MCMC) algorithm."""

    n_chains: int = 8
    # Steps of markov chain iteration
    n_steps_per_chain: int = 1000
    # Standard deviation of step geodesic distance is cap radius times this value
    step_scale: float = 1 / 3
    burnin_steps: int = 100


@jax.jit
def reflect_geodesic_step_into_cap(
    point, tangent_direction, distance, cap_center, cap_d_max
):
    """Reflect a geodesic step at the spherical-cap boundary.

    Given a starting `point` on the unit sphere, a unit `tangent_direction` in its tangent space,
    and a geodesic `distance` (can be any real number), returns the new point after moving along
    the great-circle and applying specular reflections at the boundary of the spherical cap
    defined by x·cap_center >= 1 - cap_d_max.

    The mapping preserves proposal symmetry (detailed balance) when distances are sampled from a
    symmetric distribution and tangent directions are sampled symmetrically.
    """
    threshold = 1.0 - cap_d_max
    two_pi = 2.0 * jnp.pi
    eps = 1e-12

    # Compute forward/backward distances to the cap boundary along this geodesic
    a = jnp.dot(point, cap_center)
    b = jnp.dot(tangent_direction, cap_center)
    r = jnp.sqrt(jnp.maximum(a * a + b * b, eps))

    # If the entire great circle stays inside the cap, skip reflection
    always_inside = threshold <= -r + eps
    # Clip for numerical stability
    val = jnp.clip(threshold / r, -1.0, 1.0)
    delta = jnp.arccos(val)
    phi = jnp.arctan2(b, a)

    t1_mod = jnp.mod(phi - delta, two_pi)
    t2_mod = jnp.mod(phi + delta, two_pi)

    # Forward distance to nearest boundary crossing (smallest nonnegative)
    L_plus = jnp.minimum(t1_mod, t2_mod)

    # Backward distance to nearest boundary crossing (treat boundary at 0 as distance 0)
    any_zero = jnp.logical_or(t1_mod < eps, t2_mod < eps)
    L_minus_raw = jnp.minimum(two_pi - t1_mod, two_pi - t2_mod)
    L_minus = jnp.where(any_zero, 0.0, L_minus_raw)

    # Reflect distance into the interval [-L_minus, L_plus]
    a_int = -L_minus
    b_int = L_plus
    width = b_int - a_int

    def reflect_scalar(s):
        t = s - a_int
        two_w = 2.0 * width
        t_mod = jnp.mod(t, two_w)
        return jnp.where(
            t_mod <= width,
            a_int + t_mod,
            b_int - (t_mod - width),
        )

    s_unfolded = jnp.where(width > eps, reflect_scalar(distance), 0.0)
    s_reflected = jnp.where(always_inside, distance, s_unfolded)

    # Move along geodesic with reflected distance and re-normalize to unit sphere
    result = point * jnp.cos(s_reflected) + tangent_direction * jnp.sin(s_reflected)
    return result / jnp.linalg.norm(result)


def test_reflected_moves_stay_in_cap_and_on_unit_sphere():
    """
    Reflected geodesic moves should remain inside the cap and on the unit sphere.
    """
    rng = jax.random.PRNGKey(0)
    dim = 8
    n = 512

    cap_center = jnp.zeros((dim,))
    cap_center = cap_center.at[0].set(1.0)
    cap_d_max = 0.5  # threshold = 0.5

    def _normalize(v, axis=-1, eps=1e-12):
        norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
        norm = jnp.maximum(norm, eps)
        return v / norm

    # Sample points inside the cap by rotating the cap center along random tangent directions
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    table = LogitsTable(dim - 1, 8192)
    keys = jax.random.split(r1, n)
    points = jax.vmap(lambda k: sample_from_cap(k, table, cap_center, cap_d_max))(keys)

    # Sample unit tangent directions at each point
    noise2 = jax.random.normal(r3, (n, dim))
    tangents = noise2 - jnp.sum(noise2 * points, axis=1, keepdims=True) * points
    tangents = _normalize(tangents, axis=1)

    # Sample step distances
    distances = jax.random.normal(r3, (n,)) * 0.8

    reflected = jax.vmap(
        lambda p, t, s: reflect_geodesic_step_into_cap(p, t, s, cap_center, cap_d_max)
    )(points, tangents, distances)

    # Unit sphere constraint
    norms = jnp.linalg.norm(reflected, axis=1)
    assert jnp.allclose(norms, jnp.ones_like(norms), atol=1e-6, rtol=0)

    # Cap constraint
    threshold = 1.0 - cap_d_max
    dots = reflected @ cap_center
    assert bool(jnp.all(dots >= threshold - 1e-7))


def test_reflection_reversibility_round_trip():
    """
    For MCMC to be mathematically well founded, the proposal distribution must be symmetric. In
    other words, the probability of proposing point p from point q must equal the probability of
    proposing point q from point p. In order for reflections to preserve this symmetry, if a move
    from p to q of distance d is reflected, there must be a move from q to p of distance d in the
    reflected direction. Without reflection, this is the case and the reverse move is simply the
    opposite direction. With reflection, the reverse move is the opposite direction *along the
    reflected geodesic*, not the opposite direction of the original move. This test verifies that
    this is the case.
    """
    rng = jax.random.PRNGKey(1)
    dim = 8
    n = 512

    cap_center = jnp.zeros((dim,))
    cap_center = cap_center.at[0].set(1.0)
    cap_d_max = 0.5  # threshold = 0.5

    def _normalize(v, axis=-1, eps=1e-12):
        norm = jnp.linalg.norm(v, axis=axis, keepdims=True)
        norm = jnp.maximum(norm, eps)
        return v / norm

    # Sample points inside the cap
    rng, r1, r2, r3 = jax.random.split(rng, 4)
    table = LogitsTable(dim - 1, 8192)
    keys = jax.random.split(r1, n)
    points = jax.vmap(lambda k: sample_from_cap(k, table, cap_center, cap_d_max))(keys)

    # Sample unit tangent directions at each point
    noise = jax.random.normal(r2, (n, dim))
    tangents = noise - jnp.sum(noise * points, axis=1, keepdims=True) * points
    tangents = _normalize(tangents, axis=1)

    # Sample step distances; use a wide spread to induce multiple reflections
    distances = jax.random.normal(r3, (n,)) * 8.0

    # Forward reflected step: p -> q
    q = jax.vmap(
        lambda p, t, s: reflect_geodesic_step_into_cap(p, t, s, cap_center, cap_d_max)
    )(points, tangents, distances)

    # Recover effective angle along the (p, t) great circle that produced q
    cos_comp = jnp.sum(q * points, axis=1)
    sin_comp = jnp.sum(q * tangents, axis=1)
    s_eff = jnp.arctan2(sin_comp, cos_comp)

    # Build the reverse tangent at q along the same great circle, pointing back toward p
    # t_rev = sin(s_eff) * p - cos(s_eff) * t
    t_rev = jnp.sin(s_eff)[:, None] * points - jnp.cos(s_eff)[:, None] * tangents
    t_rev = _normalize(t_rev, axis=1)

    # Reverse reflected step with the effective reflected distance along the same geodesic: q -> p_back
    p_back = jax.vmap(
        lambda qq, tt, s: reflect_geodesic_step_into_cap(
            qq, tt, s, cap_center, cap_d_max
        )
    )(q, t_rev, s_eff)

    # Check round trip accuracy
    assert jnp.allclose(p_back, points, atol=1e-6, rtol=0)


@jax.jit
def _mk_geodesic_proposals_reflected(key, positions, step_size, cap_center, cap_d_max):
    """Geodesic proposals with specular reflection at the spherical-cap boundary.

    Ensures all proposed positions remain inside the cap defined by
    x·cap_center >= 1 - cap_d_max, while preserving proposal symmetry.
    """
    directions_rng, distances_rng = jax.random.split(key)

    # Sample directions in the tangent spaces of the input positions
    directions = jax.random.normal(
        directions_rng, (positions.shape[0], positions.shape[1])
    )
    directions = (
        directions - jnp.sum(directions * positions, axis=1, keepdims=True) * positions
    )
    directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

    # Sample distances from a normal distribution (symmetric around 0)
    distances = jax.random.normal(distances_rng, (positions.shape[0],)) * step_size

    return jax.vmap(
        lambda p, t, s: reflect_geodesic_step_into_cap(p, t, s, cap_center, cap_d_max)
    )(positions, directions, distances)


def _generate_cap_constrained_samples_mcmc(
    model,
    params,
    rng,
    cap_center,
    cap_d_max,
    table,
    cond_vec,
    n_output_samples,
    mcmc_params: MCMCParams,
    flow_n_steps: int,
):
    """
    Generate samples from a trained model constrained to lie within a spherical cap using
    parallel Markov Chain Monte Carlo (MCMC).

    Args:
        model: Trained VectorField model
        params: Model parameters
        rng: JAX random key
        cap_center: Center of the spherical cap [domain_dim]
        cap_d_max: Maximum cosine distance defining the cap size
        table: LogitsTable for cap sampling
        cond_vec: Conditioning vector shared across all samples [cond_dim]
        n_output_samples: Number of final samples to return
        mcmc_params: MCMC-specific parameters

    Returns:
        samples: Cap-constrained samples [n_output_samples, domain_dim]
        log_probs: Model log densities for returned samples [n_output_samples]
        ess: Effective sample size (equals n_output_samples for MCMC)
        debug_info: Performance debugging information
    """
    assert cap_center.shape == (model.domain_dim,)
    assert cond_vec.shape == (model.conditioning_dim,)

    init_rng, chain_rng = jax.random.split(rng)

    # Initialize chains uniformly within the cap
    init_keys = jax.random.split(init_rng, mcmc_params.n_chains)
    chain_positions = jax.vmap(
        lambda key: sample_from_cap(key, table, cap_center, cap_d_max)
    )(init_keys)

    # Expand conditioning vector to all chains
    chain_cond_vecs = jnp.broadcast_to(
        cond_vec[None, :], (mcmc_params.n_chains, model.conditioning_dim)
    )

    # Compute initial log probabilities
    init_log_probs = compute_log_probability(
        model,
        params,
        chain_positions,
        chain_cond_vecs,
        n_steps=flow_n_steps,
        rng=None,
        n_projections=10,
    )

    def mcmc_step(state, step_rng):
        """Single MCMC step for all chains."""
        positions, log_probs = state

        proposals = _mk_geodesic_proposals_reflected(
            step_rng,
            positions,
            mcmc_params.step_scale * jnp.arccos(1 - cap_d_max),
            cap_center,
            cap_d_max,
        )

        # All proposals are in-cap by construction; assert and report for debugging
        threshold = 1.0 - cap_d_max
        in_cap = (proposals @ cap_center) >= threshold
        assert np.all(in_cap)

        proposal_log_probs = compute_log_probability(
            model,
            params,
            proposals,
            chain_cond_vecs,
            n_steps=flow_n_steps,
            rng=None,
            n_projections=10,
        )

        # Metropolis-Hastings acceptance
        log_accept_probs = jnp.minimum(0.0, proposal_log_probs - log_probs)
        accept_flags = (
            jnp.log(jax.random.uniform(step_rng, (mcmc_params.n_chains,)))
            < log_accept_probs
        )

        # Update positions and log probabilities
        new_positions = jnp.where(accept_flags[:, None], proposals, positions)
        new_log_probs = jnp.where(accept_flags, proposal_log_probs, log_probs)

        return (new_positions, new_log_probs), (new_positions, accept_flags)

    # Run MCMC chains with regular loop
    current_state = (chain_positions, init_log_probs)
    all_positions = []
    all_accept_flags = []

    with tqdm(
        total=mcmc_params.n_steps_per_chain, desc="MCMC sampling", unit="step"
    ) as pbar:
        for step in range(mcmc_params.n_steps_per_chain):
            # Get random key for this step
            step_key = jax.random.fold_in(chain_rng, step)

            # Take MCMC step
            new_state, (step_positions, step_accept_flags) = mcmc_step(
                current_state, step_key
            )

            # Store results
            step_positions, step_accept_flags = jax.device_get(
                (step_positions, step_accept_flags)
            )
            all_positions.append(step_positions)
            all_accept_flags.append(step_accept_flags)

            # Update state for next iteration
            current_state = new_state

            # Update progress bar
            pbar.update(1)

    # Convert lists to arrays
    all_positions = np.stack(all_positions, axis=0)  # [n_steps, n_chains, dim]
    accept_flags = np.stack(all_accept_flags, axis=0)  # [n_steps, n_chains]

    # Extract samples after burnin
    burnin = mcmc_params.burnin_steps
    post_burnin_positions = all_positions[burnin:]  # [n_steps - burnin, n_chains, dim]

    # Reshape to [n_samples, dim] where n_samples = (n_steps - burnin) * n_chains
    all_samples = post_burnin_positions.reshape(-1, model.domain_dim)

    # Randomly select n_output_samples
    indices = jax.random.choice(
        jax.random.split(chain_rng)[0],
        all_samples.shape[0],
        shape=(n_output_samples,),
        replace=True,
    )
    final_samples = jax.device_put(all_samples[indices])

    # Compute probabilities for the returned samples
    final_batch_cond_vecs = jnp.broadcast_to(
        cond_vec[None, :], (final_samples.shape[0], model.conditioning_dim)
    )
    final_log_probs = compute_log_probability(
        model,
        params,
        final_samples,
        final_batch_cond_vecs,
        n_steps=flow_n_steps,
        rng=None,
        n_projections=10,
    )

    # Compute acceptance rate for debugging
    total_accepts = jnp.sum(accept_flags)
    acceptance_rate = total_accepts / (
        mcmc_params.n_chains * mcmc_params.n_steps_per_chain
    )
    print(f"MCMC acceptance rate: {acceptance_rate:.3f}")

    # Calculate effective sample size using autocorrelation analysis
    # NOTE: this is a dubious method for high dimensions.
    total_ess, ess_per_chain = calculate_total_effective_sample_size(
        post_burnin_positions
    )

    # Report ESS statistics
    print(f"Total ESS: {total_ess:.1f}")
    print(
        f"ESS per chain: min={min(ess_per_chain):.1f}, max={max(ess_per_chain):.1f}, mean={sum(ess_per_chain)/len(ess_per_chain):.1f}"
    )

    ess = total_ess

    # Debug info: we evaluate probabilities at each step for each chain
    total_prob_evals = mcmc_params.n_chains * (
        mcmc_params.n_steps_per_chain + 1
    )  # +1 for initial
    debug_info = SamplingDebugInfo(
        n_model_samples_drawn=0, n_model_probabilities_calculated=total_prob_evals
    )

    return final_samples, final_log_probs, ess, debug_info


@dataclass
class SIRParams:
    """Parameters specific to Sampling-Importance-Resampling (SIR) algorithm."""

    n_proposal_samples: int
    n_projections: int = 10
    batch_size: int = 512


def _generate_cap_constrained_samples_sir(
    model,
    params,
    rng,
    cap_center,
    cap_d_max,
    table,
    cond_vec,
    n_output_samples,
    sir_params: SIRParams,
    flow_n_steps: int,
):
    """
    Generate samples from a trained model constrained to lie within a spherical cap using
    Sampling-Importance-Resampling (SIR).

    Args:
        model: Trained VectorField model
        params: Model parameters
        rng: JAX random key
        cap_center: Center of the spherical cap [domain_dim]
        cap_d_max: Maximum cosine distance defining the cap size
        table: LogitsTable for cap sampling
        cond_vec: Conditioning vector shared across all samples [cond_dim]
        n_output_samples: Number of final samples to return after importance resampling
        sir_params: SIR-specific parameters

    Returns:
        samples: Cap-constrained samples [n_output_samples, domain_dim]
        log_probs: Model log densities for returned samples [n_output_samples]
        ess: Effective sample size (measure of sampling efficiency)
        debug_info: Performance debugging information
    """
    assert cap_center.shape == (model.domain_dim,)
    assert cond_vec.shape == (model.conditioning_dim,)

    proposal_rng, prob_rng, resample_rng = jax.random.split(rng, 3)

    # 1. Sample uniformly from the cap
    proposal_keys = jax.random.split(proposal_rng, sir_params.n_proposal_samples)
    proposal_samples = jax.vmap(
        lambda key: sample_from_cap(key, table, cap_center, cap_d_max)
    )(proposal_keys)

    # 2. Compute model log probabilities for all proposals in batches
    n_batches = ceil(sir_params.n_proposal_samples / sir_params.batch_size)
    model_log_probs_list = []

    # Split RNG for each batch
    batch_rngs = jax.random.split(prob_rng, n_batches)

    with tqdm(
        total=sir_params.n_proposal_samples,
        desc="computing model probabilities",
        unit="sample",
    ) as pbar:
        for batch_idx in range(n_batches):
            start_idx = batch_idx * sir_params.batch_size
            end_idx = min(
                (batch_idx + 1) * sir_params.batch_size, sir_params.n_proposal_samples
            )
            actual_batch_size = end_idx - start_idx

            # Extract batch of samples
            batch_samples = proposal_samples[start_idx:end_idx]

            # Expand cond_vecs to match batch size
            batch_cond_vecs = jnp.broadcast_to(
                cond_vec[None, :], (actual_batch_size, model.conditioning_dim)
            )

            # Compute log probabilities for this batch
            batch_log_probs = compute_log_probability(
                model,
                params,
                batch_samples,
                batch_cond_vecs,
                n_steps=flow_n_steps,
                rng=batch_rngs[batch_idx],
                n_projections=sir_params.n_projections,
            )

            model_log_probs_list.append(batch_log_probs)
            pbar.update(actual_batch_size)

    # Concatenate all batch results
    model_log_probs = jnp.concatenate(model_log_probs_list, axis=0)
    del model_log_probs_list

    # 3. Compute uniform cap log density. Fraction of the total sphere area that is inside the cap,
    # times the total sphere area gets you the area of the cap, reciprocal of that is the density.
    log_cap_size_frac = table.log_cap_size(cap_d_max)
    sphere_log_area = -sphere_log_inverse_surface_area(model.domain_dim)
    log_cap_area = log_cap_size_frac + sphere_log_area
    uniform_cap_log_density = -log_cap_area

    # 4. Compute importance weights: model_density / uniform_cap_density
    # In log space: log_weight = model_log_prob - uniform_cap_log_density
    log_weights = model_log_probs - uniform_cap_log_density

    # Normalize weights for numerical stability
    log_weights = log_weights - jnp.max(log_weights)
    weights = jnp.exp(log_weights)
    weights = weights / jnp.sum(weights)

    # 5. Compute effective sample size
    ess = jnp.sum(weights) ** 2 / jnp.sum(weights**2)

    # 6. Resample according to importance weights
    indices = jax.random.choice(
        resample_rng,
        sir_params.n_proposal_samples,
        shape=(n_output_samples,),
        p=weights,
    )
    final_samples = proposal_samples[indices]

    # Log-densities for returned samples from precomputed log-probs
    selected_log_probs = model_log_probs[indices]

    debug_info = SamplingDebugInfo(
        n_model_samples_drawn=0,
        n_model_probabilities_calculated=sir_params.n_proposal_samples,
    )

    return final_samples, selected_log_probs, ess, debug_info


def generate_cap_constrained_samples(
    model,
    params,
    rng,
    cap_center,
    cap_d_max,
    table,
    cond_vec,
    n_output_samples,
    flow_n_steps: int,
    algorithm: SamplingAlgorithm,
    algorithm_params,
):
    """
    Generate samples from a trained model constrained to lie within a spherical cap.

    Args:
        model: Trained VectorField model
        params: Model parameters
        rng: JAX random key
        cap_center: Center of the spherical cap [domain_dim]
        cap_d_max: Maximum cosine distance defining the cap size
        table: LogitsTable for cap sampling
        cond_vec: Conditioning vector shared across all samples [cond_dim]
        n_output_samples: Number of final samples to return
        algorithm: Which sampling algorithm to use
        algorithm_params: Parameters specific to the chosen algorithm

    Returns:
        samples: Cap-constrained samples [n_output_samples, domain_dim]
        log_probs: Model log densities for returned samples [n_output_samples]
        ess: Effective sample size (measure of sampling efficiency)
        debug_info: Performance debugging information
    """
    if algorithm == SamplingAlgorithm.SIR:
        if not isinstance(algorithm_params, SIRParams):
            raise TypeError(
                f"Expected SIRParams for SIR algorithm, got {type(algorithm_params)}"
            )
        return _generate_cap_constrained_samples_sir(
            model,
            params,
            rng,
            cap_center,
            cap_d_max,
            table,
            cond_vec,
            n_output_samples,
            algorithm_params,
            flow_n_steps,
        )
    elif algorithm == SamplingAlgorithm.REJECTION:
        if not isinstance(algorithm_params, RejectionParams):
            raise TypeError(
                f"Expected RejectionParams for REJECTION algorithm, got {type(algorithm_params)}"
            )
        return _generate_cap_constrained_samples_rejection(
            model,
            params,
            rng,
            cap_center,
            cap_d_max,
            table,
            cond_vec,
            n_output_samples,
            algorithm_params,
            flow_n_steps,
        )
    elif algorithm == SamplingAlgorithm.MCMC:
        if not isinstance(algorithm_params, MCMCParams):
            raise TypeError(
                f"Expected MCMCParams for MCMC algorithm, got {type(algorithm_params)}"
            )
        return _generate_cap_constrained_samples_mcmc(
            model,
            params,
            rng,
            cap_center,
            cap_d_max,
            table,
            cond_vec,
            n_output_samples,
            algorithm_params,
            flow_n_steps,
        )
    else:
        raise ValueError(f"Unknown sampling algorithm: {algorithm}")


def test_generate_cap_constrained_samples_matches_rejection():
    """
    Train a 3D VectorField on a vMF distribution, then compare cap-constrained samples drawn via:
    1) rejection sampling using generate_cap_constrained_samples, and
    2) importance sampling (SIR) using generate_cap_constrained_samples.

    - Uses a fixed random cap center and d_max=0.25
    - Reports proposal counts for both methods
    - Verifies all samples are in-cap
    - Compares statistics of both sets within tolerances
    """

    # Fixed RNGs
    rng = jax.random.PRNGKey(20250713)
    train_rng, cap_rng, rs_rng, imp_rng = jax.random.split(rng, 4)

    # Train VectorField on a 3D vMF distribution
    model = replace(_baseline_model, domain_dim=3, n_layers=2)

    batch_size = 512
    n_samples = 32768

    mean_direction = np.zeros(3)
    mean_direction[0] = 1.0
    kappa = 2.0
    vmf = stats.vonmises_fisher(mean_direction, kappa)
    points = vmf.rvs(n_samples)

    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    state, train_loss = _train_loop_for_tests(model, dset, batch_size, 1e-3, 2)
    params = state.params

    # Cap setup: 2-sphere table for 3D vectors
    table = LogitsTable(2, 8192)
    cap_center = np.zeros(3)
    cap_center[1] = 1.0
    cap_center = jnp.asarray(cap_center)
    d_max = jnp.float32(0.25)
    threshold = 1.0 - d_max

    target_n = 4096

    # Rejection sampling
    cond_vec = jnp.zeros((model.conditioning_dim,))
    rejection_params = RejectionParams(proposal_batch_size=256)
    rs_samples, _rs_probs, rs_ess, rs_debug = generate_cap_constrained_samples(
        model,
        params,
        rs_rng,
        cap_center,
        d_max,
        table,
        cond_vec,
        target_n,
        16,
        SamplingAlgorithm.REJECTION,
        rejection_params,
    )

    # Importance sampling
    n_proposal_samples = 16384
    cond_vec = jnp.zeros((model.conditioning_dim,))
    sir_params = SIRParams(
        n_proposal_samples=n_proposal_samples,
        n_projections=10,
    )
    imp_samples, _imp_probs, ess, sir_debug = generate_cap_constrained_samples(
        model,
        params,
        imp_rng,
        cap_center,
        d_max,
        table,
        cond_vec,
        target_n,
        16,
        SamplingAlgorithm.SIR,
        sir_params,
    )
    # MCMC sampling (reduced parameters for testing speed)
    mcmc_rng = jax.random.split(rng, 4)[3]
    mcmc_params = MCMCParams(
        n_chains=512,
        n_steps_per_chain=128,
        burnin_steps=16,
    )
    mcmc_samples, _mcmc_probs, mcmc_ess, mcmc_debug = generate_cap_constrained_samples(
        model,
        params,
        mcmc_rng,
        cap_center,
        d_max,
        table,
        cond_vec,
        target_n,
        16,
        SamplingAlgorithm.MCMC,
        mcmc_params,
    )

    print(
        f"Rejection proposals used: {rs_debug.n_model_samples_drawn} for {target_n} accepted samples ({(rs_debug.n_model_samples_drawn / target_n):.2f} per sample)"
    )
    print(
        f"Importance sampling (SIR): {n_proposal_samples} proposals, ESS={float(ess):.2f} ({n_proposal_samples / ess:.2f} per effective sample)"
    )
    print(
        f"MCMC: {mcmc_params.n_chains} chains × {mcmc_params.n_steps_per_chain} steps, ESS={float(mcmc_ess):.2f}, {mcmc_debug.n_model_probabilities_calculated} prob evals"
    )

    assert ess >= target_n, f"SIR ESS={ess} < target_n={target_n}"
    assert (
        rs_ess == target_n
    ), f"Rejection ESS should equal output samples, got {rs_ess}"
    # MCMC ESS should be at least 10% of target samples (accounting for autocorrelation)
    assert (
        mcmc_ess >= target_n * 0.1
    ), f"MCMC ESS {mcmc_ess} too low compared to target {target_n}"

    # Verify all samples are in-cap
    for arr in [rs_samples, imp_samples, mcmc_samples]:
        cos_vals = arr @ cap_center
        assert jnp.all(cos_vals >= threshold)

    # Compare statistics: per-coordinate mean and cosine-to-center moments
    def summarize(arr, label):
        means = jnp.mean(arr, axis=0)
        cos_vals = arr @ cap_center
        cos_mean = jnp.mean(cos_vals)
        cos_std = jnp.std(cos_vals)
        print(
            f"{label} mean: {means}, cos_mean: {cos_mean:.2f}, cos_std: {cos_std:.2f}"
        )
        return means, cos_mean, cos_std

    rs_means, rs_cos_mean, rs_cos_std = summarize(rs_samples, "rejection")
    imp_means, imp_cos_mean, imp_cos_std = summarize(imp_samples, "importance")
    mcmc_means, mcmc_cos_mean, mcmc_cos_std = summarize(mcmc_samples, "MCMC")

    # Compare rejection vs SIR
    np.testing.assert_allclose(rs_means, imp_means, atol=0.1, rtol=0)
    np.testing.assert_allclose(rs_cos_mean, imp_cos_mean, atol=0.03, rtol=0)
    np.testing.assert_allclose(rs_cos_std, imp_cos_std, atol=0.03, rtol=0)

    # Compare rejection vs MCMC (looser tolerance for MCMC due to finite chain effects)
    np.testing.assert_allclose(rs_means, mcmc_means, atol=0.1, rtol=0)
    np.testing.assert_allclose(rs_cos_mean, mcmc_cos_mean, atol=0.05, rtol=0)
    np.testing.assert_allclose(rs_cos_std, mcmc_cos_std, atol=0.05, rtol=0)


def test_compute_psi_t_spherical_antipodal_path_is_geodesic():
    # Two antipodal points on S^2
    x0 = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float32)
    x1 = jnp.array([-1.0, 0.0, 0.0], dtype=jnp.float32)

    # Sample a few times in (0,1) and ensure points stay on unit sphere and nonzero
    ts = jnp.linspace(0.0, 1.0, 5)
    xs = jax.vmap(lambda t: compute_psi_t_spherical(x0, x1, t))(ts)

    # Endpoints match inputs
    np.testing.assert_allclose(xs[0], x0, atol=1e-6, rtol=0)
    np.testing.assert_allclose(xs[-1], x1, atol=1e-6, rtol=0)

    # Intermediate points should be unit-norm and non-zero
    norms = jnp.linalg.norm(xs, axis=-1)
    assert jnp.all(norms > 1e-6)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6, rtol=0)

    # Path should lie on a great circle orthogonal to a consistent tangent direction at x0
    tangent_dir = get_consistent_tangent_direction(x0)
    # Check that midpoint is close to tangent_dir (up to sign ambiguity)
    mid = xs[2]
    cos_sim = jnp.dot(mid, tangent_dir)
    assert jnp.abs(cos_sim) > 0.9


def test_spherical_ot_field_antipodal_direction_tangent_and_nonzero():
    x0 = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    x1 = jnp.array([0.0, -1.0, 0.0], dtype=jnp.float32)

    # Check a few times including the midpoint
    for t in [0.0, 0.5, 1.0]:
        t = jnp.array(t, dtype=jnp.float32)
        x, v = spherical_ot_field(x0, x1, t)

        # x should be on the sphere
        np.testing.assert_allclose(jnp.linalg.norm(x), 1.0, atol=1e-6, rtol=0)

        # v should be tangent: dot(x, v) = 0 (within tolerance)
        np.testing.assert_allclose(jnp.dot(x, v), 0.0, atol=1e-6, rtol=0)

        # At antipodal case, v should be non-zero except at exact endpoints when angle logic may return zero
        if t not in (0.0, 1.0):
            assert jnp.linalg.norm(v) > 1e-6


@pytest.mark.parametrize(
    "inject_keys",
    [
        frozenset(),
        frozenset({"x"}),
        frozenset({"t"}),
        frozenset({"x", "t"}),
    ],
)
@pytest.mark.parametrize("model_kind", ["mlp", "transformer"])
def test_train_hemisphere_density(model_kind, inject_keys):
    """
    Train on points uniformly sampled from the northern hemisphere, then verify:
    - log density on the northern hemisphere ≈ log(2) + uniform sphere log density
    - log density on the southern hemisphere (outside training support) is near zero (very small)

    Uses sample_from_cap to generate uniform-in-cap points (cap d_max=1 corresponds to hemisphere).
    """
    domain_dim = 3
    if model_kind == "mlp":
        model = _mk_model(
            "mlp",
            domain_dim=domain_dim,
            n_layers=6,
            d_model=512,
            mlp_always_inject=inject_keys,
        )
        lr = 1e-3
    else:
        model = _mk_model("transformer", domain_dim=domain_dim, n_learned_tokens=14)
        if inject_keys:
            pytest.skip("Transformer with inject_keys not supported")
        lr = 3e-3
    epochs = 16

    # Hemisphere definition: center at north pole, half-angle π/2 ⇒ d_max = 1.0
    table = LogitsTable(domain_dim - 1, 8192)
    north = jnp.zeros(domain_dim).at[domain_dim - 1].set(1.0)
    south = -north

    def sample_cap_batch(rng, center, d_max, n):
        keys = jax.random.split(rng, n)
        return jax.vmap(lambda k: sample_from_cap(k, table, center, d_max))(keys)

    # Generate training dataset on northern hemisphere
    rng = jax.random.PRNGKey(20250828)
    train_rng, test_rng1, test_rng2 = jax.random.split(rng, 3)
    n_train = 120_000
    batch_gen = 4096
    pts = []
    remaining = n_train
    while remaining > 0:
        this = int(min(batch_gen, remaining))
        sub_rng, train_rng = jax.random.split(train_rng)
        pts.append(sample_cap_batch(sub_rng, north, 1.0, this))
        remaining -= this
    train_points = jnp.concatenate(pts, axis=0)
    assert train_points.shape == (n_train, domain_dim)

    dsets = (
        Dataset.from_dict({"point_vec": train_points})
        .with_format("np")
        .train_test_split(test_size=2048)
    )
    train_dataset = dsets["train"]
    test_dataset = dsets["test"]

    # Train
    batch_size = 512
    state, final_loss, test_loss, test_nll = _train_loop_for_tests(
        model, train_dataset, batch_size, lr, epochs, test_dataset
    )

    print(
        f"Final train loss: {final_loss:.6f}, test loss: {test_loss:.6f}, test NLL: {test_nll:.6f}"
    )

    # Build two test sets: in-cap (north hemisphere) and opposite (south hemisphere)
    n_test = 1024
    north_points = sample_cap_batch(test_rng1, north, 1.0, n_test)
    # Density falloff is smooth, doesn't exactly reflect the true step-function pdf. We only test
    # points in the southern hemisphere that are a bit away from the boundary.
    south_points = sample_cap_batch(test_rng2, south, 0.9, n_test)

    cond_vecs = jnp.zeros((n_test, 0), dtype=jnp.float32)
    uniform_log_density = sphere_log_inverse_surface_area(domain_dim)

    # Compute log probabilities
    north_log_probs = compute_log_probability(
        model,
        state.params,
        north_points,
        cond_vecs,
        n_steps=100,
        rng=jax.random.PRNGKey(12345),
        n_projections=10,
    )
    south_log_probs = compute_log_probability(
        model,
        state.params,
        south_points,
        cond_vecs,
        n_steps=100,
        rng=jax.random.PRNGKey(67890),
        n_projections=10,
    )

    # Also probe along the geodesic from north to south (inclusive)
    num_geodesic_points = 10
    ts = jnp.linspace(0.0, 1.0, num_geodesic_points)
    geodesic_points = jax.vmap(lambda tau: compute_psi_t_spherical(north, south, tau))(
        ts
    )
    geodesic_log_probs = compute_log_probability(
        model,
        state.params,
        geodesic_points,
        jnp.zeros((num_geodesic_points, 0), dtype=jnp.float32),
        n_steps=100,
        rng=jax.random.PRNGKey(13579),
        n_projections=10,
    )
    print("  Geodesic path model log probs (10 points from north to south):")
    print(f"    log_probs: {geodesic_log_probs}")

    # In-hemisphere density should be 2x uniform
    expected_north = uniform_log_density + jnp.log(2.0)
    print(
        f"North hemisphere: mean={north_log_probs.mean():.4f}, expected={expected_north:.4f}"
    )
    np.testing.assert_allclose(
        jax.device_get(north_log_probs.mean()),
        jax.device_get(expected_north),
        rtol=0,
        atol=0.1,
    )

    # Outside support should be very small (close to zero density)
    MAXIMUM_DENSITY_RATIO = 0.05
    threshold = uniform_log_density + jnp.log(MAXIMUM_DENSITY_RATIO)
    print(
        f"South hemisphere log-prob stats: min={south_log_probs.min():.4f}, max={south_log_probs.max():.4f}, threshold={threshold:.4f}"
    )
    too_high = jnp.sum(south_log_probs > threshold)
    assert (
        too_high / n_test < 0.05
    ), f"{too_high} south-hemisphere points have log prob >= threshold"


@pytest.mark.parametrize(
    "inject_keys",
    [
        frozenset(),
        frozenset({"x"}),
        frozenset({"t"}),
        frozenset({"x", "t"}),
        frozenset({"x", "t", "cond"}),
    ],
)
def test_vector_field_mlp_always_inject_forward(inject_keys):
    """Sanity-check forward shapes with various per-block injection configurations."""
    rng = jax.random.PRNGKey(0)

    domain_dim = 8
    conditioning_dim = 4
    time_dim = 16

    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=domain_dim,
        conditioning_dim=conditioning_dim,
        time_dim=time_dim,
        reference_directions=16,
        n_layers=2,
        d_model=64,
        mlp_expansion_factor=2,
        mlp_dropout_rate=None,
        input_dropout_rate=None,
        use_pre_mlp_projection=False,
        mlp_always_inject=inject_keys,
    )

    batch_size = 7
    x = sample_sphere(jax.random.PRNGKey(1), batch_size, domain_dim)
    t = jax.random.uniform(jax.random.PRNGKey(2), (batch_size,))
    cond_vec = jax.random.normal(jax.random.PRNGKey(3), (batch_size, conditioning_dim))

    params = model.init(rng, *model.dummy_inputs())
    out = model.apply(params, x, t, cond_vec)

    assert out.shape == (batch_size, domain_dim)
    dots = jnp.sum(out * x, axis=1)
    assert jnp.all(jnp.abs(dots) < 1e-5)
