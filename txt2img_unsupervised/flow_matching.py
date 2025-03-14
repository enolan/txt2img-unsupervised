"""
Flow matching model for spherical data.

Explanation from Claude:

This implementation extends Flow Matching with Optimal Transport (FM-OT) from Euclidean spaces 
to the unit sphere, preserving its key theoretical properties while respecting the manifold's geometry.

## Theory and Design

In Euclidean spaces, standard FM-OT uses conditional probability paths based on Gaussian distributions 
that interpolate linearly between a noise distribution and the target. The resulting vector field
produces straight-line trajectories with constant direction - the most efficient paths in Euclidean space.

On the sphere, we adapt this paradigm by:

1. **Conditional Probability Paths**: Instead of Gaussian distributions, we use von Mises-Fisher (vMF) 
   distributions. At t=0, we begin with a uniform distribution on the sphere (kappa_0=0). As t increases, 
   the vMF distribution becomes more concentrated around the target (kappa_t = t*kappa_1).

2. **Geodesic Trajectories**: While Euclidean FM-OT follows straight lines, spherical FM-OT follows 
   geodesics (great circles) - the spherical equivalent of straight lines. These paths represent the 
   optimal transport solution on the sphere, minimizing the distance traveled along the manifold.

3. **Vector Field**: The vector field at point x points toward the target x1 along the geodesic 
   connecting them. Mathematically, this is achieved by projecting x1 onto the tangent space at x:
   u_t(x|x1) = kappa_t * normalize(x1 - (x1·x)x)
   
   This field has two key properties:
   - It is always tangent to the sphere (orthogonal to x)
   - It points along the geodesic toward x1

4. **Flow Map**: The flow map represents the position at time t of a particle starting at x0 
   and flowing toward x1. In Euclidean space, this is a linear interpolation; on the sphere, 
   it's the spherical linear interpolation (slerp):
   ψ_t(x0) = sin((1-t)θ)/sin(θ) · x0 + sin(tθ)/sin(θ) · x1
   where θ is the angle between x0 and x1.

Like Euclidean FM-OT, this spherical formulation maintains: (1) simplicity of paths, (2) constant 
tangent direction along geodesics, (3) efficient sampling with fewer integration steps, and 
(4) theoretical guarantees from continuous normalizing flows. The difference is that all computations 
respect the geometry of the sphere, ensuring flows remain on the manifold.

"""
from dataclasses import dataclass, field
from datasets import Dataset
from einops import rearrange, repeat
from flax import linen as nn
from flax.training import train_state
from functools import partial
from math import floor
from typing import Sequence, Tuple, Dict, Callable, Any
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import stats
from tqdm import tqdm
from tqdm.contrib import tenumerate

from .cap_sampling import LogitsTable, sample_cap, sample_from_cap


class MLPBlock(nn.Module):
    """
    Single block of a residual MLP a la the standard pre-norm transformer with SwiGLU.
    """

    bottleneck_dim: int
    expansion_factor: int

    activations_dtype: jnp.dtype
    weights_dtype: jnp.dtype
    kernel_init: Callable[..., jnp.ndarray]

    def setup(self) -> None:
        self.norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=jnp.float32,
        )
        self.gate_proj = nn.Dense(
            features=self.expansion_factor * self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=self.kernel_init,
        )
        self.value_proj = nn.Dense(
            features=self.expansion_factor * self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=self.kernel_init,
        )
        self.out_proj = nn.Dense(
            features=self.bottleneck_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=self.kernel_init,
        )

    def __call__(self, x, _):
        "Run the layer forward. Unused extra parameter and return value are for scan."
        assert len(x.shape) == 2
        assert x.shape[1] == self.bottleneck_dim
        x_normed = self.norm(x)
        gate = self.gate_proj(x_normed)
        value = self.value_proj(x_normed)
        gated = jax.nn.silu(gate) * value
        out = self.out_proj(gated)
        assert out.shape == x.shape
        return x + out, None


class VectorField(nn.Module):
    """
    Model of a vector field conditioned on a time t in [0, 1] and a arbitrary-dimensional vector.
    The vector field is over unit vectors in the domain.
    """

    # Dimension of the sphere and the vector field
    domain_dim: int
    # Dimension of the conditioning vector. Can be zero for unconditioned models.
    conditioning_dim: int
    # Number of MLP blocks
    n_layers: int
    # Width of the MLP
    d_model: int
    # MLP expansion factor
    mlp_expansion_factor: int

    activations_dtype: jnp.dtype
    weights_dtype: jnp.dtype

    def setup(self) -> None:
        # Initializers for our 3 inputs are specially scaled so their sum has mean 0 variance 1.
        # Each component of the sum should have mean 0 and variance either 1/3 or 1/2, depending on
        # if we have conditioning vectors or not
        input_variance_scale = 1.0 / 3.0 if self.conditioning_dim > 0 else 1.0 / 2.0

        self.domain_in_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=jnp.sqrt(input_variance_scale)),
        )

        if self.conditioning_dim > 0:
            self.conditioning_in_proj = nn.Dense(
                features=self.d_model,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
                use_bias=False,
                kernel_init=nn.initializers.variance_scaling(
                    # We assume the input components are unit normal
                    scale=input_variance_scale,
                    mode="fan_in",
                    distribution="normal",
                ),
            )
        else:
            self.conditioning_in_proj = None

        # Time projection. We subtract 0.5 in process_inputs to get mean 0, and the variance of
        # U[-0.5, 0.5] is 1/12, so we multiply by 12
        self.time_in_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            use_bias=False,
            kernel_init=nn.initializers.normal(
                stddev=jnp.sqrt(12 * input_variance_scale)
            ),
        )

        # Rather than having a bias on each of the input linear layers, which then just get summed,
        # we have them share a single one. Reduces parameter count very slightly and speeds up
        # inference and gradients. May or may not actually matter tbh.
        self.mlp_in_bias = self.param(
            "shared_bias", nn.initializers.zeros, (self.d_model,), self.weights_dtype
        )

        self.mlp_blocks = nn.scan(
            nn.remat(MLPBlock),
            variable_axes={"params": 0},
            variable_broadcast=False,
            split_rngs={"params": True},
            length=self.n_layers,
        )(
            bottleneck_dim=self.d_model,
            expansion_factor=self.mlp_expansion_factor,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="normal"
            ),
        )

        self.final_norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=jnp.float32,
        )

        self.out_proj = nn.Dense(
            features=self.domain_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(
                scale=1.0, mode="fan_in", distribution="normal"
            ),
        )

    def process_inputs(self, x, t, cond_vec):
        """
        Process the inputs to the model, returning the individual components and combined input.
        This helper method is used for testing the distribution of values.

        Args:
            x: Input unit vectors [batch_size, domain_dim]
            t: Time parameters in [0, 1] [batch_size]
            cond_vec: Conditioning vectors [batch_size, conditioning_dim]

        Returns:
            Dictionary with individual projection outputs and combined input:
            - domain_proj: Output of domain_in_proj [batch_size, d_model]
            - cond_proj: Output of conditioning_in_proj [batch_size, d_model]
            - time_proj: Output of time_in_proj [batch_size, d_model]
            - mlp_input: Combined input to the MLP [batch_size, d_model]
        """
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cond_vec.shape == (batch_size, self.conditioning_dim)

        # Convert inputs to the dimension of our MLP
        x_in = self.domain_in_proj(x)
        cond_in = (
            self.conditioning_in_proj(cond_vec)
            if self.conditioning_dim > 0
            else jnp.zeros((batch_size, self.d_model))
        )
        # important so the layer interprets the times as being batched and not just a vector for a
        # single input
        t_reshaped = rearrange(t, "b -> b 1")
        t_in = self.time_in_proj(t_reshaped - 0.5)  # Center the uniform [0,1] input
        assert x_in.shape == (batch_size, self.d_model)
        assert cond_in.shape == (batch_size, self.d_model)
        assert t_in.shape == (batch_size, self.d_model)

        mlp_in = x_in + cond_in + t_in + self.mlp_in_bias
        assert mlp_in.shape == (batch_size, self.d_model)

        return {
            "domain_proj": x_in,
            "cond_proj": cond_in,
            "time_proj": t_in,
            "mlp_input": mlp_in,
        }

    def __call__(self, x, t, cond_vec):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cond_vec.shape == (batch_size, self.conditioning_dim)

        # Process inputs so we can give them to the MLP
        inputs = self.process_inputs(x, t, cond_vec)
        mlp_in = inputs["mlp_input"]

        # Run them through our MLP and normalize the output
        mlp_out, _ = self.mlp_blocks(mlp_in, None)
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

        return tangent_outputs


def test_vector_field_projections_normalization():
    """
    Test that the outputs of domain_in_proj, conditioning_proj, and time_in_proj in the VectorField
    model have the distributions they're supposed to (mean 0, variance 1/2 or 1/3 depending on if
    conditioning_dim is > 0), and that the sum (input to MLP) has mean 0 and variance 1.

    Assumptions:
    * x is composed of uniformly distributed unit vectors in domain_dim dimensions
    * t is uniformly distributed between 0 and 1
    * cond_vec's components each have mean 0 variance 1

    """
    rng = jax.random.PRNGKey(42)

    # Test both with and without conditioning
    domain_dims = [3, 10]
    conditioning_cases = [0, 4]  # 0 for no conditioning, positive for conditioning

    for domain_dim in domain_dims:
        for conditioning_dim in conditioning_cases:
            model = VectorField(
                activations_dtype=jnp.float32,
                weights_dtype=jnp.float32,
                domain_dim=domain_dim,
                conditioning_dim=conditioning_dim,
                n_layers=1,
                d_model=2048,  # Bigger mlp_in vectors reduce the variance of our estimates
                mlp_expansion_factor=1,
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

            inputs = model.apply(
                state.params, x, t, cond_vec, method=model.process_inputs
            )

            expected_proj_variance = 1.0 / 3.0 if conditioning_dim > 0 else 1.0 / 2.0

            # Compute statistics for each projection, and their sum
            domain_proj = inputs["domain_proj"]
            domain_mean = jnp.mean(domain_proj)
            domain_variance = jnp.mean(jnp.var(domain_proj, axis=1))

            time_proj = inputs["time_proj"]
            time_mean = jnp.mean(time_proj)
            time_variance = jnp.mean(jnp.var(time_proj, axis=1))

            if conditioning_dim > 0:
                cond_proj = inputs["cond_proj"]
                cond_mean = jnp.mean(cond_proj)
                cond_variance = jnp.mean(jnp.var(cond_proj, axis=1))

            mlp_input = inputs["mlp_input"]
            mlp_input_mean = jnp.mean(mlp_input)
            mlp_input_variance = jnp.mean(jnp.var(mlp_input, axis=1))

            # Print statistics
            print(
                f"\nTesting with domain_dim={domain_dim}, conditioning_dim={conditioning_dim}"
            )
            print(f"Expected component variance: {expected_proj_variance:.6f}")
            print(
                f"Domain projection - Mean: {domain_mean:.6f}, Variance: {domain_variance:.6f}"
            )
            print(
                f"Time projection - Mean: {time_mean:.6f}, Variance: {time_variance:.6f}"
            )
            if conditioning_dim > 0:
                print(
                    f"Conditioning projection - Mean: {cond_mean:.6f}, Variance: {cond_variance:.6f}"
                )
            print(
                f"MLP input - Mean: {mlp_input_mean:.6f}, Variance: {mlp_input_variance:.6f}"
            )

            # Check if means are close to 0
            mean_tol = 0.02
            assert (
                abs(domain_mean) < mean_tol
            ), f"Domain projection mean should be close to 0, got {domain_mean}"
            assert (
                abs(domain_variance - expected_proj_variance) < 0.02
            ), f"Domain projection variance should be close to {expected_proj_variance}, got {domain_variance}"
            assert (
                abs(time_mean) < mean_tol
            ), f"Time projection mean should be close to 0, got {time_mean}"
            assert (
                abs(time_variance - expected_proj_variance) < 0.02
            ), f"Time projection variance should be close to {expected_proj_variance}, got {time_variance}"
            if conditioning_dim > 0:
                assert (
                    abs(cond_mean) < mean_tol
                ), f"Conditioning projection mean should be close to 0, got {cond_mean}"
                assert (
                    abs(cond_variance - expected_proj_variance) < 0.02
                ), f"Conditioning projection variance should be close to {expected_proj_variance}, got {cond_variance}"
            assert (
                abs(mlp_input_mean) < mean_tol
            ), f"MLP input mean should be close to 0, got {mlp_input_mean}"
            assert (
                abs(mlp_input_variance - 1) < 0.02
            ), f"MLP input variance should be close to 1, got {mlp_input_variance}"


def test_optimal_transport_field_direction():
    """
    Test that optimal_transport_field produces vectors that are aligned with
    the direct path from x to x1 after tangent space projection.
    """
    # Set random seed for reproducibility
    rng = jax.random.PRNGKey(0)

    # Test parameters
    dim = 3
    n_samples = 100
    kappa_1 = 100.0

    # Generate random points on the sphere
    key1, key2, key3 = jax.random.split(rng, 3)
    x_samples = sample_sphere(key1, n_samples, dim)
    x1_samples = sample_sphere(key2, n_samples, dim)
    t_values = jax.random.uniform(key3, (n_samples,))

    def compute_and_compare(x, x1, t):
        ot_field = spherical_ot_field(x, x1, t, kappa_1)

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
    similarities = jax.vmap(compute_and_compare)(x_samples, x1_samples, t_values)

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


def spherical_ot_field(x, x1, t, kappa_1):
    """
    Special fancy spherical version of the OT field. Compute a tangent vector field on the sphere
    that generates paths resembling optimal transport with regularization but follow geodesics
    on the sphere rather than straight lines. Based on vMF distributions instead of gaussians.

    Args:
        x: Current point on the sphere [dim]
        x1: Target point on the sphere [dim]
        t: Time parameter in [0, 1]
        kappa_1: Concentration parameter of the final vMF distribution

    Returns:
        Vector field at x, tangent to the sphere
    """
    assert len(x.shape) == 1
    assert x.shape == x1.shape
    assert t.shape == ()

    kappa_t = t * kappa_1

    cos_angle = jnp.clip(jnp.dot(x, x1), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    def handle_close_or_opposite():
        return jax.lax.cond(
            angle < 1e-8,
            # Field is 0 if x is very close to x1
            lambda: jnp.zeros_like(x),
            # If points are opposite/almost opposite, pick an orthogonal vector
            lambda: get_consistent_tangent_direction(x) * kappa_t,
        )

    def handle_general_case():
        # Vector field points towards x1 from x1, in the tangent space at x.
        proj_scalar = jnp.dot(x1, x)
        tangent_component = x1 - proj_scalar * x
        tangent_norm = jnp.linalg.norm(tangent_component)
        return jax.lax.cond(
            tangent_norm > 1e-8,
            lambda: kappa_t * tangent_component / tangent_norm,
            lambda: jnp.zeros_like(x),
        )

    return jax.lax.cond(
        jnp.logical_or(angle < 1e-8, angle > jnp.pi - 1e-8),
        handle_close_or_opposite,
        handle_general_case,
    )


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
    angle = jnp.arccos(cos_angle)

    def handle_close_or_opposite():
        return jax.lax.cond(
            angle < 1e-8,
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
        jnp.logical_or(angle < 1e-8, angle > jnp.pi - 1e-8),
        handle_close_or_opposite,
        lambda: slerp(x0, x1, t),
    )


def conditional_flow_matching_loss(
    model, params, x0, x1, t, conds=None, kappa_1=10.0, logits_table=None, caps_rng=None
):
    """
    Compute the Conditional Flow Matching loss from eq. 9 in the paper, modified for the spherical
    OT field.

    Args:
        model: Vector field model (VectorField or CapConditionedVectorField)
        params: Model parameters
        x0: Noise samples from p0 (batched) [batch_size, dim]
        x1: Target samples from p1 (batched) [batch_size, dim]
        t: Time parameters (batched) in [0, 1] [batch_size]
        conds: Conditioning vectors (batched) [batch_size, cond_dim] (only for VectorField)
        kappa_1: Concentration parameter of the final vMF distribution
        logits_table: LogitsTable for CapConditionedVectorField (required if model is CapConditionedVectorField)
        caps_rng: JAX random key for cap sampling (required if model is CapConditionedVectorField)

    Returns:
        CFM loss value (scalar)
    """
    assert len(x0.shape) == 2
    batch_size = x0.shape[0]
    assert x0.shape == x1.shape
    assert x0.shape[1] == model.domain_dim
    assert t.shape == (batch_size,)

    # Check if the model is a CapConditionedVectorField
    is_cap_conditioned = isinstance(model, CapConditionedVectorField)

    if is_cap_conditioned:
        assert (
            logits_table is not None
        ), "LogitsTable is required for CapConditionedVectorField"
        assert (
            caps_rng is not None
        ), "caps_rng is required for CapConditionedVectorField"
        assert (
            conds is None
        ), "extra conditioning is not supported for CapConditionedVectorField"
    else:
        assert conds is not None, "conds is required for regular VectorField"
        assert conds.shape == (batch_size, model.conditioning_dim)

    psi_t = jax.vmap(compute_psi_t_spherical)(x0, x1, t)
    assert psi_t.shape == x0.shape

    # Compute target vector field (ground truth OT field)
    target_field = jax.vmap(spherical_ot_field, in_axes=(0, 0, 0, None))(
        psi_t, x1, t, kappa_1
    )
    assert target_field.shape == x0.shape

    # Compute predicted vector field from our model (different for each model type)
    if is_cap_conditioned:
        predicted_field = model.apply(
            params,
            psi_t,
            t,
            logits_table,
            method=CapConditionedVectorField.forward_with_random_caps,
            rngs={"caps": caps_rng},
        )
    else:
        predicted_field = model.apply(params, psi_t, t, conds)

    assert predicted_field.shape == x0.shape

    loss = jnp.mean(jnp.sum((predicted_field - target_field) ** 2, axis=1))

    return loss


def create_train_state(rng, model, learning_rate_or_schedule):
    """Create initial training state."""
    dummy_x = jnp.ones((1, model.domain_dim))
    dummy_t = jnp.full((1,), 0.5)
    if isinstance(model, CapConditionedVectorField):
        params = model.init(
            rng, dummy_x, dummy_t, jnp.ones((1, model.domain_dim)), jnp.ones((1,))
        )
    else:
        params = model.init(
            rng, dummy_x, dummy_t, jnp.ones((1, model.conditioning_dim))
        )
    adamw = optax.adamw(learning_rate_or_schedule, weight_decay=0.001)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=adamw)


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


@partial(jax.jit, inline=True, static_argnames=("model", "kappa_1"))
def compute_batch_loss(model, params, batch, rng, kappa_1, logits_table=None):
    """
    Compute the loss for a batch of data.

    Args:
        model: The vector field model (VectorField or CapConditionedVectorField)
        params: Model parameters
        batch: Batch of data containing "point_vec" and "cond_vec" (cond_vec only needed for VectorField)
        rng: JAX random key
        kappa_1: Parameter for the flow matching loss
        logits_table: LogitsTable for CapConditionedVectorField (required if model is CapConditionedVectorField)

    Returns:
        loss: The computed loss value
    """
    x1_batch = batch["point_vec"]
    batch_size = x1_batch.shape[0]
    assert x1_batch.shape == (batch_size, model.domain_dim)

    rng, noise_rng, time_rng, caps_rng = jax.random.split(rng, 4)
    x0_batch = sample_sphere(noise_rng, batch_size, model.domain_dim)
    t = jax.random.uniform(time_rng, (batch_size,))

    is_cap_conditioned = isinstance(model, CapConditionedVectorField)

    if is_cap_conditioned:
        assert (
            "cond_vec" not in batch
        ), "cond_vec is not supported for CapConditionedVectorField"
        conds = None
    else:
        conds = batch["cond_vec"]
        assert conds.shape == (batch_size, model.conditioning_dim)

    return conditional_flow_matching_loss(
        model,
        params,
        x0_batch,
        x1_batch,
        t,
        conds=conds,
        kappa_1=kappa_1,
        logits_table=logits_table,
        caps_rng=caps_rng,
    )


@partial(
    jax.jit, static_argnames=("model", "kappa_1"), donate_argnames=("state", "rng")
)
def train_step(model, kappa_1, state, batch, rng, logits_table=None):
    """
    Train for a single step.

    Args:
        model: The vector field model
        kappa_1: Parameter for the flow matching loss
        state: Training state
        batch: Batch of data containing "point_vec" and "cond_vec"
        rng: JAX random key

    Returns:
        Updated state, loss value, gradient norm, and updated random key
    """
    rng, next_rng = jax.random.split(rng)

    def loss_fn(params):
        loss = compute_batch_loss(model, params, batch, rng, kappa_1, logits_table)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    grad_norm = optax.global_norm(grads)

    state = state.apply_gradients(grads=grads)

    return state, loss, grad_norm, next_rng


def _train_loop_for_tests(
    model,
    dataset,
    batch_size,
    learning_rate,
    epochs,
    test_dataset=None,
    kappa_1=10.0,
):
    """
    Simple training loop for unit tests.

    Args:
        model: The model to train
        dataset: Training dataset
        batch_size: Batch size for training
        learning_rate: Initial learning rate (will use cosine schedule)
        epochs: Number of epochs to train for
        test_dataset: Optional test dataset for evaluation after each epoch
        kappa_1: Parameter for the flow matching loss

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

    state = create_train_state(params_rng, model, cosine_schedule)
    np_rng = np.random.Generator(np.random.PCG64(seed=42))

    dummy_cond = jnp.zeros((batch_size, 0)) if model.conditioning_dim == 0 else None
    final_test_loss = None
    final_test_nll = None
    first_step = True
    step_count = 0

    print("Train! Loop! Go!")
    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            # Training loop
            for i, batch in enumerate(dataset.iter(batch_size, drop_last_batch=True)):
                if dummy_cond is not None:
                    batch["cond_vec"] = dummy_cond
                norms = jnp.linalg.norm(batch["point_vec"], axis=1, keepdims=True)
                np.testing.assert_allclose(np.asarray(norms), 1.0, rtol=0, atol=1e-6)

                state, train_loss, grad_norm, step_rng = train_step(
                    model, kappa_1, state, batch, step_rng
                )

                step_count += 1
                current_lr = cosine_schedule(step_count)

                if first_step or i % 10 == 0:
                    step_rng, nll_rng = jax.random.split(step_rng)
                    train_nlls = -compute_log_probability(
                        model,
                        state,
                        batch["point_vec"],
                        batch["cond_vec"],
                        100,
                        nll_rng,
                        10,
                    )
                    train_nll = float(np.mean(train_nlls))
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

                for test_batch in test_dataset.iter(batch_size, drop_last_batch=True):
                    if dummy_cond is not None:
                        test_batch["cond_vec"] = dummy_cond

                    test_batch_rng, test_nll_rng, test_rng = jax.random.split(
                        test_rng, 3
                    )

                    batch_test_loss = compute_batch_loss(
                        model, state.params, test_batch, test_batch_rng, kappa_1
                    )

                    test_losses.append(batch_test_loss)

                    test_nlls.append(
                        -compute_log_probability(
                            model,
                            state,
                            test_batch["point_vec"],
                            test_batch["cond_vec"],
                            100,
                            test_nll_rng,
                            10,
                        )
                    )

                if len(test_losses) > 0:
                    test_losses = jnp.asarray(test_losses)
                    avg_test_loss = jax.device_get(jnp.mean(test_losses))
                    final_test_loss = avg_test_loss

                    test_nlls = jnp.asarray(test_nlls)
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


def test_train_trivial():
    "Train a model with a single example"
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=0,
        n_layers=2,
        d_model=16,
        mlp_expansion_factor=4,
    )
    batch_size = 512
    points = repeat(jnp.array([1, 0, 0]), "v -> b v", b=batch_size * 11)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    train_dset = dset.select(range(batch_size * 10))
    test_dset = dset.select(range(batch_size * 10, batch_size * 11))
    state, loss, test_loss, test_nll = _train_loop_for_tests(
        model, train_dset, batch_size, 1e-3, 20, kappa_1=10.0, test_dataset=test_dset
    )
    print(f"Final loss: {loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test nll: {test_nll:.6f}")
    samples = generate_samples(
        model, state, jax.random.PRNGKey(0), 20, jnp.zeros((20, 0)), 1000, "rk4"
    )
    cos_sims = samples @ points[0]

    for sample, cos_sim in zip(samples, cos_sims):
        print(
            f"Sample: [{sample[0]:9.6f}, {sample[1]:9.6f}, {sample[2]:9.6f}]  Cosine similarity: {cos_sim:9.6f}"
        )

    assert cos_sims.shape == (20,)
    # It's very hard for the network to learn the vector field near the antipode of our target
    # point. That region of the sphere is very sparsely covered by paths. So any samples that start
    # there get stuck. Hopefully this isn't an issue with nontrivial distributions.
    high_sims = cos_sims > 0.99
    assert high_sims.mean() >= 0.95
    assert test_nll < -10.0


def _vmf_differential_entropy(kappa):
    "Calculate the differential entropy of a von Mises-Fisher distribution. Used for tests."
    sinh_k = np.sinh(kappa)
    coth_k = np.cosh(kappa) / sinh_k
    return np.log(4 * np.pi * sinh_k / kappa) - kappa * coth_k + 1


def test_train_vmf():
    """
    Train a model with data from a von Mises-Fisher distribution and evaluate the samples.
    """
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=0,
        n_layers=3,
        d_model=128,
        mlp_expansion_factor=4,
    )

    batch_size = 512
    n_samples = 32768

    mean_direction = np.array([1.0, 0.0, 0.0])
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
        model, dset, batch_size, 1e-3, 30, test_dset, kappa_1=5.0
    )
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test nll: {test_nll:.6f}")

    assert test_nll < differential_entropy + 0.5

    n_test_samples = 1_000
    samples = generate_samples(
        model,
        state,
        jax.random.PRNGKey(42),
        n_test_samples,
        jnp.zeros((n_test_samples, 0)),
        1000,
        "rk4",
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
        print(
            f"Sample: [{sample[0]:9.6f}, {sample[1]:9.6f}, {sample[2]:9.6f}]  "
            f"Log prob: {log_prob:9.6f}  Cosine similarity: {cos_sim:9.6f}"
        )

    # Allow some tolerance due to sampling variability
    assert (
        abs(sample_nll - differential_entropy) < 1.0
    ), f"Sample NLL {sample_nll} too far from differential entropy {differential_entropy}"


def test_train_conditional_vmf():
    """
    Train a model with data from two different von Mises-Fisher distributions
    conditioned on a binary conditioning vector.

    When conditioning vector is 0, samples should come from the first vMF distribution.
    When conditioning vector is 1, samples should come from the second vMF distribution.
    """
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=1,
        n_layers=2,
        d_model=32,
        mlp_expansion_factor=4,
    )

    # Define two different vMF distributions
    mean_direction1 = np.array([1.0, 0.0, 0.0])  # First distribution along x-axis
    mean_direction2 = np.array([0.0, 0.0, 1.0])  # Second distribution along z-axis
    kappa = 2.0

    vmf1 = stats.vonmises_fisher(mean_direction1, kappa)
    vmf2 = stats.vonmises_fisher(mean_direction2, kappa)

    # Sample from our distributions
    np_rng = np.random.Generator(np.random.PCG64(seed=42))
    total_samples = 20_000
    train_samples = int(total_samples * 0.9)
    test_samples = total_samples - train_samples
    points1 = vmf1.rvs(total_samples // 2)
    points2 = vmf2.rvs(total_samples // 2)

    cond_vec1 = np.zeros((total_samples // 2, 1))
    cond_vec2 = np.ones((total_samples // 2, 1))

    all_points = np.vstack([points1, points2])
    all_cond_vecs = np.vstack([cond_vec1, cond_vec2])

    dataset = Dataset.from_dict(
        {"point_vec": all_points, "cond_vec": all_cond_vecs}
    ).with_format("np")
    train_dataset = dataset.select(range(train_samples))
    test_dataset = dataset.select(range(train_samples, total_samples))

    batch_size = 256
    state, train_loss, test_loss, test_nll = _train_loop_for_tests(
        model, train_dataset, batch_size, 1e-2, 20, test_dataset, kappa_1=5.0
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
        model, state, seed1, n_eval_samples, cond_vec_0, 500, "rk4"
    )
    samples_1 = generate_samples(
        model, state, seed2, n_eval_samples, cond_vec_1, 500, "rk4"
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

    distribution_nll = _vmf_differential_entropy(kappa)
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


def spherical_rk4_step(f, x, t, dt):
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
    k1 = f(x, t)

    x2 = geodesic_step(x, k1, dt / 2)
    k2 = f(x2, t + dt / 2)
    k2_at_x = parallel_transport(k2, x2, x)

    x3 = geodesic_step(x, k2, dt / 2)
    k3 = f(x3, t + dt / 2)
    k3_at_x = parallel_transport(k3, x3, x)

    x4 = geodesic_step(x, k3, dt)
    k4 = f(x4, t + dt)
    k4_at_x = parallel_transport(k4, x4, x)

    combined_direction = (k1 + 2 * k2_at_x + 2 * k3_at_x + k4_at_x) / 6

    return geodesic_step(x, combined_direction, dt)


@partial(jax.jit, static_argnames=("model",), inline=True)
def spherical_rk4_step_with_model(
    model, state, x, t, dt, cond_vecs=None, cap_centers=None, cap_d_maxes=None
):
    vector_field_fn = lambda x, t: _compute_vector_field_for_sampling(
        model, state, x, t, cond_vecs, cap_centers, cap_d_maxes
    )
    return spherical_rk4_step(vector_field_fn, x, t, dt)


@partial(jax.jit, static_argnames=("model",), inline=True)
def _compute_vector_field_for_sampling(
    model, state, x, t, cond_vecs=None, cap_centers=None, cap_d_maxes=None
):
    if isinstance(model, CapConditionedVectorField):
        assert cond_vecs is None
        return model.apply(
            state.params, x, jnp.full((x.shape[0],), t), cap_centers, cap_d_maxes
        )
    else:
        return model.apply(state.params, x, jnp.full((x.shape[0],), t), cond_vecs)


def generate_samples(
    model,
    state,
    rng,
    batch_size,
    cond_vecs=None,
    cap_centers=None,
    cap_d_maxes=None,
    n_steps=100,
    method="rk4",
):
    """
    Generate samples from the flow matching model by solving the ODE.

    Args:
        model: Vector field model
        state: Training state with model parameters
        rng: JAX random key
        batch_size: Number of samples to generate
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        method: ODE solver method ('euler', 'midpoint', or 'rk4')

    Returns:
        Generated samples [batch_size, dim]
    """
    if isinstance(model, CapConditionedVectorField):
        assert cond_vecs is None
        assert cap_centers is not None
        assert cap_d_maxes is not None
        assert cap_centers.shape == (batch_size, model.domain_dim)
        assert cap_d_maxes.shape == (batch_size,)
    else:
        assert cond_vecs.shape == (batch_size, model.conditioning_dim)
        assert cap_centers is None
        assert cap_d_maxes is None

    # Sample initial points uniformly from the sphere
    x0 = sample_sphere(rng, batch_size, model.domain_dim)

    vector_field_fn = lambda x, t: _compute_vector_field_for_sampling(
        model, state, x, t, cond_vecs, cap_centers, cap_d_maxes
    )

    # Solve ODE
    dt = 1.0 / n_steps
    x = x0

    if method == "euler":
        # Forward Euler method
        for i in range(n_steps):
            t = i * dt
            v = vector_field_fn(x, t)
            x = geodesic_step(x, v, dt)
    elif method == "midpoint":
        # Midpoint method
        for i in range(n_steps):
            t = i * dt
            # First half-step
            v1 = vector_field_fn(x, t)
            x_mid = geodesic_step(x, v1, dt / 2)

            # Second half-step using midpoint derivative
            v2 = vector_field_fn(x_mid, t + 0.5 * dt)
            x = geodesic_step(x, v2, dt)
    elif method == "rk4":
        # 4th order Runge-Kutta method
        for i in range(n_steps):
            t = i * dt
            x = spherical_rk4_step_with_model(
                model, state, x, t, dt, cond_vecs, cap_centers, cap_d_maxes
            )
    else:
        raise ValueError(f"Unknown ODE solver method: {method}")

    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(x, axis=1, keepdims=True)), 1.0, atol=1e-6, rtol=0
    )
    return x


@partial(jax.jit, static_argnames=("model", "n_projections"))
def hutchinson_estimator(model, state, x, t, cond_vecs, step_rng, n_projections):
    """
    Estimate the divergence of a vector field on a spherical manifold using Hutchinson's trace
    estimator. Properly handles the (d-1)-dimensional tangent space of the d-dimensional sphere.

    Args:
        model: Vector field model
        state: Training state with model parameters
        x: Current points on the sphere [batch_size, dim]
        t: Current time
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        step_rng: JAX random key for this step
        n_projections: Number of random projections to use

    Returns:
        Divergence estimate [batch_size]
    """
    pass


@partial(jax.jit, static_argnames=("model",))
def exact_divergence(model, state, x, t, cond_vecs, step_rng=None, n_projections=None):
    """
    Compute the exact divergence of a vector field on a spherical manifold.
    Has the same interface as hutchinson_estimator for easy substitution.

    Args:
        model: Vector field model
        state: Training state with model parameters
        x: Current points on the sphere [batch_size, dim]
        t: Current time (scalar)
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        step_rng: JAX random key (unused, but kept for interface compatibility)
        n_projections: Number of random projections (unused, but kept for interface compatibility)

    Returns:
        Exact divergence [batch_size]
    """
    batch_size, dim = x.shape
    assert dim == model.domain_dim
    assert isinstance(t, float) or (isinstance(t, jnp.ndarray) and t.shape == ())
    assert cond_vecs.shape == (batch_size, model.conditioning_dim)

    # Helper to compute divergence for a single instance.
    def divergence_single(x_i, cond):
        # Define a function f: R^(dim) -> R^(dim) that wraps model.apply.
        # We add a batch dimension of 1 to x, t, and cond.
        def f(x_single):
            # x_single has shape [dim], so we reshape to [1, dim]
            # t is a scalar; we wrap it as [t] to form a batch of one.
            # cond has shape [cond_dim] and we reshape it to [1, cond_dim].
            return model.apply(
                state.params, x_single[None, :], jnp.array([t]), cond[None, :]
            )[0]

        # Compute the Jacobian (of shape [dim, dim]) with respect to x_i.
        jac = jax.jacfwd(f)(x_i)
        # The intrinsic divergence on the sphere is given by the trace of the projected Jacobian:
        #   divergence = trace(Df) - x^T (Df) x
        return jnp.trace(jac) - jnp.dot(x_i, jac @ x_i)

    # Vectorize the divergence computation over the batch dimension.
    return jax.vmap(divergence_single)(x, cond_vecs)


@partial(jax.jit, static_argnames=("model", "n_steps", "n_projections"))
def _reverse_path_and_compute_divergence(
    model, model_state, samples, cond_vecs, n_steps, rng, n_projections=1
):
    """
    Compute the reverse path and integrate the divergence.

    Args:
        model: Vector field model
        model_state: Training state with model parameters
        samples: Points on the sphere to evaluate [batch_size, dim]
        cond_vecs: Conditioning vectors [batch_size, cond_dim]
        n_steps: Number of integration steps
        rng: JAX random key for stochastic estimation
        n_projections: Number of random projections to use for divergence estimation

    Returns:
        div_sum: Integrated divergence along the path [batch_size]
    """
    batch_size = samples.shape[0]
    assert samples.shape == (batch_size, model.domain_dim)
    assert cond_vecs.shape == (batch_size, model.conditioning_dim)

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
        step_rng, rng = jax.random.split(rng)
        div_t = exact_divergence(
            model, model_state, x_t, t, cond_vecs, step_rng, n_projections
        )
        div_sum = div_sum + div_t * (1.0 / n_steps)

        # Take a step backward along the path
        next_x = spherical_rk4_step_with_model(
            model,
            model_state,
            x_t,
            t,
            -1.0 / n_steps,
            cond_vecs,
        )

        return next_x, div_sum, rng

    # Run the loop
    init_state = (x_t, div_sum, rng)
    final_x, final_div_sum, final_rng = jax.lax.fori_loop(
        0, n_steps, body_fun, init_state
    )

    return final_div_sum


def compute_log_probability(
    model, state, samples, cond_vecs, n_steps=100, rng=None, n_projections=1
):
    """
    Compute the log probability of samples under the flow-matching model.

    Args:
        model: Vector field model
        state: Training state with model parameters
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

    if rng is None:
        rng = jax.random.PRNGKey(0)
    # Normalize samples to ensure they're on the unit sphere
    samples = samples / jnp.linalg.norm(samples, axis=1, keepdims=True)

    div_sum = _reverse_path_and_compute_divergence(
        model, state, samples, cond_vecs, n_steps, rng, n_projections
    )

    # Density of the base distribution
    log_p0 = -(
        jnp.log(2 * jnp.power(jnp.pi, model.domain_dim / 2))
        - jax.lax.lgamma(model.domain_dim / 2)
    )
    log_p1 = log_p0 - div_sum
    return log_p1


@pytest.mark.parametrize(
    "divergence_fn,n_projections,field",
    [
        pytest.param(
            hutchinson_estimator,
            10,
            "zero_divergence",
            marks=pytest.mark.xfail(reason="Hutchinson estimator not implemented yet"),
        ),
        pytest.param(
            hutchinson_estimator,
            10,
            "variable_divergence",
            marks=pytest.mark.xfail(reason="Hutchinson estimator not implemented yet"),
        ),
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

    # Create a simple vector field with known divergence on the sphere
    def simple_vector_field(params, x, t, cond_vecs):
        if field == "zero_divergence":
            # Always 0
            return jax.vmap(lambda x_i: jnp.array([-x_i[1], x_i[0], 0]))(x)
        elif field == "variable_divergence":
            # -2 * x[2]
            return jax.vmap(lambda x_i: jnp.array([0, 0, 1]) - x_i * x_i[2])(x)
        else:
            raise ValueError(f"Unknown field: {field}")

    # Create a dummy model
    model = VectorField(
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        domain_dim=3,
        conditioning_dim=0,
        n_layers=1,
        d_model=16,
        mlp_expansion_factor=2,
    )

    # Create state with patched apply function
    rng = jax.random.PRNGKey(42)
    state = create_train_state(rng, model, 1e-3)

    # Replace the model's apply function with our test field
    original_apply = model.apply
    model.apply = simple_vector_field

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
    div_estimates = divergence_fn(model, state, x, t, cond_vecs, rng, n_projections)

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

    # Restore original apply function
    model.apply = original_apply


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
        n_layers=2,
        d_model=64,
        mlp_expansion_factor=4,
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


@dataclass
class CapConditionedVectorField(VectorField):
    """
    Model of a vector field that learns a distribution on the unit sphere and can sample conditional
    on the sample being within an arbitrary spherical cap.
    """

    def __post_init__(self):
        correct_conditioning_dim = self.domain_dim + 1
        assert (
            self.conditioning_dim is None
            or self.conditioning_dim == correct_conditioning_dim
        ), f"conditioning_dim can't be user controlled in CapConditionedVectorFields, but was {self.conditioning_dim}"
        self.conditioning_dim = correct_conditioning_dim
        super().__post_init__()

    def process_cap_inputs(self, cap_centers, cap_d_maxes):
        """
        Process the cap centers and max distances to have appropriate statistics.
        This helper method is used for testing the distribution of values.

        Args:
            cap_centers: Unit vectors defining cap centers [batch_size, domain_dim]
            cap_d_maxes: Maximum cosine distances [batch_size]

        Returns:
            Dictionary with:
            - cap_centers_transformed: Transformed cap centers [batch_size, domain_dim]
            - cap_d_maxes_transformed: Transformed max distances [batch_size]
            - cap_info: Combined conditioning vector for the base model [batch_size, conditioning_dim]
        """
        batch_size = cap_centers.shape[0]
        assert cap_centers.shape == (batch_size, self.domain_dim)
        assert cap_d_maxes.shape == (batch_size,)

        # Norm the input to have mean 0 variance 1
        # Unit vector components have std 1/sqrt(d), mean 0
        cap_centers_transformed = cap_centers * jnp.sqrt(self.domain_dim)
        # U[0, 2] has variance 1/3, mean 1
        cap_d_maxes_transformed = (cap_d_maxes - 1) * np.sqrt(3)

        # Convert the spherical cap info into a vector we can pass
        cap_info = jnp.concatenate(
            [cap_centers_transformed, cap_d_maxes_transformed[:, None]], axis=1
        )
        assert cap_info.shape == (batch_size, self.conditioning_dim)

        return {
            "cap_centers_transformed": cap_centers_transformed,
            "cap_d_maxes_transformed": cap_d_maxes_transformed,
            "cap_info": cap_info,
        }

    def __call__(self, x, t, cap_centers, cap_d_maxes):
        assert len(x.shape) == 2
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)
        assert cap_centers.shape == (batch_size, self.domain_dim)
        assert cap_d_maxes.shape == (batch_size,)

        processed = self.process_cap_inputs(cap_centers, cap_d_maxes)
        return super().__call__(x, t, processed["cap_info"])

    def forward_with_random_caps(self, x, t, logits_table):
        """
        Run the model forward on a batch of examples, generating random containing caps for each
        example beforehand. Useful exclusively for training.
        """
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)

        assert logits_table.d == self.domain_dim - 1

        rngs = jax.random.split(self.make_rng("caps"), batch_size)

        # idk if bias_d_max should be true or not. I think the data efficiency argument is weaker
        # in a design where you have a model dedicated to generating cap-conditioned embeddings
        # than in a design where you have a single model handling embeddings and actual data
        # generation.
        my_sample_cap = lambda rng, x: sample_cap(
            logits_table, rng, x, bias_d_max=False
        )
        cap_centers, cap_d_maxes = jax.vmap(my_sample_cap)(rngs, x)

        assert cap_centers.shape == (batch_size, self.domain_dim)
        assert cap_d_maxes.shape == (batch_size,)
        return self.__call__(x, t, cap_centers, cap_d_maxes)


def test_cap_conditioned_vector_field_normalization():
    """
    Test that CapConditionedVectorField properly normalizes the inputs passed to the superclass.
    This verifies that the cap centers and max distances are properly transformed to have the
    statistical properties expected by the base VectorField class.

    The test checks:
    1. That cap centers are scaled to have variance 1 per dimension
    2. That cap distances are transformed to have mean 0 and variance 1
    3. That the combined conditioning vector has appropriate statistics
    """
    rng = jax.random.PRNGKey(42)

    # Test with different dimensions like in the VectorField test
    domain_dims = [3, 5, 8]

    for domain_dim in domain_dims:
        # Create the model
        model = CapConditionedVectorField(
            activations_dtype=jnp.float32,
            weights_dtype=jnp.float32,
            domain_dim=domain_dim,
            n_layers=1,
            d_model=2048,
            mlp_expansion_factor=1,
            conditioning_dim=None,  # Should be automatically set to domain_dim + 1
        )

        params_rng, sample_rng = jax.random.split(rng)
        state = create_train_state(params_rng, model, 1e-3)

        # Generate test inputs
        n_samples = 10_000
        keys = jax.random.split(sample_rng, 3)

        cap_centers = sample_sphere(keys[0], n_samples, domain_dim)
        cap_d_maxes = jax.random.uniform(
            keys[1], shape=(n_samples,), minval=0.0, maxval=2.0
        )

        # Get the processed conditioning data using the model's helper method
        processed = model.apply(
            state.params, cap_centers, cap_d_maxes, method=model.process_cap_inputs
        )

        cap_centers_transformed = processed["cap_centers_transformed"]
        cap_d_maxes_transformed = processed["cap_d_maxes_transformed"]
        cap_info = processed["cap_info"]

        assert cap_centers_transformed.shape == (n_samples, domain_dim)
        assert cap_d_maxes_transformed.shape == (n_samples,)
        assert cap_info.shape == (n_samples, model.conditioning_dim)

        # Check statistics of transformed values
        cap_centers_mean = jnp.mean(cap_centers_transformed, axis=0)
        cap_centers_var = jnp.var(cap_centers_transformed, axis=0)
        cap_d_maxes_mean = jnp.mean(cap_d_maxes_transformed)
        cap_d_maxes_var = jnp.var(cap_d_maxes_transformed)

        print(
            f"\nTesting CapConditionedVectorField normalization with domain_dim={domain_dim}"
        )
        print(f"Cap centers - Mean: {jnp.mean(cap_centers_mean):.6f}, Expected: ~0.0")
        print(
            f"Cap centers - Variance: {jnp.mean(cap_centers_var):.6f}, Expected: ~1.0"
        )
        print(f"Cap distances - Mean: {cap_d_maxes_mean:.6f}, Expected: ~0.0")
        print(f"Cap distances - Variance: {cap_d_maxes_var:.6f}, Expected: ~1.0")

        assert jnp.all(
            jnp.abs(cap_centers_mean) < 0.05
        ), f"Cap center mean should be close to 0, got {cap_centers_mean}"
        assert jnp.all(
            jnp.abs(cap_centers_var - 1.0) < 0.05
        ), f"Cap center variance should be close to 1, got {cap_centers_var}"
        assert (
            jnp.abs(cap_d_maxes_mean) < 0.05
        ), f"Cap distances mean should be close to 0, got {cap_d_maxes_mean}"
        assert (
            jnp.abs(cap_d_maxes_var - 1.0) < 0.05
        ), f"Cap distances variance should be close to 1, got {cap_d_maxes_var}"


def test_train_cap_conditioned_model():
    """
    Train a cap conditioned model on a simple 3d training distribution and check the samples are
    inside the input caps and correctly distributed.
    """

    model = CapConditionedVectorField(
        domain_dim=3,
        n_layers=4,
        d_model=256,
        mlp_expansion_factor=4,
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        conditioning_dim=None,
    )

    distribution_rng, shuffle_rng, params_rng, train_rng, sample_rng = jax.random.split(
        jax.random.PRNGKey(0), 5
    )

    logits_table = LogitsTable(d=model.domain_dim - 1, n=8192)

    kappa_1 = 5.0

    # Generate a training set from a discrete uniform distribution of 5 fixed random points on the
    # sphere.
    n_distribution_points = 5
    distribution_points = sample_sphere(distribution_rng, n_distribution_points, 3)
    training_points = jnp.repeat(distribution_points, 2000, axis=0)
    assert training_points.shape == (10_000, 3)
    shuffle_indices = jax.random.permutation(
        shuffle_rng, jnp.arange(training_points.shape[0])
    )
    training_points = training_points[shuffle_indices]

    train_set_size = floor(training_points.shape[0] * 0.9)
    dset = Dataset.from_dict({"point_vec": training_points}).with_format("np")
    train_dset = dset.select(range(train_set_size))
    test_dset = dset.select(range(train_set_size, training_points.shape[0]))

    # Train a model
    epochs = 20
    batch_size = 256
    batches_per_epoch = len(train_dset) // batch_size
    total_steps = epochs * batches_per_epoch

    cosine_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=floor(total_steps * 0.1),
        decay_steps=total_steps,
    )
    train_state = create_train_state(
        params_rng, model, learning_rate_or_schedule=cosine_schedule
    )

    for epoch in tqdm(range(epochs), desc="Epochs"):
        with tqdm(total=batches_per_epoch, desc=f"Epoch {epoch}") as pbar:
            for i, batch in enumerate(
                train_dset.iter(batch_size, drop_last_batch=True)
            ):
                train_state, loss, grad_norm, train_rng = train_step(
                    model, kappa_1, train_state, batch, train_rng, logits_table
                )
                pbar.set_postfix(loss=loss, grad_norm=grad_norm)
                pbar.update(1)

            test_losses = []
            for batch in tqdm(
                test_dset.iter(batch_size, drop_last_batch=True),
                total=len(test_dset) // batch_size,
                desc="Testing",
            ):
                test_losses.append(
                    compute_batch_loss(
                        model,
                        train_state.params,
                        batch,
                        train_rng,
                        kappa_1,
                        logits_table,
                    )
                )
            test_loss = jnp.mean(jnp.array(test_losses))
            tqdm.write(f"Test loss: {test_loss}")

    print(f"distribution points: {distribution_points}")

    n_samples = 20

    print("Sampling from full sphere...")
    full_sample_rng, center_rng, sample_rng = jax.random.split(sample_rng, 3)
    centers = sample_sphere(center_rng, n_samples, 3)
    d_maxes = jnp.full((n_samples,), 1.0)
    full_samples = generate_samples(
        model,
        train_state,
        full_sample_rng,
        n_samples,
        cap_centers=centers,
        cap_d_maxes=d_maxes,
    )
    print(f"full samples: {full_samples}")

    # Compute the cosine similarity of each distribution point/sample pair
    cos_similarities = jnp.dot(full_samples, distribution_points.T)
    assert cos_similarities.shape == (n_samples, n_distribution_points)

    closest_distribution_points = jnp.argmax(cos_similarities, axis=1)
    assert closest_distribution_points.shape == (n_samples,)
    print(f"closest distribution points: {closest_distribution_points}")
    print(
        f"closest cos similarity: {cos_similarities[jnp.arange(n_samples), closest_distribution_points]}"
    )
    assert jnp.all(
        cos_similarities[jnp.arange(n_samples), closest_distribution_points] > 0.999
    )

    print("Sampling from hemisphere...")
    hemisphere_sample_rng, center_rng, sample_rng = jax.random.split(sample_rng, 3)
    # Sample a random center for our hemisphere to have
    center = sample_sphere(center_rng, 1, 3)[0]
    centers = jnp.repeat(center[None, :], n_samples, axis=0)
    d_maxes = jnp.full((n_samples,), 1.0)

    hemisphere_samples = generate_samples(
        model,
        train_state,
        hemisphere_sample_rng,
        n_samples,
        cap_centers=centers,
        cap_d_maxes=d_maxes,
    )
    print(f"hemisphere samples: {hemisphere_samples}")
    cos_similarity_targets = jnp.concatenate(
        [center[None, :], distribution_points], axis=0
    )
    assert cos_similarity_targets.shape == (n_distribution_points + 1, 3)
    cos_similarities = jnp.dot(hemisphere_samples, cos_similarity_targets.T)
    print(f"cosine similarities to cap center:\n{cos_similarities[:, 0]}")
    print(f"cosine similarities to distribution points:\n{cos_similarities[:, 1:]}")
    # All samples should be within the hemisphere
    assert jnp.all(cos_similarities[:, 0] > 0)
