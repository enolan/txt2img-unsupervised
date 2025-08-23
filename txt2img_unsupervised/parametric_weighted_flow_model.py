"""
Parametric weighted flow matching models for spherical data.

This module contains the WeightedFlowModel class and related functionality that allows training
flow matching models with parametric weighting functions. This enables learning distributions
that are weighted versions of the base flow distribution, specified at inference time.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Union
import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import pytest

from .cap_sampling import (
    LogitsTable,
    process_d_max_dist,
    random_pt_with_cosine_similarity,
    sample_cap,
)
from .flow_matching import VectorField, sample_sphere


class WeightingFunction(Enum):
    """The function that weights a WeightedFlowModel's distribution."""

    CONSTANT = "constant"
    """Constant weight - all samples weighted equally, no conditioning."""

    CAP_INDICATOR = "cap_indicator"
    """
    Indicator function for a spherical cap - 1 inside the cap, 0 outside, conditioning data is cap
    center and d_max.
    PROS: Straightforward, intuitively what we want for EGCG.
    CONS: Doesn't god damn work. Flow models do not want to learn sharp discontinuous distributions,
          you get samples outside the cap and (I think) distortion inside the cap. Especially bad
          with higher dimensionality and/or smaller caps.
    """

    SMOOTHED_CAP_INDICATOR = "smoothed_cap_indicator"
    """
    Smoothed indicator function for a spherical cap - 1 inside the cap, linear falloff to 0 outside
    the cap up to a boundary, then 0 beyond the boundary. Conditioning data is the same as
    CAP_INDICATOR.
    PROS: Distribution is now continuous, hopefully more learnable. And the distribution is very
          close to what we want for EGCG, so SIR is efficient.
    CONS: Now we have an extra tunable parameter (distance to boundary).
    """

    VMF_DENSITY = "vmf_density"
    """
    vMF density function (unnormalized) - exp(kappa * mu.T @ x). More weight for samples closer to
    the mean, smooth falloff, density is positive everywhere. Conditioning data is mu and kappa.
    Not implemented yet.
    PROS: Continuous, differentiable, very smooth, hopefully even easier to learn.
    CONS: Divergence between the weighted distribution and the target distribution is higher, SIR is
          less efficient.
    """


@dataclass(frozen=True)
class CapIndicatorExtraParams:
    """
    Extra hyperparameters for WeightedFlowModels that use the cap indicator weighting function.
    """

    d_max_dist: Tuple[Tuple[float, float], ...] = ((0.95, 1.0), (0.05, 2.0))
    """
    Training distribution of maximum cosine distances, specified as a mixture of uniform
    distributions. Each tuple contains the weight of the mixture component and the upper d_max
    bound for that component. Note this does not actually affect the *weighting function* per se,
    but is used during training to choose d_max and may affect how well the models learns the
    weighted distribution at different d_max values.
    """


@dataclass(frozen=True)
class SmoothedCapIndicatorExtraParams:
    """
    Extra hyperparameters for WeightedFlowModels that use the smoothed cap indicator weighting function.
    """

    d_max_dist: Tuple[Tuple[float, float], ...] = ((0.95, 1.0), (0.05, 2.0))
    """
    Training distribution of maximum cosine distances, specified as a mixture of uniform
    distributions. Each tuple contains the weight of the mixture component and the upper d_max
    bound for that component. Note this does not actually affect the *weighting function* per se,
    but is used during training to choose d_max and may affect how well the models learns the
    weighted distribution at different d_max values.
    """

    boundary_width: float = jnp.pi / 10
    """
    Width of the linear falloff in the smoothed cap indicator, in radians.
    """


class WeightedFlowModel(nn.Module):
    """A flow model trained to sample from a distribution, weighted by a function specified at
    inference time. The *family* of weighting functions is specified at initialization, before
    training, including the choice of WeightingFunction and some hyperparameters included in
    weighting_function_extra_params. At inference time, further parameters are passed to the model,
    fully specifying the weighting function.
    """

    # Some notes for the paper:

    # Weighted generative models are a superset of class conditional generative models:
    # We can recover class-conditional generative models with a very simple extension of this. For
    # concreteness, imagine a model that generates pictures of cats or dogs. And we have a labeled
    # dataset. If we think of the label as part of the example we're generating, then our weighting
    # function can consider it. We augment each cat/dog picture with a one hot vector that says what
    # kind of animal it's a picture of, and use a weighting function that's an indicator of which
    # value in the one-hot is hot. Very roundabout, but shows that regular class conditional
    # generative models are a special case of this. And, just because we're "considering" the class
    # as part of the example doesn't mean our model needs to actually output classes again at
    # sampling time. Each example can be split into a) the item of the domain we're generating and
    # b) arbitrary data that the weighting function may consider. b would be used to generate
    # conditioning data during training, but not be reproduced.

    # Requirements for the weighting function:
    # 1. With the obviously implementation, there must be at least one set of weighting function
    #    parameters that produces positive weight for every point in the domain. But you can drop
    #    this requirement if you simply ignore training examples that cannot have positive weight.
    #    This would be kinda dumb but nevertheless.
    # 2. Weights must be nonnegative & finite everywhere.
    # 3. ??? idk, in general the reversed distribution must be valid.

    # Hyperparameters for the underlying VectorField
    domain_dim: int
    reference_directions: Optional[int]
    time_dim: int
    use_pre_mlp_projection: bool
    n_layers: int
    d_model: int
    mlp_expansion_factor: int
    mlp_dropout_rate: Optional[float]
    input_dropout_rate: Optional[float]
    activations_dtype: jnp.dtype = jnp.float32
    weights_dtype: jnp.dtype = jnp.float32
    d_model_base: int = 512
    variance_base: float = 1 / 512
    alpha_input: float = 1.0
    alpha_output: float = 1.0

    # Weighting-related hyperparameters.
    weighting_function: WeightingFunction = WeightingFunction.CONSTANT
    weighting_function_extra_params: Optional[
        Union[CapIndicatorExtraParams, SmoothedCapIndicatorExtraParams]
    ] = None

    @property
    def conditioning_dim(self) -> int:
        if self.weighting_function == WeightingFunction.CONSTANT:
            return 0
        elif self.weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ]:
            return (
                self.reference_directions
                if self.reference_directions
                else self.domain_dim
            ) + 1
        elif self.weighting_function == WeightingFunction.VMF_DENSITY:
            return NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")

    @nn.nowrap
    def mk_vector_field(self) -> VectorField:
        # We need a VectorField object for various calculations, some of which should happen outside
        # of Flax init/apply, when setup() won't have run yet. So we define a function to create
        # one. They're immutable and pure functions of the constructor params, so we can make as
        # many as we feel like and they'll all be identical.
        return VectorField(
            domain_dim=self.domain_dim,
            reference_directions=self.reference_directions,
            conditioning_dim=self.conditioning_dim,
            time_dim=self.time_dim,
            use_pre_mlp_projection=self.use_pre_mlp_projection,
            n_layers=self.n_layers,
            d_model=self.d_model,
            mlp_expansion_factor=self.mlp_expansion_factor,
            mlp_dropout_rate=self.mlp_dropout_rate,
            input_dropout_rate=self.input_dropout_rate,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            d_model_base=self.d_model_base,
            variance_base=self.variance_base,
            alpha_input=self.alpha_input,
            alpha_output=self.alpha_output,
        )

    def setup(self) -> None:
        if self.weighting_function == WeightingFunction.VMF_DENSITY:
            raise NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        elif self.weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ]:
            self.logits_table = LogitsTable(self.domain_dim - 1, 8192)
            # Precompute bucket angles α for the logits table indices. Using convention cos(α) = -h
            # where h are the table heights in [-1, 1].
            buckets = self.logits_table.buckets
            idxs = jnp.arange(buckets, dtype=jnp.int32)
            heights = 2.0 * (idxs.astype(jnp.float32) / (buckets - 1.0)) - 1.0
            cos_thetas = -heights
            self._bucket_alphas = jnp.arccos(jnp.clip(cos_thetas, -1.0, 1.0))

        self.vector_field = self.mk_vector_field()

    @nn.nowrap
    def _build_smoothed_weighted_logits_table(self, d_max: jax.Array) -> LogitsTable:
        """Construct a weighted ``LogitsTable`` reflecting the smoothed-cap indicator around angle
        0 with edge at θ_cap and linear falloff of width ``boundary_width``.

        For bucket angles α corresponding to table heights, weights are
        1 for α ≤ θ_cap; linearly decreasing to 0 over (θ_cap, θ_cap + boundary_width]; and 0
        beyond. This reweights the base slice-area logits so that sampling heights is equivalent to
        sampling directions with density proportional to area × smoothed-weight.
        """
        boundary_width = self.weighting_function_extra_params.boundary_width
        cap_theta = jnp.arccos(jnp.clip(1.0 - d_max, -1.0, 1.0))
        alpha_max = jnp.minimum(jnp.pi, cap_theta + boundary_width)
        alphas = self._bucket_alphas
        within_support = alphas <= alpha_max
        linear_falloff = jnp.maximum(
            0.0, 1.0 - (alphas - cap_theta) / jnp.maximum(boundary_width, 1e-12)
        )
        weight_vals = jnp.where(
            alphas <= cap_theta, 1.0, jnp.where(within_support, linear_falloff, 0.0)
        )
        return self.logits_table.weighted(weight_vals)

    def dummy_inputs(self) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Create dummy inputs for model initialization with the correct shapes.

        Returns:
            Tuple of (x, t, cap_centers, cap_d_maxes) with appropriate shapes for initialization.
        """
        x = jnp.ones((1, self.domain_dim))
        t = jnp.ones((1,))
        if self.weighting_function == WeightingFunction.CONSTANT:
            weighting_function_params = None
        elif self.weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ]:
            weighting_function_params = (jnp.ones((1, self.domain_dim)), jnp.ones((1,)))
        elif self.weighting_function == WeightingFunction.VMF_DENSITY:
            raise NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")
        return x, t, weighting_function_params

    @property
    def d_model_scale_factor(self) -> float:
        "m_d in muP."
        return self.mk_vector_field().d_model_scale_factor

    @nn.nowrap
    def mk_partition_map(self, use_muon: bool):
        """
        Create a partition map for optimizer configuration with muP scaling.
        Delegates to the underlying VectorField.
        """
        return {
            "params": {
                "vector_field": self.mk_vector_field().mk_partition_map(use_muon)[
                    "params"
                ]
            }
        }

    @nn.nowrap
    def scale_lr(self, lr: float) -> float:
        "Scaled learning rate for hidden layers. Delegates to the underlying VectorField."
        return self.mk_vector_field().scale_lr(lr)

    def process_weighting_function_params(
        self, cond_vecs: jax.Array, cond_scalars: jax.Array
    ) -> jax.Array:
        """
        Convert weighting-function parameters to the model's conditioning vector. In principle,
        there could be other shapes of inference-time weighting function parameters, but ATM all our
        (non-constant) weighting functions take a single vector and scalar.

        Args:
            cond_vecs: Unit vectors in R^{domain_dim} (cap center / direction parameter).
            cond_scalars: Scalars in [0, 2] representing d_max (maximum cosine distance).

        Returns:
            Conditioning vectors of shape `(batch, conditioning_dim)` with approximately mean 0 and
            variance 1 per component, suitable for feeding to the VectorField.
        """
        assert len(cond_vecs.shape) == 2, f"cond_vecs.shape: {cond_vecs.shape}"
        batch_size = cond_vecs.shape[0]
        assert cond_vecs.shape == (
            batch_size,
            self.domain_dim,
        ), f"cond_vecs.shape: {cond_vecs.shape}"
        assert cond_scalars.shape == (
            batch_size,
        ), f"cond_scalars.shape: {cond_scalars.shape}"
        assert self.weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ], f"process_weighting_function_params called with unsupported weighting function: {self.weighting_function}"

        # Encode the direction parameter to match how inputs are encoded, then scale to unit variance.
        # - If using reference directions, project onto them and scale by sqrt(domain_dim) so that
        #   each component has variance ~1 for isotropic inputs.
        # - Otherwise, pass through coordinates scaled by sqrt(domain_dim).
        if self.reference_directions is not None:
            # [ref_dirs, domain_dim] <- reference vectors; produce [ref_dirs]
            dir_features = (
                cond_vecs @ self.vector_field.reference_vectors.T
            ) * jnp.sqrt(self.domain_dim)
        else:
            dir_features = cond_vecs * jnp.sqrt(self.domain_dim)
        assert dir_features.shape == (
            batch_size,
            self.conditioning_dim - 1,
        ), f"dir_features.shape: {dir_features.shape}"

        # Normalize d_max scalars using the training distribution specified by d_max_dist
        weights, range_starts, range_ends = process_d_max_dist(
            self.weighting_function_extra_params.d_max_dist
        )
        component_means = (range_starts + range_ends) / 2.0
        component_vars = (range_ends - range_starts) ** 2 / 12.0
        mixture_mean = jnp.sum(weights * component_means)
        mixture_var = jnp.sum(
            weights * (component_vars + (component_means - mixture_mean) ** 2)
        )
        mixture_std = jnp.sqrt(mixture_var)
        assert jnp.all(mixture_std > 0), "mixture_std must be positive"

        scalar_features = (cond_scalars - mixture_mean) / mixture_std
        scalar_features = scalar_features[:, None]
        assert scalar_features.shape == (batch_size, 1)

        processed_cond_vecs = jnp.concatenate([dir_features, scalar_features], axis=1)
        assert processed_cond_vecs.shape == (batch_size, self.conditioning_dim)
        return processed_cond_vecs

    def __call__(self, x, t, weighting_function_params):
        if self.weighting_function == WeightingFunction.CONSTANT:
            assert weighting_function_params is None
            cond_vecs_for_inner_model = jnp.zeros((x.shape[0], 0))
        elif (
            self.weighting_function == WeightingFunction.CAP_INDICATOR
            or self.weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR
        ):
            assert isinstance(weighting_function_params, tuple)
            assert len(weighting_function_params) == 2
            cond_vecs, cond_scalars = weighting_function_params
            cond_vecs_for_inner_model = self.process_weighting_function_params(
                cond_vecs, cond_scalars
            )
        elif self.weighting_function == WeightingFunction.VMF_DENSITY:
            raise NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")
        return self.vector_field(x, t, cond_vecs_for_inner_model)

    def compute_weight(
        self,
        x: jax.Array,
        weighting_function_params: Optional[Tuple[jax.Array, jax.Array]],
    ) -> jax.Array:
        """
        Compute the weight of a point under the weighting function defined by the parameters.
        Returns a float32 scalar in [0, 1]. (In principle this could be any nonnegative finite
        value, but this implementation returns weights in [0, 1].)
        """
        assert x.shape == (self.domain_dim,)
        if self.weighting_function == WeightingFunction.CONSTANT:
            assert weighting_function_params is None
            return jnp.asarray(1.0, dtype=jnp.float32)
        elif (
            self.weighting_function == WeightingFunction.CAP_INDICATOR
            or self.weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR
        ):
            assert isinstance(weighting_function_params, tuple)
            assert len(weighting_function_params) == 2
            cap_center, d_max = weighting_function_params

            assert cap_center.shape == (self.domain_dim,)
            assert d_max.shape == ()

            cos_sim_to_cap_center = jnp.dot(x, cap_center)
            cos_distance_to_cap_center = 1 - cos_sim_to_cap_center
            in_cap = cos_distance_to_cap_center <= d_max
            if self.weighting_function == WeightingFunction.CAP_INDICATOR:
                return in_cap.astype(jnp.float32)  # convert boolean to 1.0 or 0.0
            elif self.weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
                # If the point is in the cap, the weight is 1.0. If it's outside the cap, weight
                # falls off linearly to zero at boundary_width radians away from the cap's edge.
                boundary_width = self.weighting_function_extra_params.boundary_width
                cap_theta = jnp.arccos(1 - d_max)
                x_theta = jnp.arccos(cos_sim_to_cap_center)
                x_theta_to_cap_edge = x_theta - cap_theta
                x_theta_to_cap_edge_weight = jnp.minimum(
                    1.0, jnp.maximum(0.0, 1 - x_theta_to_cap_edge / boundary_width)
                )
                return x_theta_to_cap_edge_weight.astype(jnp.float32)
        elif self.weighting_function == WeightingFunction.VMF_DENSITY:
            raise NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")

    def sample_weighting_function_params(
        self, x: jax.Array
    ) -> Optional[Tuple[jax.Array, jax.Array]]:
        """
        Given a point x, sample a set of parameters for the weighting function from a distribution
        that's density is proportional to the weighting function and independent of any attribute of
        x that does not affect the weighting function. Used at training time to make the conditioned
        weighting work.

        This is the clever bit that makes the weighted distribution work. We want our model to learn
        to sample from a family of distributions, all of which are weighted versions of the same
        base distribution. In other words, depending on the weighting function parameters, the
        probability of any training example should be scaled by the weighting function.

        The naive method would be to draw the weighting function's parameters independently of the
        data and scale the loss for each example by the weighting function. In principle this should
        work but if the vast vast majority of examples will end up having very low weights (consider
        what happens in high dimensions where the caps we care about and are training on can have
        fractional areas as small as e^-262), then sample and compute efficiency will be horrible.
        Our EGCG setting is exactly that, so we need a better method.

        The clever method is, as above, to draw the weighting function's parameters from a
        distribution that's density is proportional to the weight that would apply given those. This
        means that when we flip it around the density at any given point is weighted proportionally
        to the weight at that point.
        """
        if self.weighting_function == WeightingFunction.CONSTANT:
            return None
        rng = self.make_rng("sample_weighting_params")
        if self.weighting_function == WeightingFunction.CAP_INDICATOR:
            return sample_cap(
                self.logits_table,
                rng,
                x,
                self.weighting_function_extra_params.d_max_dist,
            )
        elif self.weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
            # Sample d_max from the configured mixture (independent of x)
            weights, range_starts, range_ends = process_d_max_dist(
                self.weighting_function_extra_params.d_max_dist
            )

            comp_rng, dmax_rng, angle_rng, dir_rng = jax.random.split(rng, 4)
            component_idx = jax.random.categorical(comp_rng, jnp.log(weights))
            d_max = jax.random.uniform(
                dmax_rng,
                minval=range_starts[component_idx],
                maxval=range_ends[component_idx],
            )

            # Build weighted logits table using helper and sample with existing interpolation logic
            weighted_tbl = self._build_smoothed_weighted_logits_table(d_max)
            # Pass d_max=2.0 so no additional truncation; support is encoded in weighted_tbl logits
            sampled_dist = weighted_tbl.sample_cap_cos_distance(angle_rng, 2.0)
            cap_center = random_pt_with_cosine_similarity(dir_rng, x, sampled_dist)

            return cap_center, d_max
        elif self.weighting_function == WeightingFunction.VMF_DENSITY:
            raise NotImplementedError(
                "VMF density weighting function not implemented yet."
            )
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")


@pytest.mark.parametrize(
    "weighting_function",
    [
        WeightingFunction.CONSTANT,
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ],
)
@pytest.mark.parametrize("domain_dim", [3, 8])
def test_compute_weight(weighting_function, domain_dim):
    """
    Test the compute_weight function for different weighting functions and edge cases.
    """
    rng = jax.random.PRNGKey(42)
    test_points = sample_sphere(rng, 10, domain_dim)

    # Create model with the specified weighting function
    if weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
        extra_params = SmoothedCapIndicatorExtraParams(boundary_width=0.2)
    else:
        extra_params = CapIndicatorExtraParams()

    model = WeightedFlowModel(
        domain_dim=domain_dim,
        time_dim=16,
        reference_directions=8,
        n_layers=1,
        d_model=32,
        mlp_expansion_factor=2,
        activations_dtype=jnp.float32,
        weights_dtype=jnp.float32,
        weighting_function=weighting_function,
        weighting_function_extra_params=extra_params,
        input_dropout_rate=None,
        mlp_dropout_rate=None,
        use_pre_mlp_projection=True,
    )

    if weighting_function == WeightingFunction.CONSTANT:
        weights = jax.vmap(lambda point: model.compute_weight(point, None))(test_points)
        assert weights.shape == (test_points.shape[0],)
        assert weights.dtype == jnp.float32
        np.testing.assert_array_equal(weights, 1.0)

    elif weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        # Test cap-based weighting functions
        cap_center = jnp.zeros(domain_dim).at[0].set(1.0)
        d_max = jnp.array(0.5)
        params = (cap_center, d_max)

        weights = jax.vmap(lambda point: model.compute_weight(point, params))(
            test_points
        )
        assert weights.shape == (test_points.shape[0],)
        assert weights.dtype == jnp.float32
        assert jnp.all(
            (weights >= 0.0) & (weights <= 1.0)
        )  # All weights should be in [0, 1]

        # Compute expected weights manually using vectorized operations
        cos_sims = test_points @ cap_center
        cos_distances = 1 - cos_sims
        in_caps = cos_distances <= d_max

        if weighting_function == WeightingFunction.CAP_INDICATOR:
            expected_weights = in_caps.astype(jnp.float32)
            np.testing.assert_array_equal(weights, expected_weights)
        elif weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
            # Check that points in the cap have weight 1, points in the boundary have a positive
            # weight less than 1, and points outside have weight 0.
            boundary_width = model.weighting_function_extra_params.boundary_width
            cap_theta = jnp.arccos(1 - d_max)
            x_thetas = jnp.arccos(cos_sims)
            x_thetas_to_cap_edges = x_thetas - cap_theta
            in_cap_idxs = jnp.where(in_caps)[0]
            assert jnp.all(weights[in_cap_idxs] == 1.0)
            in_boundary = (x_thetas_to_cap_edges <= boundary_width) & (
                x_thetas_to_cap_edges >= 0
            )
            assert jnp.all((weights[in_boundary] > 0.0) & (weights[in_boundary] < 1.0))
            neither = ~in_caps & ~in_boundary
            assert jnp.all(weights[neither] == 0.0)

        # Test edge cases
        # Point exactly at cap center
        center_weight = model.compute_weight(cap_center, params)
        np.testing.assert_array_equal(center_weight, 1.0)

        # Point at antipode (maximum distance)
        antipode = -cap_center
        antipode_weight = model.compute_weight(antipode, params)
        if weighting_function == WeightingFunction.CAP_INDICATOR:
            np.testing.assert_array_equal(antipode_weight, 0.0)
        else:  # SMOOTHED_CAP_INDICATOR
            # Should be 0 since antipode is far outside boundary
            np.testing.assert_array_equal(antipode_weight, 0.0)

        # Test point exactly on cap boundary
        # Create a point that is exactly d_max cosine distance from center
        # For a point on the cap boundary: cos_sim = 1 - d_max
        target_cos_sim = 1 - d_max

        # Create an orthogonal vector in the subspace perpendicular to cap_center
        orthogonal = jnp.zeros(domain_dim).at[1].set(1.0)
        orthogonal = orthogonal - jnp.dot(orthogonal, cap_center) * cap_center
        orthogonal = orthogonal / jnp.linalg.norm(orthogonal)

        # Create boundary point using spherical geometry
        boundary_angle = jnp.arccos(target_cos_sim)
        boundary_point = (
            jnp.cos(boundary_angle) * cap_center + jnp.sin(boundary_angle) * orthogonal
        )
        boundary_point = boundary_point / jnp.linalg.norm(
            boundary_point
        )  # Ensure unit norm

        boundary_weight = model.compute_weight(boundary_point, params)

        if weighting_function == WeightingFunction.CAP_INDICATOR:
            # Should be exactly 1.0 since point is on boundary (inclusive)
            np.testing.assert_allclose(boundary_weight, 1.0, rtol=0, atol=1e-6)
        elif weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
            # Should be 1.0 since point is exactly on cap edge
            np.testing.assert_allclose(boundary_weight, 1.0, rtol=0, atol=1e-6)
        else:
            raise ValueError(f"Unknown weighting function: {weighting_function}")

        if weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
            # Check that a point in the middle of the boundary has weight 0.5
            midpoint_angle = boundary_angle + boundary_width / 2
            midpoint = (
                jnp.cos(midpoint_angle) * cap_center
                + jnp.sin(midpoint_angle) * orthogonal
            )
            midpoint = midpoint / jnp.linalg.norm(midpoint)  # Ensure unit norm
            midpoint_weight = model.compute_weight(midpoint, params)
            np.testing.assert_allclose(midpoint_weight, 0.5, rtol=0, atol=1e-6)


@pytest.mark.parametrize("domain_dim", [3, 8, 16])
@pytest.mark.parametrize("reference_directions", [None, 8, 16])
@pytest.mark.parametrize(
    "d_max_dist",
    [
        None,  # Default: uniform U[0, 2]
        [(1.0, 2.0)],  # Explicit uniform U[0, 2]
        [(0.95, 1.0), (0.05, 2.0)],  # Biased distribution
        [(0.45, 0.8), (0.45, 1.2), (0.1, 2.0)],  # Triangular distribution
    ],
)
def test_process_weighting_function_params(
    domain_dim, reference_directions, d_max_dist
):
    "Verify that process_weighting_function_params returns normalized vectors of the right shape."

    # Create CapIndicatorExtraParams with the specified d_max_dist
    if d_max_dist is not None:
        extra_params = CapIndicatorExtraParams(
            d_max_dist=tuple(tuple(x) for x in d_max_dist)
        )
    else:
        extra_params = CapIndicatorExtraParams()

    model = WeightedFlowModel(
        domain_dim=domain_dim,
        reference_directions=reference_directions,
        time_dim=16,
        use_pre_mlp_projection=False,
        n_layers=1,
        d_model=128,
        mlp_expansion_factor=4,
        mlp_dropout_rate=None,
        input_dropout_rate=None,
        weighting_function=WeightingFunction.CAP_INDICATOR,
        weighting_function_extra_params=extra_params,
    )

    params = model.init(jax.random.PRNGKey(0), *model.dummy_inputs())

    # Generate a batch of unit vectors and d_max values, the former uniformly distributed and the
    # latter from the same d_max training distribution used by the model.
    table = LogitsTable(2, 8192)
    n = 8192
    rng = jax.random.PRNGKey(20250820)
    d_max_rng, unit_vec_rng = jax.random.split(rng, 2)

    weights, range_starts, range_ends = process_d_max_dist(
        model.weighting_function_extra_params.d_max_dist
    )
    component_rng, d_max_rng = jax.random.split(d_max_rng, 2)
    component_idxs = jax.random.categorical(component_rng, jnp.log(weights), shape=(n,))
    d_maxes = jax.random.uniform(
        d_max_rng,
        minval=range_starts[component_idxs],
        maxval=range_ends[component_idxs],
        shape=(n,),
    )

    unit_vecs = sample_sphere(unit_vec_rng, n, model.domain_dim)

    # Process to conditioning vectors
    processed = model.apply(
        params, unit_vecs, d_maxes, method=model.process_weighting_function_params
    )
    assert processed.shape == (n, model.conditioning_dim)

    processed_np = jax.device_get(processed)
    means = processed_np.mean(axis=0)
    stds = processed_np.std(axis=0)

    # Check approximately zero-mean and unit-std per component
    np.testing.assert_allclose(means, 0.0, atol=0.05, rtol=0)
    np.testing.assert_allclose(stds, 1.0, atol=0.05, rtol=0)