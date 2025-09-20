from dataclasses import dataclass, field, replace
from datasets import Dataset
from enum import Enum
from functools import partial
from tqdm import tqdm
from typing import FrozenSet, Literal, Optional, Tuple, Union
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from . import flow_matching
from .cap_sampling import (
    LogitsTable,
    process_d_max_dist,
    random_pt_with_cosine_similarity,
    sample_cap,
    sample_from_cap,
    sphere_log_inverse_surface_area,
)
from .flow_matching import VectorField, sample_sphere


class WeightingFunction(Enum):
    """The function that weights a FunctionWeightedFlowModel's distribution."""

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
    Extra hyperparameters for FunctionWeightedFlowModels that use the cap indicator weighting
    function.
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
    Extra hyperparameters for FunctionWeightedFlowModels that use the smoothed cap indicator
    weighting function.
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


class FunctionWeightedFlowModel(nn.Module):
    """A function-weighted flow model - i.e. one trained to sample from a distribution, weighted
    by a function specified at inference time. The *family* of weighting functions is specified at
    initialization, before training, including the choice of WeightingFunction and some
    hyperparameters included in weighting_function_extra_params. At inference time, further
    parameters are passed to the model, fully specifying the weighting function.

    This implements the general concept of a function-weighted generative model.
    """

    # Some notes for the paper:

    # Function-weighted generative models are a superset of class conditional generative models:
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
    time_dim: Optional[int]
    use_pre_mlp_projection: bool
    n_layers: int
    d_model: int
    mlp_expansion_factor: int
    mlp_dropout_rate: Optional[float]
    input_dropout_rate: Optional[float]
    mlp_always_inject: FrozenSet[Literal["x", "t", "cond"]] = field(
        default_factory=frozenset
    )
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

    # Base distribution is uniform over the conditioning cap. Only makes sense with CAP_INDICATOR.
    cap_conditioned_base: bool = False

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
            mlp_always_inject=self.mlp_always_inject,
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
            if self.cap_conditioned_base:
                if (
                    max(p[1] for p in self.weighting_function_extra_params.d_max_dist)
                    > 1.0
                ):
                    raise ValueError(
                        "d_max_dist max value must be <= 1.0 for cap-conditioned base"
                    )

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

    def prepare_training_conditioning(self, batch):
        """Prepare conditioning data for training - generate weighting parameters for each point.

        Args:
            batch: Training batch containing "point_vec"

        Returns:
            Weighting function parameters for the batch
        """
        if self.weighting_function == WeightingFunction.CONSTANT:
            return None

        x1_batch = batch["point_vec"]
        return jax.vmap(lambda x1: self.sample_weighting_function_params(x1))(x1_batch)

    def sample_base_distribution(self, weighting_function_params, batch_size):
        """Sample base distribution x0 for training.
        Depends on cap_conditioned_base:
        - False: uniform on the sphere.
        - True: uniform within the cap specified by each example's weighting-function parameters.
          Supports caps of size up to a hemisphere (d_max <= 1.0) only. This ensures all paths are
          inside the cap.
        """
        rng = self.make_rng("sample_base")
        if self.cap_conditioned_base:
            assert (
                self.weighting_function == WeightingFunction.CAP_INDICATOR
            ), "cap_conditioned_base only supported with CAP_INDICATOR weighting"
            assert (
                isinstance(weighting_function_params, tuple)
                and len(weighting_function_params) == 2
            )
            cap_centers, d_maxes = weighting_function_params
            assert cap_centers.shape == (batch_size, self.domain_dim)
            assert d_maxes.shape == (batch_size,)
            keys = jax.random.split(rng, batch_size)
            x0 = jax.vmap(lambda k, c, d: sample_from_cap(k, self.logits_table, c, d))(
                keys, cap_centers, d_maxes
            )
            return x0
        else:
            return sample_sphere(rng, batch_size, self.domain_dim)

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

        scalar_features = (cond_scalars - mixture_mean) / mixture_std
        scalar_features = scalar_features[:, None]
        assert scalar_features.shape == (batch_size, 1)

        processed_cond_vecs = jnp.concatenate([dir_features, scalar_features], axis=1)
        assert processed_cond_vecs.shape == (batch_size, self.conditioning_dim)
        return processed_cond_vecs

    def __call__(self, x, t, weighting_function_params):
        batch_size = x.shape[0]
        assert x.shape == (batch_size, self.domain_dim)
        assert t.shape == (batch_size,)

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
            assert cond_vecs.shape == (batch_size, self.domain_dim)
            assert cond_scalars.shape == (batch_size,)

            cond_vecs_for_inner_model = self.process_weighting_function_params(
                cond_vecs, cond_scalars
            )
            assert cond_vecs_for_inner_model.shape == (
                batch_size,
                self.conditioning_dim,
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

    model = FunctionWeightedFlowModel(
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

    model = FunctionWeightedFlowModel(
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


def generate_samples(
    model,
    params,
    rng,
    weighting_function_params,
    n_steps=20,
    method="tsit5",
    batch_size=None,
):
    """
    Generate samples from a FunctionWeightedFlowModel.

    Args:
        model: FunctionWeightedFlowModel
        params: Model parameters
        rng: JAX random key
        weighting_function_params: Parameters for the weighting function. PyTree with leading dim
            batch_size, None if the weighting function is constant.
        n_steps: Number of integration steps
        method: Method of integration
        batch_size: Batch size. If None, will be inferred from the shape of the weighting function
            parameters.
    """
    leading_dims = jax.tree.leaves(
        jax.tree.map(lambda x: x.shape[0], weighting_function_params)
    )
    if len(leading_dims) == 0:
        if weighting_function_params is not None:
            raise ValueError(
                "weighting_function_params must be None (unparameterized weighting function) or a "
                "PyTree of arrays with leading dimension batch_size (all others)."
            )
        if batch_size is None:
            raise ValueError(
                "batch_size must be specified if weighting_function_params is None"
            )
    else:
        if all(x == leading_dims[0] for x in leading_dims):
            inferred_batch_size = leading_dims[0]
        else:
            raise ValueError(
                "All leading dimensions of weighting function parameters must be the same, got "
                f"{leading_dims}"
            )
        if batch_size is None:
            batch_size = inferred_batch_size
        elif batch_size != inferred_batch_size:
            raise ValueError(
                "batch_size must match the leading dimension of the weighting function parameters "
                f"got {batch_size}, but inferred {inferred_batch_size} from the weighting function "
                "parameters."
            )
    x0_rng, path_rng = jax.random.split(rng)
    x0 = model.apply(
        params,
        x0_rng,
        method=model.sample_base_distribution,
        weighting_function_params=weighting_function_params,
    )

    samples, _eval_counts = flow_matching.generate_samples_inner(
        path_rng,
        n_steps,
        batch_size,
        method,
        _compute_vector_field,
        model,
        params,
        weighting_function_params,
        model.domain_dim,
        initial_x0=x0,
    )
    return samples


# lifted to top level so JIT gets cached
_log_cap_size_batch = jax.jit(lambda tbl, d_maxes: jax.vmap(tbl.log_cap_size)(d_maxes))


def compute_log_probability(
    model,
    params,
    samples,
    weighting_function_params,
    n_steps=20,
    rng=None,
    n_projections=10,
):
    """
    Compute the log probability of samples under a function-weighted flow model.
    """
    if model.cap_conditioned_base:
        assert (
            model.weighting_function == WeightingFunction.CAP_INDICATOR
        ), "cap_conditioned_base only supported with CAP_INDICATOR weighting"
        assert (
            isinstance(weighting_function_params, tuple)
            and len(weighting_function_params) == 2
        )
        assert rng is not None, "rng must be provided for cap-conditioned base"
        cap_centers, d_maxes = weighting_function_params
        assert d_maxes.shape[0] == samples.shape[0]
        # log density of uniform-in-cap = -log(cap_area)
        # log(cap_area) = log_cap_size_frac + log(sphere_area)
        log_sphere_area = -sphere_log_inverse_surface_area(model.domain_dim)
        # FIXME should pass logits table as parameter
        table = LogitsTable(model.domain_dim - 1, 8192)
        log_cap_size_frac = _log_cap_size_batch(table, d_maxes)
        base_log_densities = -(log_cap_size_frac + log_sphere_area)
    else:
        base_log_densities = sphere_log_inverse_surface_area(model.domain_dim)

    # Reverse flow to get source points and divergence integral
    x0, div_sum = flow_matching.reverse_path_and_compute_divergence(
        _compute_vector_field,
        model,
        params,
        weighting_function_params,
        samples,
        n_steps,
        rng,
        n_projections,
        method="tsit5",
        tsit5_settings=flow_matching.Tsit5Settings(atol=1e-2, rtol=1e-2),
    )

    # If using a cap-conditioned base, zero out density for sources outside the cap
    if model.cap_conditioned_base:
        assert model.weighting_function == WeightingFunction.CAP_INDICATOR
        assert (
            isinstance(weighting_function_params, tuple)
            and len(weighting_function_params) == 2
        )
        cap_centers, d_maxes = weighting_function_params
        cos_dists = 1 - jnp.sum(x0 * cap_centers, axis=1)
        in_support = cos_dists <= d_maxes
        log_p1 = base_log_densities - div_sum
        return jnp.where(in_support, log_p1, -jnp.inf)
    else:
        return base_log_densities - div_sum


@partial(jax.jit, static_argnames=("model",), inline=True)
def _compute_vector_field(model, params, weighting_function_params, x, t, rng=None):
    rngs_dict = {"dropout": rng} if rng is not None else {}
    return model.apply(
        params,
        x,
        jnp.full((x.shape[0],), t),
        weighting_function_params,
        rngs=rngs_dict,
    )


# Baseline configuration used for tests
_baseline_model = FunctionWeightedFlowModel(
    domain_dim=3,
    reference_directions=None,
    time_dim=128,
    use_pre_mlp_projection=True,
    n_layers=4,
    d_model=256,
    mlp_expansion_factor=4,
    mlp_dropout_rate=None,
    input_dropout_rate=None,
    weighting_function=WeightingFunction.CONSTANT,
    weighting_function_extra_params=None,
)


@pytest.mark.usefixtures("starts_with_progressbar")
@pytest.mark.parametrize("domain_dim", [3, 16])
@pytest.mark.parametrize(
    "weighting_function",
    [
        WeightingFunction.CONSTANT,
        pytest.param(
            WeightingFunction.CAP_INDICATOR,
            marks=[],  # pytest.mark.xfail(reason="Models can't learn this well"),
        ),
        pytest.param(
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
            marks=pytest.mark.skip(reason="Models can't learn this well"),
        ),
    ],
)
@pytest.mark.parametrize(
    "mlp_always_inject",
    [
        pytest.param(frozenset(), id="none"),
        pytest.param(frozenset({"x"}), id="x", marks=pytest.mark.skip(reason="slow")),
        pytest.param(frozenset({"t"}), id="t", marks=pytest.mark.skip(reason="slow")),
        pytest.param(
            frozenset({"cond"}), id="cond", marks=pytest.mark.skip(reason="slow")
        ),
        pytest.param(frozenset({"x", "t"}), id="x&t", marks=pytest.mark.skip(reason="slow")),
        pytest.param(
            frozenset({"x", "cond"}), id="x&cond",
        ),
        pytest.param(
            frozenset({"t", "cond"}), id="t&cond", marks=pytest.mark.skip(reason="slow")
        ),
        pytest.param(
            frozenset({"x", "t", "cond"}),
            id="x&t&cond",
            marks=pytest.mark.skip(reason="slow"),
        ),
    ],
)
def test_train_uniform(domain_dim, weighting_function, mlp_always_inject):
    """
    Train a function-weighted model on uniform distribution, then verify it produces correct weighted distributions.
    """
    # Set up model with appropriate extra parameters
    d_max_dist = ((1.0, 1.0),)
    if weighting_function == WeightingFunction.SMOOTHED_CAP_INDICATOR:
        extra_params = SmoothedCapIndicatorExtraParams(
            d_max_dist=d_max_dist, boundary_width=jnp.pi / 10
        )
    elif weighting_function == WeightingFunction.CAP_INDICATOR:
        extra_params = CapIndicatorExtraParams(d_max_dist=d_max_dist)
    else:
        extra_params = None

    model = replace(
        _baseline_model,
        domain_dim=domain_dim,
        d_model=512,
        n_layers=16,
        time_dim=None,
        weighting_function=weighting_function,
        weighting_function_extra_params=extra_params,
        use_pre_mlp_projection=True,
        mlp_always_inject=mlp_always_inject,
        cap_conditioned_base=weighting_function == WeightingFunction.CAP_INDICATOR,
    )
    rng = jax.random.PRNGKey(20250823)
    train_rng, test_rng = jax.random.split(rng)

    # Generate uniform training distribution
    n_train_examples = 1_000_000
    train_points = sample_sphere(train_rng, n_train_examples, domain_dim)
    assert train_points.shape == (n_train_examples, domain_dim)

    # Create dataset
    dsets = (
        Dataset.from_dict({"point_vec": train_points})
        .with_format("np")
        .train_test_split(test_size=4096)
    )
    train_dataset = dsets["train"]
    test_dataset = dsets["test"]

    # Train the model using the shared infrastructure
    batch_size = 512
    learning_rate = 1e-4
    if domain_dim == 3 and weighting_function == WeightingFunction.CONSTANT:
        epochs = 2
    elif domain_dim == 3 and weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        epochs = 6
    elif domain_dim == 16 and weighting_function == WeightingFunction.CONSTANT:
        epochs = 6
    elif domain_dim == 16 and weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        epochs = 32
    else:
        raise ValueError(
            f"Unknown domain_dim: {domain_dim} and weighting_function: {weighting_function}"
        )

    print(
        f"Training FWFM for domain_dim={domain_dim}, weighting_function={weighting_function}"
    )

    # Use the generic training loop via partial application
    state, final_loss, test_loss, test_nll = _train_loop_for_tests(
        model=model,
        dataset=train_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        test_dataset=test_dataset,
    )

    print(
        f"Training completed. Final loss: {final_loss:.6f}, test loss: {test_loss:.6f}, test NLL: {test_nll:.6f}"
    )

    # Generate independent test points
    n_test_points = 1000
    test_points = sample_sphere(test_rng, n_test_points, domain_dim)

    # Get uniform sphere log density
    uniform_log_density = sphere_log_inverse_surface_area(domain_dim)

    # Test different parameter sets based on weighting function
    if weighting_function == WeightingFunction.CONSTANT:
        param_sets = [None]
    elif weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        # Test multiple cap configurations
        center = jnp.zeros(domain_dim).at[0].set(1.0)  # [1, 0, 0, ...]
        param_sets = [
            (center, jnp.array(d_max, dtype=jnp.float32)) for d_max in [1.0, 0.5]
        ]
    else:
        raise ValueError(f"Unknown weighting function: {weighting_function}")
    # Test each parameter set
    for i, params in enumerate(param_sets):
        print(f"Testing parameter set {i+1}/{len(param_sets)}: {params}")

        # Compute true weights
        if params is None:
            true_weights = jnp.ones(n_test_points)  # CONSTANT weighting function
        else:
            true_weights = jax.vmap(lambda point: model.compute_weight(point, params))(
                test_points
            )

        assert jnp.all(true_weights >= 0.0), "All weights must be non-negative"
        assert jnp.any(true_weights > 0.0), "At least one weight must be positive"

        # Multiply uniform density by weight by this to get weighted density
        weight_normalization_factor = 1 / jnp.mean(true_weights)

        print(
            f"  Weight stats: min={true_weights.min():.3f}, max={true_weights.max():.3f}, mean={true_weights.mean():.3f}"
        )
        print(
            f"  Positive weights: {jnp.sum(true_weights > 0)}/{len(true_weights)} points"
        )
        print(f"  Weight normalization factor: {weight_normalization_factor:.3f}")

        # Compute model log probabilities
        if params is None:
            model_params_batch = None
        else:
            center, d_max = params
            model_params_batch = (
                jnp.broadcast_to(center, (n_test_points, domain_dim)),
                jnp.full((n_test_points,), d_max),
            )
        if weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ]:
            # Test density along a geodesic from cap center to antipode (inclusive).
            num_geodesic_points = 10
            ts = jnp.linspace(0.0, 1.0, num_geodesic_points)
            geodesic_points = jax.vmap(
                lambda tau: flow_matching.compute_psi_t_spherical(center, -center, tau)
            )(ts)
            extra_test_points = geodesic_points
            extended_test_points = jnp.concatenate(
                [extra_test_points, test_points], axis=0
            )
            assert extended_test_points.shape == (
                num_geodesic_points + n_test_points,
                domain_dim,
            )

            extended_model_params_batch = (
                jnp.broadcast_to(
                    center, (num_geodesic_points + n_test_points, domain_dim)
                ),
                jnp.full((num_geodesic_points + n_test_points,), d_max),
            )
        else:
            extended_test_points = test_points
            extended_model_params_batch = model_params_batch

        extended_model_log_probs = compute_log_probability(
            model=model,
            params=state.params,
            samples=extended_test_points,
            weighting_function_params=extended_model_params_batch,
            n_steps=20,
            rng=jax.random.PRNGKey(12345),
            n_projections=10,
        )
        model_log_probs = extended_model_log_probs[
            extended_test_points.shape[0] - n_test_points :
        ]

        if weighting_function in [
            WeightingFunction.CAP_INDICATOR,
            WeightingFunction.SMOOTHED_CAP_INDICATOR,
        ]:
            print(
                "  Geodesic path model log probs (10 points from center to antipode):"
            )
            print(f"    log_probs: {extended_model_log_probs[:10]}")

        # Split into zero-weight and positive-weight cases
        zero_weight_mask = true_weights == 0.0
        positive_weight_mask = ~zero_weight_mask

        # For positive weights: check the relationship model_log_prob ≈ log(weight) + uniform_log_density
        if jnp.any(positive_weight_mask):
            expected_log_probs = (
                jnp.log(
                    true_weights[positive_weight_mask] * weight_normalization_factor
                )
                + uniform_log_density
            )

            model_valid_probs_for_positive_weights = model_log_probs[
                positive_weight_mask
            ]

            absdiffs = jnp.abs(
                model_valid_probs_for_positive_weights - expected_log_probs
            )
            count_diffs_over_15pct = np.sum(absdiffs > jnp.log(1.15))

            print("  Checking densities for points with positive weight...")
            print(
                f"    Model range: [{model_valid_probs_for_positive_weights.min():.3f}, {model_valid_probs_for_positive_weights.max():.3f}]"
            )
            print(
                f"  Model deciles: {np.percentile(model_valid_probs_for_positive_weights, np.linspace(0, 100, 11))}"
            )
            print(
                f"    Model mean: {model_valid_probs_for_positive_weights.mean():.3f} Model std: {model_valid_probs_for_positive_weights.std():.3f}"
            )
            print(
                f"    Expected range: [{expected_log_probs.min():.3f}, {expected_log_probs.max():.3f}]"
            )
            print(f"    Mean abs diff: {jnp.mean(absdiffs):.3f}")
            print(
                f"    Number of points with abs diff > 15%: {count_diffs_over_15pct}/{len(absdiffs)}"
            )
            print(
                f"    Mean diff: {jnp.mean(model_valid_probs_for_positive_weights - expected_log_probs):.3f}"
            )

            assert count_diffs_over_15pct < 0.1 * len(
                absdiffs
            ), f"Too many likelihoods differ by more than 15%: {count_diffs_over_15pct}/{len(absdiffs)}"
        else:
            print(f"  WARNING: No positive weights found for parameter set {params}")

        # For zero weights the theoretical log probability would be -inf, which is only possible if
        # the base distribution has compact support i.e. only with cap_conditioned_base.
        if jnp.any(zero_weight_mask):
            if model.cap_conditioned_base:
                sufficiently_negative_logprob = -jnp.inf
            else:
                MAXIMUM_DENSITY_RATIO = 0.05  # VERY generous ratio... :'(
                sufficiently_negative_logprob = uniform_log_density + jnp.log(
                    MAXIMUM_DENSITY_RATIO
                )
            zero_weight_log_probs = model_log_probs[zero_weight_mask]
            num_too_high = jnp.sum(
                zero_weight_log_probs > sufficiently_negative_logprob
            )

            print(
                f"  Checking densities for points with zero weight - should be less than {sufficiently_negative_logprob}"
            )
            print(
                f"    Model range: [{zero_weight_log_probs.min():.3f}, {zero_weight_log_probs.max():.3f}]"
            )
            print(
                f"    Model mean: {zero_weight_log_probs.mean():.3f} Model std: {zero_weight_log_probs.std():.3f}"
            )
            print(
                f"    Mean diff: {jnp.mean(zero_weight_log_probs - sufficiently_negative_logprob):.3f}"
            )
            print(
                f"    Number of points with log prob > {sufficiently_negative_logprob}: {num_too_high}/{len(zero_weight_log_probs)}"
            )

            assert (
                num_too_high / len(zero_weight_log_probs) < 0.05
            ), f"{num_too_high} zero-weight points have log prob >= {sufficiently_negative_logprob}"


def compute_hemisphere_probability_masses(
    model, params, rng, n_samples, n_steps, n_projections
):
    """Compute the model probability masses for the northern and southern hemispheres."""

    # We compute the likelihood of n_samples points that are in both the northern and eastern
    # hemispheres, conditioned on the cap being the nothern hemisphere, then do the same conditioned
    # on the cap being the eastern hemisphere, then repeat the process for n_samples points that
    # are in both the southern and eastern hemispheres. Since the ratio of likelihoods is the
    # ratio of the probability masses of the caps, and because we know the masses of the northern
    # and southern hemispheres sum to 1, we can find the masses of all three. We return only the
    # masses of the northern and southern hemispheres, since that's all that will actually be used
    # downstream.

    # In principle n_samples could be one and the ratios would work out but these models are
    # approximations.

    assert model.weighting_function == WeightingFunction.CAP_INDICATOR

    samples_rng, logprob_rng = jax.random.split(rng)

    north = jnp.zeros(model.domain_dim).at[0].set(1.0)
    south = -north
    east = jnp.zeros(model.domain_dim).at[1].set(1.0)

    # Generate samples uniform over full sphere
    initial_samples = sample_sphere(samples_rng, 2 * n_samples, model.domain_dim)
    # reflect across n-s axis to put everything in eastern hemisphere
    eastern_samples = initial_samples.at[:, 1].set(jnp.abs(initial_samples[:, 1]))
    # reflect across e-w axis to put everything in northern/southern hemisphere
    northeast_samples = (
        eastern_samples[:n_samples]
        .at[:, 0]
        .set(jnp.abs(eastern_samples[:n_samples, 0]))
    )
    southeast_samples = (
        eastern_samples[n_samples:]
        .at[:, 0]
        .set(-jnp.abs(eastern_samples[n_samples:, 0]))
    )

    # We have 2 * n_samples points, each of which needs to be checked in 2 caps.
    cap_centers = jnp.concatenate(
        [
            jnp.broadcast_to(north, (n_samples, model.domain_dim)),
            jnp.broadcast_to(east, (2 * n_samples, model.domain_dim)),
            jnp.broadcast_to(south, (n_samples, model.domain_dim)),
        ],
        axis=0,
    )
    cap_d_maxes = jnp.full((4 * n_samples,), 1.0)

    logprobs = compute_log_probability(
        model=model,
        params=params,
        samples=jnp.concatenate(
            [
                northeast_samples,
                northeast_samples,
                southeast_samples,
                southeast_samples,
            ],
            axis=0,
        ),
        weighting_function_params=(cap_centers, cap_d_maxes),
        n_steps=n_steps,
        rng=logprob_rng,
        n_projections=n_projections,
    )

    assert logprobs.shape == (4 * n_samples,)
    north_east_in_north_logprobs = logprobs[:n_samples]
    north_east_in_east_logprobs = logprobs[n_samples : 2 * n_samples]
    south_east_in_east_logprobs = logprobs[2 * n_samples : 3 * n_samples]
    south_east_in_south_logprobs = logprobs[3 * n_samples :]
    north_east_in_north_finite_mask = jnp.isfinite(north_east_in_north_logprobs)
    north_east_in_east_finite_mask = jnp.isfinite(north_east_in_east_logprobs)
    south_east_in_east_finite_mask = jnp.isfinite(south_east_in_east_logprobs)
    south_east_in_south_finite_mask = jnp.isfinite(south_east_in_south_logprobs)

    northeast_both_finite_mask = (
        north_east_in_north_finite_mask & north_east_in_east_finite_mask
    )
    southeast_both_finite_mask = (
        south_east_in_east_finite_mask & south_east_in_south_finite_mask
    )

    north_east_ratios_log = north_east_in_north_logprobs - north_east_in_east_logprobs
    south_east_ratios_log = south_east_in_east_logprobs - south_east_in_south_logprobs
    # Ignore points for which one or both likelihoods are -inf. Kind of a hack.
    north_east_ratios_log = jnp.where(
        northeast_both_finite_mask, north_east_ratios_log, 0.0
    )
    south_east_ratios_log = jnp.where(
        southeast_both_finite_mask, south_east_ratios_log, 0.0
    )

    # Average the ratios in linear space
    north_east_ratio_log = jax.nn.logsumexp(north_east_ratios_log) - jnp.log(n_samples)
    south_east_ratio_log = jax.nn.logsumexp(south_east_ratios_log) - jnp.log(n_samples)

    # We have N/E and S/E, N/S is (N/E) / (S/E)
    north_south_ratio_log = north_east_ratio_log - south_east_ratio_log

    # N / S = r
    # N = r * S
    # N + S = 1
    # r * S + S = 1
    # (r + 1) * S = 1
    # S = 1 / (r + 1)
    # same for north but reciprocate r

    south_log_mass = -(jnp.logaddexp(north_south_ratio_log, 0.0))
    north_log_mass = -(jnp.logaddexp(-north_south_ratio_log, 0.0))

    output_dict = {
        "north_log_mass": north_log_mass,
        "south_log_mass": south_log_mass,
        "northeast_both_finite_frac": jnp.mean(northeast_both_finite_mask),
        "southeast_both_finite_frac": jnp.mean(southeast_both_finite_mask),
        "north_east_log_ratios_std": jnp.std(north_east_ratios_log),
        "south_east_log_ratios_std": jnp.std(south_east_ratios_log),
    }

    tqdm.write(
        f"north mass: {float(jnp.exp(output_dict['north_log_mass'])):.3f}, south mass: {float(jnp.exp(output_dict['south_log_mass'])):.3f}"
    )
    tqdm.write(
        f"northeast both finite frac: {output_dict['northeast_both_finite_frac']:.3f}, southeast both finite frac: {output_dict['southeast_both_finite_frac']:.3f}"
    )
    tqdm.write(
        f"north east log ratios std: {output_dict['north_east_log_ratios_std']:.3f}, south east log ratios std: {output_dict['south_east_log_ratios_std']:.3f}"
    )


    return output_dict


def _precompute_hemisphere_masses_fwfm(model, params, rng, n_steps, n_projections):
    """Precompute hemisphere probability masses for cap_conditioned_base FWFM models.

    This is extracted from _compute_nll_fwfm to avoid recomputing expensive hemisphere masses
    for every test batch when using cap_conditioned_base.

    Args:
        model: FunctionWeightedFlowModel with cap_conditioned_base=True
        params: Model parameters
        rng: JAX random key
        n_steps: Number of integration steps
        n_projections: Number of divergence projections

    Returns:
        Dict containing hemisphere probability masses, or None if not applicable
    """
    if (
        model.weighting_function == WeightingFunction.CAP_INDICATOR
        and model.cap_conditioned_base
    ):
        return compute_hemisphere_probability_masses(
            model=model,
            params=params,
            rng=rng,
            n_samples=64,
            n_steps=n_steps,
            n_projections=n_projections,
        )
    return None


def _compute_nll_fwfm(model, params, batch, n_steps, rng, n_projections, precomputed_stats=None):
    """Compute NLL for FunctionWeightedFlowModel - use 'evenest weights'."""
    batch_size = batch["point_vec"].shape[0]

    if model.weighting_function == WeightingFunction.CONSTANT:
        weighting_function_params = None
    elif model.weighting_function in [
        WeightingFunction.CAP_INDICATOR,
        WeightingFunction.SMOOTHED_CAP_INDICATOR,
    ]:
        if not model.cap_conditioned_base:
            # Use full sphere as cap
            arbitrary_center = jnp.zeros(model.domain_dim).at[0].set(1.0)
            d_max = 2.0
            weighting_function_params = (
                jnp.broadcast_to(arbitrary_center, (batch_size, model.domain_dim)),
                jnp.full((batch_size,), d_max),
            )
        else:
            # We only support hemisphere or smaller caps, so we can't do as above. Instead, use
            # northern or southern hemisphere depending on which the point fits in. The likelhood of
            # a point is its likelhood under its hemiphere's distribution times the total mass of
            # its hemisphere.

            points = batch["point_vec"]
            assert points.shape == (batch_size, model.domain_dim)

            # Choose hemisphere cap per sample
            north = jnp.zeros(model.domain_dim).at[0].set(1.0)
            south = -north
            north_mask = points[:, 0] >= 0.0
            cap_centers = jnp.where(
                north_mask[:, None],
                jnp.broadcast_to(north, (batch_size, model.domain_dim)),
                jnp.broadcast_to(south, (batch_size, model.domain_dim)),
            )
            cap_d_maxes = jnp.full((batch_size,), 1.0)
            weighting_function_params = (cap_centers, cap_d_maxes)

            # Estimate hemisphere masses and combine to get full-sphere likelihoods
            if precomputed_stats is not None:
                hemisphere_probability_masses_dict = precomputed_stats
                prob_rng = rng
            else:
                masses_rng, prob_rng = jax.random.split(rng)
                hemisphere_probability_masses_dict = compute_hemisphere_probability_masses(
                    model=model,
                    params=params,
                    rng=masses_rng,
                    n_samples=64,
                    n_steps=n_steps,
                    n_projections=n_projections,
                )

            conditional_logprobs = compute_log_probability(
                model=model,
                params=params,
                samples=points,
                weighting_function_params=weighting_function_params,
                n_steps=n_steps,
                rng=prob_rng,
                n_projections=n_projections,
            )

            hemisphere_log_masses = jnp.where(
                north_mask,
                hemisphere_probability_masses_dict["north_log_mass"],
                hemisphere_probability_masses_dict["south_log_mass"],
            )
            adjusted_logprobs = conditional_logprobs + hemisphere_log_masses
            hemisphere_probability_masses_dict = jax.device_get(
                hemisphere_probability_masses_dict
            )

            # If we don't do this then reported NLL is +inf until the network learns the cap
            # function
            min_logprob = sphere_log_inverse_surface_area(model.domain_dim) - jnp.log(
                1_000_000_000
            )
            adjusted_logprobs = jnp.maximum(adjusted_logprobs, min_logprob)
            return adjusted_logprobs
    elif model.weighting_function == WeightingFunction.VMF_DENSITY:
        raise NotImplementedError("VMF density weighting function not implemented yet.")
    else:
        raise ValueError(f"Unknown weighting function: {model.weighting_function}")

    return compute_log_probability(
        model=model,
        params=params,
        samples=batch["point_vec"],
        weighting_function_params=weighting_function_params,
        n_steps=n_steps,
        rng=rng,
        n_projections=n_projections,
    )


# Create FunctionWeightedFlowModel-specific training loop using partial application
_train_loop_for_tests = partial(
    flow_matching._train_loop_for_tests_generic,
    compute_nll_fn=_compute_nll_fwfm,
    precompute_test_stats_fn=_precompute_hemisphere_masses_fwfm,
)
