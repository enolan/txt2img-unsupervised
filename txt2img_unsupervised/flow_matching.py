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
from datasets import Dataset
from einops import rearrange, repeat
from flax import linen as nn
from flax.training import train_state
from functools import partial
from math import floor
from typing import Sequence, Tuple, Dict, Callable, Any
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from scipy import stats
from tqdm import tqdm


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
        # Default initialization for internal layers
        default_kernel_init = nn.initializers.variance_scaling(
            scale=1.0, mode="fan_in", distribution="normal"
        )

        # Initializers for our 3 conditioning inputs are specially scaled so their sum has mean 0
        # variance 1. Each component of the sum should have mean 0 and variance 1/3.
        self.domain_in_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(
                # The variance of a uniformly distributed unit vector is approximeatly 1/d, so we
                # multiply to cancel
                scale=self.domain_dim / 3.0,
                mode="fan_in",
                distribution="normal",
            ),
        )

        if self.conditioning_dim > 0:
            self.conditioning_in_proj = nn.Dense(
                features=self.d_model,
                dtype=self.activations_dtype,
                param_dtype=self.weights_dtype,
                kernel_init=nn.initializers.variance_scaling(
                    # We assume the input is unit normal
                    scale=1.0 / 3.0,
                    mode="fan_in",
                    distribution="normal",
                ),
            )
        else:
            self.conditioning_in_proj = None

        self.time_in_proj = nn.Dense(
            features=self.d_model,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=nn.initializers.variance_scaling(
                # The variance of U[0, 1] is 1/12, so me multiply by 12 and divide by 3 to get 4.
                scale=4.0,
                mode="fan_in",
                distribution="normal",
            ),
            bias_init=nn.initializers.constant(-0.5),  # Center the uniform [0,1] input
        )

        self.mlp_blocks = nn.scan(
            nn.remat(MLPBlock),
            variable_axes={"params": 0},
            variable_broadcast=False,
            split_rngs={"params": True, "dropout": True},
            length=self.n_layers,
        )(
            bottleneck_dim=self.d_model,
            expansion_factor=self.mlp_expansion_factor,
            activations_dtype=self.activations_dtype,
            weights_dtype=self.weights_dtype,
            kernel_init=default_kernel_init,
        )

        self.final_norm = nn.LayerNorm(
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
        )

        self.out_proj = nn.Dense(
            features=self.domain_dim,
            dtype=self.activations_dtype,
            param_dtype=self.weights_dtype,
            kernel_init=default_kernel_init,
        )

    def __call__(self, x, t, cond_vec):
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
        t = rearrange(t, "b -> b 1")
        t_in = self.time_in_proj(t)
        assert x_in.shape == (batch_size, self.d_model)
        assert cond_in.shape == (batch_size, self.d_model)
        assert t_in.shape == (batch_size, self.d_model)

        mlp_in = x_in + cond_in + t_in
        assert mlp_in.shape == (batch_size, self.d_model)

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
            lambda: generate_orthogonal_vector(x) * kappa_t,
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


def generate_orthogonal_vector(v):
    """Generate a unit vector orthogonal to v."""
    assert len(v.shape) == 1
    assert v.shape[0] > 0

    # Find the component with smallest absolute value
    idx = jnp.argmin(jnp.abs(v))

    # Create a basis vector in that direction
    e = jnp.zeros_like(v).at[idx].set(1.0)

    # Make it orthogonal to v
    orthogonal = e - jnp.dot(e, v) * v

    # Normalize and return
    return orthogonal / jnp.linalg.norm(orthogonal)


def compute_psi_t_spherical(x0, x1, t):
    """
    Compute the flow map for the spherical OT field.

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

    # Compute the angle between the points
    cos_angle = jnp.clip(jnp.dot(x0, x1), -1.0, 1.0)
    angle = jnp.arccos(cos_angle)

    def handle_close_or_opposite():
        return jax.lax.cond(
            angle < 1e-8,
            lambda: x0,
            lambda: generate_orthogonal_vector(x0),
        )

    return jax.lax.cond(
        jnp.logical_or(angle < 1e-8, angle > jnp.pi - 1e-8),
        handle_close_or_opposite,
        lambda: slerp(x0, x1, t),
    )


def conditional_flow_matching_loss(
    model: VectorField, params, x0, x1, t, conds, kappa_1
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
        conds: Conditioning vectors (batched) [batch_size, cond_dim]
        kappa_1: Concentration parameter of the final vMF distribution

    Returns:
        CFM loss value (scalar)
    """
    assert len(x0.shape) == 2
    batch_size = x0.shape[0]
    assert x0.shape == x1.shape
    assert x0.shape[1] == model.domain_dim
    assert t.shape == (batch_size,)
    assert conds.shape == (batch_size, model.conditioning_dim)

    psi_t = jax.vmap(compute_psi_t_spherical)(x0, x1, t)
    assert psi_t.shape == x0.shape

    # Compute target vector field (ground truth OT field)
    target_field = jax.vmap(spherical_ot_field, in_axes=(0, 0, 0, None))(
        psi_t, x1, t, kappa_1
    )
    assert target_field.shape == x0.shape

    # Compute predicted vector field from our model
    predicted_field = model.apply(params, psi_t, t, conds)
    assert predicted_field.shape == x0.shape

    # Compute MSE loss
    loss = jnp.mean(jnp.sum((predicted_field - target_field) ** 2, axis=1))

    return loss


def create_train_state(rng, model, learning_rate_or_schedule):
    """Create initial training state."""
    dummy_x = jnp.ones((1, model.domain_dim))
    dummy_t = jnp.full((1,), 0.5)
    dummy_cond = jnp.ones((1, model.conditioning_dim))
    params = model.init(rng, dummy_x, dummy_t, dummy_cond)
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
def compute_batch_loss(model, params, batch, rng, kappa_1):
    """
    Compute the loss for a batch of data.

    Args:
        model: The vector field model
        params: Model parameters
        batch: Batch of data containing "point_vec" and "cond_vec"
        rng: JAX random key

    Returns:
        loss: The computed loss value
    """
    x1_batch = batch["point_vec"]
    cond_batch = batch["cond_vec"]
    batch_size = x1_batch.shape[0]
    assert x1_batch.shape == (batch_size, model.domain_dim)
    assert cond_batch.shape == (batch_size, model.conditioning_dim)

    rng, noise_rng, time_rng = jax.random.split(rng, 3)
    x0_batch = sample_sphere(noise_rng, batch_size, model.domain_dim)
    t = jax.random.uniform(time_rng, (batch_size,))

    # Compute the loss
    loss = conditional_flow_matching_loss(
        model, params, x0_batch, x1_batch, t, cond_batch, kappa_1
    )

    return loss


@partial(
    jax.jit, static_argnames=("model", "kappa_1"), donate_argnames=("state", "rng")
)
def train_step(model, kappa_1, state, batch, rng):
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
        loss = compute_batch_loss(model, params, batch, rng, kappa_1)
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
                    f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Grad Norm: {grad_norm:.6f}"
                )

            # Shuffle dataset for next epoch
            dataset.shuffle(generator=np_rng)

    if test_dataset is not None:
        return state, train_loss, final_test_loss
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
    points = repeat(jnp.array([1, 0, 0]), "v -> b v", b=batch_size * 10)
    dset = Dataset.from_dict({"point_vec": points}).with_format("np")
    state, loss, _ = _train_loop_for_tests(
        model, dset, batch_size, 1e-3, 10, kappa_1=10.0, test_dataset=dset
    )
    print(f"Final loss: {loss:.6f}")

    samples = generate_samples(
        model, state, jax.random.PRNGKey(0), 20, jnp.zeros((20, 0)), 1000, "rk4"
    )
    for sample in samples:
        cos_sim = jnp.dot(sample, points[0])
        print(
            f"Sample: [{sample[0]:9.6f}, {sample[1]:9.6f}, {sample[2]:9.6f}]  Cosine similarity: {cos_sim:9.6f}"
        )


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

    dset = Dataset.from_dict(
        {"point_vec": points, "cond_vec": np.zeros((n_samples, 0))}
    ).with_format("np")

    test_n_samples = 2048
    test_points = vmf.rvs(test_n_samples)
    test_dset = Dataset.from_dict({"point_vec": test_points}).with_format("np")

    state, train_loss, test_loss = _train_loop_for_tests(
        model, dset, batch_size, 1e-3, 30, test_dset, kappa_1=20.0
    )
    print(f"Final train loss: {train_loss:.6f}")
    print(f"Final test loss: {test_loss:.6f}")

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
    nll = -np.mean(log_probs)

    print(f"Negative log-likelihood: {nll:.6f}")

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

    # Assert that the NLL is reasonable (should be close to the theoretical value)
    # For VMF(mu, kappa) in 3D, the theoretical NLL is approximately:
    # log(4*pi*sinh(kappa)/kappa) - kappa
    theoretical_nll = np.log(4 * np.pi * np.sinh(kappa) / kappa) - kappa
    print(f"Theoretical NLL: {theoretical_nll:.6f}")

    # Allow some tolerance due to sampling variability
    assert (
        abs(nll - theoretical_nll) < 1.0
    ), f"NLL {nll} too far from theoretical {theoretical_nll}"


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

    # Train the model (smaller batch size)
    batch_size = 256
    state, train_loss, test_loss = _train_loop_for_tests(
        model, train_dataset, batch_size, 1e-2, 20, test_dataset, kappa_1=10.0
    )
    print(f"Final loss - Train: {train_loss:.4f}, Test: {test_loss:.4f}")

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

    # Check if conditioning affects samples by comparing with same seed
    seed_check = jax.random.PRNGKey(456)
    check_samples_0 = generate_samples(
        model, state, seed_check, 50, jnp.zeros((50, 1)), 500, "rk4"
    )
    check_samples_1 = generate_samples(
        model, state, seed_check, 50, jnp.ones((50, 1)), 500, "rk4"
    )
    mean_dist = jnp.mean(
        jnp.sqrt(jnp.sum((check_samples_0 - check_samples_1) ** 2, axis=1))
    )

    assert mean_dist > 0.1, "Conditioning has no effect on samples"

    # Calculate average cosine similarities
    cos_sim_0_dir1 = np.mean([np.dot(s, mean_direction1) for s in samples_0])
    cos_sim_0_dir2 = np.mean([np.dot(s, mean_direction2) for s in samples_0])
    cos_sim_1_dir1 = np.mean([np.dot(s, mean_direction1) for s in samples_1])
    cos_sim_1_dir2 = np.mean([np.dot(s, mean_direction2) for s in samples_1])

    # Validate correct conditioning behavior
    print(
        f"Conditioning=0 - Avg similarity with dir1: {cos_sim_0_dir1:.4f}, dir2: {cos_sim_0_dir2:.4f}"
    )
    print(
        f"Conditioning=1 - Avg similarity with dir1: {cos_sim_1_dir1:.4f}, dir2: {cos_sim_1_dir2:.4f}"
    )

    # Verify that samples align with their respective target distributions
    # Samples with cond=0 should align better with direction1
    assert (
        cos_sim_0_dir1 > cos_sim_0_dir2
    ), "Samples with cond=0 don't align with direction1"
    # Samples with cond=1 should align better with direction2
    assert (
        cos_sim_1_dir2 > cos_sim_1_dir1
    ), "Samples with cond=1 don't align with direction2"

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

    distribution_nll = np.log(4 * np.pi * np.sinh(kappa) / kappa) - kappa
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


@partial(jax.jit, static_argnames=("model",))
def spherical_rk4_step_with_model(
    model, state, x, t, dt, cond_vecs, negate_vector_field=False
):
    vector_field_multiplier = jax.lax.cond(
        negate_vector_field, lambda: -1.0, lambda: 1.0
    )
    vector_field_fn = (
        lambda x, t: vector_field_multiplier
        * _compute_vector_field_for_sampling(model, state, x, t, cond_vecs)
    )
    return spherical_rk4_step(vector_field_fn, x, t, dt)


@partial(jax.jit, static_argnames=("model",))
def _compute_vector_field_for_sampling(model, state, x, t, cond_vecs):
    return model.apply(state.params, x, jnp.full((x.shape[0],), t), cond_vecs)


def generate_samples(
    model, state, rng, batch_size, cond_vecs, n_steps=100, method="rk4"
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
    assert cond_vecs.shape == (batch_size, model.conditioning_dim)
    # Sample initial points uniformly from the sphere
    x0 = sample_sphere(rng, batch_size, model.domain_dim)

    vector_field_fn = lambda x, t: _compute_vector_field_for_sampling(
        model, state, x, t, cond_vecs
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
                model, state, x, t, dt, cond_vecs, negate_vector_field=False
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

    # Normalize samples to ensure they're on the unit sphere
    samples = samples / jnp.linalg.norm(samples, axis=1, keepdims=True)

    # Initialize RNG if not provided
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Find the path from the samples to their origin points in x_0 by backward integration
    def vector_field_fn(x, t):
        # return jnp.zeros_like(x)
        return -_compute_vector_field_for_sampling(model, state, x, t, cond_vecs)

    ts = np.linspace(1, 0, n_steps)

    xs = np.zeros((n_steps, batch_size, model.domain_dim))
    xs[0] = samples
    for i in range(n_steps - 1):
        t = ts[i]
        xs[i + 1] = spherical_rk4_step_with_model(
            model, state, xs[i], t, -1.0 / n_steps, cond_vecs, negate_vector_field=True
        )

    # Compute divergence integrals along paths
    div_sum = np.zeros(batch_size)

    for i in range(n_steps - 1):
        x_t = xs[i]
        t = ts[i]

        step_rng, rng = jax.random.split(rng)
        div_t = exact_divergence(
            model, state, x_t, t, cond_vecs, step_rng, n_projections
        )
        assert div_t.shape == (batch_size,)
        div_sum += jax.device_get(div_t) * (1.0 / n_steps)

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
        (hutchinson_estimator, 10, "zero_divergence"),
        (hutchinson_estimator, 10, "variable_divergence"),
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
