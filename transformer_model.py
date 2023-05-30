import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import]
from copy import copy
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from functools import partial
from typing import Any, Optional, Tuple
from tqdm import trange


class ImageModel(nn.Module):
    """A transformer model for images encoded to a discrete representation."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    seq_len: int

    def setup(self) -> None:
        self.in_embed = nn.Embed(num_embeddings=8192, features=self.d_model)
        self.positional_encoding = nn.Embed(
            num_embeddings=self.seq_len, features=self.d_model
        )
        self.transformer_layers = [
            TransformerLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
            )
            for _ in range(self.n_layers)
        ]
        self.logits_decoder = nn.Dense(features=8192)

    def __call__(self, image: jax.Array) -> jax.Array:
        """Run the model, returning log probabilities. Input should be padded to seq_len."""
        assert image.shape == (self.seq_len,)
        assert image.dtype == jnp.int32 or image.dtype == jnp.int64
        embeds = self.in_embed(image)
        # Have to shift the input right so causality isn't violated
        embeds = jnp.concatenate([jnp.zeros((1, self.d_model)), embeds[:-1]], axis=0)
        h: jax.Array = embeds + self.positional_encoding(jnp.arange(self.seq_len))
        mask = nn.attention.make_causal_mask(image)
        for tl in self.transformer_layers:
            h = tl(h, mask=mask)
        return self.logits_decoder(h)  # type: ignore[no-any-return]


@partial(jax.jit, static_argnums=(0,))
def sample(
    mdl: ImageModel,
    params: FrozenDict[str, Any],
    rng: jax.random.KeyArray,
    top_p: float = 0.95,
) -> jax.Array:
    """Sample a single image from the model. Returns an array of codes to be passed to the
    LDM decoder."""

    # This needs to be outside the linen module because the fori_loop combinator doesn't work
    # inside them.
    def loop_iter(
        i: int, acc: Tuple[jax.Array, jax.random.KeyArray]
    ) -> Tuple[jax.Array, jax.random.KeyArray]:
        image, rng = acc
        logits: jax.Array = mdl.apply(params, image=image)[i]  # type: ignore[assignment]
        assert logits.shape == (8192,)
        filtered_logits = _filter_top_p(logits, top_p)
        rng2, rng3 = jax.random.split(rng, 2)
        tok = jax.random.categorical(rng2, filtered_logits)
        image = image.at[i].set(tok)
        return (image, rng3)

    image, _ = jax.lax.fori_loop(  # type: ignore[no-untyped-call]
        0,
        mdl.seq_len,
        loop_iter,
        (jnp.zeros((mdl.seq_len,), dtype=jnp.int32), rng),
    )
    return image  # type: ignore[no-any-return]


def _filter_top_p(logits: jax.Array, top_p: float) -> jax.Array:
    """Filter an array of logits to include the smallest subset of possibilities that has
    proability mass at least p i.e. top p/nucleus sampling. Returns the filtered array.
    """
    probs = jax.nn.softmax(logits)
    sorted_probs, sorted_indices = (
        jnp.sort(probs)[::-1],
        jnp.argsort(probs)[::-1],
    )
    cumulative_probs = jnp.cumsum(sorted_probs)
    # collect the minimal set of possibilites with probability <= top_p
    mask = cumulative_probs <= top_p
    # Find the index of the first possibility that has cumulative probability >= top_p
    # this might be the last element we found above or might be the one after it.
    # we could do only the argmax and set the mask with a range but that's not JIT-able so we do
    # this.
    last_idx = jnp.argmax(cumulative_probs >= top_p)
    mask = mask.at[last_idx].set(True)

    # permute the mask back to the original order
    mask = mask[sorted_indices.argsort()]

    filtered_logits = jnp.where(mask, logits, -np.inf)
    return filtered_logits


def test_filter_top_p_10() -> None:
    """Test that filter_top_p is the identity function when top_p = 1.0."""
    logits = jnp.arange(10)
    filtered_logits = _filter_top_p(logits, 1.0)
    assert jnp.allclose(
        filtered_logits, logits
    ), "filter_top_p doesn't match the identity function when top_p = 1.0"


def test_filter_top_p_05() -> None:
    """Test that filter_top_p removes low-probability elements when top_p = 0.5."""
    probabilities = jnp.array([0.35, 0.35, 0.1, 0.1, 0.1])
    assert jnp.isclose(jnp.sum(probabilities), 1.0)
    logits = jnp.log(probabilities)
    filtered_logits = _filter_top_p(logits, 0.5)
    np.testing.assert_allclose(
        np.array(jax.nn.softmax(filtered_logits)), np.array([0.5, 0.5, 0, 0, 0])
    )


def test_filter_top_p_out_of_order() -> None:
    """Test that filter_top_p removes low-probability elements when inputs do not start sorted."""
    probabilities = np.repeat(1000.0, 7)
    big_indices = np.array([3, 5])
    medium_indices = np.array([2, 4])
    small_indices = np.array([0, 1, 6])
    probabilities[big_indices] = 0.25
    probabilities[medium_indices] = 0.2
    probabilities[small_indices] = 0.1 / 3.0
    np.testing.assert_allclose(np.sum(probabilities), 1.0)

    logits = jnp.log(probabilities)
    filtered_logits = _filter_top_p(logits, 0.75)
    filtered_probabilities = np.array(jax.nn.softmax(filtered_logits))

    np.testing.assert_allclose(filtered_probabilities[small_indices], 0.0)
    np.testing.assert_allclose(filtered_probabilities[medium_indices], 0.2 / 0.9)
    np.testing.assert_allclose(filtered_probabilities[big_indices], 0.25 / 0.9)


class TransformerLayer(nn.Module):
    """A single transformer layer."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]

    def setup(self) -> None:
        self.mha = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            # dropout in the attention matrix was introduced in
            # https://arxiv.org/abs/1907.11065, it's *not* the normal thing
            # from Attention is All You Need.
            dropout_rate=0,
            deterministic=False,
        )
        self.layer_norm_1 = nn.LayerNorm()
        self.linear_1 = nn.Dense(features=self.ff_dim)
        self.linear_2 = nn.Dense(features=self.d_model)
        self.layer_norm_2 = nn.LayerNorm()
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout, deterministic=False)
        else:
            self.dropout_layer = nn.Dropout(rate=0, deterministic=True)

    def __call__(self, embeds: jax.Array, mask: jax.Array) -> jax.Array:
        out_block_1 = self.layer_norm_1(self.mha(embeds, mask=mask))
        in_block_2: jax.Array = embeds + self.dropout_layer(out_block_1)
        out_block_2: jax.Array = self.layer_norm_2(
            self.dropout_layer(
                self.linear_2(nn.activation.relu(self.linear_1(in_block_2)))  # type: ignore[attr-defined]
            )
        )
        return in_block_2 + out_block_2


def loss(
    model: ImageModel,
    params: FrozenDict[str, Any],
    dropout_rng: jax.random.KeyArray,
    ex: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a single example."""
    logits: jax.Array = model.apply(params, rngs={"dropout": dropout_rng}, image=ex)  # type: ignore[assignment]
    return optax.softmax_cross_entropy(logits, jax.nn.one_hot(ex, 8192))  # type: ignore[no-any-return]


def loss_batch(
    model: ImageModel,
    params: FrozenDict[str, Any],
    dropout_rng: jax.random.KeyArray,
    batch: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a batch of examples."""
    return jnp.mean(
        jax.vmap(loss, in_axes=(None, None, 0, 0))(
            model, params, jax.random.split(dropout_rng, batch.shape[0]), batch
        )
    )


@dataclass
class ModelConfig:
    """Configuration for the transformer models."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    seq_len: int


# Parameters taken from GPT-1, except seq_len is 256 instead of 1024
gpt_1_config = ModelConfig(
    d_model=768, num_heads=12, ff_dim=3072, dropout=0.1, n_layers=12, seq_len=256
)


def train_loop_simple(
    data: jax.Array, mdl: ImageModel, iters: int
) -> Tuple[jax.Array, FrozenDict[str, Any]]:
    """Train the model repeatedly on a single batch for testing."""
    params = mdl.init(
        rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
        image=jnp.zeros((gpt_1_config.seq_len,), dtype=jnp.int32),
    )
    opt = optax.adam(learning_rate=optax.linear_onecycle_schedule(iters, 1e-3))
    opt_state = opt.init(params)
    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    def opt_step(
        params: FrozenDict[str, Any],
        opt_state: Any,
        dropout_rng: jax.random.KeyArray,
        batch: jax.Array,
    ) -> Tuple[FrozenDict[str, Any], Any, jax.random.KeyArray, jax.Array]:
        rng1, rng2 = jax.random.split(dropout_rng)
        loss, grads = loss_grad_fn(mdl, params, rng1, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params: FrozenDict[str, Any] = optax.apply_updates(params, updates)
        return new_params, opt_state, rng2, loss

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    dropout_rng = jax.random.PRNGKey(0)
    for i in range(iters):
        params, opt_state, dropout_rng, loss = opt_step(  # type:ignore[misc]
            params, opt_state, dropout_rng, data
        )
        print(f"iter {i} loss: {loss}")
    return loss, params  # type:ignore[return-value]


def test_learn_zeros() -> None:
    """Test whether the model can learn to predict all zeros."""
    mdl = ImageModel(**gpt_1_config.__dict__)
    data = jnp.zeros((16, gpt_1_config.seq_len), dtype=jnp.int32)
    loss, params = train_loop_simple(data, mdl, 10)
    assert loss < 1e-10
    cfg = copy(gpt_1_config)
    cfg.dropout = None
    mdl = ImageModel(**cfg.__dict__)
    sampled_arr = sample(mdl, params, jax.random.PRNGKey(0))
    assert jnp.all(sampled_arr == 0)


def test_learn_ranges() -> None:
    """Test whether the model can learn to predict a range of integers."""
    mdl = ImageModel(**gpt_1_config.__dict__)
    data = jnp.arange(16 * gpt_1_config.seq_len).reshape((16, gpt_1_config.seq_len))
    # It's annoying how many iterations we need to get to minimal loss on such a trivial dataset
    loss, params = train_loop_simple(data, mdl, 500)
    assert loss < 0.6
    cfg = copy(gpt_1_config)
    cfg.dropout = None
    mdl = ImageModel(**cfg.__dict__)
    sample_jv = jax.jit(jax.vmap(lambda rng: sample(mdl, params, rng, top_p=0.99)))
    print("Generating samples...")
    total_samples = 256
    samples_per_batch = 64
    assert total_samples % samples_per_batch == 0
    sample_batches = []
    rng = jax.random.PRNGKey(0)
    for _ in trange(total_samples // samples_per_batch):
        rng, rng2 = jax.random.split(rng)
        sample_batches.append(sample_jv(jax.random.split(rng2, samples_per_batch)))
    sampled_arr: jax.Array = jnp.concatenate(sample_batches, axis=0)
    print(f"Generated samples, shape: {sampled_arr.shape}")
    counts = dict([(i, 0) for i in range(16)])
    for i, s in enumerate(sampled_arr):
        assert s[0] % 256 == 0
        np.testing.assert_equal(np.array(s), np.array(s[0] + np.arange(256)))
        counts[int(s[0] // 256)] += 1
    print(f"counts: {counts}")
    assert all([c > 0 for c in counts.values()])
