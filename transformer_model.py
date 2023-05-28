import flax.linen as nn
import jax
import jax.numpy as jnp
import optax  # type: ignore[import]
from copy import copy
from dataclasses import dataclass
from flax.core.frozen_dict import FrozenDict
from typing import Any, Optional, Tuple


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

    def sample(self, top_p: float = 0.95) -> jax.Array:
        """Sample a single image from the model."""
        image = jnp.zeros((self.seq_len,), dtype=jnp.int32)
        for i in range(self.seq_len):
            logits = self(image)[i]
            probs = jax.nn.softmax(logits)
            sorted_probs, sorted_indices = (
                jnp.sort(probs)[::-1],
                jnp.argsort(probs)[::-1],
            )
            cumulative_probs = jnp.cumsum(sorted_probs)
            mask = cumulative_probs <= top_p
            filtered_probs, filtered_indices = sorted_probs[mask], sorted_indices[mask]
            if not jnp.any(mask):
                # This shouldn't happen, but if it does, just take the most probable
                # token.
                image = image.at[i].set(sorted_indices[0])
            else:
                rng = self.make_rng("sampling")
                tok = sorted_indices[jax.random.categorical(rng, filtered_probs)]
                image = image.at[i].set(tok)
        return image


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
    opt = optax.adam(learning_rate=3e-4)
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
    sample = mdl.apply(
        params, rngs={"sampling": jax.random.PRNGKey(0)}, method=ImageModel.sample
    )
    assert jnp.all(sample == 0)


def test_learn_ranges() -> None:
    """Test whether the model can learn to predict a range of integers."""
    mdl = ImageModel(**gpt_1_config.__dict__)
    data = jnp.arange(16 * gpt_1_config.seq_len).reshape((16, gpt_1_config.seq_len))
    loss, params = train_loop_simple(data, mdl, 300)
    assert loss < 0.6
    cfg = copy(gpt_1_config)
    cfg.dropout = None
    mdl = ImageModel(**cfg.__dict__)
    sample: jax.Array = mdl.apply(  # type: ignore[assignment]
        params,
        rngs={"sampling": jax.random.PRNGKey(69_420)},
        method=ImageModel.sample,
        top_p=0.95,
    )
    print(f"sample: {sample}")
    # TODO when sampling isn't hideously slow, generate enough samples to check that each
    # possibility is generated at least once
    assert any([jnp.array_equal(sample, data[i]) for i in range(16)])
