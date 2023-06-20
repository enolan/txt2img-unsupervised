import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import]
from config import ModelConfig
from copy import copy
from dataclasses import dataclass
from flax import struct
from flax.core.frozen_dict import FrozenDict
from functools import partial
from typing import Any, Callable, Optional, Tuple
from tqdm import trange


class ImageModel(nn.Module):
    """A transformer model for images encoded to a discrete representation."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    seq_len: int
    use_biases: bool
    activations_dtype: jnp.dtype
    activation_function: Callable[[jax.Array], jax.Array]
    decode: bool = False

    def setup(self) -> None:
        default_kernel_init = nn.initializers.normal(
            stddev=0.02 / jnp.sqrt(self.n_layers)
        )
        self.in_embed = nn.Embed(
            num_embeddings=8192,
            features=self.d_model,
            embedding_init=default_kernel_init,
            dtype=self.activations_dtype,
        )
        self.positional_encoding = nn.Embed(
            num_embeddings=self.seq_len,
            features=self.d_model,
            embedding_init=default_kernel_init,
            dtype=self.activations_dtype,
        )
        self.transformer_layers = [
            TransformerLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                use_biases=self.use_biases,
                activations_dtype=self.activations_dtype,
                activation_function=self.activation_function,
                kernel_init=default_kernel_init,
                decode=self.decode,
            )
            for _ in range(self.n_layers)
        ]
        self.logits_decoder = nn.Dense(
            features=8192,
            kernel_init=default_kernel_init,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
        )

    def __call__(self, image: jax.Array) -> jax.Array:
        """Run the model, returning log probabilities."""
        assert image.shape == (self.seq_len,)
        assert image.dtype == jnp.int32 or image.dtype == jnp.int64
        embeds = self.in_embed(image)
        # Have to shift the input right so causality isn't violated
        embeds = jnp.concatenate([jnp.zeros((1, self.d_model)), embeds[:-1]], axis=0)
        h: jax.Array = embeds + self.positional_encoding(jnp.arange(self.seq_len))
        mask = nn.attention.make_causal_mask(image, dtype=self.activations_dtype)
        for tl in self.transformer_layers:
            h = tl(h, mask=mask)
        return self.logits_decoder(h)  # type: ignore[no-any-return]

    def decode_step(self, tok: jax.Array, idx: jax.Array) -> jax.Array:
        """Do a step of iterative decoding from the model. Returns the logits for the next token.
        See below tests for usage examples. N.B. You need to start from "token 0" which is nothing
        to get logits for output token 0. So when idx is 0, tok is ignored.
        Both inputs are scalars.
        """
        assert (
            self.decode
        ), "Can't call decode_step on a model that wasn't set up for decoding."
        assert tok.shape == ()
        assert tok.dtype == jnp.int32 or tok.dtype == jnp.int64

        # This bit is analagous to the right shift above.
        def zeros_fn(mdl: ImageModel) -> jax.Array:
            # Have to use the embeddding variable in both branches to make Flax happy
            _ = mdl.in_embed(jnp.array(0))
            return jnp.zeros((self.d_model,))

        def embed_fn(mdl: ImageModel) -> jax.Array:
            return mdl.in_embed(tok)

        embed = nn.cond(idx == 0, zeros_fn, embed_fn, self)
        assert embed.shape == (self.d_model,)

        h = embed + self.positional_encoding(idx)
        assert h.shape == (self.d_model,)
        h = h[None, :]

        for tl in self.transformer_layers:
            h = tl(h)
        return self.logits_decoder(h[0])  # type: ignore[no-any-return]


def _assert_frozen_dicts_equal(d1, d2, name) -> None:
    assert isinstance(d1, FrozenDict)
    assert isinstance(d2, FrozenDict)
    assert d1.keys() == d2.keys()
    for k in d1.keys():
        if isinstance(d1[k], FrozenDict):
            _assert_frozen_dicts_equal(d1[k], d2[k], f"{name}.{k}")
        elif isinstance(d1[k], jax.Array):
            np.testing.assert_allclose(
                np.array(d1[k]), np.array(d2[k]), atol=1e-8, rtol=0
            )
        else:
            assert False, f"unknown type {type(d1[k])} for {name}.{k}"


def _setup_test_sample() -> (
    Tuple[ImageModel, ImageModel, FrozenDict, jax.Array, jax.Array]
):
    """Shared setup code for iterative sampling tests."""
    cfg_nodec = copy(gpt_1_config)
    cfg_nodec.dropout = None
    # smaller model makes debug output easier to read
    cfg_nodec.n_layers = 2
    cfg_nodec.d_model = 64
    cfg_nodec.num_heads = 4
    mdl_nodec = ImageModel(**cfg_nodec.__dict__)
    cfg_dec = copy(cfg_nodec)
    cfg_dec.decode = True
    mdl_dec = ImageModel(**cfg_dec.__dict__)

    toks = jax.random.randint(jax.random.PRNGKey(420), (256,), 0, 8192)
    params = mdl_nodec.init(jax.random.PRNGKey(69), toks)
    # IMPORTANT: use regular __call__ here, not decode_step. The cache needs to be initialized to
    # the full seq_len size.
    params_dec = mdl_dec.init(jax.random.PRNGKey(69), toks)

    _assert_frozen_dicts_equal(params["params"], params_dec["params"], "params")
    # params_dec = params.copy({"cache": params_dec["cache"]})

    logits_all = mdl_nodec.apply(params, image=toks)

    return mdl_nodec, mdl_dec, params, params_dec["cache"], toks, logits_all


def test_sample_tok_0() -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 0."""
    mdl_nodec, mdl_dec, params, cache, toks, logits_all = _setup_test_sample()

    params = params.copy({"cache": cache})
    logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        idx=jnp.array(0),
    )

    np.testing.assert_allclose(logits_all[0], logits_0, rtol=0, atol=1e-5)


def test_sample_tok_1() -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 1."""
    mdl_nodec, mdl_dec, params, cache, toks, logits_all = _setup_test_sample()

    params = params.copy({"cache": cache})

    _logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        idx=jnp.array(0),
    )
    params = params.copy(cache)
    logits_1, _cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=toks[0],
        idx=jnp.array(1),
    )

    np.testing.assert_allclose(logits_all[1], logits_1, rtol=0, atol=1e-5)


def test_sample_tok_all() -> None:
    """Test that step-by-step decoding is equivalent to all at once for all tokens."""
    mdl_nodec, mdl_dec, params, cache, toks, logits_all = _setup_test_sample()

    decoded_logits = []
    params = params.copy({"cache": cache})

    # step 0
    logits, new_cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        idx=jnp.array(0),
    )
    decoded_logits.append(logits)
    params = params.copy(new_cache)

    # steps 1-255
    for i in range(1, 256):
        logits, new_cache = mdl_dec.apply(
            params,
            mutable=["cache"],
            method=mdl_dec.decode_step,
            tok=toks[i - 1],
            idx=jnp.array(i),
        )
        assert logits.shape == (8192,)
        decoded_logits.append(logits)
        params = params.copy(new_cache)

    decoded_logits = jnp.stack(decoded_logits, axis=0)
    assert decoded_logits.shape == (256, 8192)
    np.testing.assert_allclose(logits_all, decoded_logits, rtol=0, atol=1e-6)


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
    use_biases: bool
    activations_dtype: jnp.dtype
    activation_function: Callable[[jax.Array], jax.Array]
    kernel_init: Callable[..., jnp.ndarray]
    decode: bool

    def setup(self) -> None:
        self.mha = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            # dropout in the attention matrix was introduced in
            # https://arxiv.org/abs/1907.11065, it's *not* the normal thing
            # from Attention is All You Need.
            dropout_rate=0,
            deterministic=False,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            kernel_init=self.kernel_init,
            decode=self.decode,
        )
        self.layer_norm_1 = nn.LayerNorm(dtype=self.activations_dtype)
        self.linear_1 = nn.Dense(
            features=self.ff_dim,
            use_bias=self.use_biases,
            kernel_init=self.kernel_init,
            dtype=self.activations_dtype,
        )
        self.linear_2 = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            kernel_init=self.kernel_init,
            dtype=self.activations_dtype,
        )
        self.layer_norm_2 = nn.LayerNorm(dtype=self.activations_dtype)
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout, deterministic=False)
        else:
            self.dropout_layer = nn.Dropout(rate=0, deterministic=True)

    def __call__(
        self, embeds: jax.Array, mask: Optional[jax.Array] = None
    ) -> jax.Array:
        out_block_1 = self.layer_norm_1(self.mha(embeds, mask=mask))
        in_block_2: jax.Array = embeds + self.dropout_layer(out_block_1)
        out_block_2: jax.Array = self.layer_norm_2(
            self.dropout_layer(
                self.linear_2(self.activation_function(self.linear_1(in_block_2)))  # type: ignore[attr-defined]
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


# Parameters taken from GPT-1, except seq_len is 256 instead of 1024
gpt_1_config = ModelConfig(
    d_model=768,
    num_heads=12,
    ff_dim=3072,
    dropout=0.1,
    n_layers=12,
    seq_len=256,
    use_biases=True,
    activation_function=jax.nn.relu,
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
