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
    image_tokens: int
    clip_conditioning: bool
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
        self.clip_proj = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
        )
        self.positional_encoding = nn.Embed(
            num_embeddings=self.seq_len(),
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

    def seq_len(self) -> int:
        # How many tokens are fed to the model at once and equally how many are output. This
        # function will actually do something once we have more than one conditioning token.
        return self.image_tokens

    def __call__(self, image: jax.Array, clip_embedding: jax.Array) -> jax.Array:
        """Run the model, returning log probabilities of the image tokens. No probabilities are computed
        for any CLIP conditioning tokens."""
        assert image.shape == (
            self.image_tokens,
        ), f"Expected image shape {(self.image_tokens,)}, got {image.shape}"
        assert image.dtype == jnp.int32 or image.dtype == jnp.int64

        embeds = self.in_embed(image)

        # We either insert a zero token at the start, or a CLIP token at the start. In either case
        # the nth output predicts the nth token, conditional on all *previous* tokens. In future
        # when there's more than one conditioning token, we'll have to do something more
        # complicated.
        if self.clip_conditioning:
            assert clip_embedding.shape == (768,)
            clip_tok = self.clip_proj(clip_embedding)
            toks = jnp.concatenate([clip_tok[None, :], embeds[:-1]], axis=0)
        else:
            assert clip_embedding.shape == (0,)
            toks = jnp.concatenate([jnp.zeros((1, self.d_model)), embeds[:-1]], axis=0)

        h: jax.Array = toks + self.positional_encoding(jnp.arange(self.seq_len()))
        mask = jnp.tril(jnp.ones((self.num_heads, self.seq_len(), self.seq_len())))
        for tl in self.transformer_layers:
            h = tl(h, mask=mask)

        return self.logits_decoder(h)  # type: ignore[no-any-return]

    def decode_step(
        self, tok: jax.Array, idx: jax.Array, clip_embedding: jax.Array
    ) -> jax.Array:
        """Do a step of iterative decoding from the model. Returns the logits for the next token.
        See below tests for usage examples. N.B. You need to start from "token 0" which is either
        nothing or the CLIP embedding to get logits for output token 0. So when idx is 0, tok is
        ignored.
        tok and idx are scalars, clip_embed is a 768-d vector or an empty array depending on
        whether the model is conditioned on CLIP or not.
        """
        assert (
            self.decode
        ), "Can't call decode_step on a model that wasn't set up for decoding."
        assert tok.shape == ()
        assert tok.dtype == jnp.int32 or tok.dtype == jnp.int64
        if self.clip_conditioning:
            assert clip_embedding.shape == (768,)
        else:
            assert clip_embedding.shape == (0,)

        # This bit is analagous to the right shift above.
        def init_tok_fn(mdl: ImageModel) -> jax.Array:
            # Have to use the embed variable in both branches to make Flax happy
            _ = mdl.in_embed(jnp.array(0))
            if self.clip_conditioning:
                return mdl.clip_proj(clip_embedding)
            else:
                return jnp.zeros((self.d_model,), dtype=self.activations_dtype)

        def embed_fn(mdl: ImageModel) -> jax.Array:
            return mdl.in_embed(tok)

        embed = nn.cond(idx == 0, init_tok_fn, embed_fn, self)
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


def _setup_test_sample(
    clip_conditioning: bool = False,
) -> Tuple[ImageModel, ImageModel, FrozenDict, jax.Array, jax.Array]:
    """Shared setup code for iterative sampling tests."""
    cfg_nodec = copy(gpt_1_config)
    cfg_nodec.dropout = None
    # smaller model makes debug output easier to read
    cfg_nodec.n_layers = 2
    cfg_nodec.d_model = 64
    cfg_nodec.num_heads = 4
    if clip_conditioning:
        cfg_nodec.clip_conditioning = True
    mdl_nodec = ImageModel(**cfg_nodec.__dict__)
    cfg_dec = copy(cfg_nodec)
    cfg_dec.decode = True
    mdl_dec = ImageModel(**cfg_dec.__dict__)

    toks = jax.random.randint(jax.random.PRNGKey(420), (256,), 0, 8192)
    if clip_conditioning:
        clip_embedding = jax.random.normal(jax.random.PRNGKey(1337), (768,))
        clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)
        # print(f"clip_embedding norm: {jnp.linalg.norm(clip_embedding)}")
    else:
        clip_embedding = jnp.zeros((0,), dtype=jnp.float32)
    params = mdl_nodec.init(jax.random.PRNGKey(69), toks, clip_embedding=clip_embedding)
    # IMPORTANT: use regular __call__ here, not decode_step. The cache needs to be initialized to
    # the full seq_len size.
    params_dec = mdl_dec.init(
        jax.random.PRNGKey(69), toks, clip_embedding=clip_embedding
    )

    _assert_frozen_dicts_equal(params["params"], params_dec["params"], "params")

    logits_all = mdl_nodec.apply(params, image=toks, clip_embedding=clip_embedding)

    return (
        mdl_nodec,
        mdl_dec,
        params,
        params_dec["cache"],
        toks,
        clip_embedding,
        logits_all,
    )


def _test_sample_tok_0(clip_conditioning: bool) -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 0."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        logits_all,
    ) = _setup_test_sample(clip_conditioning)

    params = params.copy({"cache": cache})
    logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        clip_embedding=clip_embedding,
        idx=jnp.array(0),
    )

    np.testing.assert_allclose(logits_all[0], logits_0, rtol=0, atol=1e-5)


def test_sample_tok_0_no_clip() -> None:
    _test_sample_tok_0(False)


def test_sample_tok_0_clip() -> None:
    _test_sample_tok_0(True)


def _test_sample_tok_1(clip_conditioning: bool) -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 1."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        logits_all,
    ) = _setup_test_sample(clip_conditioning)

    params = params.copy({"cache": cache})

    _logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        clip_embedding=clip_embedding,
        idx=jnp.array(0),
    )
    params = params.copy(cache)
    logits_1, _cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=toks[0],
        clip_embedding=clip_embedding,
        idx=jnp.array(1),
    )

    np.testing.assert_allclose(logits_all[1], logits_1, rtol=0, atol=1e-5)


def test_sample_tok_1_no_clip() -> None:
    _test_sample_tok_1(False)


def test_sample_tok_1_clip() -> None:
    _test_sample_tok_1(True)


def _test_sample_tok_all(clip_conditioning: bool) -> None:
    """Test that step-by-step decoding is equivalent to all at once for all tokens."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        logits_all,
    ) = _setup_test_sample(clip_conditioning)

    decoded_logits = []
    params = params.copy({"cache": cache})

    # step 0
    logits, new_cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=jnp.array(0),
        idx=jnp.array(0),
        clip_embedding=clip_embedding,
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
            clip_embedding=clip_embedding,
        )
        assert logits.shape == (8192,)
        decoded_logits.append(logits)
        params = params.copy(new_cache)

    decoded_logits = jnp.stack(decoded_logits, axis=0)
    assert decoded_logits.shape == (256, 8192)
    np.testing.assert_allclose(logits_all, decoded_logits, rtol=0, atol=1e-6)


def test_sample_tok_all_no_clip() -> None:
    _test_sample_tok_all(False)


def test_sample_tok_all_clip() -> None:
    _test_sample_tok_all(True)


def test_clip_does_anything() -> None:
    """Test that changing the CLIP embedding changes the logits."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        logits_all,
    ) = _setup_test_sample(True)

    clip_embedding = jnp.zeros_like(clip_embedding)
    logits_all_zero = mdl_nodec.apply(params, image=toks, clip_embedding=clip_embedding)

    assert not jnp.allclose(logits_all, logits_all_zero, rtol=0, atol=1e-3)


@partial(jax.jit, static_argnums=(0,))
def sample(
    mdl: ImageModel,
    params: FrozenDict[str, Any],
    clip_embedding: jax.Array,
    rng: jax.random.KeyArray,
    top_p: float = 0.95,
) -> jax.Array:
    """Sample a single image from the model. Returns an array of codes to be passed to the
    LDM decoder."""
    if mdl.clip_conditioning:
        assert clip_embedding.shape == (768,)
    else:
        assert clip_embedding.shape == (0,)

    # This needs to be outside the linen module because the fori_loop combinator doesn't work
    # inside them.
    def loop_iter(
        i: int, acc: Tuple[jax.Array, jax.random.KeyArray]
    ) -> Tuple[jax.Array, jax.random.KeyArray, FrozenDict[str, Any]]:
        image_toks, rng, params = acc
        logits, new_cache = mdl.apply(
            params,
            mutable=["cache"],
            method=mdl.decode_step,
            tok=image_toks[i - 1],
            idx=i,
            clip_embedding=clip_embedding,
        )
        assert logits.shape == (8192,)
        params = params.copy(new_cache)
        filtered_logits = _filter_top_p(logits, top_p)
        rng_sample, rng_loop = jax.random.split(rng, 2)
        tok = jax.random.categorical(rng_sample, filtered_logits)
        image_toks = image_toks.at[i].set(tok)
        return (image_toks, rng_loop, params)

    rng0, rng_loop = jax.random.split(rng, 2)
    logits_0, cache = mdl.apply(
        params,
        mutable=["cache"],
        method=mdl.decode_step,
        tok=jnp.array(0),
        idx=jnp.array(0),
        clip_embedding=clip_embedding,
    )
    assert logits_0.shape == (8192,)
    filtered_logits_0 = _filter_top_p(logits_0, top_p)
    tok_0 = jax.random.categorical(rng0, filtered_logits_0)

    params = params.copy(cache)

    image_toks = jnp.zeros((mdl.image_tokens,), dtype=jnp.int32).at[0].set(tok_0)
    image_toks, _, _ = jax.lax.fori_loop(  # type: ignore[no-untyped-call]
        1,
        mdl.image_tokens,
        loop_iter,
        (image_toks, rng_loop, params),
    )
    return image_toks  # type: ignore[no-any-return]


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
    ex_img: jax.Array,
    ex_clip: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a single example."""
    assert ex_img.shape == (model.image_tokens,)
    if model.clip_conditioning:
        assert ex_clip.shape == (768,)
    else:
        assert ex_clip.shape == (0,)
    logits: jax.Array = model.apply(params, rngs={"dropout": dropout_rng}, image=ex_img, clip_embedding=ex_clip)  # type: ignore[assignment]
    return optax.softmax_cross_entropy(logits, jax.nn.one_hot(ex_img, 8192))  # type: ignore[no-any-return]


def loss_batch(
    model: ImageModel,
    params: FrozenDict[str, Any],
    dropout_rng: jax.random.KeyArray,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a batch of examples."""
    assert batch_imgs.shape[0] == batch_clips.shape[0]
    return jnp.mean(
        jax.vmap(loss, in_axes=(None, None, 0, 0, 0))(
            model,
            params,
            jax.random.split(dropout_rng, batch_imgs.shape[0]),
            batch_imgs,
            batch_clips,
        )
    )


# Parameters taken from GPT-1, except seq_len is 256 instead of 1024
gpt_1_config = ModelConfig(
    d_model=768,
    num_heads=12,
    ff_dim=3072,
    dropout=0.1,
    n_layers=12,
    image_tokens=256,
    use_biases=True,
    activation_function=jax.nn.relu,
    clip_conditioning=False,
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
