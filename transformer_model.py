import cone_sampling
import flax.core
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax  # type: ignore[import]
from config import ModelConfig
from copy import copy
from dataclasses import dataclass
from datetime import datetime
from einops import rearrange, repeat
from flash_attention_jax import causal_flash_attention
from flax import struct
from functools import partial
from typing import Any, Callable, Optional, Tuple
from tqdm import tqdm, trange
from triangle_schedule import triangle_schedule


class ImageModel(nn.Module):
    """A transformer model for images encoded to a discrete representation."""

    d_model: int
    num_heads: int
    ff_dim: int
    dropout: Optional[float]
    n_layers: int
    image_tokens: int
    clip_conditioning: bool
    clip_cones: bool
    clip_cone_count: Optional[int]
    use_biases: bool
    activations_dtype: jnp.dtype
    activation_function: Callable[[jax.Array], jax.Array]
    decode: bool = False
    flash_attention: bool = True

    def setup(self) -> None:
        default_stddev = 0.02 / jnp.sqrt(self.n_layers)
        default_kernel_init = nn.initializers.normal(stddev=default_stddev)
        self.in_embed = nn.Embed(
            num_embeddings=8192,
            features=self.d_model,
            embedding_init=default_kernel_init,
            dtype=self.activations_dtype,
        )
        # A note on how CLIP conditioning works:
        # There are three modes:
        # 1) No conditioning. We prepend a zero token to the input sequence.
        # 2) Conditioning on one CLIP embedding. We project the CLIP embedding into d_model
        # and prepend it to the input sequence. This works, and the model learns to produce images
        # that look like an image with that CLIP embedding.
        # 3) Conditioning on multiple cones, corresponding to areas of CLIP space centered on
        # (but not necessarily containing) CLIP embedings. Each cone is a CLIP embedding along with
        # lower and upper bound cosine similarity the image's embedding should have to that
        # embedding. We project the CLIP embedding and the lower and upper bounds into d_model and
        # sum them, producing one conditioning token per cone, which we then prepend to the input.
        # So the image's embedding must be in the intersection of all the cones. I'm trying this
        # for a few reasons: firstly, because an ideal model conditioning on a single exact
        # embedding would learn to produce images with that exact embedding, when in reality
        # image-text prompt cosine similarities are usually between like 0.15 and 0.4 and
        # image-very similiar image cosine similarities are around 0.8. Secondly, because the
        # intersections may allow you to do things with prompting that are impossible with a single
        # prompt. Other models do allow multiple prompts but AFAICT they're not doing
        # *intersections* they're just taking an average. Thirdly, this gives us something similar
        # to classifier free guidance - prompting with bigger cones is like lower cfg scale and
        # will hopefully lead to more realistic - though less prompt-aligned - images.
        # N.B. the more cones we condition on the more the model learns about the CLIP embeddings
        # of the examples at train time. One cone mayb be a very large area, but more allows it to
        # effectively "triangulate".

        # At training time we take the CLIP embedding and generate n random cones that contain it,
        # using those are our conditioning cones.

        # 1 & 2 work, we'll find out whether 3 does soon, hopefully.

        assert (
            self.clip_conditioning or not self.clip_cones
        ), "Can't use clip_cones without clip_conditioning"

        if self.clip_cones:
            assert self.clip_cone_count is not None, "clip_cone_count must be set"
            assert self.clip_cone_count > 0, "clip_cone_count must be positive"

        # The initializers for CLIP conditioning are chosen such that the projected clip emedding
        # has the same distribution as the token embeddings and the projected cosine similarities
        # have the same average magnitude as the token embeddings, assuming the cos sims are drawn
        # from U[-1, 1].
        self.clip_proj = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            kernel_init=default_kernel_init,
        )
        cos_sim_kernel_init = nn.initializers.normal(stddev=2 * default_stddev)
        self.cos_sim_lower_proj = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            kernel_init=cos_sim_kernel_init,
        )
        self.cos_sim_upper_proj = nn.Dense(
            features=self.d_model,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
            kernel_init=cos_sim_kernel_init,
        )
        self.positional_encoding = nn.Embed(
            num_embeddings=self.seq_len(),
            features=self.d_model,
            embedding_init=default_kernel_init,
            dtype=self.activations_dtype,
        )
        # it'd potentially be better to use nn.remat_scan here, but it makes inference massively
        # slower for some reason. Even though checkpointing should only affect gradient computation.
        # Might have to do with the fact that remat_scan creates a scan-of-scans? Could cause bad
        # optimization in JAX or XLA.
        self.transformer_layers = nn.scan(
            nn.remat(TransformerLayer),
            variable_axes={"params": 0, "cache": 0},
            variable_broadcast=False,
            split_rngs={"params": True, "dropout": True},
            length=self.n_layers,
        )(
            d_model=self.d_model,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            dropout=self.dropout,
            use_biases=self.use_biases,
            activations_dtype=self.activations_dtype,
            activation_function=self.activation_function,
            kernel_init=default_kernel_init,
            decode=self.decode,
            flash_attention=self.flash_attention,
        )

        self.logits_decoder = nn.Dense(
            features=8192,
            kernel_init=default_kernel_init,
            use_bias=self.use_biases,
            dtype=self.activations_dtype,
        )

        tokens_res = int(self.image_tokens**0.5)
        assert tokens_res * tokens_res == self.image_tokens

    def seq_len(self) -> int:
        """How many tokens are in the sequence being modeled."""
        return self.image_tokens + self.prepended_tokens() - 1

    def prepended_tokens(self) -> int:
        """How many tokens are prepended to the image tokens."""
        # We always prepend at least one, becuase the first image token must be conditioned on
        # *something* even when that's a constant.
        if self.clip_conditioning:
            if self.clip_cones:
                return self.clip_cone_count
            else:
                return 1
        else:
            return 1

    def gen_conditioning_tokens(
        self,
        clip_embedding: jax.Array,
        cos_sim_lower: jax.Array,
        cos_sim_upper: jax.Array,
    ) -> jax.Array:
        """Generate the conditioning tokens that should be prepended to the image tokens. Returns a
        (self.prepended_tokens(), self.d_model) shaped array."""
        if not self.clip_conditioning:
            assert (
                clip_embedding.shape
                == cos_sim_lower.shape
                == cos_sim_upper.shape
                == (0,)
            )
            res = jnp.zeros((1, self.d_model), dtype=self.activations_dtype)
        else:
            if not self.clip_cones:
                assert clip_embedding.shape == (768,)
                assert cos_sim_lower.shape == cos_sim_upper.shape == (0,)  # both empty
                res = self.clip_proj(clip_embedding)[None, :]
            else:
                assert clip_embedding.shape == (self.clip_cone_count, 768)
                assert (
                    cos_sim_lower.shape
                    == cos_sim_upper.shape
                    == (self.clip_cone_count,)
                )
                res = (
                    self.clip_proj(clip_embedding)
                    + self.cos_sim_lower_proj(cos_sim_lower)
                    + self.cos_sim_upper_proj(cos_sim_upper)
                ) / 3
        assert res.shape == (self.prepended_tokens(), self.d_model)
        return res

    def output_shape_tokens(self) -> int:
        """What (2-D) shape of tokens is output by the model."""
        res = int(self.image_tokens**0.5)
        return (res, res)

    def __call__(
        self, image: jax.Array, clip_embedding: jax.Array, cos_sim_lower, cos_sim_upper
    ) -> jax.Array:
        """Run the model, returning log probabilities of the image tokens. No probabilities are computed
        for any CLIP conditioning tokens."""
        assert image.shape == (
            self.image_tokens,
        ), f"Expected image shape {(self.image_tokens,)}, got {image.shape}"
        assert image.dtype == jnp.int32 or image.dtype == jnp.int64

        embeds = self.in_embed(image)

        cond_tokens = self.gen_conditioning_tokens(
            clip_embedding, cos_sim_lower, cos_sim_upper
        )
        toks = jnp.concatenate([cond_tokens, embeds[:-1]], axis=0)
        assert toks.shape == (self.seq_len(), self.d_model)

        h: jax.Array = toks + self.positional_encoding(jnp.arange(self.seq_len()))
        h, _ = self.transformer_layers(h, None)
        h = h[self.prepended_tokens() - 1 :]
        h = self.logits_decoder(h)
        assert h.shape == (self.image_tokens, 8192)

        return h

    def decode_init(
        self,
        clip_embedding: jax.Array,
        cos_sim_lower: jax.Array,
        cos_sim_upper: jax.Array,
    ):
        """Initialize the cache for decoding by computing and feeding the conditioning tokens. Returns
        the logits for the first image token. The cache should be ready for use with decode_step
        when this is done."""
        assert self.decode
        assert not self.flash_attention, "Flash attention doesn't work with decoding."

        cond_tokens = self.gen_conditioning_tokens(
            clip_embedding, cos_sim_lower, cos_sim_upper
        )

        h = cond_tokens + self.positional_encoding(jnp.arange(self.prepended_tokens()))
        assert h.shape == (self.prepended_tokens(), self.d_model)

        # TODO vectorize, don't loop
        for tok in h:
            h, _ = self.transformer_layers(tok[None, :], None)
            last_tok = h[0]
        assert last_tok.shape == (self.d_model,)

        logits_out = self.logits_decoder(last_tok)
        assert logits_out.shape == (8192,)
        return logits_out

    def decode_step(self, tok: jax.Array, idx: jax.Array) -> jax.Array:
        """Do a step of iterative decoding from the model. Returns the logits for the next token.
        See below tests for usage examples.
        """
        assert (
            self.decode
        ), "Can't call decode_step on a model that wasn't set up for decoding."
        assert not self.flash_attention, "Flash attention doesn't work with decoding."
        assert tok.shape == ()
        assert tok.dtype == jnp.int32 or tok.dtype == jnp.int64
        assert idx.shape == ()

        embed = self.in_embed(tok)
        assert embed.shape == (self.d_model,)

        h = embed + self.positional_encoding(idx + self.prepended_tokens())
        assert h.shape == (self.d_model,)
        h = h[None, :]

        h, _ = self.transformer_layers(h, None)
        return self.logits_decoder(h[0])  # type: ignore[no-any-return]


def _assert_dicts_equal(d1, d2, name) -> None:
    assert isinstance(d1, dict)
    assert isinstance(d2, dict)
    assert d1.keys() == d2.keys()
    for k in d1.keys():
        if isinstance(d1[k], dict):
            _assert_dicts_equal(d1[k], d2[k], f"{name}.{k}")
        elif isinstance(d1[k], jax.Array):
            np.testing.assert_allclose(
                np.array(d1[k]), np.array(d2[k]), atol=1e-8, rtol=0
            )
        else:
            assert False, f"unknown type {type(d1[k])} for {name}.{k}"


def _setup_test_sample(
    clip_conditioning: bool = False,
    clip_cones: bool = False,
    clip_cone_count: Optional[int] = None,
    image_tokens: int = 256,
) -> Tuple[ImageModel, ImageModel, dict, jax.Array, jax.Array]:
    """Shared setup code for iterative sampling tests."""
    cfg_nodec = copy(gpt_1_config)
    cfg_nodec.dropout = None
    cfg_nodec.image_tokens = image_tokens
    # smaller model makes debug output easier to read
    cfg_nodec.n_layers = 2
    cfg_nodec.d_model = 64
    cfg_nodec.num_heads = 4
    if clip_conditioning:
        cfg_nodec.clip_conditioning = True
    if clip_cones:
        cfg_nodec.clip_cones = True
        if clip_cone_count is None:
            clip_cone_count = 2
        cfg_nodec.clip_cone_count = clip_cone_count
    mdl_nodec = ImageModel(**cfg_nodec.__dict__)
    mdl_dec = mdl_nodec.clone(decode=True, flash_attention=False)

    img_toks = jax.random.randint(jax.random.PRNGKey(420), (image_tokens,), 0, 8192)
    if clip_conditioning:
        if clip_cones:
            clip_embedding = jax.random.normal(
                jax.random.PRNGKey(1337), (clip_cone_count, 768)
            )
            clip_embedding = clip_embedding / jnp.linalg.norm(
                clip_embedding, axis=-1, keepdims=True
            )
            cos_sim_lower = jnp.full(clip_cone_count, 0.1)
            cos_sim_upper = jnp.full(clip_cone_count, 0.5)
        else:
            clip_embedding = jax.random.normal(jax.random.PRNGKey(1337), (768,))
            clip_embedding = clip_embedding / jnp.linalg.norm(clip_embedding)
            cos_sim_lower = cos_sim_upper = jnp.array([])
    else:
        clip_embedding = cos_sim_lower = cos_sim_upper = jnp.array([])

    params = mdl_nodec.init(
        jax.random.PRNGKey(69),
        img_toks,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    # IMPORTANT: use regular __call__ here, not decode_step. The cache needs to be initialized to
    # the full seq_len size.
    params_dec = mdl_dec.init(
        jax.random.PRNGKey(69),
        img_toks,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_lower,
    )

    _assert_dicts_equal(params["params"], params_dec["params"], "params")

    logits_all = mdl_nodec.apply(
        params,
        image=img_toks,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )

    return (
        mdl_nodec,
        mdl_dec,
        params,
        params_dec["cache"],
        img_toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    )


def _test_sample_tok_0(
    clip_conditioning: bool, clip_cones: bool, clip_cone_count: Optional[int] = None
) -> None:
    """Test that step-by-step decoding is equivalent to all at once for image token 0."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_cones, clip_cone_count)

    params = flax.core.copy(params, {"cache": cache})
    logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    assert logits_0.shape == (8192,)

    np.testing.assert_allclose(logits_all[0], logits_0, rtol=0, atol=1e-5)


def test_sample_tok_0_no_clip() -> None:
    _test_sample_tok_0(False, False)


def test_sample_tok_0_clip() -> None:
    _test_sample_tok_0(True, False)


def test_sample_tok_0_clip_cones_1() -> None:
    _test_sample_tok_0(True, True, 1)


def test_sample_tok_0_clip_cones_2() -> None:
    _test_sample_tok_0(True, True, 2)


def _test_sample_tok_1(clip_conditioning: bool, clip_cones: bool) -> None:
    """Test that step-by-step decoding is equivalent to all at once for token 1."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_cones)

    params = flax.core.copy(params, {"cache": cache})

    _logits_0, cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    params = flax.core.copy(params, cache)
    logits_1, _cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_step,
        tok=toks[0],
        idx=jnp.array(0),
    )

    np.testing.assert_allclose(logits_all[1], logits_1, rtol=0, atol=1e-5)


def test_sample_tok_1_no_clip() -> None:
    _test_sample_tok_1(False, False)


def test_sample_tok_1_clip() -> None:
    _test_sample_tok_1(True, False)


def test_sample_tok_1_clip_cones() -> None:
    _test_sample_tok_1(True, True)


def _test_sample_tok_all(
    clip_conditioning: bool, clip_cones: bool, image_tokens: int = 256
) -> None:
    """Test that step-by-step decoding is equivalent to all at once for all tokens."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    ) = _setup_test_sample(clip_conditioning, clip_cones, None, image_tokens)

    decoded_logits = []
    params = flax.core.copy(params, {"cache": cache})

    # compute logits for image tok 0
    logits, new_cache = mdl_dec.apply(
        params,
        mutable=["cache"],
        method=mdl_dec.decode_init,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    assert logits.shape == (8192,)
    decoded_logits.append(logits)
    params = flax.core.copy(params, new_cache)

    step_j = jax.jit(
        lambda params, i: mdl_dec.apply(
            params,
            mutable=["cache"],
            method=mdl_dec.decode_step,
            tok=toks[i],
            idx=jnp.array(i),
        )
    )

    # compute logits for image toks 1-255 (inputting toks 0-254)
    for i in range(image_tokens - 1):
        logits, new_cache = step_j(params, i)
        assert logits.shape == (8192,)
        decoded_logits.append(logits)
        params = flax.core.copy(params, new_cache)

    decoded_logits = jnp.stack(decoded_logits, axis=0)
    assert decoded_logits.shape == (image_tokens, 8192)
    np.testing.assert_allclose(logits_all, decoded_logits, rtol=0, atol=1e-6)


def test_sample_tok_all_no_clip() -> None:
    _test_sample_tok_all(False, False)


def test_sample_tok_all_clip() -> None:
    _test_sample_tok_all(True, False)


def test_sample_tok_all_clip_cones() -> None:
    _test_sample_tok_all(True, True)


def test_sample_tok_all_clip_cones_1024() -> None:
    # There was a boundary issue with flash attention that broke with sequence lengths that > 1024
    # & not multiples of 1024.
    _test_sample_tok_all(True, True, 1024)


def test_clip_does_anything() -> None:
    """Test that changing the CLIP embedding changes the logits."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    ) = _setup_test_sample(True, False)

    clip_embedding = jnp.zeros_like(clip_embedding)
    logits_all_zero = mdl_nodec.apply(
        params,
        image=toks,
        clip_embedding=clip_embedding,
        cos_sim_lower=jnp.array([]),
        cos_sim_upper=jnp.array([]),
    )

    assert not jnp.allclose(logits_all, logits_all_zero, rtol=0, atol=1e-3)


def test_clip_cones_do_anything() -> None:
    """Test that changing the CLIP cone bounds changes the logits."""
    (
        mdl_nodec,
        mdl_dec,
        params,
        cache,
        toks,
        clip_embedding,
        cos_sim_lower,
        cos_sim_upper,
        logits_all,
    ) = _setup_test_sample(True, True)

    logits_full_range = mdl_nodec.apply(
        params,
        image=toks,
        clip_embedding=clip_embedding,
        cos_sim_lower=jnp.array([-1.0, 0.1]),
        cos_sim_upper=jnp.array([1.0, 0.4]),
    )

    assert not jnp.allclose(logits_all, logits_full_range, rtol=0, atol=1e-3)


@partial(jax.jit, static_argnums=(0,))
def sample(
    mdl: ImageModel,
    params: dict[str, Any],
    clip_embedding: jax.Array,
    cos_sim_lower: jax.Array,
    cos_sim_upper: jax.Array,
    rng: jax.Array,
    top_p: float = 0.95,
) -> jax.Array:
    """Sample a single image from the model. Returns an array of codes to be passed to the
    LDM decoder."""
    if mdl.clip_conditioning and mdl.clip_cones:
        assert clip_embedding.shape == (mdl.clip_cone_count, 768)
        assert cos_sim_lower.shape == cos_sim_upper.shape == (mdl.clip_cone_count,)
    elif mdl.clip_conditioning and not mdl.clip_cones:
        assert clip_embedding.shape == (768,)
        assert cos_sim_lower.shape == cos_sim_upper.shape == (0,)
    else:
        assert (
            clip_embedding.shape == cos_sim_lower.shape == cos_sim_upper.shape == (0,)
        )

    # Flash attention doesn't work with Flax's fast decoding. Something to do with how masks are
    # handled. Would be nice to fix it, but for now we just use the slower attention when sampling.
    mdl_decode = mdl.clone(decode=True, flash_attention=False, dropout=0.0)
    params_fake = mdl_decode.init(
        jax.random.PRNGKey(0),
        image=jnp.zeros((mdl.image_tokens,), dtype=jnp.int32),
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    params = flax.core.copy(params, {"cache": params_fake["cache"]})
    del params_fake

    # This needs to be outside the linen module because the fori_loop combinator doesn't work
    # inside them.
    def loop_iter(
        i: int, acc: Tuple[jax.Array, jax.Array]
    ) -> Tuple[jax.Array, jax.Array, dict[str, Any]]:
        image_toks, rng, params = acc
        logits, new_cache = mdl_decode.apply(
            params,
            mutable=["cache"],
            method=mdl_decode.decode_step,
            tok=image_toks[i],
            idx=i,
        )
        assert logits.shape == (8192,)
        params = flax.core.copy(params, new_cache)
        filtered_logits = _filter_top_p(logits, top_p)
        rng_sample, rng_loop = jax.random.split(rng, 2)
        tok = jax.random.categorical(rng_sample, filtered_logits)
        image_toks = image_toks.at[i + 1].set(tok)
        return (image_toks, rng_loop, params)

    rng0, rng_loop = jax.random.split(rng, 2)
    logits_0, cache = mdl_decode.apply(
        params,
        mutable=["cache"],
        method=mdl_decode.decode_init,
        clip_embedding=clip_embedding,
        cos_sim_lower=cos_sim_lower,
        cos_sim_upper=cos_sim_upper,
    )
    assert logits_0.shape == (8192,)
    filtered_logits_0 = _filter_top_p(logits_0, top_p)
    tok_0 = jax.random.categorical(rng0, filtered_logits_0)

    params = flax.core.copy(params, cache)

    image_toks = jnp.zeros((mdl.image_tokens,), dtype=jnp.int32).at[0].set(tok_0)
    image_toks, _, _ = jax.lax.fori_loop(  # type: ignore[no-untyped-call]
        0,
        mdl.image_tokens - 1,
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
    flash_attention: bool

    def setup(self) -> None:
        if self.flash_attention:
            # Use fast flash attention implementation
            def attn_function(
                q,
                k,
                v,
                bias=None,
                mask=None,
                broadcast_dropout=True,
                dropout_rng=None,
                dropout_rate=0.0,
                deterministic=False,
                dtype=None,
                precision=None,
            ):
                assert len(q.shape) == 3, "batch dimensions not implemented"
                assert q.shape[1] == k.shape[1] == v.shape[1]
                assert q.shape[2] == k.shape[2] == v.shape[2]
                assert k.shape[0] == v.shape[0]

                rearrange_qkv = lambda x: rearrange(
                    x, "seq_len heads head_dim -> 1 heads seq_len head_dim"
                )
                q, k, v = map(rearrange_qkv, (q, k, v))

                assert bias == None, "attention bias not implemented"
                assert (
                    mask == None
                ), "attention mask is redundant with causal_flash_attention"
                assert dropout_rate == 0.0, "attention dropout not implemented"

                try:
                    res = causal_flash_attention(q, k, v)
                except TypeError as e:
                    if "cannot reshape array of shape" in str(e):
                        raise ValueError(
                            (
                                "Got an exception from causal_flash_attention: {}. You may have "
                                "run into its bug with sequence lengths that are not a multiple of "
                                "the chunk size."
                            ).format(e)
                        )
                if dtype != None:
                    assert res.dtype == dtype

                return rearrange(
                    res, "1 heads seq_len head_dim -> seq_len heads head_dim"
                )

        else:
            attn_function = nn.attention.dot_product_attention
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
            attention_fn=attn_function,
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

    def __call__(self, embeds: jax.Array, _) -> jax.Array:
        if self.flash_attention:
            mask = None
        else:
            mask = jnp.tril(
                jnp.ones((self.num_heads, embeds.shape[0], embeds.shape[0]))
            )
        out_block_1 = self.layer_norm_1(self.mha(embeds, mask=mask))
        in_block_2: jax.Array = embeds + self.dropout_layer(out_block_1)
        out_block_2: jax.Array = self.layer_norm_2(
            self.dropout_layer(
                self.linear_2(self.activation_function(self.linear_1(in_block_2)))  # type: ignore[attr-defined]
            )
        )
        return in_block_2 + out_block_2, None


def test_flash_attention_equals_standard() -> None:
    """Test that flash attention gives the same results as Flax's standard attention."""
    mdl_std = TransformerLayer(
        d_model=768,
        num_heads=12,
        ff_dim=3072,
        dropout=None,
        use_biases=False,
        activations_dtype=jnp.float32,
        activation_function=jax.nn.relu,
        kernel_init=jax.nn.initializers.xavier_uniform(),
        decode=False,
        flash_attention=False,
    )

    input_shape = (64, 768)
    input_vals = jax.random.normal(jax.random.PRNGKey(0), input_shape)

    params = mdl_std.init(
        jax.random.PRNGKey(1), jnp.ones(input_shape, dtype=jnp.float32), None
    )

    out_std, _ = mdl_std.apply(params, input_vals, None)

    mdl_flash = mdl_std.clone(flash_attention=True)
    out_flash, _ = mdl_flash.apply(params, input_vals, None)

    np.testing.assert_allclose(out_std, out_flash, atol=3e-6, rtol=0)


# A cache for the cosine similarity table is important. It's possible to create it in
# ImageModel.setup but that gets called every time apply is called which makes the tests slow. So
# we use a global one.
_cos_sim_table = None


def loss(
    model: ImageModel,
    params: dict[str, Any],
    dropout_rng: jax.Array,
    cone_rng: jax.Array,
    ex_img: jax.Array,
    ex_clip: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a single example."""
    global _cos_sim_table
    assert ex_img.shape == (
        model.image_tokens,
    ), f"ex_img.shape: {ex_img.shape}, expected: {(model.image_tokens,)}"
    if model.clip_conditioning:
        assert ex_clip.shape == (768,)
    else:
        assert ex_clip.shape == (0,)
    if model.clip_cones:
        if _cos_sim_table is None:
            print("Creating cosine similarity table...")
            start_time = datetime.now()
            # 64 KiB ought to be enough for anybody
            _cos_sim_table = cone_sampling.LogitsTable(767, 64 // 4 * 1024)
            end_time = datetime.now()
            print(f"Created cosine similarity table in {end_time - start_time}")
        else:
            print("Using cached cosine similarity table")
        cone_rngs = jax.random.split(cone_rng, model.clip_cone_count)
        cond_clip, lower_bound, upper_bound = jax.vmap(
            cone_sampling.generate_clip_cone, in_axes=(None, 0, None)
        )(_cos_sim_table, cone_rngs, ex_clip)
    else:
        cond_clip = ex_clip
        lower_bound = upper_bound = jnp.array([], dtype=jnp.float32)
    logits: jax.Array = model.apply(
        params,
        rngs={"dropout": dropout_rng},
        image=ex_img,
        clip_embedding=cond_clip,
        cos_sim_lower=lower_bound,
        cos_sim_upper=upper_bound,
    )
    return optax.softmax_cross_entropy(logits, jax.nn.one_hot(ex_img, 8192))  # type: ignore[no-any-return]


def loss_batch(
    model: ImageModel,
    params: dict[str, Any],
    dropout_rng: jax.Array,
    cones_rng: jax.Array,
    batch_imgs: jax.Array,
    batch_clips: jax.Array,
) -> jax.Array:
    """Compute the cross-entropy loss for a batch of examples."""
    assert batch_imgs.shape[0] == batch_clips.shape[0]
    return jnp.mean(
        jax.vmap(loss, in_axes=(None, None, 0, 0, 0, 0))(
            model,
            params,
            jax.random.split(dropout_rng, batch_imgs.shape[0]),
            jax.random.split(cones_rng, batch_imgs.shape[0]),
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


def test_cone_train() -> None:
    """Test the model can memorize some image/clip pairs."""
    mdl_cfg = copy(gpt_1_config)
    mdl_cfg.clip_conditioning = True
    mdl_cfg.clip_cones = True
    mdl_cfg.clip_cone_count = 9
    mdl_cfg.dropout = None
    # This may or may not actually need a model this big, I'm tired of fucking with it.
    mdl_cfg.d_model = 1024
    mdl_cfg.num_heads = 16
    mdl_cfg.n_layers = 24

    n_imgs = 8

    mdl = ImageModel(**mdl_cfg.__dict__)

    img_rng, clip_rng, params_rng, train_rng, test_rng = jax.random.split(
        jax.random.PRNGKey(0), 5
    )

    imgs = jax.random.randint(img_rng, (n_imgs, mdl.image_tokens), 0, 8192)
    clips = jax.random.normal(clip_rng, (n_imgs, 768))
    clips = clips / jnp.linalg.norm(clips, axis=-1, keepdims=True)

    params = mdl.init(
        {"params": params_rng, "dropout": jax.random.PRNGKey(0)},
        image=jnp.zeros((mdl.image_tokens,), dtype=jnp.int32),
        clip_embedding=jnp.zeros((mdl.clip_cone_count, 768), dtype=jnp.float32),
        cos_sim_lower=jnp.zeros((mdl.clip_cone_count,), dtype=jnp.float32),
        cos_sim_upper=jnp.zeros((mdl.clip_cone_count,), dtype=jnp.float32),
    )

    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    steps = 2_000
    adam = optax.adam(learning_rate=triangle_schedule(3e-5, steps))
    opt = optax.chain(optax.clip_by_global_norm(0.25), adam)
    opt_state = opt.init(params)

    def opt_step(params, opt_state, rng):
        dropout_rng, cones_rng, rng2 = jax.random.split(rng, 3)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            cones_rng,
            batch_imgs=imgs,
            batch_clips=clips,
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        norm = optax.global_norm(grads)
        return new_params, opt_state, rng2, loss, norm

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    for i in trange(steps):
        params, opt_state, train_rng, loss, norm = opt_step(
            params, opt_state, train_rng
        )
        tqdm.write(f"iter {i:04d} loss: {loss:0.4f} grad norm: {norm:7.2f}")

    sampled_imgs = []
    for i in trange(n_imgs):
        sample_rng, tgt_rng, fill_rng, test_rng = jax.random.split(test_rng, 4)

        # The model should've memorized the dataset at this point, but it can only generate images
        # with a specified similarity to a given embedding. It can't generate an image with an
        # exact embedding. So we generate a random target with a specified similarity.
        tgt_v = cone_sampling.random_pt_with_cosine_similarity(tgt_rng, clips[i], 0.5)[
            None, :
        ]
        tgt_lower_bound = jnp.array([0.5])
        tgt_upper_bound = jnp.array([1.0])

        fill_vs = jax.random.normal(fill_rng, (mdl.clip_cone_count - 1, 768))
        fill_vs = fill_vs / jnp.linalg.norm(fill_vs, axis=-1, keepdims=True)

        fill_lower_bounds = jnp.full((mdl.clip_cone_count - 1,), -1.0)
        fill_upper_bounds = jnp.full((mdl.clip_cone_count - 1,), 1.0)

        cond_vs = jnp.concatenate([tgt_v, fill_vs], axis=0)
        cond_lower_bounds = jnp.concatenate(
            [tgt_lower_bound, fill_lower_bounds], axis=0
        )
        cond_upper_bounds = jnp.concatenate(
            [tgt_upper_bound, fill_upper_bounds], axis=0
        )

        assert cond_vs.shape == (mdl.clip_cone_count, 768)
        assert (
            cond_lower_bounds.shape == cond_upper_bounds.shape == (mdl.clip_cone_count,)
        )

        toks = sample(
            mdl,
            params,
            cond_vs,
            cond_lower_bounds,
            cond_upper_bounds,
            sample_rng,
            top_p=0.25,
        )
        sampled_imgs.append(toks)

    sampled_imgs = jnp.stack(sampled_imgs, axis=0)

    # Test that sampled images are the training images
    correct_imgs = [0] * n_imgs
    for img in sampled_imgs:
        for i, train_img in enumerate(imgs):
            if jnp.all(img == train_img):
                correct_imgs[i] += 1
                break
    correct_img_cnt = sum(correct_imgs)
    print(f"correct_imgs: {correct_imgs}, count {correct_img_cnt}")
    assert correct_img_cnt == n_imgs

    # Test that sampled images match the CLIP conditioning
    correct_conds = [False] * n_imgs
    for i in range(n_imgs):
        correct_conds[i] = jnp.array_equal(imgs[i], sampled_imgs[i])
    correct_conds = np.array(correct_conds)
    print(f"correct_conds: {correct_conds}")
    assert all(correct_conds)


def train_loop_simple(
    data: jax.Array, mdl: ImageModel, iters: int
) -> Tuple[jax.Array, dict[str, Any]]:
    """Train the model repeatedly on a single batch for testing."""
    assert mdl.clip_conditioning is False
    params = mdl.init(
        rngs={"dropout": jax.random.PRNGKey(0), "params": jax.random.PRNGKey(1)},
        image=jnp.zeros((gpt_1_config.image_tokens,), dtype=jnp.int32),
        clip_embedding=jnp.zeros((0,), dtype=jnp.float32),
        cos_sim_lower=jnp.array([]),
        cos_sim_upper=jnp.array([]),
    )
    opt = optax.adam(learning_rate=3e-5)
    opt_state = opt.init(params)
    loss_grad_fn = jax.value_and_grad(loss_batch, argnums=1)

    def opt_step(
        params: dict[str, Any],
        opt_state: Any,
        rng: jax.Array,
        batch_imgs: jax.Array,
    ) -> Tuple[dict[str, Any], Any, jax.Array, jax.Array]:
        dropout_rng, cones_rng, rng2 = jax.random.split(rng, 3)
        loss, grads = loss_grad_fn(
            mdl,
            params,
            dropout_rng,
            cones_rng,
            batch_imgs=batch_imgs,
            batch_clips=jnp.zeros((batch_imgs.shape[0], 0), dtype=jnp.float32),
        )
        updates, opt_state = opt.update(grads, opt_state)
        new_params: dict[str, Any] = optax.apply_updates(params, updates)
        return new_params, opt_state, rng2, loss

    opt_step = jax.jit(opt_step, donate_argnums=(0, 1, 2))

    train_rng = jax.random.PRNGKey(0)
    for i in range(iters):
        params, opt_state, train_rng, loss = opt_step(  # type:ignore[misc]
            params, opt_state, train_rng, data
        )
        print(f"iter {i} loss: {loss}")
    return loss, params  # type:ignore[return-value]


def test_learn_zeros() -> None:
    """Test whether the model can learn to predict all zeros."""
    mdl = ImageModel(**gpt_1_config.__dict__)
    data = jnp.zeros((16, gpt_1_config.image_tokens), dtype=jnp.int32)
    loss, params = train_loop_simple(data, mdl, 10)
    assert loss < 1e-10

    sampled_arr = sample(
        mdl,
        params,
        jnp.zeros((0,), dtype=jnp.float32),
        jnp.array([], dtype=jnp.float32),
        jnp.array([], dtype=jnp.float32),
        jax.random.PRNGKey(0),
    )
    assert jnp.all(sampled_arr == 0)


def test_learn_ranges() -> None:
    """Test whether the model can learn to predict a range of integers."""
    mdl = ImageModel(**gpt_1_config.__dict__)
    data = jnp.arange(16 * gpt_1_config.image_tokens).reshape(
        (16, gpt_1_config.image_tokens)
    )
    # It's annoying how many iterations we need to get to minimal loss on such a trivial dataset
    loss, params = train_loop_simple(data, mdl, 1000)
    assert loss < 0.6
    empty_arr = jnp.zeros((0,), dtype=jnp.float32)
    sample_jv = jax.jit(
        lambda params, rng: jax.vmap(
            lambda rng: sample(
                mdl, params, empty_arr, empty_arr, empty_arr, rng, top_p=0.98
            )
        )(rng)
    )
    print("Generating samples...")
    total_samples = 256
    samples_per_batch = 64
    assert total_samples % samples_per_batch == 0
    sample_batches = []
    rng = jax.random.PRNGKey(0)
    for _ in trange(total_samples // samples_per_batch):
        rng, rng2 = jax.random.split(rng)
        sample_batches.append(
            sample_jv(params, jax.random.split(rng2, samples_per_batch))
        )
    sampled_arr: jax.Array = jnp.concatenate(sample_batches, axis=0)
    print(f"Generated samples, shape: {sampled_arr.shape}")
    counts = dict([(i, 0) for i in range(16)])
    for i, s in enumerate(sampled_arr):
        assert s[0] % 256 == 0, f"sample {i} starts with {s[0]}"
        np.testing.assert_equal(np.array(s), np.array(s[0] + np.arange(256)))
        counts[int(s[0] // 256)] += 1
    print(f"counts: {counts}")
    assert all([c > 0 for c in counts.values()])
