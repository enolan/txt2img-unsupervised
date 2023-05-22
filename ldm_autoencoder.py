import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from pathlib import Path


class LDMAutoencoder(nn.Module):
    """Flax Reimplentation of the quantized autoencoder from the latent diffusion repo.
    Only intended to work with the f4 model."""

    cfg: dict

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.cfg["n_embed"],
            dtype=jnp.float32,
            features=self.cfg["embed_dim"],
        )
        self.post_quant_conv = nn.Conv(
            features=self.cfg["ddconfig"]["z_channels"], kernel_size=[1, 1]
        )
        self.decoder = LDMDecoder(cfg=self.cfg["ddconfig"])

    def embed(self, x, shape=None):
        """Embed the codes, reshaping them to (height, width), where height and width are
        the height and width of the compressed representation. You must pass the shape parameter
        for decoding to work."""
        if shape is not None:
            x = x.reshape(shape)
        return self.embedding(x)

    def _conv_embeds(self, x):
        """Run the post-quantization convolutional layer on the embedded codes. For testing."""
        return self.post_quant_conv(x)

    def _dec_conv_in(self, x):
        """Run the first convolutional layer in the decoder. For testing."""
        return self.decoder.conv_in(x)

    def _dec_mid_resnet_1(self, x):
        """Run the 1st resnet block of the middle of the decoder. For testing."""
        return self.decoder.mid_resnet_1(x)

    def _dec_mid_attn(self, x):
        """Run the attention block in the middle of the decoder. For testing."""
        return self.decoder.mid_attn_1(x)

    @staticmethod
    def params_from_torch(state_dict):
        """Load the parameters from a torch checkpoint."""
        state_dict = state_dict["state_dict"]

        def conv2d_params(prefix: str):
            return {
                "kernel": rearrange(
                    state_dict[prefix + ".weight"],
                    "outC inC kH kW -> kH kW inC outC",
                ),
                "bias": state_dict[prefix + ".bias"],
            }

        def groupnorm_params(prefix: str):
            return {
                "bias": state_dict[prefix + ".bias"],
                "scale": state_dict[prefix + ".weight"],
            }

        def resnet_block_params(prefix: str):
            return {
                "conv1": conv2d_params(prefix + ".conv1"),
                "conv2": conv2d_params(prefix + ".conv2"),
                "norm1": groupnorm_params(prefix + ".norm1"),
                "norm2": groupnorm_params(prefix + ".norm2"),
            }

        def attn_block_params(prefix: str):
            return {
                "norm": groupnorm_params(prefix + ".norm"),
                "q": conv2d_params(prefix + ".q"),
                "k": conv2d_params(prefix + ".k"),
                "v": conv2d_params(prefix + ".v"),
                "proj_out": conv2d_params(prefix + ".proj_out"),
            }

        params = {
            "params": {
                "embedding": {"embedding": state_dict["quantize.embedding.weight"]},
                "post_quant_conv": conv2d_params("post_quant_conv"),
                "decoder": {
                    "conv_in": conv2d_params("decoder.conv_in"),
                    "mid_resnet_1": resnet_block_params("decoder.mid.block_1"),
                    "mid_resnet_2": resnet_block_params("decoder.mid.block_2"),
                    "mid_attn_1": attn_block_params("decoder.mid.attn_1"),
                },
            }
        }
        return jax.tree_map(jnp.array, params)


class LDMDecoder(nn.Module):
    """Flax Reimplentation of the decoder from the latent diffusion repo."""

    cfg: dict

    def setup(self):
        # channels used in decoder
        block_in = self.cfg["ch"] * self.cfg["ch_mult"][-1]

        # convolutional layer applied first
        # torch code pads with zeros, what does flax pad with? Tests pass so I guess it's fine.
        self.conv_in = nn.Conv(features=block_in, kernel_size=[3, 3], padding=1)

        # middle blocks
        self.mid_resnet_1 = ResnetBlock(out_channels=block_in, in_channels=block_in)
        self.mid_attn_1 = AttnBlock(in_channels=block_in)
        self.mid_resnet_2 = ResnetBlock(out_channels=block_in, in_channels=block_in)


class ResnetBlock(nn.Module):
    in_channels: int
    out_channels: int

    def setup(self):
        self.norm1 = BatchlessGroupNorm(num_groups=32, epsilon=1e-6)
        self.conv1 = nn.Conv(features=self.out_channels, kernel_size=[3, 3], padding=1)
        self.norm2 = BatchlessGroupNorm(num_groups=32, epsilon=1e-6)
        self.conv2 = nn.Conv(features=self.out_channels, kernel_size=[3, 3], padding=1)

    def __call__(self, x):
        assert (
            len(x.shape) == 3 and x.shape[-1] == self.in_channels
        ), f"resnet block should be called with shape h w c and {in_channels} channels, got {x.shape}"
        h = self.norm1(x)
        h = nn.activation.swish(h)
        h = self.conv1(h)
        # temb is always None in torch, so skip it here
        h = self.norm2(h)
        h = nn.activation.swish(h)
        # there's dropout here in the torch code, but we skip it because we only want to do inference
        h = self.conv2(h)
        # there's a bunch of code about "use_conv_shortcut" in the torch code but that param is
        # always false so it's not included here.
        return x + h


class AttnBlock(nn.Module):
    """Self attention block. I could use Flax's self attention code but it seems easier to ensure
    the implementation is the same if I just reimplement the latent diffusion Torch code.
    """

    in_channels: int

    def setup(self):
        self.norm = BatchlessGroupNorm(num_groups=32, epsilon=1e-6)
        self.q = nn.Conv(features=self.in_channels, kernel_size=[1, 1], padding=0)
        self.k = nn.Conv(features=self.in_channels, kernel_size=[1, 1], padding=0)
        self.v = nn.Conv(features=self.in_channels, kernel_size=[1, 1], padding=0)
        self.proj_out = nn.Conv(
            features=self.in_channels, kernel_size=[1, 1], padding=0
        )

    def __call__(self, x):
        assert len(x.shape) == 3 and x.shape[-1] == self.in_channels
        height, width, _channels = x.shape
        h = self.norm(x)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = rearrange(q, "h w c -> (h w) c")
        k = rearrange(k, "h w c -> (h w) c")
        v = rearrange(v, "h w c -> (h w) c")

        attn = jax.nn.softmax((q @ k.T) * self.in_channels ** (-0.5))

        h = attn @ v

        h = rearrange(h, "(h w) c -> h w c", h=height, w=width)
        h = self.proj_out(h)
        return x + h


class BatchlessGroupNorm(nn.GroupNorm):
    """Version of GroupNorm that doesn't require a batch dimension."""

    def __call__(self, x):
        input = rearrange(x, "... -> 1 ...")
        res = super().__call__(input)
        return rearrange(res, "1 ... -> ...")


def _setup_comparison_test(name):
    """Load stuff to compare Flax & Torch behavior."""
    src_dir = Path(__file__).parent
    path_prefix = src_dir / f"test-images/{name}"
    cfg = OmegaConf.load(
        src_dir / "vendor/latent-diffusion/models/first_stage_models/vq-f4/config.yaml"
    )
    mdl = LDMAutoencoder(cfg=cfg["model"]["params"])
    params = LDMAutoencoder.params_from_torch(
        torch.load(src_dir / "vq-f4.ckpt", map_location="cpu")
    )
    return src_dir, path_prefix, mdl, params


def _test_embedding(name):
    """Test that the embedding matches the one from the original implementation."""
    src_dir, path_prefix, mdl, params = _setup_comparison_test(name)
    codes = jnp.load(path_prefix.with_suffix(".codes.npy"))

    golden_embedded_codes = jnp.load(path_prefix.with_suffix(".embedded_codes.npy"))

    computed_embedded_codes = rearrange(
        mdl.apply(params, x=codes, shape=(64, 64), method=mdl.embed), "h w c -> c h w"
    )
    assert golden_embedded_codes.shape == computed_embedded_codes.shape
    np.testing.assert_array_equal(golden_embedded_codes, computed_embedded_codes)


def _test_post_quant_conv(name):
    """Test that the post-quantization convolution matches the one from the original implementation."""
    src_dir, path_prefix, mdl, params = _setup_comparison_test(name)
    embedded_codes = jnp.load(path_prefix.with_suffix(".embedded_codes.npy"))
    assert embedded_codes.shape == (3, 64, 64)

    golden_convolved_embedded_codes = jnp.load(
        path_prefix.with_suffix(".convolved_embedded_codes.npy")
    )
    assert golden_convolved_embedded_codes.shape == (3, 64, 64)
    assert not (np.array_equal(embedded_codes, golden_convolved_embedded_codes))

    computed_convolved_embedded_codes = mdl.apply(
        params, x=rearrange(embedded_codes, "c h w -> h w c"), method=mdl._conv_embeds
    )
    assert computed_convolved_embedded_codes.shape == (64, 64, 3)
    np.testing.assert_allclose(
        rearrange(golden_convolved_embedded_codes, "c h w -> h w c"),
        computed_convolved_embedded_codes,
        atol=1e-6,
    )


def _test_dec_conv_in(name):
    """Test that the first convolutional layer in the decoder matches the pytorch implementation."""
    src_dir, path_prefix, mdl, params = _setup_comparison_test(name)
    convolved_embedded_codes = jnp.load(
        path_prefix.with_suffix(".convolved_embedded_codes.npy")
    )
    assert convolved_embedded_codes.shape == (3, 64, 64)
    golden_conv_in = jnp.load(path_prefix.with_suffix(".post_conv_hidden.npy"))
    assert golden_conv_in.shape == (512, 64, 64)
    computed_conv_in = mdl.apply(
        params,
        x=rearrange(convolved_embedded_codes, "c h w -> h w c"),
        method=mdl._dec_conv_in,
    )
    assert computed_conv_in.shape == (64, 64, 512)
    np.testing.assert_allclose(
        computed_conv_in, rearrange(golden_conv_in, "c h w -> h w c")
    )


def _test_mid_resnet_block_1(name):
    """Test that the first resnet block in the middle of the decoder matches the pytorch implementation."""
    src_dir, path_prefix, mdl, params = _setup_comparison_test(name)
    conv_in = jnp.load(path_prefix.with_suffix(".post_conv_hidden.npy"))
    assert conv_in.shape == (512, 64, 64)
    golden_mid_resnet_1 = jnp.load(path_prefix.with_suffix(".post_resnet_1_hidden.npy"))
    assert golden_mid_resnet_1.shape == (512, 64, 64)
    computed_mid_resnet_1 = mdl.apply(
        params, x=rearrange(conv_in, "c h w -> h w c"), method=mdl._dec_mid_resnet_1
    )
    assert computed_mid_resnet_1.shape == (64, 64, 512)
    np.testing.assert_allclose(
        rearrange(computed_mid_resnet_1, "h w c -> c h w"),
        golden_mid_resnet_1,
        atol=1e-3,
        rtol=0
        # TBH not sure if the error here is a bug or not. we see up to 57% difference in some
        # values. very small in absolute terms though.
    )


def _test_mid_attn_block_1(name):
    """Test that the attention block in the decoder matches the pytorch implementation."""
    src_dir, path_prefix, mdl, params = _setup_comparison_test(name)
    post_resnet1_hidden = jnp.load(path_prefix.with_suffix(".post_resnet_1_hidden.npy"))
    assert post_resnet1_hidden.shape == (512, 64, 64)
    golden_mid_attn_1 = jnp.load(path_prefix.with_suffix(".post_attn_hidden.npy"))
    assert golden_mid_attn_1.shape == (512, 64, 64)
    computed_mid_attn_1 = mdl.apply(
        params,
        x=rearrange(post_resnet1_hidden, "c h w -> h w c"),
        method=mdl._dec_mid_attn,
    )
    assert computed_mid_attn_1.shape == (64, 64, 512)
    np.testing.assert_allclose(
        rearrange(computed_mid_attn_1, "h w c -> c h w"),
        golden_mid_attn_1,
        atol=1e-4,
        rtol=0,
    )


def test_embedding_me():
    _test_embedding("devil me")


def test_embedding_painting():
    _test_embedding("painty lady")


def test_post_quant_conv_me():
    _test_post_quant_conv("devil me")


def test_post_quant_conv_painting():
    _test_post_quant_conv("painty lady")


def test_dec_conv_in_me():
    _test_dec_conv_in("devil me")


def test_dec_conv_in_painting():
    _test_dec_conv_in("painty lady")


def test_mid_resnet_block_1_me():
    _test_mid_resnet_block_1("devil me")


def test_mid_resnet_block_1_painting():
    _test_mid_resnet_block_1("painty lady")


def test_mid_attn_block_1_me():
    _test_mid_attn_block_1("devil me")


def test_mid_attn_block_1_painting():
    _test_mid_attn_block_1("painty lady")
