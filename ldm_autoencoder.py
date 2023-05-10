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
            features=self.cfg["ddconfig"]["z_channels"], kernel_size=[1]
        )

    def embed(self, x, shape=None):
        """Embed the codes, reshaping them to (height, width), where height and width are
        the height and width of the compressed representation. You must pass the shape parameter
        for decoding to work."""
        if shape is not None:
            x = x.reshape(shape)
        return self.embedding(x)

    def conv_embeds(self, x):
        return self.post_quant_conv(x)

    @staticmethod
    def params_from_torch(state_dict):
        """Load the parameters from a torch checkpoint."""
        state_dict = state_dict["state_dict"]
        params = {
            "params": {
                "embedding": {"embedding": state_dict["quantize.embedding.weight"]},
                "post_quant_conv": {
                    "kernel": rearrange(
                        # not sure why there's an extra dimension.
                        state_dict["post_quant_conv.weight"],
                        "outC inC 1 1 -> 1 inC outC",
                    ),
                    "bias": state_dict["post_quant_conv.bias"],
                },
            }
        }
        return jax.tree_map(jnp.array, params)


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
        params, x=rearrange(embedded_codes, "c h w -> h w c"), method=mdl.conv_embeds
    )
    assert computed_convolved_embedded_codes.shape == (64, 64, 3)
    np.testing.assert_allclose(
        rearrange(golden_convolved_embedded_codes, "c h w -> h w c"),
        computed_convolved_embedded_codes,
        atol=1e-6,
    )


def test_embedding_me():
    _test_embedding("devil me")


def test_embedding_painting():
    _test_embedding("painty lady")


def test_post_quant_conv_me():
    _test_post_quant_conv("devil me")

def test_post_quant_conv_painting():
    _test_post_quant_conv("painty lady")
