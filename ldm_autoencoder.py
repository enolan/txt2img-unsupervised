import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import torch
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

    def embed(self, x):
        return self.embedding(x)

    @staticmethod
    def params_from_torch(state_dict):
        """Load the parameters from a torch checkpoint."""
        state_dict = state_dict["state_dict"]
        return {
            "params": {
                "embedding": {"embedding": state_dict["quantize.embedding.weight"]}
            }
        }


def _test_embedding(name):
    """Test that the embedding matches the one from the original implementation."""
    src_dir = Path(__file__).parent
    path_prefix = src_dir / f"test-images/{name}"
    cfg = OmegaConf.load(
        src_dir / "vendor/latent-diffusion/models/first_stage_models/vq-f4/config.yaml"
    )
    mdl = LDMAutoencoder(cfg=cfg["model"]["params"])
    params = LDMAutoencoder.params_from_torch(
        torch.load(src_dir / "vq-f4.ckpt", map_location="cpu")
    )
    codes = jnp.load(path_prefix.with_suffix(".codes.npy"))

    golden_embedded_codes = jnp.load(path_prefix.with_suffix(".embedded_codes.npy"))

    computed_embedded_codes = mdl.apply(params, x=codes, method=mdl.embed)
    assert golden_embedded_codes.shape == computed_embedded_codes.shape
    np.testing.assert_array_equal(golden_embedded_codes, computed_embedded_codes)


def test_embedding_me():
    _test_embedding("devil me")


def test_embedding_painting():
    _test_embedding("painty lady")
