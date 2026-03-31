# Conveniences for ipython

from copy import copy

import orbax.checkpoint
import torch
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf
from transformer_model import gpt_1_config

gpt_1_config_no_dropout = copy(gpt_1_config)
gpt_1_config_no_dropout.dropout = None

checkpointer = orbax.checkpoint.PyTreeCheckpointer()


def load_ckpt_params(ckpt_dir):
    return checkpointer.restore(ckpt_dir)["params"]


ae_cfg = OmegaConf.load(
    "vendor/latent-diffusion/models/first_stage_models/vq-f4/config.yaml"
)["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)


def load_autoencoder_params():
    return LDMAutoencoder.params_from_torch(
        torch.load("vq-f4.ckpt", map_location="cpu"), cfg=ae_cfg
    )
