# Conveniences for ipython

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint
import PIL.Image
import transformer_model
import torch
from transformer_model import ImageModel, gpt_1_config
from copy import copy
from ldm_autoencoder import LDMAutoencoder
from omegaconf import OmegaConf

gpt_1_config_no_dropout = copy(gpt_1_config)
gpt_1_config_no_dropout.dropout = None

checkpointer = orbax.checkpoint.PyTreeCheckpointer()

def load_ckpt_params(ckpt_dir):
    return checkpointer.restore(ckpt_dir)["params"]

ae_cfg = OmegaConf.load("vendor/latent-diffusion/models/first_stage_models/vq-f4/config.yaml")["model"]["params"]
ae_mdl = LDMAutoencoder(ae_cfg)

def load_autoencoder_params():
    return LDMAutoencoder.params_from_torch(torch.load("vq-f4.ckpt", map_location="cpu"), cfg=ae_cfg)