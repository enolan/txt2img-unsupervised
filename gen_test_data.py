"""
Generate some test data for testing the Jax implementation of the autoencoder.
"""
import PIL
import numpy as np
import torch
import sys
from pathlib import Path
import argparse
from omegaconf import OmegaConf
from einops import rearrange

from ldm.models.autoencoder import VQModel

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("img", type=str)
parser.add_argument("ckpt", type=str)
parser.add_argument("cfg", type=str)
args = parser.parse_args()

img_path = Path(args.img).absolute()
out_dir = img_path.parent

# Load the image
img = PIL.Image.open(img_path)
w, h = img.size
assert w == 256 and h == 256
assert img.mode == "RGB"

img = np.array(img, dtype=np.float32) / 127.5 - 1.0
img = torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)

# Load the config
cfg = OmegaConf.load(args.cfg)

# Load the model
model = VQModel(**cfg["model"]["params"])
model.init_from_ckpt(args.ckpt)

# Encode the image
z = model.encode(img)
codes = z[2][2].numpy()

# Save the encoded representation
codes_path = img_path.with_suffix(".codes.npy")
print(f"Saving codes to {codes_path}, shape {codes.shape}")
np.save(codes_path, codes)
latents_path = img_path.with_suffix(".latents.npy")
print(f"Saving latents to {latents_path}, shape {z[0].shape}")
np.save(latents_path, z[0].detach().numpy())

# Save embedded codes
embedded_codes = model.quantize.get_codebook_entry(z[2][2], [1, 64, 64, 3])
embedded_codes_np = embedded_codes.squeeze(0).detach().numpy()
embedded_codes_path = img_path.with_suffix(".embedded_codes.npy")
print(
    f"Saving embedded codes to {embedded_codes_path}, shape {embedded_codes_np.shape}"
)
np.save(embedded_codes_path, embedded_codes_np)

# Save convolved embedded codes
convolved_embedded_codes = model.post_quant_conv(embedded_codes).detach()
convolved_embedded_codes_np = convolved_embedded_codes.numpy().squeeze(0)
convolved_embedded_codes_path = img_path.with_suffix(".convolved_embedded_codes.npy")
print(
    f"Saving convolved embedded codes to {convolved_embedded_codes_path}, shape {convolved_embedded_codes_np.shape}"
)
np.save(convolved_embedded_codes_path, convolved_embedded_codes_np)

# Save hidden representation after 1st convolution in decoder
post_conv_hidden = model.decoder.conv_in(convolved_embedded_codes)
post_conv_hidden_np = post_conv_hidden.detach().numpy().squeeze(0)
post_conv_hidden_path = img_path.with_suffix(".post_conv_hidden.npy")
print(
    f"Saving post conv hidden to {post_conv_hidden_path}, shape {post_conv_hidden_np.shape}"
)
np.save(post_conv_hidden_path, post_conv_hidden_np)

# Save hidden representation after 1st resnet block in decoder
post_resnet_1_hidden = model.decoder.mid.block_1(post_conv_hidden, None)
post_resnet_1_hidden_np = post_resnet_1_hidden.detach().numpy().squeeze(0)
post_resnet_1_hidden_path = img_path.with_suffix(".post_resnet_1_hidden.npy")
print(
    f"Saving post resnet 1 hidden to {post_resnet_1_hidden_path}, shape {post_resnet_1_hidden_np.shape}"
)
np.save(post_resnet_1_hidden_path, post_resnet_1_hidden_np)

# Save hidden representation after attention block in decoder
post_attn_hidden = model.decoder.mid.attn_1(post_resnet_1_hidden)
post_attn_hidden_np = post_attn_hidden.detach().numpy().squeeze(0)
post_attn_hidden_path = img_path.with_suffix(".post_attn_hidden.npy")
print(
    f"Saving post attn hidden to {post_attn_hidden_path}, shape {post_attn_hidden_np.shape}"
)
np.save(post_attn_hidden_path, post_attn_hidden_np)

# Save hidden representation after mid blocks
post_mid_hidden = model.decoder.mid.block_2(
    model.decoder.mid.attn_1(model.decoder.mid.block_1(post_conv_hidden, None)), None
)
post_mid_hidden_np = post_mid_hidden.detach().numpy().squeeze(0)
post_mid_hidden_path = img_path.with_suffix(".post_mid_hidden.npy")
print(
    f"Saving post mid hidden to {post_mid_hidden_path}, shape {post_mid_hidden_np.shape}"
)
np.save(post_mid_hidden_path, post_mid_hidden_np)

# Save hidden representation after upsampling
post_up_hidden = model.decoder.upsample(post_mid_hidden)
post_up_hidden_np = post_up_hidden.detach().numpy().squeeze(0)
post_up_hidden_path = img_path.with_suffix(".post_upsample_hidden.npy")
print(
    f"Saving post up hidden to {post_up_hidden_path}, shape {post_up_hidden_np.shape}"
)
np.save(post_up_hidden_path, post_up_hidden_np)

# Save decoded image
full_decode = model.decode(model.quantize.get_codebook_entry(z[2][2], [1, 64, 64, 3]))
full_decode_np = full_decode.detach().numpy().squeeze(0)
full_decode_path = img_path.with_suffix(".full_decode.npy")
print(f"Saving full decode to {full_decode_path}, shape {full_decode_np.shape}")
np.save(full_decode_path, full_decode_np)
