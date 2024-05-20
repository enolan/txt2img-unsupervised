#!/usr/bin/env bash

set -euo pipefail

# Set up dev environment
git clone --recursive https://github.com/enolan/txt2img-unsupervised.git /root/txt2img-unsupervised
ln -sf /root/vq-f4/model.ckpt /root/txt2img-unsupervised/vq-f4.ckpt
cd /root/txt2img-unsupervised
poetry install --with dev --with cuda --no-root