#!/bin/bash
# Setup script for TPU VMs. Needs to run as a regular user, and the regular users don't exist
# until they're created by a google service after you `gcloud compute tpus tpu-vm ssh`.

set -euo pipefail

echo "Installing pyenv"
curl https://pyenv.run | bash

# copy config into bashrc
cat >> ~/.bashrc << EOF
export PYENV_ROOT="\$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init -)"
EOF

echo "Activating pyenv"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

echo "Installing Python 3.11.3"
pyenv install 3.11.3
pyenv global 3.11.3

echo "Installing Poetry & poet-plugin"
pip install poetry
pip install poet-plugin

echo "Cloning git repo"
git clone https://github.com/enolan/txt2img-unsupervised.git ~/txt2img-unsupervised
cd ~/txt2img-unsupervised
git submodule update --init

echo "Installing txt2img dependencies"
cd ~/txt2img-unsupervised || exit 1
ln -sf poetry-tpu.lock poetry.lock
poetry install --with tpu --without cuda

echo "Fetching rclone config"
mkdir -p ~/.config/rclone
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/rclone_config > ~/.config/rclone/rclone.conf

echo "Downloading dataset"
mkdir -p ~/datasets/preprocessed
rclone copy \
    --fast-list --size-only --multi-thread-streams 16 --include '*.tar' --include '*.tar.zst' \
    --include '*.parquet' -P \
    r2:txt2img-unsupervised-dataset/preprocessed/ ~/datasets/preprocessed/

echo "Downloading VQGAN"
wget https://ommer-lab.com/files/latent-diffusion/vq-f4.zip -O ~/vq-f4.zip
unzip ~/vq-f4.zip -d ~/vq-f4
mv ~/vq-f4/model.ckpt ~/txt2img-unsupervised/vq-f4.ckpt

echo "Done. Now relogin, and don't forget to start tmux."