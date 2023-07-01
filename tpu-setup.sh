#!/bin/bash
# Startup script for TPU VMs

set -euo pipefail

echo "TXT2IMG-UNSUPERVISED: running as $(whoami)"
echo "TXT2IMG-UNSUPERVISED: Installing apt dependencies"
apt-get update
apt-get install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev \
        libxmlsec1-dev libffi-dev liblzma-dev rclone magic-wormhole zstd jq atop fish -y

echo "TXT2IMG-UNSUPERVISED: Fetching user setup script"
curl -H "Metadata-Flavor: Google" \
    http://metadata.google.internal/computeMetadata/v1/instance/attributes/txt2img-user-setup > /usr/local/bin/txt2img-user-setup
chmod +x /usr/local/bin/txt2img-user-setup
