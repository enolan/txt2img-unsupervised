#!/usr/bin/env bash

# Set up a session on an SF Compute node. This script runs on the SFC node.

set -euo pipefail

docker pull enolan/txt2img-unsupervised-vast:latest

# We want the container to run indefinitely. If we just do `docker run [...] tmux new-session` then
# the container stops if your ssh session dies or you detach the tmux session on purpose, stopping
# the training run.
docker run -d --name txt2img-container --gpus all enolan/txt2img-unsupervised-vast:latest tail -f /dev/null

docker exec -it -e TERM txt2img-container tmux new-session \
    -n "fish" 'bash -c "/root/setup-repo.sh; exec fish"' \; \
    new-window -n "fish" 'fish' \; \
    new-window -n "htop" 'htop' \; \
    new-window -n "atop" 'atop 1' \; \
    new-window -n "nvidia-smi" 'watch -n 1 nvidia-smi'
