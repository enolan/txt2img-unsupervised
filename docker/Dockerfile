# Generic parameterized Dockerfile for Vast.ai and RunPod

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install common packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    atop bc build-essential curl fish git htop jq less \
    magic-wormhole nano rclone tmux unzip vmtouch wget zstd

RUN usermod -s /bin/fish root

# https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
ADD docker/vq-f4.zip /root/vq-f4.zip
RUN unzip /root/vq-f4.zip -d /root/vq-f4

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY .python-version /root/.python-version
RUN uv python install $(cat /root/.python-version)

ADD docker/setup-repo.sh /root/setup-repo.sh
ADD docker/get-dataset.sh /root/get-dataset.sh
ADD docker/tmux.conf /root/.tmux.conf