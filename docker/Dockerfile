# Generic parameterized Dockerfile for Vast.ai and RunPod

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Install common packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    atop bc build-essential curl fish git htop jq less libbz2-dev libffi-dev liblzma-dev \
    libncursesw5-dev libreadline-dev libsqlite3-dev libssl-dev libxml2-dev libxmlsec1-dev \
    magic-wormhole nano rclone tk-dev tmux unzip vmtouch wget xz-utils zlib1g-dev zstd

RUN usermod -s /bin/fish root

# https://ommer-lab.com/files/latent-diffusion/vq-f4.zip
ADD vq-f4.zip /root/vq-f4.zip
RUN unzip /root/vq-f4.zip -d /root/vq-f4

RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
RUN echo $(pyenv init) >> /root/.bashrc
RUN pyenv install 3.11.13 && pyenv global 3.11.13

ADD fish-pyenv.fish /root/.config/fish/conf.d/fish-pyenv.fish

RUN pip install poetry

ADD setup-repo.sh /root/setup-repo.sh
ADD get-dataset.sh /root/get-dataset.sh
ADD tmux.conf /root/.tmux.conf