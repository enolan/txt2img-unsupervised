FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev curl libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev rclone magic-wormhole zstd atop fish \
    unzip git wget htop nano

RUN usermod -s /bin/fish root

ADD vq-f4.zip /root/vq-f4.zip
RUN unzip /root/vq-f4.zip -d /root/vq-f4

RUN curl https://pyenv.run | bash
ENV PATH="/root/.pyenv/bin:/root/.pyenv/shims:$PATH"
RUN echo $(pyenv init) >> /root/.bashrc

RUN pyenv install 3.11.5 && pyenv global 3.11.5

ADD fish-pyenv.fish /root/.config/fish/conf.d/fish-pyenv.fish

RUN pip install poetry poet-plugin