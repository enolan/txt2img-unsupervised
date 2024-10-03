#!/usr/bin/env bash

set -euo pipefail

echo "Building Vast.ai image"
docker build \
  -t enolan/txt2img-unsupervised-vast \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 \
  "$(dirname "$0")"

echo "Building RunPod image"
docker build \
  -t enolan/txt2img-unsupervised-runpod \
  --build-arg BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  "$(dirname "$0")"

for image in enolan/txt2img-unsupervised-vast enolan/txt2img-unsupervised-runpod; do
  echo "Pushing $image"
  docker push $image
done
