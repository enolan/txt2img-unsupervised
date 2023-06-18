#!/usr/bin/env bash

set -euo pipefail


if [ "$#" -ne 3 ]; then
  echo "Error: This script requires three arguments, the name of the TPU VM to create, the GCP zone, and the accelerator type."
  exit 1
fi

TPU_VM_NAME=$1
ZONE=$2
ACCELERATOR_TYPE=$3

this_dir=$(dirname "$0")

if [ ! -f "$this_dir"/tpu-rclone.conf ]; then
  echo "Error: rclone config not found at $this_dir/tpu-rclone.conf"
  echo "You either need credentials to get the dataset or you need to use another one."
  exit 1
fi

gcloud compute tpus tpu-vm create "$TPU_VM_NAME" --zone "$ZONE" \
  --accelerator-type "$ACCELERATOR_TYPE" --version tpu-vm-base \
  --metadata-from-file=startup-script="$this_dir"/tpu-setup.sh,rclone_config="$this_dir"/tpu-rclone.conf,txt2img-user-setup="$this_dir"/tpu-setup-user.sh,netrc="$this_dir"/wandb-netrc

echo "VM up, waiting 90s for root setup script to run."
sleep 90

echo "Running user setup script"
gcloud compute tpus tpu-vm ssh "$TPU_VM_NAME" --zone "$ZONE" -- txt2img-user-setup

echo "Opening ssh session to TPU VM"
gcloud compute tpus tpu-vm ssh "$TPU_VM_NAME" --zone "$ZONE" -- tmux
