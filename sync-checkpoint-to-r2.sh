#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Error: Please provide a path as an argument."
    echo "Usage: $0 <path_to_checkpoint>"
    exit 1
fi

once_mode=false
if [ "$1" = "--once" ]; then
    once_mode=true
    shift
fi

checkpoint_path="$1"

if [[ ! "$checkpoint_path" =~ ^checkpoints/ ]]; then
    echo "Error: The checkpoint path must start with 'checkpoints/'."
    echo "Usage: $0 checkpoints/foo"
    exit 1
fi

if [[ "$checkpoint_path" == "checkpoints/" || "$checkpoint_path" == "checkpoints" ]]; then
    echo "Error: The checkpoint path must include a subdirectory or file after 'checkpoints/'."
    echo "Usage: $0 checkpoints/foo"
    exit 1
fi

rclone_path="r2-ckpt:txt2img-unsupervised-checkpoints/$(basename "$checkpoint_path")"


echo "Will sync $checkpoint_path to $rclone_path"

read -r -p "Do you want to proceed with the sync? (y/n) " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Sync cancelled."
    exit 0
fi

echo "Starting sync process..."

sync_function() {
    echo "Syncing at $(date)"
    rclone sync -P --fast-list --size-only --transfers 24 "$checkpoint_path" "$rclone_path"
}

if $once_mode; then
    sync_function
else
    while true; do
        sync_function
        sleep 15m
    done
fi
