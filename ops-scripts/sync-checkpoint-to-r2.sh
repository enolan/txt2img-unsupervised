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

if [[ ! -d "$checkpoint_path" ]]; then
    echo "Error: The checkpoint path must be a directory."
    echo "Usage: $0 checkpoints/foo"
    exit 1
fi

if [[ "$checkpoint_path" == "checkpoints/" || "$checkpoint_path" == "checkpoints" ]]; then
    echo "Error: The checkpoint path must include a subdirectory after 'checkpoints/'."
    echo "Usage: $0 checkpoints/foo"
    exit 1
fi

if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' is not installed. Please install it to continue."
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

last_hash=""
sync_if_changed() {
    # Create a list of every file along with its mod time, excluding temporary Orbax files
    current_hash=$(find "$checkpoint_path" -type f -not -path '*.orbax-checkpoint-tmp-*' -printf '%P %T@\n' | sort | sha1sum | awk '{print $1}')

    # If there are new files, files have been deleted, or files have been modified, the hash will
    # have changed, so we sync.
    if [ "$current_hash" != "$last_hash" ]; then
        echo "Changes detected. Syncing at $(date)"
        while true; do
            if rclone sync -P --fast-list --transfers 24 "$checkpoint_path" "$rclone_path"; then
                last_hash=$current_hash
                break
            else
                echo "Sync failed. Retrying immediately..."
            fi
        done
    fi
}

if $once_mode; then
    sync_if_changed
else
    echo "Starting continuous sync process..."
    echo "Press Ctrl+C to stop the sync process."
    while true; do
        sync_if_changed
        sleep 5s
    done
fi
