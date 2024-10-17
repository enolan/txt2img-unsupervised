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

last_mod_time=0
sync_if_changed() {
    # Get the latest mod time of a file that isn't one of the temporary files orbax makes while
    # checkpointing is happening.
    current_mod_time=$(find "$checkpoint_path" -type f -not -path '*.orbax-checkpoint-tmp-*' -printf '%T@\n' | sort -n | tail -1)

    if (( $(echo "$current_mod_time > $last_mod_time" | bc -l) )); then
        echo "Changes detected. Syncing at $(date)"
        while true; do
            if rclone sync -P --fast-list --transfers 24 "$checkpoint_path" "$rclone_path"; then
                last_mod_time=$current_mod_time
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
