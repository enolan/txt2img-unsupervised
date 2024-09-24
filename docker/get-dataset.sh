#!/usr/bin/env bash

set -euo pipefail

# Default path
default_r2_path="txt2img-unsupervised-dataset/preprocessed/capexamples/128x128-randomcrops-merged-2caps/"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --r2-path)
            if [[ -n "$2" ]]; then
                r2_path="$2"
                shift 2
            else
                echo "Error: --r2-path requires a non-empty string argument." >&2
                exit 1
            fi
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

# Set r2_path to default if not provided
r2_path=${r2_path:-$default_r2_path}

rclone config

rclone -P --fast-list --transfers 32 copy "r2:$r2_path" /root/dataset

echo "Download done"

SYS_RAM=$(free -b | awk '/Mem:/ {print $2}')
DATASET_SIZE=$(du -sb /root/dataset | awk '{print $1}')

if [ "$SYS_RAM" -gt "$DATASET_SIZE" ]; then
    echo "RAM larger than dataset, caching..."
    vmtouch -tv /root/dataset
else
    echo "RAM smaller than dataset, not caching"
fi
