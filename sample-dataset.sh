#!/bin/bash

# Set variables
REMOTE="r2:txt2img-unsupervised-dataset/preprocessed/capexamples/128x128-randomcrops-merged-2caps"
LOCAL_DIR="./random_sample"
SAMPLE_SIZE=100

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# List all files, shuffle them, and take the first 100
files=$(rclone lsf "$REMOTE" | shuf -n "$SAMPLE_SIZE")

# Download each file
count=0
total="$SAMPLE_SIZE"
for file in $files; do
    count=$((count + 1))
    echo "Downloading file $count of $total: $file"
    rclone copy "$REMOTE/$file" "$LOCAL_DIR" --progress
done

echo "Download complete. $SAMPLE_SIZE files have been downloaded to $LOCAL_DIR"
