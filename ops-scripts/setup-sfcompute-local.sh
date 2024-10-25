#!/usr/bin/env bash

# Set up a session on an SF Compute node. This script runs on the local machine and will create a
# tmux session on the node inside an ssh session.

set -euo pipefail
# Check if both IP address and port are provided
if [ $# -lt 2 ]; then
    echo "Error: Please provide an IP address and port as arguments."
    echo "Usage: $0 <IP_ADDRESS> <SSH_PORT>"
    exit 1
fi

IP_ADDRESS="$1"
SSH_PORT="$2"
SSH_USER="ubuntu"

# SCP the setup script to the remote machine
scp -P "$SSH_PORT" "$(dirname "$0")/setup-sfcompute-remote.sh" "$SSH_USER@$IP_ADDRESS:/home/$SSH_USER/setup-sfcompute-remote.sh"

# SSH into the remote machine and run the setup script
ssh -p "$SSH_PORT" -t "$SSH_USER@$IP_ADDRESS" "bash /home/$SSH_USER/setup-sfcompute-remote.sh"
