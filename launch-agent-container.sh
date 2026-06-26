#!/usr/bin/env bash
# Launch a persistent Docker container for AI coding agents with GPU access.
#
# The container runs detached and hosts a tmux server, so your work survives client
# disconnects and Docker daemon restarts: the daemon's live-restore keeps the container's
# processes (and therefore the tmux server and anything running in it) alive while it
# restarts. Re-running this script reattaches to the same container and tmux session
# instead of starting fresh.
#
# Creates a container workspace at <repo>-docker[-<name>]/ on first run (a git clone of this repo).
# The host repo is mounted read-only at /repo-readonly.
# Use --name <name> to run multiple independent containers with separate git checkouts.
# Run a one-off command instead of attaching to tmux: ./launch-agent-container.sh [--name foo] bash
# To rebuild from an updated image, remove the container first: docker rm -f <container-name>
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "${REPO_DIR}")"
IMAGE_NAME="txt2img-unsupervised-dev"
HOST_UID="$(id -u)"
PYTHON_VERSION="$(tr -d '[:space:]' < "${REPO_DIR}/.python-version")"
LOCKFILE="/run/user/${HOST_UID}/claude-gpu-lock"

# Parse --name argument; remaining args are run as a one-off command instead of attaching to tmux.
CONTAINER_NAME="${REPO_NAME}-agent-container"
WORKSPACE="${REPO_DIR}-docker"
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            if [[ $# -lt 2 ]]; then
                echo "Error: --name requires an argument" >&2
                exit 1
            fi
            CONTAINER_NAME="$2"
            WORKSPACE="${REPO_DIR}-docker-${2}"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# Create the container the first time, building the image and workspace as needed. On later runs we
# reuse the existing one so in-flight work is preserved.
if ! docker container inspect "${CONTAINER_NAME}" &>/dev/null; then
    echo "Building image..."
    docker build \
        --build-arg USER_UID="${HOST_UID}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        -t "${IMAGE_NAME}" \
        - < "${REPO_DIR}/docker/Dockerfile.agent"

    if [ ! -d "${WORKSPACE}/.git" ]; then
        echo "Creating container workspace at ${WORKSPACE}..."
        git clone "${REPO_DIR}" "${WORKSPACE}"
        git -C "${WORKSPACE}" submodule init
        git -C "${WORKSPACE}" submodule update --reference "${REPO_DIR}" --dissociate
    fi

    touch "${LOCKFILE}"
    mkdir -p "$HOME/.claude" "$HOME/.codex"
    touch "$HOME/.claude.json"

    # Files always persist on the host through the mounts below, so what we're protecting here is
    # running processes. The daemon's live-restore keeps them alive across a docker daemon restart;
    # a host reboot or crash kills them regardless, and --restart unless-stopped just brings the
    # container back up afterward so you can reattach and restart your work.
    DOCKER_ARGS=(
        -d --init --restart unless-stopped --gpus all --ipc=host --name "${CONTAINER_NAME}"
        -v "${REPO_DIR}:/repo-readonly:ro"
        -v "${WORKSPACE}:/home/devuser/txt2img-unsupervised"
        -v "${LOCKFILE}:${LOCKFILE}"
        -v txt2img-uv-cache:/home/devuser/.cache/uv
        -v "$HOME/.claude:/home/devuser/.claude"
        -v "$HOME/.claude.json:/home/devuser/.claude.json"
        -v "$HOME/.codex:/home/devuser/.codex"
        -v "${REPO_DIR}/docker/tmux.conf:/home/devuser/.tmux.conf:ro"
    )
    [ -f "$HOME/.gitconfig" ] && DOCKER_ARGS+=(-v "$HOME/.gitconfig:/home/devuser/.gitconfig:ro")

    echo "Starting container ${CONTAINER_NAME}..."
    # tail -f /dev/null keeps the container running independently of any client; the tmux server is
    # created lazily on first attach below and reparents to PID 1, so it outlives every docker exec.
    docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" tail -f /dev/null
else
    echo "Reusing existing container ${CONTAINER_NAME} (run 'docker rm -f ${CONTAINER_NAME}' to rebuild from the image)."
    docker start "${CONTAINER_NAME}" >/dev/null
fi

# Run a one-off command if given, otherwise attach to (or create) the persistent tmux session.
if [[ $# -gt 0 ]]; then
    exec docker exec -it -e TERM "${CONTAINER_NAME}" "$@"
else
    exec docker exec -it -e TERM "${CONTAINER_NAME}" tmux new-session -A -s main
fi
