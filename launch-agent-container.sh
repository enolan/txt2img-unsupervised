#!/usr/bin/env bash
# Launch a Docker container for AI coding agents with GPU access.
#
# Creates a container workspace at <repo>-docker[-<name>]/ on first run (a git clone of this repo).
# The host repo is mounted read-only at /repo-readonly.
# Use --name <name> to run multiple independent containers with separate git checkouts.
# Pass extra args or override the default fish shell: ./launch-agent-container.sh [--name foo] bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "${REPO_DIR}")"
IMAGE_NAME="txt2img-unsupervised-dev"
HOST_UID="$(id -u)"
PYTHON_VERSION="$(tr -d '[:space:]' < "${REPO_DIR}/.python-version")"
LOCKFILE="/run/user/${HOST_UID}/claude-gpu-lock"

# Parse --name argument; remaining args are passed to docker run.
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

DOCKER_ARGS=(
    -it --rm --gpus all --ipc=host --name "${CONTAINER_NAME}"
    -v "${REPO_DIR}:/repo-readonly:ro"
    -v "${WORKSPACE}:/home/devuser/txt2img-unsupervised"
    -v "${LOCKFILE}:${LOCKFILE}"
    -v txt2img-uv-cache:/home/devuser/.cache/uv
    -v "$HOME/.claude:/home/devuser/.claude"
    -v "$HOME/.claude.json:/home/devuser/.claude.json"
    -v "$HOME/.codex:/home/devuser/.codex"
)
[ -f "$HOME/.gitconfig" ] && DOCKER_ARGS+=(-v "$HOME/.gitconfig:/home/devuser/.gitconfig:ro")

exec docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}" "$@"
