#!/usr/bin/env bash
# Launch a Docker container for AI coding agents with GPU access.
#
# Creates a container workspace at <repo>-docker/ on first run (a git clone of this repo).
# The host repo is mounted read-only at /repo-readonly.
# Pass extra args or override the default fish shell: ./launch-agent-container.sh bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="txt2img-unsupervised-dev"
HOST_UID="$(id -u)"
PYTHON_VERSION="$(tr -d '[:space:]' < "${REPO_DIR}/.python-version")"
WORKSPACE="${REPO_DIR}-docker"
LOCKFILE="/run/user/${HOST_UID}/claude-gpu-lock"

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
    -it --rm --gpus all --ipc=host
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
