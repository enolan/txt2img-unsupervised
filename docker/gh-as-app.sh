#!/usr/bin/env bash
# Run gh authenticated as the repository's GitHub App.
#
# launch-agent-container.sh mounts this ahead of the real gh on PATH, so it applies whatever shell
# an agent happens to use - fish interactively, bash for tool calls. A token is minted per
# invocation rather than exported once, so shells that outlive an hour never hold an expired one.
#
# GH_REPO is set because the workspace's origin is the read-only host checkout, which gh can't
# resolve to a GitHub repository on its own.
set -euo pipefail

GH_TOKEN="$(/repo-readonly/docker/mint-github-token.sh)"
export GH_TOKEN
export GH_REPO="${GH_REPO:-${GITHUB_APP_REPO}}"
exec /usr/bin/gh "$@"
