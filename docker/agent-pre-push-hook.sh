#!/usr/bin/env bash
# Refuse to push anything but agents/** branches to GitHub.
#
# Repository rulesets enforce this server-side (see ops-scripts/setup-github-rulesets.sh); the hook
# exists to turn that rejection into an immediate, legible local error. Installed into the
# container workspace's .git/hooks/pre-push by launch-agent-container.sh.
set -euo pipefail

case "${2:-}" in
    *github.com*) ;;
    *) exit 0 ;;
esac

while read -r _local_ref _local_sha remote_ref _remote_sha; do
    if [[ "${remote_ref}" != refs/heads/agents/* ]]; then
        echo "pre-push: refusing to push ${remote_ref} to GitHub." >&2
        echo "Only branches under agents/ may be pushed, e.g. agents/$(hostname -s)/my-change." >&2
        exit 1
    fi
done
