#!/usr/bin/env bash
# Create the repository rulesets that keep coding agents from breaking anything on GitHub.
#
# Agents authenticate as a GitHub App with write access to contents and pull requests on this
# repository alone (see docker/mint-github-token.sh). That is enough to push branches and open pull
# requests - and, since merging a pull request is just a write to its target branch, it would
# otherwise be enough to merge them too. No token scope can separate the two, so these rulesets are
# what makes merging human-only. The app is deliberately absent from every bypass list, leaving it
# able to create and update agents/** branches and nothing else: master, every other branch and
# every tag can only be written by the repository owner.
#
# Idempotent - rulesets that already exist are left alone. Run it with the gh CLI authenticated as
# the owner of the repository.
set -euo pipefail

REPO="${1:-enolan/txt2img-unsupervised}"
OWNER_ID="$(gh api user --jq .id)"
EXISTING="$(gh api "repos/${REPO}/rulesets" --jq '.[].name')"

# Takes a ruleset as JSON on stdin and creates it unless the repository already has one by that
# name.
create_ruleset() {
    local ruleset name
    ruleset="$(cat)"
    name="$(jq -r .name <<<"${ruleset}")"
    if grep -qxF "${name}" <<<"${EXISTING}"; then
        echo "ruleset '${name}' already exists, leaving it alone"
        return
    fi
    gh api --method POST "repos/${REPO}/rulesets" --input - <<<"${ruleset}" >/dev/null
    echo "created ruleset '${name}'"
}

# Deliberately redundant with agent-branches-only below: the gate on merging is important enough
# that widening the branch fence later should not be able to open it by accident.
create_ruleset <<EOF
{
  "name": "human-only-master",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [{"actor_id": ${OWNER_ID}, "actor_type": "User", "bypass_mode": "always"}],
  "conditions": {"ref_name": {"include": ["refs/heads/master"], "exclude": []}},
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}, {"type": "non_fast_forward"}]
}
EOF

create_ruleset <<EOF
{
  "name": "agent-branches-only",
  "target": "branch",
  "enforcement": "active",
  "bypass_actors": [{"actor_id": ${OWNER_ID}, "actor_type": "User", "bypass_mode": "always"}],
  "conditions": {"ref_name": {"include": ["~ALL"], "exclude": ["refs/heads/agents/**"]}},
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}, {"type": "non_fast_forward"}]
}
EOF

create_ruleset <<EOF
{
  "name": "human-only-tags",
  "target": "tag",
  "enforcement": "active",
  "bypass_actors": [{"actor_id": ${OWNER_ID}, "actor_type": "User", "bypass_mode": "always"}],
  "conditions": {"ref_name": {"include": ["~ALL"], "exclude": []}},
  "rules": [{"type": "creation"}, {"type": "update"}, {"type": "deletion"}, {"type": "non_fast_forward"}]
}
EOF
