#!/usr/bin/env bash
# Print a GitHub App installation token for this repository, minting a new one when needed.
#
# Runs inside the agent container, where it is the only source of GitHub credentials: git uses it
# through a credential helper and `gh` through the PATH shim in gh-as-app.sh. The app grants write
# access to contents and pull requests on one repository and nothing else, and repository rulesets
# confine it to `agents/**` branches, so a token cannot touch master, other branches, tags, or
# repository settings. See ops-scripts/setup-github-rulesets.sh.
#
# Note what this means for the key itself: anything running in the container can mint tokens for as
# long as the key is valid, so revoking an agent's access means rotating the key in the app
# settings, not just removing the container. The rulesets are what bound the damage.
#
# Tokens last an hour; we cache one and re-mint with ten minutes to spare, so the frequent calls
# from git and gh cost nothing.
set -euo pipefail

: "${GITHUB_APP_ID:?not set - the container was launched without the GitHub App configured}"
: "${GITHUB_APP_REPO:?not set - the container was launched without the GitHub App configured}"
PEM="${GITHUB_APP_PEM:-/home/devuser/.github-app.pem}"
CACHE="${XDG_RUNTIME_DIR:-/tmp}/github-app-token"
CACHE_LIFETIME_SECONDS=3000

if [ -s "${CACHE}" ] && [ "$(( $(date +%s) - $(stat -c %Y "${CACHE}") ))" -lt "${CACHE_LIFETIME_SECONDS}" ]; then
    cat "${CACHE}"
    exit 0
fi

# A JWT signed with the app's private key authenticates us as the app itself, which is only good for
# looking up installations and exchanging them for installation tokens.
b64url() { openssl base64 -A | tr '+/' '-_' | tr -d '='; }
now="$(date +%s)"
header="$(printf '{"alg":"RS256","typ":"JWT"}' | b64url)"
payload="$(printf '{"iat":%d,"exp":%d,"iss":"%s"}' "$((now - 60))" "$((now + 540))" "${GITHUB_APP_ID}" | b64url)"
signature="$(printf '%s.%s' "${header}" "${payload}" | openssl dgst -sha256 -sign "${PEM}" | b64url)"
jwt="${header}.${payload}.${signature}"

app_api() {
    curl -fsS -H "Authorization: Bearer ${jwt}" -H "Accept: application/vnd.github+json" "$@"
}

installation_id="$(app_api "https://api.github.com/repos/${GITHUB_APP_REPO}/installation" | jq -r .id)"
token="$(app_api -X POST "https://api.github.com/app/installations/${installation_id}/access_tokens" | jq -r .token)"

if [ -z "${token}" ] || [ "${token}" = "null" ]; then
    echo "Failed to mint an installation token for ${GITHUB_APP_REPO}" >&2
    exit 1
fi

# Write through a temporary file so an interrupted write can't leave a truncated token cached.
(umask 077; printf '%s' "${token}" > "${CACHE}.new" && mv "${CACHE}.new" "${CACHE}")
printf '%s' "${token}"
