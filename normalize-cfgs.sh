#!/usr/bin/env bash

set -euo pipefail

for cfg in configs/model/*.json; do
  tmpfile=$(mktemp)
  jq -S . "$cfg" > "$tmpfile" && mv "$tmpfile" "$cfg"
done