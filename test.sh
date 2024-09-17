#!/usr/bin/env bash
set -euo pipefail

# Check if -x option is provided
if [[ "$*" == *"-x"* ]]; then
    PYTEST_ARGS="-x"
else
    PYTEST_ARGS=""
fi

poetry run pytest $PYTEST_ARGS -vs txt2img_unsupervised/*.py captree_sweep.py concat_pqs.py \
  dedup_by_clip.py find_by_clip.py find_by_name.py gen_captree.py merge_capexamples.py \
  process_imgur.py rand_imgs_json.py