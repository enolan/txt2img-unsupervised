#!/usr/bin/env bash
set -euo pipefail

# Set default pytest arguments
PYTEST_ARGS="-vs"

# Add -x if provided as an argument
if [[ "$*" == *"-x"* ]]; then
    PYTEST_ARGS="-x $PYTEST_ARGS"
fi


uv run pytest $PYTEST_ARGS -vs txt2img_unsupervised/*.py captree_sweep.py concat_pqs.py \
  dedup_by_clip.py find_by_clip.py find_by_name.py gen_captree.py merge_capexamples.py \
  process_imgur.py rand_imgs_json.py sample_dset_imgs.py