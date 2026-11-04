#!/bin/bash
# Reset the 90-day purge clock: touch (access + modification time) every file and directory of IG_work on
# SCRATCH and PROJECT. Symlinks are followed, so the files they point to are refreshed too. Run on the login
# node, from anywhere (no GPU, ~1 min for ~17k files). The MAGI-1 training jobs touch SCRATCH on their own
# (touch_scratch in magi1/magi1_env.sh); this is for long pauses without jobs.
#
# Usage:
#   bash jobs/jupiter/touch_storage.sh

set -euo pipefail

DIRS=(
    /e/scratch/e-dev-2026d09-262/ig_work
    /e/project1/e-dev-2026d09-262/ig_work
)

for dir in "${DIRS[@]}"; do
    [[ -d "${dir}" ]] || { echo "missing: ${dir}"; continue; }
    start=$(date -Iseconds)
    find "${dir}" -exec touch -a -m {} +
    stale=$(find -L "${dir}" -not -newermt "${start}" | wc -l)  # what the touch did not reach
    echo "${dir}: $(find "${dir}" | wc -l) entries touched, ${stale} not touched"
done
