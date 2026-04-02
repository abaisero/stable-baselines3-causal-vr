#!/bin/bash
set -euo pipefail

POSITIONAL_ARGS=()
for arg in "$@"; do
    [[ $arg == --* ]] || POSITIONAL_ARGS+=("$arg")
done

EXPERIMENT="${POSITIONAL_ARGS[0]:?"Usage: $0 [options] <experiment>"}"

sbatch_kwargs=(
    --job-name=halfcheetah-v5:a2c:"$EXPERIMENT"
)
sbatch "${sbatch_kwargs[@]}" job.sbatch run-halfcheetah-a2c.py "$@"
