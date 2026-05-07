#!/bin/bash
set -euo pipefail

POSITIONAL_ARGS=()
for arg in "$@"; do
    [[ $arg == --* ]] || POSITIONAL_ARGS+=("$arg")
done

EXPERIMENT="${POSITIONAL_ARGS[0]:?"Usage: $0 [options] <experiment>"}"

sbatch_kwargs=(
    --job-name=tracking-v0:causal-a2c:"$EXPERIMENT"
)
sbatch "${sbatch_kwargs[@]}" job.sbatch run-tracking-causal-a2c.py "$@"
