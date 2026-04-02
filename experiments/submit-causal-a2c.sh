#!/bin/bash
sbatch_kwargs=(
    --job-name=halfcheetah-v5:causal-a2c:$1
)
sbatch "${sbatch_kwargs[@]}" experiments/job.sbatch experiments/run-halfcheetah-causal-a2c.py "$@"
