#!/bin/bash
sbatch_kwargs=(
    --job-name=halfcheetah-v5:a2c:$1
)
sbatch "${sbatch_kwargs[@]}" experiments/job.sbatch experiments/run-halfcheetah-a2c.py "$@"
