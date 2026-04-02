#!/bin/bash
#SBATCH --job-name=causal-vr
#SBATCH --array=0-4
#SBATCH --time 08:00:00
#SBATCH --partition ava_f.p
#SBATCH --nodelist ava-f0
#SBATCH --cpus-per-task 8
#SBATCH --gpus 1
#SBATCH --mem 16GB
#SBATCH --output=logs/%A_%a:%x:%a.out

python "$@" $SLURM_ARRAY_TASK_ID
