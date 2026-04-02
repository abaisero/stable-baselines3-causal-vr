#!/bin/bash
#SBATCH --job-name=causal-vr
#SBATCH --array=0-4
#SBATCH --partition ava_f.p
#SBATCH --cpus-per-task 8
#SBATCH --gpus 1
#SBATCH --mem 16GB
#SBATCH --output=logs/%A_%a:%x:%a.out

source source-this.sh
python "$@" $SLURM_ARRAY_TASK_ID
