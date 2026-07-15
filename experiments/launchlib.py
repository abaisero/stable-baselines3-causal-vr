"""Submit experiments to SLURM or run them locally.

A launch / sweep file is a Python script that imports `submit` and calls it
once. Run it directly:

    python launch-<name>.py <experiment> [--local] [--dry-run]   # single config
    python sweep-<name>.py  <experiment> [--local] [--dry-run]   # hp grid

`submit` fans out the Cartesian product of `grid` (may be empty), launching
one sbatch job per hp combination. Seeds become SLURM array tasks within
each job. With `--local`, each (hp combination, seed) pair runs sequentially
on this machine instead; the `slurm` settings are ignored.
"""

import argparse
import itertools as itt
import subprocess
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import more_itertools as mitt

EXPERIMENTS_DIR = Path(__file__).resolve().parent
JOB_SBATCH = EXPERIMENTS_DIR / "job.sbatch"


def make_slurm_array_spec(seeds: Iterable[int]) -> str:
    seeds = sorted(set(seeds))
    splits = mitt.split_when(seeds, lambda x, y: y - x > 1)
    splits = [(split[0], split[-1]) for split in splits]
    specs = ",".join(f"{first}-{last}" if first != last else str(first) for (first, last) in splits)

    return specs


def submit(
    *,
    script: str,
    grid: Mapping[str, Iterable[Any]] | None = None,
    seeds: Iterable[int],
    slurm: Mapping[str, Any] | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="sweep label")
    parser.add_argument("--local", action="store_true", help="run sequentially on this machine instead of submitting to SLURM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment

    grid = {} if grid is None else dict(grid)
    slurm = {} if slurm is None else dict(slurm)
    array_spec = make_slurm_array_spec(seeds)

    script_path = EXPERIMENTS_DIR / script
    hp_combos = list(itt.product(*grid.values())) if grid else [()]

    for combo in hp_combos:
        params = dict(zip(grid.keys(), combo))
        hp_flags = [f"--{k.replace('_', '-')}={v}" for k, v in params.items()]
        hp_suffix = ":".join(f"{k}{v}" for k, v in params.items())

        run_args = [str(script_path), experiment]
        if hp_suffix:
            run_args.append(f"--hp-suffix={hp_suffix}")
        run_args += hp_flags

        if args.local:
            # Seed goes last, matching job.sbatch (`python "$@" $SLURM_ARRAY_TASK_ID`).
            for seed in sorted(set(seeds)):
                cmd = [sys.executable, *run_args, str(seed)]
                print(" ".join(cmd))
                if not args.dry_run:
                    subprocess.run(cmd, check=True)
        else:
            job_name = f"{slurm.get('job_name_prefix', 'sweep')}:{experiment}"
            if hp_suffix:
                job_name += f":{hp_suffix}"

            cmd = ["sbatch", f"--job-name={job_name}", f"--array={array_spec}"]
            if "time" in slurm:
                cmd.append(f"--time={slurm['time']}")
            cmd += [str(JOB_SBATCH), *run_args]

            print(" ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
