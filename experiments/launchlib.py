"""Submit experiments to SLURM.

A launch / sweep file is a Python script that imports `submit` and calls it
once. Run it directly:

    python launch-<name>.py <experiment> [--dry-run]   # single config
    python sweep-<name>.py  <experiment> [--dry-run]   # hp grid

`submit` fans out the Cartesian product of `grid` (may be empty), launching
one sbatch job per hp combination. Seeds become SLURM array tasks within
each job.
"""
import argparse
import itertools
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

EXPERIMENTS_DIR = Path(__file__).resolve().parent
JOB_SBATCH = EXPERIMENTS_DIR / "job.sbatch"


def submit(
    *,
    script: str,
    grid: Mapping[str, Iterable[Any]] | None = None,
    seeds: Iterable[int],
    slurm: Mapping[str, Any] | None = None,
) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="sweep label")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = args.experiment

    grid = dict(grid or {})
    slurm = dict(slurm or {})
    seeds = list(seeds)

    if seeds == list(range(min(seeds), max(seeds) + 1)):
        array_spec = f"{min(seeds)}-{max(seeds)}"
    else:
        array_spec = ",".join(map(str, seeds))

    script_path = EXPERIMENTS_DIR / script
    hp_combos = list(itertools.product(*grid.values())) if grid else [()]

    for combo in hp_combos:
        params = dict(zip(grid.keys(), combo))
        hp_flags = [f"--{k.replace('_', '-')}={v}" for k, v in params.items()]
        hp_suffix = ":".join(f"{k}{v}" for k, v in params.items())

        job_name = f"{slurm.get('job_name_prefix', 'sweep')}:{experiment}"
        if hp_suffix:
            job_name += f":{hp_suffix}"

        cmd = ["sbatch", f"--job-name={job_name}", f"--array={array_spec}"]
        if "time" in slurm:
            cmd.append(f"--time={slurm['time']}")
        cmd += [str(JOB_SBATCH), str(script_path), experiment, *hp_flags]

        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
