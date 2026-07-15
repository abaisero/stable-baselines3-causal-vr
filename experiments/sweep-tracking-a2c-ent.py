#!/usr/bin/env python
from launchlib import submit

submit(
    script="run-tracking-a2c.py",
    grid={
        "ent_coef": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    seeds=range(20),
    slurm={
        "job_name_prefix": "tracking-a2c",
        "time": "00:30:00",
    },
)
