#!/usr/bin/env python
from launchlib import submit

submit(
    script="run-tracking-a2c.py",
    seeds=range(500),
    slurm={
        "job_name_prefix": "tracking-a2c",
        "time": "00:30:00",
    },
)
