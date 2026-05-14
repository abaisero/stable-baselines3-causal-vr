from launchlib import submit

submit(
    script="run-tracking-a2c.py",
    seeds=range(5),
    slurm={
        "job_name_prefix": "tracking-a2c",
        "time": "00:30:00",
    },
)
