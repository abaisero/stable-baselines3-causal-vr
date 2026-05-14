from launchlib import submit

submit(
    script="run-tracking-causal-a2c.py",
    seeds=range(5),
    slurm={
        "job_name_prefix": "tracking-causal-a2c",
        "time": "00:30:00",
    },
)
