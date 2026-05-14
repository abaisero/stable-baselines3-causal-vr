from launchlib import submit

submit(
    script="run-tracking-causal-a2c.py",
    grid={
        "ent_coef": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    seeds=range(20),
    slurm={
        "job_name_prefix": "tracking-causal-a2c",
        "time": "00:30:00",
    },
)
