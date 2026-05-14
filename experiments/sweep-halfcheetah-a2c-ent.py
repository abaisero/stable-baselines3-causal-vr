from launchlib import submit

submit(
    script="run-halfcheetah-a2c.py",
    grid={
        "ent_coef": [0.0, 1e-4, 1e-3, 1e-2, 1e-1],
    },
    seeds=range(5),
    slurm={
        "job_name_prefix": "halfcheetah-a2c",
    },
)
