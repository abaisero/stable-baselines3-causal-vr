from launchlib import submit

submit(
    script="run-halfcheetah-a2c.py",
    seeds=range(5),
    slurm={
        "job_name_prefix": "halfcheetah-a2c",
    },
)
