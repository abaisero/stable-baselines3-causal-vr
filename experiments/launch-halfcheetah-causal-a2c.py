from launchlib import submit

submit(
    script="run-halfcheetah-causal-a2c.py",
    seeds=range(5),
    slurm={
        "job_name_prefix": "halfcheetah-causal-a2c",
    },
)
