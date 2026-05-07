import argparse

import wandb
from wandb.integration.sb3 import WandbCallback

import stable_baselines3.common.envs  # noqa: F401  # registers CausalTracking-v0
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

N_ENVS = 4
N_STEPS = 128
TOTAL_TIMESTEPS = 1_000_000

VERBOSE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    env = "CausalTracking-v0"
    algo = "a2c"
    experiment = args.experiment
    seed = args.seed

    wandb_name = f"{env}:{algo}:{experiment}:{seed}"
    wandb_group = f"{env}:{algo}:{experiment}"
    wandb_mode = "disabled" if args.dry_run else "online"

    config = {
        "n_envs": N_ENVS,
        "n_steps": N_STEPS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "env": env,
        "algo": algo,
        "experiment": experiment,
        "seed": seed,
    }

    with wandb.init(
        name=wandb_name,
        group=wandb_group,
        config=config,
        sync_tensorboard=True,
        save_code=True,
        mode=wandb_mode,
    ) as run:
        venv = make_vec_env(env, n_envs=N_ENVS)

        model = A2C(
            ActorCriticPolicy,
            venv,
            n_steps=N_STEPS,
            verbose=VERBOSE,
            seed=seed,
            tensorboard_log=f"runs/{run.id}",
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(verbose=2),
        )
        venv.close()


if __name__ == "__main__":
    main()
