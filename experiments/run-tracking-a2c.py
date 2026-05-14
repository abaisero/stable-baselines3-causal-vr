import argparse

import wandb
from wandb.integration.sb3 import WandbCallback

import stable_baselines3.common.envs  # noqa: F401  # registers CausalTracking-* envs
from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

N_ENVS = 4
N_STEPS = 128
TOTAL_TIMESTEPS = 1_000_000
MAX_EPISODE_STEPS = 100

VERBOSE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    env = "CausalTracking-TransitionRewards-v0"
    algo = "a2c"
    experiment = args.experiment
    seed = args.seed

    hp_str = f"ec{args.ent_coef}"
    wandb_group = f"{env}:{algo}:{experiment}:{hp_str}"
    wandb_name = f"{wandb_group}:s{seed}"
    wandb_tags = [f"env={env}", f"algo={algo}", f"sweep={experiment}"]
    wandb_mode = "disabled" if args.dry_run else "online"

    config = {
        "n_envs": N_ENVS,
        "n_steps": N_STEPS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "env": env,
        "algo": algo,
        "experiment": experiment,
        "seed": seed,
        "ent_coef": args.ent_coef,
    }

    with wandb.init(
        name=wandb_name,
        group=wandb_group,
        config=config,
        tags=wandb_tags,
        sync_tensorboard=True,
        save_code=True,
        mode=wandb_mode,
    ) as run:
        venv = make_vec_env(
            env,
            n_envs=N_ENVS,
            env_kwargs={"max_episode_steps": MAX_EPISODE_STEPS},
        )

        model = A2C(
            ActorCriticPolicy,
            venv,
            n_steps=N_STEPS,
            ent_coef=args.ent_coef,
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
