import argparse

import wandb
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.a2c.causal_a2c import CausalA2C
from stable_baselines3.common.causal_env import CausalEnv
from stable_baselines3.common.causal_policies import CausalActorCriticPolicy
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs.causal_half_cheetah import CausalHalfCheetah

N_ENVS = 4
N_STEPS = 256
TOTAL_TIMESTEPS = 10_000_000

VERBOSE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_mask_manager(env_class: type[CausalEnv]) -> CausalMaskManager:
    _env = env_class()
    mask_manager = CausalMaskManager(_env.adjacency_as(), _env.adjacency_ss())
    _env.close()
    return mask_manager


def main():
    args = parse_args()

    env = "halfcheetah-v5"
    algo = "causal-a2c"
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
        venv = make_vec_env(CausalHalfCheetah, n_envs=N_ENVS)
        mask_manager = make_mask_manager(CausalHalfCheetah)

        model = CausalA2C(
            CausalActorCriticPolicy,
            venv,
            n_steps=N_STEPS,
            ent_coef=args.ent_coef,
            verbose=VERBOSE,
            seed=seed,
            tensorboard_log=f"runs/{run.id}",
            policy_kwargs={"mask_manager": mask_manager},
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(verbose=2),
        )
        venv.close()


if __name__ == "__main__":
    main()
