import argparse

import numpy as np
import wandb
from wandb.integration.sb3 import WandbCallback

import stable_baselines3.common.envs  # noqa: F401  # registers CausalTracking-* envs
from stable_baselines3.a2c.causal_a2c import CausalA2C
from stable_baselines3.common.causal_env import CausalEnv
from stable_baselines3.common.causal_policies import CausalActorCriticPolicy
from stable_baselines3.common.causal_utils import CausalMaskManager
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.envs.causal_tracking import CausalTrackingEnv
from stable_baselines3.common.vec_env import VecNormalize

N_ENVS = 4
N_STEPS = 128
TOTAL_TIMESTEPS = 2_000_000
MAX_EPISODE_STEPS = 100

GAMMA = 0.5
SIGMA_Z = 2.0

VERBOSE = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("--hp-suffix", type=str, default=None)
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

    env = "CausalTracking-TransitionRewards-v0"
    algo = "causal-a2c"
    experiment = args.experiment
    seed = args.seed

    wandb_group = f"{env}:{algo}:{experiment}"
    if args.hp_suffix is not None:
        wandb_group += f":{args.hp_suffix}"
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
            env_kwargs={"max_episode_steps": MAX_EPISODE_STEPS, "sigma_z": SIGMA_Z},
        )
        # Tracking obs (x, y) are an unbounded random walk; normalize observations so the
        # critics (especially the causal critic, which sees y's full future window) get
        # stable-scale inputs. Reward is left unnormalized to keep return/EV diagnostics
        # directly comparable across runs.
        # clip_obs=inf: clipping would alias large y excursions (partial observability).
        # venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=np.inf)
        mask_manager = make_mask_manager(CausalTrackingEnv)

        model = CausalA2C(
            CausalActorCriticPolicy,
            venv,
            n_steps=N_STEPS,
            gamma=GAMMA,
            ent_coef=args.ent_coef,
            verbose=VERBOSE,
            seed=seed,
            tensorboard_log=f"runs/{run.id}",
            policy_kwargs={"mask_manager": mask_manager},
        )
        run.config.update(model.policy.param_counts())

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=WandbCallback(verbose=2),
        )
        venv.close()


if __name__ == "__main__":
    main()
