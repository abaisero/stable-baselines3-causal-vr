import gymnasium as gym

from stable_baselines3.common.envs.bit_flipping_env import BitFlippingEnv
from stable_baselines3.common.envs.identity_env import (
    FakeImageEnv,
    IdentityEnv,
    IdentityEnvBox,
    IdentityEnvMultiBinary,
    IdentityEnvMultiDiscrete,
)
from stable_baselines3.common.envs.causal_tracking import CausalTrackingEnv
from stable_baselines3.common.envs.multi_input_envs import SimpleMultiObsEnv

gym.register(
    id="CausalTracking-StateActionRewards-v0",
    entry_point="stable_baselines3.common.envs.causal_tracking:CausalTrackingEnv",
    max_episode_steps=500,
    kwargs={"reward_type": "state-action"},
)

gym.register(
    id="CausalTracking-TransitionRewards-v0",
    entry_point="stable_baselines3.common.envs.causal_tracking:CausalTrackingEnv",
    max_episode_steps=500,
    kwargs={"reward_type": "transition"},
)

__all__ = [
    "CausalTrackingEnv",
    "BitFlippingEnv",
    "FakeImageEnv",
    "IdentityEnv",
    "IdentityEnvBox",
    "IdentityEnvMultiBinary",
    "IdentityEnvMultiDiscrete",
    "SimpleMultiObsEnv",
    "SimpleMultiObsEnv",
]
