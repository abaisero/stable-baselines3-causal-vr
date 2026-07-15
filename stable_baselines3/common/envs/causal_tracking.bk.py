import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3.common.causal_env import CausalEnv

# Toy tracking environment that cleanly illustrates the causal RL setting:
#
#   x — controllable: action directly causes x' (action-descendant)
#   y — uncontrollable: pure random walk, independent of action
#
# Task: control x to track y.  Reward penalizes the gap |x - y|.
#
# Why this is maximally illustrative:
#   y's future trajectory is entirely independent of the action, so knowing
#   it in advance perfectly predicts the baseline return.  A causal critic
#   that conditions on y's future collapses all reward variance due to y's
#   random walk.  A standard critic cannot do this because it cannot
#   distinguish causal from non-causal variance sources.
#
# Observation (2 dims):
#   0: x — position of controllable variable
#   1: y — position of uncontrollable target variable
#
# Action (discrete, len(ACTIONS) choices):
#   0: move x left  (-1.0)
#   1: stay         ( 0.0)
#   2: move x right (+1.0)
#
# Dynamics:
#   x' = x + ACTIONS[a] + noise_x,   noise_x ~ N(0, sigma_x)
#   y' = y              + noise_y,   noise_y ~ N(0, sigma_y)
#
# Reward (depends on `reward_type`):
#   "state-action": r = -|x_post_action - y|   [function of (s, a) only]
#   "transition":   r = -|x' - y'|             [function of (s, a, s'); noise leaks in]
#
# Causal structure:
#   adjacency_as:  action -> x only  (action does not affect y)
#   adjacency_ss:  x -> x, y -> y   (independent random walks, no cross-causation)

N_OBS = 2

# Maps discrete action index to displacement applied to x
ACTIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)


class CausalTrackingEnv(CausalEnv):
    """
    Minimal tracking environment for causal RL.

    x (controllable) must track y (uncontrollable random walk).
    The causal critic can condition on y's future to collapse reward variance
    from y's random walk — variance the standard critic cannot eliminate.

    Episode termination is handled externally via TimeLimit wrapper.

    :param sigma_x: std of process noise for x
    :param sigma_y: std of process noise for y (drives the tracking difficulty)
    :param reward_type: "state-action" for R(s, a) — reward computed before noise.
        "transition" for R(s, a, s') — reward computed after noise (next-state-dependent).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        sigma_x: float = 0.1,
        sigma_y: float = 1.0,
        *,
        reward_type: str = "transition",
    ):
        if reward_type not in ("state-action", "transition"):
            raise ValueError(f"reward_type must be 'state-action' or 'transition', got {reward_type!r}")

        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.reward_type = reward_type

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(N_OBS,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTIONS))

        self._state: np.ndarray = np.zeros(N_OBS, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = self.np_random.standard_normal(N_OBS).astype(np.float32)
        # self._state.shape == (N_OBS,)
        return self._state.copy(), {}

    def step(self, action: int):
        action_state = np.array([ACTIONS[action], 0.0], dtype=np.float32)
        # action_state.shape == (N_OBS,)
        self._state = self._state + action_state
        # self._state.shape == (N_OBS,)

        if self.reward_type == "state-action":
            reward = float(-abs(self._state[0] - self._state[1]))

        noise = self.np_random.normal([0.0, 0.0], [self.sigma_x, self.sigma_y]).astype(np.float32)
        # noise.shape == (N_OBS,)
        self._state = self._state + noise
        # self._state.shape == (N_OBS,)

        if self.reward_type == "transition":
            reward = float(-abs(self._state[0] - self._state[1]))

        return self._state.copy(), reward, False, False, {}

    def adjacency_as(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (1, N_OBS).
        adjacency_as[0, j] = 1 if the action directly causes next-state feature j.

        The action (discrete displacement) causes x (index 0) but not y (index 1).
        """
        return np.array(
            [
                #  x  y
                [1, 0],  # displacement -> x only
            ],
            dtype=np.int8,
        )

    def adjacency_ss(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (N_OBS, N_OBS).
        adjacency_ss[i, j] = 1 if state feature i directly causes next-state feature j.

        x and y are independent random walks with no cross-causation.
        """
        return np.array(
            [
                #  x  y
                [1, 0],  # x -> x'
                [0, 1],  # y -> y'
            ],
            dtype=np.int8,
        )
