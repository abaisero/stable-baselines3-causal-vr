from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np


class CausalEnv(gym.Env, ABC):
    """
    Abstract base class for gym environments that also expose the causal
    structure of the MDP as a Structural Causal Model (SCM) over a single
    (s, a, s') transition.

    Subclasses must implement step, reset (from gym.Env) and adjacency_as,
    adjacency_ss. Obs feature indices refer to positions in the flat
    observation vector.
    """

    @abstractmethod
    def adjacency_as(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (n_actions, n_obs).
        adjacency_as[i, j] = 1 if action dimension i directly causes next-state feature j.
        """

    @abstractmethod
    def adjacency_ss(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (n_obs, n_obs).
        adjacency_ss[i, j] = 1 if state feature i directly causes next-state feature j.
        """
