import numpy as np


class CausalMaskManager:
    """
    Encapsulates the causal structure of an MDP transition (s, a, s') and
    provides memoized computation of descendant and non-descendant masks.

    Args:
        adjacency_as: binary matrix of shape (n_actions, n_obs).
                      adjacency_as[i, j] = 1 if action dim i directly causes next-state feature j.
        adjacency_ss: binary matrix of shape (n_obs, n_obs).
                      adjacency_ss[i, j] = 1 if state feature i directly causes next-state feature j.
    """

    def __init__(self, adjacency_as: np.ndarray, adjacency_ss: np.ndarray):
        # adjacency_as.shape == (n_actions, n_obs)
        # adjacency_ss.shape == (n_obs, n_obs)
        self._adjacency_as = adjacency_as.astype(bool)
        # self._adjacency_as.shape == (n_actions, n_obs)
        self._adjacency_ss = adjacency_ss.astype(bool)
        # self._adjacency_ss.shape == (n_obs, n_obs)
        self._descendant_mask_per_dim_cache: dict[int, np.ndarray] = {}
        self._descendant_mask_cache: dict[int, np.ndarray] = {}

    def descendant_mask_per_dim(self, horizon: int) -> np.ndarray:
        """
        Boolean mask of shape (n_actions, horizon, obs_dim).
        mask[i, k, j] is True if obs feature j in s_{t+k+1} is a descendant
        of action dimension i, a_{t,i}, for k = 0, ..., horizon-1.
        """
        if horizon not in self._descendant_mask_per_dim_cache:
            d = self._adjacency_as.copy()
            # d.shape == (n_actions, n_obs)
            masks = []
            for _ in range(horizon):
                masks.append(d.copy())
                d = (d @ self._adjacency_ss).astype(bool)
                # d.shape == (n_actions, n_obs)
            self._descendant_mask_per_dim_cache[horizon] = np.stack(masks, axis=1)
            # self._descendant_mask_per_dim_cache[horizon].shape == (n_actions, horizon, n_obs)
        return self._descendant_mask_per_dim_cache[horizon]

    def descendant_mask(self, horizon: int) -> np.ndarray:
        """
        Boolean mask of shape (horizon, obs_dim).
        mask[k, j] is True if obs feature j in s_{t+k+1} is a descendant
        of the joint action a_t, for k = 0, ..., horizon-1.
        """
        if horizon not in self._descendant_mask_cache:
            self._descendant_mask_cache[horizon] = self.descendant_mask_per_dim(horizon).any(axis=0)
            # self._descendant_mask_cache[horizon].shape == (horizon, n_obs)
        return self._descendant_mask_cache[horizon]

    def non_descendant_mask_per_dim(self, horizon: int) -> np.ndarray:
        """
        Boolean mask of shape (n_actions, horizon, obs_dim).
        mask[i, k, j] is True if obs feature j in s_{t+k+1} is not a descendant
        of action dimension i, a_{t,i}, for k = 0, ..., horizon-1.
        """
        return ~self.descendant_mask_per_dim(horizon)

    def non_descendant_mask(self, horizon: int) -> np.ndarray:
        """
        Boolean mask of shape (horizon, obs_dim).
        mask[k, j] is True if obs feature j in s_{t+k+1} is not a descendant
        of the joint action a_t, for k = 0, ..., horizon-1.
        """
        return ~self.descendant_mask(horizon)
