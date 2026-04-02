import gymnasium as gym
import numpy as np

from stable_baselines3.common.causal_env import CausalEnv

# Axes convention:
#   x: forward direction (direction of locomotion)
#   y: lateral direction (perpendicular to locomotion, into the screen for 2D)
#   z: vertical direction (up)
#
# HalfCheetah-v4 observation space (17 dims):
#   0:  rootz — absolute linear position along z (torso height)
#   1:  rooty — absolute angular position around y (torso pitch)
#   2:  bthigh (back-thigh)  — relative angular position w.r.t. torso
#   3:  bshin (back-shin)    — relative angular position w.r.t. bthigh
#   4:  bfoot (back-foot)    — relative angular position w.r.t. bshin
#   5:  fthigh (front-thigh) — relative angular position w.r.t. torso
#   6:  fshin (front-shin)   — relative angular position w.r.t. fthigh
#   7:  ffoot (front-foot)   — relative angular position w.r.t. fshin
#   8:  rootx — absolute linear velocity along x (forward speed) — main reward driver
#   9:  rootz — absolute linear velocity along z (vertical speed)
#   10: rooty — absolute angular velocity around y (pitch rate)
#   11: bthigh (back-thigh)  — relative angular velocity
#   12: bshin (back-shin)    — relative angular velocity
#   13: bfoot (back-foot)    — relative angular velocity
#   14: fthigh (front-thigh) — relative angular velocity
#   15: fshin (front-shin)   — relative angular velocity
#   16: ffoot (front-foot)   — relative angular velocity
#
# Action space (6 dims):
#   0: bthigh (back-thigh)  torque
#   1: bshin (back-shin)    torque
#   2: bfoot (back-foot)    torque
#   3: fthigh (front-thigh) torque
#   4: fshin (front-shin)   torque
#   5: ffoot (front-foot)   torque

N_ACTIONS = 6
N_OBS = 17


class CausalHalfCheetah(gym.Wrapper, CausalEnv):
    """
    Causal wrapper for HalfCheetah-v5.

    adjacency_as and adjacency_ss encode the SCM over a single (s, a, s')
    transition based on the physical causal structure of HalfCheetah.
    """

    # -------------------------------------------------------------------------
    # TEMPORARY DEBUG HOOK — NOT FOR TRAINING USE
    # Set _debug_reset_prob to a value in (0, 1) to inject random early episode
    # resets, making it easier to verify future_lengths / future_mask computation.
    # Must be 0.0 (disabled) for any real experiment.
    # -------------------------------------------------------------------------
    _debug_reset_prob: float = 0.0

    def __init__(self, **kwargs):
        env = gym.make("HalfCheetah-v5", **kwargs)
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # TEMPORARY DEBUG: inject random resets — remove before real training.
        if not terminated and self._debug_reset_prob > 0.0 and np.random.rand() < self._debug_reset_prob:
            terminated = True
        return obs, reward, terminated, truncated, info

    def adjacency_as(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (6, 17).
        adjacency_as[i, j] = 1 if action dim i directly causes next-state feature j.

        Computed by scripts/compute_causal_structure.py using MuJoCo's mjd_transitionFD
        over 1000 random states (REL_THRESHOLD=1e-3 per column).
        Rows: actions (bthigh, bshin, bfoot, fthigh, fshin, ffoot torques).
        Cols: obs (rootz_pos, rooty_pos, bthigh_pos, bshin_pos, bfoot_pos,
                   fthigh_pos, fshin_pos, ffoot_pos,
                   rootx_vel, rootz_vel, rooty_vel,
                   bthigh_vel, bshin_vel, bfoot_vel,
                   fthigh_vel, fshin_vel, ffoot_vel).
        """
        return np.array([
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bthigh_torque
            [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bshin_torque
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bfoot_torque
            [0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fthigh_torque
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fshin_torque
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # ffoot_torque
        ], dtype=np.int8)

    def adjacency_ss(self) -> np.ndarray:
        """
        Binary adjacency matrix of shape (17, 17).
        adjacency_ss[i, j] = 1 if state feature i directly causes next-state feature j.

        Computed by scripts/compute_causal_structure.py using MuJoCo's mjd_transitionFD
        over 1000 random states (REL_THRESHOLD=1e-3 per column).
        Rows and cols follow the same obs ordering as adjacency_as.
        """
        return np.array([
            #rz rp bt bs bf ft fs ff xv zv rv bv sv bfv ftv fsv ffv
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # rootz_pos
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # rooty_pos
            [0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bthigh_pos
            [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bshin_pos
            [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bfoot_pos
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fthigh_pos
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fshin_pos
            [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # ffoot_pos
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # rootx_vel
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # rootz_vel
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # rooty_vel
            [0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bthigh_vel
            [0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bshin_vel
            [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # bfoot_vel
            [0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fthigh_vel
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # fshin_vel
            [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # ffoot_vel
        ], dtype=np.int8)
