"""
Compute the causal adjacency matrices for HalfCheetah-v5 by evaluating
MuJoCo's transition Jacobian (mjd_transitionFD) across many random states.

An entry is considered causally active if it exceeds REL_THRESHOLD times the
maximum entry in its column.  Each column corresponds to one cause (action or
obs feature), and normalizing per-column accounts for the fact that different
causes operate at very different scales (e.g. positions vs velocities).

Usage:
    python scripts/compute_causal_structure.py
"""

import mujoco
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
XML_PATH = "/home/abaisero/venvs/crl-cvr/lib/python3.13/site-packages/gymnasium/envs/mujoco/assets/half_cheetah.xml"
N_SAMPLES = 1000
REL_THRESHOLD = 1e-3   # minimum fraction of the per-column max
RNG_SEED = 42
EPS = 1e-6  # finite-difference step size for mjd_transitionFD

# ──────────────────────────────────────────────────────────────────────────────
# Dimension labels
# ──────────────────────────────────────────────────────────────────────────────
# Full MuJoCo state: qpos (9) + qvel (9) = 18 dims, ordered as follows:
STATE_LABELS = [
    "rootx_pos",   # 0  — excluded from obs
    "rootz_pos",   # 1
    "rooty_pos",   # 2
    "bthigh_pos",  # 3
    "bshin_pos",   # 4
    "bfoot_pos",   # 5
    "fthigh_pos",  # 6
    "fshin_pos",   # 7
    "ffoot_pos",   # 8
    "rootx_vel",   # 9
    "rootz_vel",   # 10
    "rooty_vel",   # 11
    "bthigh_vel",  # 12
    "bshin_vel",   # 13
    "bfoot_vel",   # 14
    "fthigh_vel",  # 15
    "fshin_vel",   # 16
    "ffoot_vel",   # 17
]

# Observation indices into the full 18-dim state (rootx_pos at index 0 is excluded)
OBS_TO_STATE = list(range(1, 18))
OBS_LABELS = [STATE_LABELS[i] for i in OBS_TO_STATE]
N_OBS = len(OBS_LABELS)  # 17

ACTION_LABELS = [
    "bthigh_torque",  # 0
    "bshin_torque",   # 1
    "bfoot_torque",   # 2
    "fthigh_torque",  # 3
    "fshin_torque",   # 4
    "ffoot_torque",   # 5
]
N_ACTIONS = len(ACTION_LABELS)  # 6


# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)
rng = np.random.default_rng(RNG_SEED)

nv = model.nv  # 9 — size of velocity / tangent space
nu = model.nu  # 6 — number of actuators

# Accumulate max absolute Jacobian values across all sampled states.
max_abs_A = np.zeros((2 * nv, 2 * nv))  # ∂s'/∂s, shape (18, 18)
# max_abs_A.shape == (18, 18)
max_abs_B = np.zeros((2 * nv, nu))      # ∂s'/∂a, shape (18, 6)
# max_abs_B.shape == (18, 6)

# ──────────────────────────────────────────────────────────────────────────────
# Sample random states and accumulate Jacobians
# ──────────────────────────────────────────────────────────────────────────────
for _ in range(N_SAMPLES):
    mujoco.mj_resetData(model, data)

    # Perturb qpos and qvel with noise typical of gymnasium's reset_noise_scale=0.1
    data.qpos[:] += rng.uniform(-0.1, 0.1, size=model.nq)
    data.qvel[:] += rng.normal(0, 0.1, size=model.nv)
    data.ctrl[:] = rng.uniform(-1, 1, size=nu)

    mujoco.mj_forward(model, data)

    A = np.zeros((2 * nv, 2 * nv))
    # A.shape == (18, 18)
    B = np.zeros((2 * nv, nu))
    # B.shape == (18, 6)

    mujoco.mjd_transitionFD(model, data, EPS, True, A, B, None, None)

    np.maximum(max_abs_A, np.abs(A), out=max_abs_A)
    np.maximum(max_abs_B, np.abs(B), out=max_abs_B)

# ──────────────────────────────────────────────────────────────────────────────
# Extract obs-space submatrices
# ──────────────────────────────────────────────────────────────────────────────

# B_obs[obs_j, action_i] = max |∂obs'[j] / ∂action[i]| over all sampled states
B_obs = max_abs_B[np.ix_(OBS_TO_STATE, range(N_ACTIONS))]
# B_obs.shape == (N_OBS=17, N_ACTIONS=6)

# A_obs[obs_j, obs_i] = max |∂obs'[j] / ∂obs[i]| over all sampled states
A_obs = max_abs_A[np.ix_(OBS_TO_STATE, OBS_TO_STATE)]
# A_obs.shape == (N_OBS=17, N_OBS=17)

# Binarize per-column: entry is active if it exceeds REL_THRESHOLD * column_max.
# Columns correspond to causes (actions / obs features); each cause has its own
# scale, so normalizing per-column avoids cross-scale comparisons.
B_obs_T = B_obs.T
# B_obs_T.shape == (N_ACTIONS=6, N_OBS=17)
col_max_as = B_obs_T.max(axis=1, keepdims=True)
# col_max_as.shape == (N_ACTIONS=6, 1)
adjacency_as = (B_obs_T > REL_THRESHOLD * col_max_as).astype(np.int8)
# adjacency_as.shape == (N_ACTIONS=6, N_OBS=17)

A_obs_T = A_obs.T
# A_obs_T.shape == (N_OBS=17, N_OBS=17)
col_max_ss = A_obs_T.max(axis=1, keepdims=True)
# col_max_ss.shape == (N_OBS=17, 1)
adjacency_ss = (A_obs_T > REL_THRESHOLD * col_max_ss).astype(np.int8)
# adjacency_ss.shape == (N_OBS=17, N_OBS=17)


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────────────────────────────────────
COL_W = 12  # column width for labels


def print_matrix(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = " " * COL_W + "".join(f"{lbl[:COL_W]:>{COL_W}}" for lbl in col_labels)
    print(header)
    for i, row_lbl in enumerate(row_labels):
        row_str = f"{row_lbl[:COL_W]:<{COL_W}}"
        for j in range(len(col_labels)):
            row_str += f"{'1' if matrix[i, j] else '.':>{COL_W}}"
        print(row_str)


def print_max_abs(matrix: np.ndarray, row_labels: list[str], col_labels: list[str], title: str) -> None:
    """Print max-abs Jacobian values in scientific notation, skipping near-zero entries."""
    print(f"\n{'='*60}")
    print(f"  {title}  (max |value| across {N_SAMPLES} states, rel_per_col={REL_THRESHOLD:.0e})")
    print(f"{'='*60}")
    W = 14
    header = " " * COL_W + "".join(f"{lbl[:W]:>{W}}" for lbl in col_labels)
    print(header)
    for i, row_lbl in enumerate(row_labels):
        row_str = f"{row_lbl[:COL_W]:<{COL_W}}"
        for j in range(len(col_labels)):
            v = matrix[i, j]
            row_str += f"{'.' if v == 0 else f'{v:.2e}':>{W}}"
        print(row_str)


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
print_max_abs(B_obs.T, ACTION_LABELS, OBS_LABELS, "∂obs' / ∂action  (max-abs, rows=actions, cols=obs)")
print_matrix(adjacency_as, ACTION_LABELS, OBS_LABELS, "adjacency_as  [action → obs']  (binarized)")

print_max_abs(A_obs.T, OBS_LABELS, OBS_LABELS, "∂obs' / ∂obs  (max-abs, rows=cause obs, cols=effect obs)")
print_matrix(adjacency_ss, OBS_LABELS, OBS_LABELS, "adjacency_ss  [obs → obs']  (binarized)")

# ──────────────────────────────────────────────────────────────────────────────
# Compare against current implementation
# ──────────────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, "/home/abaisero/git/stable-baselines3-causal-vr")
import gymnasium as gym
from stable_baselines3.common.envs.causal_half_cheetah import CausalHalfCheetah

env = CausalHalfCheetah()
current_as = env.adjacency_as()
# current_as.shape == (N_ACTIONS=6, N_OBS=17)
current_ss = env.adjacency_ss()
# current_ss.shape == (N_OBS=17, N_OBS=17)
env.close()

diff_as = adjacency_as.astype(int) - current_as.astype(int)
diff_ss = adjacency_ss.astype(int) - current_ss.astype(int)

print_matrix((diff_as == 1).astype(np.int8), ACTION_LABELS, OBS_LABELS,
             "adjacency_as: edges in COMPUTED but NOT in current  (+1 = missing edge)")
print_matrix((diff_as == -1).astype(np.int8), ACTION_LABELS, OBS_LABELS,
             "adjacency_as: edges in current but NOT in COMPUTED  (-1 = spurious edge)")
print_matrix((diff_ss == 1).astype(np.int8), OBS_LABELS, OBS_LABELS,
             "adjacency_ss: edges in COMPUTED but NOT in current  (+1 = missing edge)")
print_matrix((diff_ss == -1).astype(np.int8), OBS_LABELS, OBS_LABELS,
             "adjacency_ss: edges in current but NOT in COMPUTED  (-1 = spurious edge)")

print(f"\nadjacency_as  missing={int((diff_as == 1).sum())}  spurious={int((diff_as == -1).sum())}")
print(f"adjacency_ss  missing={int((diff_ss == 1).sum())}  spurious={int((diff_ss == -1).sum())}")
