"""
go1_deploy_common.py -- SINGLE SOURCE OF TRUTH for Go1 deployment.
Imported identically by mujoco_run.py (--mode deploy_sim) and the real
deploy_policy.py. Same code both places = no sim-vs-real obs drift.

RAW hardware LCM order (Unitree SDK / MuJoCo Menagerie):
  0-2 FR(hip,thigh,calf)  3-5 FL  6-8 RR  9-11 RL
ISAAC LAB policy order (grouped by joint type):
  FL_hip,FR_hip,RL_hip,RR_hip, FL_thigh,FR_thigh,RL_thigh,RR_thigh, FL_calf,FR_calf,RL_calf,RR_calf
"""
import numpy as np
# policy runs on pure numpy -- no onnxruntime/torch needed (robot has Python 3.6, torch 1.4)

POLICY_JOINT_NAMES = [
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
]
RAW_JOINT_NAMES = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]
HW_TO_POLICY = np.array([RAW_JOINT_NAMES.index(n) for n in POLICY_JOINT_NAMES])
POLICY_TO_HW = np.array([POLICY_JOINT_NAMES.index(n) for n in RAW_JOINT_NAMES])

DEFAULT_POS_POLICY = np.array([
    0.1, -0.1, 0.1, -0.1,
    0.8, 0.8, 1.0, 1.0,
    -1.5, -1.5, -1.5, -1.5,
])

ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05
ACTION_SCALE = 0.25
KP = 20.0
KD = 0.5


def sanity_check_permutations():
    assert np.array_equal(HW_TO_POLICY[POLICY_TO_HW], np.arange(12)), "perm mismatch A"
    assert np.array_equal(POLICY_TO_HW[HW_TO_POLICY], np.arange(12)), "perm mismatch B"


sanity_check_permutations()


def _elu(x):
    """ELU activation, matches torch.nn.ELU(alpha=1.0)."""
    return np.where(x > 0, x, np.exp(x) - 1.0)


class Go1Policy:
    """Pure-numpy MLP policy. Loads weights from .npz (version-independent).
    Architecture: 45 -> 512 -> 256 -> 128 -> 12, ELU between layers.
    Runs on any Python/numpy -- no onnxruntime, no torch, no version issues."""

    def __init__(self, weights_path):
        data = np.load(weights_path, allow_pickle=True)
        # Linear layers at mlp indices 0,2,4,6 (ELU at 1,3,5 has no weights)
        self.W0, self.b0 = data["mlp.0.weight"], data["mlp.0.bias"]
        self.W2, self.b2 = data["mlp.2.weight"], data["mlp.2.bias"]
        self.W4, self.b4 = data["mlp.4.weight"], data["mlp.4.bias"]
        self.W6, self.b6 = data["mlp.6.weight"], data["mlp.6.bias"]
        self.last_action = np.zeros(12, dtype=np.float32)

    def reset(self):
        self.last_action = np.zeros(12, dtype=np.float32)

    def build_observation(self, base_ang_vel, projected_gravity, velocity_command,
                          joint_pos_policy, joint_vel_policy):
        joint_pos_rel = joint_pos_policy - DEFAULT_POS_POLICY
        obs = np.concatenate([
            np.asarray(base_ang_vel) * ANG_VEL_SCALE,
            np.asarray(projected_gravity),
            np.asarray(velocity_command),
            joint_pos_rel,
            np.asarray(joint_vel_policy) * JOINT_VEL_SCALE,
            self.last_action,
        ]).astype(np.float32)
        return obs

    def infer(self, obs):
        """Forward pass through the MLP. obs: (45,) -> action: (12,)."""
        x = obs.astype(np.float32)
        x = _elu(self.W0 @ x + self.b0)
        x = _elu(self.W2 @ x + self.b2)
        x = _elu(self.W4 @ x + self.b4)
        action = self.W6 @ x + self.b6   # no activation on output
        self.last_action = action.astype(np.float32)
        return action

    def action_to_joint_target_policy(self, action):
        return DEFAULT_POS_POLICY + ACTION_SCALE * action


def raw_to_policy(vec_raw):
    return np.asarray(vec_raw)[HW_TO_POLICY]


def policy_to_raw(vec_policy):
    return np.asarray(vec_policy)[POLICY_TO_HW]


# ---- WTW-order <-> Isaac-Lab-policy-order permutation ----
# WTW get_dof_pos()/get_dof_vel() return joints in WTW order (grouped by leg):
#   FL(hip,thigh,calf), FR(...), RL(...), RR(...)
# publish also expects WTW order (it applies joint_idxs internally to reach raw).
WTW_JOINT_NAMES = [
    "FL_hip", "FL_thigh", "FL_calf",
    "FR_hip", "FR_thigh", "FR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
]
WTW_TO_POLICY = np.array([WTW_JOINT_NAMES.index(n) for n in POLICY_JOINT_NAMES])
POLICY_TO_WTW = np.array([POLICY_JOINT_NAMES.index(n) for n in WTW_JOINT_NAMES])
assert np.array_equal(WTW_TO_POLICY[POLICY_TO_WTW], np.arange(12)), "WTW perm mismatch"


def wtw_to_policy(vec_wtw):
    return np.asarray(vec_wtw)[WTW_TO_POLICY]


def policy_to_wtw(vec_policy):
    return np.asarray(vec_policy)[POLICY_TO_WTW]
