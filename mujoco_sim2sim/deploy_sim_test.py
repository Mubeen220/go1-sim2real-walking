"""Validate the REAL deployment code path in MuJoCo before hardware.
Uses go1_deploy_common (same module the robot will use) for all obs/action logic.
Reads MuJoCo joints in RAW hardware order, so this exercises the joint
permutations exactly as the robot will."""
import os
import numpy as np
import mujoco
import go1_deploy_common as dep

_HERE = os.path.dirname(os.path.abspath(__file__))

CONTROL_DT = 0.02
PHYSICS_SUBSTEPS = 4
MODEL_PATH = os.path.join(_HERE, "mujoco_menagerie", "unitree_go1", "scene.xml")
POLICY_PATH = os.environ.get(
    "GO1_POLICY_WEIGHTS",
    os.path.join(_HERE, "..", "exported_policies", "go1_deploy_proven", "policy_weights.npz"),
)

# MuJoCo joint/actuator names in RAW hardware order (FR,FL,RR,RL)
RAW_JOINT_MJ = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
RAW_ACT_MJ = [n.replace("_joint", "") for n in RAW_JOINT_MJ]


def projected_gravity(quat_wxyz):
    qi = np.zeros(4); mujoco.mju_negQuat(qi, quat_wxyz)
    g = np.zeros(3); mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), qi)
    return g


def world_to_body(quat_wxyz, v):
    qi = np.zeros(4); mujoco.mju_negQuat(qi, quat_wxyz)
    o = np.zeros(3); mujoco.mju_rotVecQuat(o, v, qi)
    return o


class DeploySim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.policy = dep.Go1Policy(POLICY_PATH)
        # RAW-order addresses
        self.qadr = [self.model.joint(n).qposadr[0] for n in RAW_JOINT_MJ]
        self.vadr = [self.model.joint(n).dofadr[0] for n in RAW_JOINT_MJ]
        self.act_id = [self.model.actuator(n).id for n in RAW_ACT_MJ]
        # default pose in RAW order for reset
        self.default_raw = dep.policy_to_raw(dep.DEFAULT_POS_POLICY)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, a in enumerate(self.qadr):
            self.data.qpos[a] = self.default_raw[i]
        mujoco.mj_forward(self.model, self.data)
        self.policy.reset()

    def step(self, cmd):
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()          # body frame
        grav = projected_gravity(quat)
        jpos_raw = np.array([self.data.qpos[a] for a in self.qadr])
        jvel_raw = np.array([self.data.qvel[a] for a in self.vadr])

        # ---- THE DEPLOYMENT PATH (shared module) ----
        jpos_pol = dep.raw_to_policy(jpos_raw)
        jvel_pol = dep.raw_to_policy(jvel_raw)
        obs = self.policy.build_observation(ang_vel, grav, cmd, jpos_pol, jvel_pol)
        action = self.policy.infer(obs)
        target_pol = self.policy.action_to_joint_target_policy(action)
        target_raw = dep.policy_to_raw(target_pol)
        # --------------------------------------------

        for _ in range(PHYSICS_SUBSTEPS):
            for i, aid in enumerate(self.act_id):
                self.data.ctrl[aid] = target_raw[i]
            mujoco.mj_step(self.model, self.data)

        vb = world_to_body(quat, self.data.qvel[0:3].copy())
        return self.data.qpos[2], grav[2], vb


if __name__ == "__main__":
    sim = DeploySim()
    print("Speed sweep through the REAL deployment code path:")
    for v in [0.1, 0.3, 0.5, 0.7, 1.0]:
        sim.reset()
        cmd_s = np.zeros(3); cmd_t = np.array([v, 0.0, 0.0])
        tail = []
        for step in range(150):
            d = np.clip(cmd_t - cmd_s, -0.06, 0.06); cmd_s += d
            h, gz, vb = sim.step(cmd_s)
            if step >= 100:
                tail.append(vb[0])
        print(f"  cmd={v:+.2f}  achieved_vx={np.mean(tail):+.3f}  trunk_h={h:.3f}  gz={gz:.3f}")
