"""
Go1 MuJoCo sim-to-sim runner -- PROVEN RECIPE version.
Observation is 45-dim, single frame (no history), matching the unitree_rl_lab
Go2 recipe adapted to Go1. Scale factors (ang_vel*0.2, joint_vel*0.05) match
the training config exactly.
"""
import argparse
import os
import numpy as np
import mujoco
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))

CONTROL_DT = 0.02
PHYSICS_SUBSTEPS = 4
KP, KD = 20.0, 0.5
TORQUE_LIMIT = 23.7
ACTION_SCALE = 0.25

JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
ACTUATOR_NAMES = [j.replace("_joint", "") for j in JOINT_ORDER]
DEFAULT_POS = np.array([0.1, -0.1, 0.1, -0.1,
                         0.8,  0.8,  1.0,  1.0,
                        -1.5, -1.5, -1.5, -1.5])

# Full command range at end of curriculum (matches Go1WalkEnvCfg_PLAY)
CMD_MAX = np.array([1.0, 0.4, 1.0])
MAX_ACCEL = np.array([3.0, 2.0, 4.0])

ANG_VEL_SCALE = 0.2
JOINT_VEL_SCALE = 0.05

POLICY_PATH = os.environ.get(
    "GO1_POLICY_ONNX",
    os.path.join(_HERE, "..", "exported_policies", "go1_deploy_proven", "policy.onnx"),
)
MODEL_PATH = os.path.join(_HERE, "mujoco_menagerie", "unitree_go1", "scene.xml")


def world_to_body(quat_wxyz, vec_world):
    quat_inv = np.zeros(4)
    mujoco.mju_negQuat(quat_inv, quat_wxyz)
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, vec_world, quat_inv)
    return out


def projected_gravity(quat_wxyz):
    quat_inv = np.zeros(4)
    mujoco.mju_negQuat(quat_inv, quat_wxyz)
    g = np.zeros(3)
    mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), quat_inv)
    return g


class Go1Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        self.policy = ort.InferenceSession(POLICY_PATH)
        self.last_action = np.zeros(12)
        self.joint_qpos_adr = [self.model.joint(name).qposadr[0] for name in JOINT_ORDER]
        self.joint_qvel_adr = [self.model.joint(name).dofadr[0] for name in JOINT_ORDER]
        self.joint_act_id = [self.model.actuator(name).id for name in ACTUATOR_NAMES]
        self.foot_geom_id = [self.model.geom(name).id for name in ["FL", "FR", "RL", "RR"]]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for i, adr in enumerate(self.joint_qpos_adr):
            self.data.qpos[adr] = DEFAULT_POS[i]
        mujoco.mj_forward(self.model, self.data)
        self.last_action[:] = 0.0

    def get_joint_state(self):
        qpos = np.array([self.data.qpos[a] for a in self.joint_qpos_adr])
        qvel = np.array([self.data.qvel[a] for a in self.joint_qvel_adr])
        return qpos, qvel

    def step(self, cmd_smoothed):
        base_quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()
        grav = projected_gravity(base_quat)
        jpos, jvel = self.get_joint_state()
        jpos_rel = jpos - DEFAULT_POS

        obs = np.concatenate([
            ang_vel * ANG_VEL_SCALE,
            grav,
            cmd_smoothed,
            jpos_rel,
            jvel * JOINT_VEL_SCALE,
            self.last_action,
        ]).astype(np.float32)

        action = self.policy.run(None, {"obs": obs[None, :]})[0][0]
        self.last_action = action.copy()

        target_pos = DEFAULT_POS + ACTION_SCALE * action
        for _ in range(PHYSICS_SUBSTEPS):
            for i, act_id in enumerate(self.joint_act_id):
                self.data.ctrl[act_id] = target_pos[i]
            mujoco.mj_step(self.model, self.data)

        trunk_h = self.data.qpos[2]
        lin_vel_world = self.data.qvel[0:3].copy()
        act_vel_b = world_to_body(base_quat, lin_vel_world)
        xy = self.data.qpos[0:2].copy()
        foot_z = [self.data.geom_xpos[gid][2] for gid in self.foot_geom_id]
        return trunk_h, grav, action, act_vel_b, xy, np.array(foot_z)


def run_interactive():
    import mujoco.viewer
    import glfw

    sim = Go1Sim()
    cmd_target = np.zeros(3)
    cmd_smoothed = np.zeros(3)

    def key_callback(keycode):
        if keycode == glfw.KEY_UP:
            cmd_target[0] = min(cmd_target[0] + 0.1, CMD_MAX[0])
        elif keycode == glfw.KEY_DOWN:
            cmd_target[0] = max(cmd_target[0] - 0.1, -CMD_MAX[0])
        elif keycode == glfw.KEY_A:
            cmd_target[1] = min(cmd_target[1] + 0.1, CMD_MAX[1])
        elif keycode == glfw.KEY_D:
            cmd_target[1] = max(cmd_target[1] - 0.1, -CMD_MAX[1])
        elif keycode == glfw.KEY_LEFT:
            cmd_target[2] = min(cmd_target[2] + 0.1, CMD_MAX[2])
        elif keycode == glfw.KEY_RIGHT:
            cmd_target[2] = max(cmd_target[2] - 0.1, -CMD_MAX[2])
        elif keycode == glfw.KEY_SPACE:
            cmd_target[:] = 0.0
        elif keycode == glfw.KEY_R:
            sim.reset()
            cmd_target[:] = 0.0

    print("Controls: UP/DOWN=fwd/back  A/D=strafe  LEFT/RIGHT=yaw  SPACE=stop  R=reset")
    with mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            delta = np.clip(cmd_target - cmd_smoothed, -MAX_ACCEL*CONTROL_DT, MAX_ACCEL*CONTROL_DT)
            cmd_smoothed += delta
            sim.step(cmd_smoothed)
            viewer.sync()


def run_scripted(log_csv):
    sim = Go1Sim()
    cmd_smoothed = np.zeros(3)
    schedule = [
        (2.0, 0.0, 0.0, 0.0),
        (2.0, 0.3, 0.0, 0.0),
        (3.0, 1.0, 0.0, 0.0),
        (0.02, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0, 1.0),
        (2.0, 0.0, 0.0, 0.0),
    ]
    rows = []
    t = 0.0
    for dur, tx, ty, tyaw in schedule:
        cmd_target = np.array([tx, ty, tyaw])
        for _ in range(int(dur / CONTROL_DT)):
            delta = np.clip(cmd_target - cmd_smoothed, -MAX_ACCEL*CONTROL_DT, MAX_ACCEL*CONTROL_DT)
            cmd_smoothed += delta
            trunk_h, grav, action, act_vel_b, xy, foot_z = sim.step(cmd_smoothed)
            rows.append([t, *cmd_smoothed, trunk_h, *grav, *act_vel_b, *xy, *foot_z])
            t += CONTROL_DT
    rows = np.array(rows)
    header = "t,cmd_vx,cmd_vy,cmd_yaw,trunk_h,gx,gy,gz,act_vx,act_vy,act_vz,x,y,fl_z,fr_z,rl_z,rr_z"
    np.savetxt(log_csv, rows, delimiter=",", header=header, comments="")

    vx_err = np.abs(rows[:,1] - rows[:,8])
    dist = np.sqrt((rows[-1,11]-rows[0,11])**2 + (rows[-1,12]-rows[0,12])**2)
    print(f"Logged {len(rows)} steps to {log_csv}")
    print(f"trunk_h mean/min: {rows[:,4].mean():.3f} / {rows[:,4].min():.3f}")
    print(f"gz mean: {rows[:,7].mean():.3f}")
    print(f"vx tracking err mean/max: {vx_err.mean():.3f} / {vx_err.max():.3f}")
    print(f"total distance: {dist:.2f} m")
    for i, leg in enumerate(["fl","fr","rl","rr"]):
        col = rows[:, 13+i]
        print(f"  {leg}_foot swing: {col.max()-col.min():.4f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["interactive", "scripted"], default="scripted")
    parser.add_argument("--log_csv", default="traj_mujoco_proven.csv")
    args = parser.parse_args()
    if args.mode == "interactive":
        run_interactive()
    else:
        run_scripted(args.log_csv)
