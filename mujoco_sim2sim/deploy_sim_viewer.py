"""Interactive MuJoCo viewer running the REAL deployment code path
(go1_deploy_common). Watch the policy walk before hardware.
Controls: UP/DOWN=fwd/back  A/D=strafe  LEFT/RIGHT=turn  SPACE=stop  R=reset"""
import os
import numpy as np
import mujoco
import mujoco.viewer
import glfw
import go1_deploy_common as dep

_HERE = os.path.dirname(os.path.abspath(__file__))

CONTROL_DT = 0.02
PHYSICS_SUBSTEPS = 4
MODEL_PATH = os.path.join(_HERE, "mujoco_menagerie", "unitree_go1", "scene.xml")
# NOTE: dep.Go1Policy loads the numpy .npz archive, not ONNX -- the original
# script pointed this at policy.onnx, which np.load() cannot read. Fixed to
# point at the .npz weights, matching how Go1Policy is actually used below.
POLICY_PATH = os.environ.get(
    "GO1_POLICY_WEIGHTS",
    os.path.join(_HERE, "..", "exported_policies", "go1_deploy_proven", "policy_weights.npz"),
)
MAX_ACCEL = np.array([2.0, 2.0, 3.0])
CMD_MAX = np.array([1.0, 0.4, 1.0])

RAW_JOINT_MJ = [
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]
RAW_ACT_MJ = [n.replace("_joint", "") for n in RAW_JOINT_MJ]


def projected_gravity(q):
    qi = np.zeros(4); mujoco.mju_negQuat(qi, q)
    g = np.zeros(3); mujoco.mju_rotVecQuat(g, np.array([0.,0.,-1.]), qi)
    return g


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
policy = dep.Go1Policy(POLICY_PATH)
qadr = [model.joint(n).qposadr[0] for n in RAW_JOINT_MJ]
vadr = [model.joint(n).dofadr[0] for n in RAW_JOINT_MJ]
act_id = [model.actuator(n).id for n in RAW_ACT_MJ]
default_raw = dep.policy_to_raw(dep.DEFAULT_POS_POLICY)

def reset():
    mujoco.mj_resetData(model, data)
    for i, a in enumerate(qadr):
        data.qpos[a] = default_raw[i]
    mujoco.mj_forward(model, data)
    policy.reset()

reset()
cmd_target = np.zeros(3)
cmd_smoothed = np.zeros(3)

def key_cb(keycode):
    global cmd_target
    if keycode == glfw.KEY_UP:    cmd_target[0] = min(cmd_target[0]+0.1, CMD_MAX[0])
    elif keycode == glfw.KEY_DOWN:  cmd_target[0] = max(cmd_target[0]-0.1, -CMD_MAX[0])
    elif keycode == glfw.KEY_A:     cmd_target[1] = min(cmd_target[1]+0.1, CMD_MAX[1])
    elif keycode == glfw.KEY_D:     cmd_target[1] = max(cmd_target[1]-0.1, -CMD_MAX[1])
    elif keycode == glfw.KEY_LEFT:  cmd_target[2] = min(cmd_target[2]+0.1, CMD_MAX[2])
    elif keycode == glfw.KEY_RIGHT: cmd_target[2] = max(cmd_target[2]-0.1, -CMD_MAX[2])
    elif keycode == glfw.KEY_SPACE: cmd_target[:] = 0.0
    elif keycode == glfw.KEY_R:     reset(); cmd_target[:] = 0.0

print("Controls: UP/DOWN=fwd/back  A/D=strafe  LEFT/RIGHT=turn  SPACE=stop  R=reset")
print("Click the window first so it captures keys.")

with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
    while viewer.is_running():
        d = np.clip(cmd_target - cmd_smoothed, -MAX_ACCEL*CONTROL_DT, MAX_ACCEL*CONTROL_DT)
        cmd_smoothed += d

        quat = data.qpos[3:7].copy()
        ang_vel = data.qvel[3:6].copy()
        grav = projected_gravity(quat)
        jpos_raw = np.array([data.qpos[a] for a in qadr])
        jvel_raw = np.array([data.qvel[a] for a in vadr])
        jpos_pol = dep.raw_to_policy(jpos_raw)
        jvel_pol = dep.raw_to_policy(jvel_raw)

        obs = policy.build_observation(ang_vel, grav, cmd_smoothed, jpos_pol, jvel_pol)
        action = policy.infer(obs)
        target_raw = dep.policy_to_raw(policy.action_to_joint_target_policy(action))

        for _ in range(PHYSICS_SUBSTEPS):
            for i, aid in enumerate(act_id):
                data.ctrl[aid] = target_raw[i]
            mujoco.mj_step(model, data)
        viewer.sync()
