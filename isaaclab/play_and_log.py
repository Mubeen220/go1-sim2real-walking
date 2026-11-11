# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Play a trained RSL-RL Go1 policy and LOG the trajectory for gait diagnosis."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play + log a Go1 RSL-RL policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--log_steps", type=int, default=500, help="How many steps to record.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import os

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    DirectMARLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_walk.tasks  # noqa: F401

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    task_name = args_cli.task.split(":")[-1]

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    env_cfg.log_dir = log_dir
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading checkpoint: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    uenv = env.unwrapped
    robot = uenv.scene["robot"]
    contact = uenv.scene["contact_forces"]
    dt = uenv.step_dt

    foot_body_ids, foot_names = robot.find_bodies(".*_foot")
    trunk_id, _ = robot.find_bodies("trunk")
    trunk_id = trunk_id[0]
    contact_foot_ids, _ = contact.find_bodies(".*_foot")
    joint_names = robot.data.joint_names

    print(f"[INFO] Feet: {foot_names}")
    print(f"[INFO] Joints ({len(joint_names)}): {joint_names}")

    records = []
    prev_action = None

    obs = env.get_observations()
    n = args_cli.log_steps
    for step in range(n):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)

        cmd = uenv.command_manager.get_command("base_velocity")[0].detach().cpu().numpy()
        lin_b = robot.data.root_lin_vel_b[0].detach().cpu().numpy()
        ang_b = robot.data.root_ang_vel_b[0].detach().cpu().numpy()
        trunk_h = float(robot.data.root_pos_w[0, 2].detach().cpu().numpy())
        proj_g = robot.data.projected_gravity_b[0].detach().cpu().numpy()
        jp = robot.data.joint_pos[0].detach().cpu().numpy()
        jv = robot.data.joint_vel[0].detach().cpu().numpy()

        fforce = contact.data.net_forces_w[0, contact_foot_ids].detach().cpu().numpy()
        fmag = np.linalg.norm(fforce, axis=-1)
        in_contact = (fmag > 1.0).astype(np.float32)
        fvel = robot.data.body_lin_vel_w[0, foot_body_ids].detach().cpu().numpy()
        fspeed = np.linalg.norm(fvel[:, :2], axis=-1)
        slip = fspeed * in_contact

        act = actions[0].detach().cpu().numpy()
        arate = np.zeros_like(act) if prev_action is None else act - prev_action
        prev_action = act

        row = {
            "step": step, "t": step * dt,
            "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_yaw": cmd[2],
            "act_vx": lin_b[0], "act_vy": lin_b[1], "act_vz": lin_b[2], "act_wz": ang_b[2],
            "trunk_h": trunk_h, "gx": proj_g[0], "gy": proj_g[1], "gz": proj_g[2],
        }
        for i in range(len(in_contact)):
            row[f"contact_{i}"] = in_contact[i]
            row[f"slip_{i}"] = slip[i]
        for i in range(len(jp)):
            row[f"jp_{i}"] = jp[i]
            row[f"jv_{i}"] = jv[i]
        for i in range(len(act)):
            row[f"act_{i}"] = act[i]
        row["arate_l2"] = float(np.sum(arate ** 2))
        records.append(row)

    env.close()

    out_dir = os.path.join(log_dir, "trajectory")
    os.makedirs(out_dir, exist_ok=True)
    keys = list(records[0].keys())
    arr = {k: np.array([r[k] for r in records]) for k in keys}
    np.savez(os.path.join(out_dir, f"traj_{task_name}.npz"), **arr)
    csv_path = os.path.join(out_dir, f"traj_{task_name}.csv")
    with open(csv_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in records:
            f.write(",".join(f"{r[k]:.5f}" for k in keys) + "\n")

    a = arr
    vx_err = np.abs(a["cmd_vx"] - a["act_vx"])
    vy_err = np.abs(a["cmd_vy"] - a["act_vy"])
    yaw_err = np.abs(a["cmd_yaw"] - a["act_wz"])
    duty = [float(a[f"contact_{i}"].mean()) for i in range(4)]
    slip_mean = [float(a[f"slip_{i}"].mean()) for i in range(4)]

    print("\n" + "=" * 62)
    print("TRAJECTORY SUMMARY  (paste this whole block)")
    print("=" * 62)
    print(f"steps logged            : {len(records)}  ({len(records)*dt:.1f}s)")
    print(f"cmd vx range            : [{a['cmd_vx'].min():.2f}, {a['cmd_vx'].max():.2f}]")
    print(f"vx tracking err  mean/max: {vx_err.mean():.3f} / {vx_err.max():.3f}")
    print(f"vy tracking err  mean/max: {vy_err.mean():.3f} / {vy_err.max():.3f}")
    print(f"yaw tracking err mean/max: {yaw_err.mean():.3f} / {yaw_err.max():.3f}")
    print(f"trunk height   mean/min  : {a['trunk_h'].mean():.3f} / {a['trunk_h'].min():.3f}  (target 0.34)")
    print(f"proj gravity z mean      : {a['gz'].mean():.3f}  (-1.0 = upright)")
    print(f"foot duty factor (4)     : {[round(d,2) for d in duty]}  (trot ~0.5)")
    print(f"foot slip mean (4) m/s   : {[round(s,3) for s in slip_mean]}  (lower better)")
    print(f"action rate L2 mean      : {a['arate_l2'].mean():.4f}  (lower smoother)")
    print(f"joint vel abs mean       : {np.mean([np.abs(a[f'jv_{i}']).mean() for i in range(12)]):.3f}")
    print("=" * 62)
    print(f"CSV : {csv_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
    simulation_app.close()
