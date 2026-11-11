# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def feet_swing_clearance(env, sensor_cfg, asset_cfg, target_height: float = 0.08):
    """Reward feet for lifting to target_height while NOT in contact (i.e. during swing).
    This directly counteracts a dragging/shuffling gait by making foot lift a real reward,
    not just an indirect side-effect of air-time timing.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    # only reward height while airborne; contribute 0 while foot is in contact
    swing_error = torch.square(foot_z - target_height) * (~in_contact).float()
    swing_error = torch.clamp(swing_error, max=1.0)  # bound against any physics-glitch outlier
    return -torch.sum(swing_error, dim=1)


def energy(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Unitree recipe: penalize power = |joint_vel * applied_torque|."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel * asset.data.applied_torque), dim=1)


def lin_vel_cmd_levels(env, env_ids, num_steps: int = 20000):
    """Command curriculum: expand ALL command ranges (x, y, AND yaw) from tiny
    to full over the first ~1/3 of training, then hold at full for the rest so
    the policy has ample time to master hard commands.

    Fixes vs. old version:
      - ramps ang_vel_z (yaw) too -- old one forgot it, so turning never trained
      - reaches full range by ~iter 500 (num_steps=12000 / 24 steps-per-env),
        leaving ~1000 iters at full range to converge (the failed run faced full
        range from step 0 and never converged; the old ramp never trained yaw)
    """
    cmd_cfg = env.command_manager.get_term("base_velocity").cfg
    progress = min(env.common_step_counter / num_steps, 1.0)
    x = 0.1 + (1.5 - 0.1) * progress      # SPEED: ramp to 1.5 m/s (evidence-based ceiling)
    y = 0.1 + (0.7 - 0.1) * progress
    yaw = 0.1 + (1.0 - 0.1) * progress
    cmd_cfg.ranges.lin_vel_x = (-x, x)
    cmd_cfg.ranges.lin_vel_y = (-y, y)
    cmd_cfg.ranges.ang_vel_z = (-yaw, yaw)
    return torch.tensor(x)
