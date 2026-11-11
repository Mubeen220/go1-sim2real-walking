# Go1 deployable flat walking -- proven recipe + foot-clearance fix for
# real-hardware tripping/falling, moderate DR for robustness.

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

import isaaclab.envs.mdp as base_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as vel_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot import mdp as spot_mdp
from . import mdp as local_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG


@configclass
class Go1WalkEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # ---- Robot / timing / terrain ----
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 20.0
        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.use_default_offset = True
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None

        # ---- Go1 trunk naming fixes ----
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.4)
        self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names="trunk")
        self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="trunk")
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="trunk")
        self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="trunk")

        # ---- OBSERVATIONS: proven deployment contract (45-dim, no history) ----
        self.observations.policy.base_lin_vel = None
        self.observations.policy.history_length = 0
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_vel.scale = 0.05

        # ---- COMMANDS: start tiny, curriculum expands ALL dims (incl. yaw) ----
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.curriculum.lin_vel_cmd_levels = CurrTerm(func=local_mdp.lin_vel_cmd_levels)

        # ---- REWARDS: proven walking base + foot-clearance fix ----
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -2.0e-4
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_pos_limits.weight = -10.0
        self.rewards.flat_orientation_l2.weight = -2.5

        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=".*_foot")
        self.rewards.feet_air_time.params["threshold"] = 0.5

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"])

        self.rewards.joint_vel = RewTerm(
            func=base_mdp.joint_vel_l2, weight=-0.001,
            params={"asset_cfg": SceneEntityCfg("robot")})
        self.rewards.energy = RewTerm(func=local_mdp.energy, weight=-2.0e-5)
        self.rewards.joint_pos = RewTerm(
            func=spot_mdp.joint_position_penalty, weight=-0.7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                    "stand_still_scale": 5.0, "velocity_threshold": 0.3})
        self.rewards.air_time_variance = RewTerm(
            func=spot_mdp.air_time_variance_penalty, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")})
        self.rewards.feet_slide = RewTerm(
            func=vel_mdp.feet_slide, weight=-0.1,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")})

        # ---- NEW: foot-clearance reward -- fixes low-foot tripping on hardware ----
        # Rewards lifting the swing foot toward target_height (8cm, close to
        # Unitree's own official footRaiseHeight=0.1). Bounded/clamped inside
        # the function to avoid the NaN blowup we hit with an earlier
        # unbounded version.
        self.rewards.feet_clearance = RewTerm(
            func=local_mdp.feet_swing_clearance,
            weight=-0.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": 0.08,
            },
        )

        # ---- EVENTS: moderate domain randomization (proven to converge) ----
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.5)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.5)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.15)
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.base_com.params["com_range"] = {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)}

        # ---- TERMINATIONS ----
        self.terminations.bad_orientation = DoneTerm(
            func=base_mdp.bad_orientation, params={"limit_angle": 0.8})


@configclass
class Go1WalkEnvCfg_PLAY(Go1WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 4.0
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.curriculum.lin_vel_cmd_levels = None
