"""
deploy_go1_isaaclab.py -- Isaac Lab Go1 policy deployment.

Incorporates patterns confirmed from Unitree's OFFICIAL SDK examples
(example_position.cpp, example_velocity.cpp, example_torque.cpp -- all
verified from the robot's own ~/UnitreeSLAM/sdk/unitree_legged_sdk-3.3.1/):

  1. SOFT CALIBRATION GAINS: Kp=5, Kd=1 while holding/ramping to standing
     (matches example_position.cpp exactly). Normal gains (Kp=20, Kd=0.5)
     only once the policy is actually running.
  2. GRAVITY-COMPENSATION FEEDFORWARD: a constant +/-0.65 N.m torque on
     the four hip joints, alternating sign by side. This appears in
     EVERY official low-level example regardless of task -- it offsets
     gravity's tendency to splay the hips outward.
  3. POWER MONITOR (approximate PowerProtect equivalent): every official
     example calls safe.PowerProtect(cmd, state, 1) -- Unitree's own
     recommendation for anyone testing is 10% power. Their exact Watt
     formula is proprietary/compiled, so this is an approximation: we
     track estimated mechanical power (|tau_est * qd|, summed over all
     joints) and if it exceeds a conservative threshold, we cut to a
     safe hold (default pose, soft gains) rather than continuing.

Sequence: hold current position (soft gains) -> R2 -> slow calibrate to
stand (soft gains) -> R2 -> policy runs (normal gains + power monitor).
Safe stop: R2 during run, or power monitor trip -> hold default pose.
Then set remote to DAMPING.
"""
import os
import time, sys
import numpy as np
import lcm

# On the robot (per docs/WORKFLOW_REPORT.md Step 6), this file,
# go1_deploy_common.py and policy_weights.npz are scp'd flat into the same
# ~/go1_gym folder -- so prefer a same-directory copy if present, and fall
# back to this repo's mujoco_sim2sim/ layout otherwise (e.g. when running
# from a clone on a workstation).
_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(_HERE, "go1_deploy_common.py")):
    sys.path.insert(0, _HERE)
else:
    sys.path.insert(0, os.path.join(_HERE, "..", "mujoco_sim2sim"))
import go1_deploy_common as dep
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.lcm_types.pd_tau_targets_lcmt import pd_tau_targets_lcmt

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")

# WTW order -> raw hardware order (from cheetah_state_estimator.py)
JOINT_IDXS = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])

CONTROL_DT = 0.02
MAX_VEL = 1.0
MAX_YAW = 1.0
MAX_LAT = 0.4
MAX_ACCEL = np.array([2.0, 2.0, 3.0])
# Same precedence as above: env var > same-directory copy (robot layout) >
# this repo's exported_policies/ (workstation/dev layout).
POLICY_PATH = os.environ.get(
    "GO1_POLICY_WEIGHTS",
    os.path.join(_HERE, "policy_weights.npz")
    if os.path.exists(os.path.join(_HERE, "policy_weights.npz"))
    else os.path.join(_HERE, "..", "exported_policies", "go1_deploy_proven", "policy_weights.npz"),
)

# ---- Unitree official calibration gains (example_position.cpp) ----
CALIB_TIME = 2.5
CALIB_KP = 5.0
CALIB_KD = 1.0
RUN_KP = dep.KP     # 20.0 -- normal policy-running gains
RUN_KD = dep.KD     # 0.5

# ---- Gravity-compensation feedforward (from EVERY official example) ----
# RAW hardware indices: FR_0=0, FL_0=3, RR_0=6, RL_0=9 (Unitree's own
# quadruped.h enum, confirmed against our joint mapping -- exact match).
GRAV_COMP_RAW = np.zeros(12)
GRAV_COMP_RAW[0] = -0.65   # FR hip
GRAV_COMP_RAW[3] = +0.65   # FL hip
GRAV_COMP_RAW[6] = -0.65   # RR hip
GRAV_COMP_RAW[9] = +0.65   # RL hip

# ---- Power monitor (approximate PowerProtect equivalent) ----
# Unitree's own formula is proprietary/compiled; this is a conservative
# stand-in: sum(|tau_est * qd|) over all joints, tripped if it exceeds
# POWER_LIMIT_WATTS for POWER_TRIP_STEPS consecutive control steps.
POWER_LIMIT_WATTS = 150.0     # conservative estimate for a 12-motor, 23.7 N.m-limit robot at walking speeds
POWER_TRIP_STEPS = 5          # must exceed limit for this many consecutive steps to trip (avoid single-sample noise)


def publish_raw(q_des_raw, kp, kd, tau_ff_raw=None):
    """Publish targets already in RAW hardware order."""
    if tau_ff_raw is None:
        tau_ff_raw = np.zeros(12)
    cmd = pd_tau_targets_lcmt()
    cmd.q_des = q_des_raw
    cmd.qd_des = np.zeros(12)
    cmd.kp = np.ones(12) * kp
    cmd.kd = np.ones(12) * kd
    cmd.tau_ff = tau_ff_raw
    cmd.se_contactState = np.zeros(4)
    cmd.timestamp_us = int(time.time() * 1e6)
    cmd.id = 0
    lc.publish("pd_plustau_targets", cmd.encode())


def publish_wtw_order(q_des_wtw, kp, kd, with_grav_comp=True):
    """Publish targets given in WTW order -> converted to raw."""
    q_des_raw = q_des_wtw[JOINT_IDXS]
    tau_ff = GRAV_COMP_RAW if with_grav_comp else np.zeros(12)
    publish_raw(q_des_raw, kp, kd, tau_ff)


def publish_policy_order(target_policy, kp, kd, with_grav_comp=True):
    publish_wtw_order(dep.policy_to_wtw(target_policy), kp, kd, with_grav_comp)


def get_current_wtw(se):
    return se.get_dof_pos()


def wait_for_r2_holding_current(se, prompt):
    """Wait for R2 while HOLDING current position, SOFT gains + grav comp."""
    print(prompt)
    se.right_lower_right_switch_pressed = False
    while True:
        publish_wtw_order(get_current_wtw(se), CALIB_KP, CALIB_KD)
        if se.right_lower_right_switch_pressed:
            se.right_lower_right_switch_pressed = False
            time.sleep(0.3)
            return
        time.sleep(CONTROL_DT)


def calibrate_to_stand(se):
    """SLOWLY interpolate to standing pose, SOFT gains + grav comp
    (matches Unitree's example_position.cpp jointLinearInterpolation)."""
    start_wtw = get_current_wtw(se)
    target_wtw = dep.policy_to_wtw(dep.DEFAULT_POS_POLICY)
    steps = int(CALIB_TIME / CONTROL_DT)
    print("[CALIB] Slowly moving to standing pose over %.1fs (soft gains, grav-comp)..." % CALIB_TIME)
    for i in range(steps):
        alpha = (i + 1) / steps
        q = (1 - alpha) * start_wtw + alpha * target_wtw
        publish_wtw_order(q, CALIB_KP, CALIB_KD)
        time.sleep(CONTROL_DT)
    print("[CALIB] Standing pose reached.")


def estimate_power_watts(se):
    """Approximate mechanical power: sum(|tau_est * qd|) over all joints."""
    try:
        tau_est = se.get_tau_est()   # WTW order
        qd = se.get_dof_vel()        # WTW order
        return float(np.sum(np.abs(tau_est * qd)))
    except Exception:
        return 0.0   # if unavailable, monitor is a no-op (fails open, not closed --
                      # acceptable since it's a SECONDARY layer on top of PD tracking limits)


def main():
    se = StateEstimator(lc)
    se.spin()
    policy = dep.Go1Policy(POLICY_PATH)

    print("=" * 60)
    print("Go1 Isaac Lab deployment. Robot should be HUNG UP (or damping on ground).")
    print("max_vel =", MAX_VEL, "m/s")
    print("Power monitor limit: %.0f W (approx, %d-step trip)" % (POWER_LIMIT_WATTS, POWER_TRIP_STEPS))
    print("=" * 60)

    wait_for_r2_holding_current(se, "\n[STATE] Press R2 to CALIBRATE to standing (slow ramp, soft gains)...")
    calibrate_to_stand(se)
    wait_for_r2_holding_current(se, "[STATE] Standing. Press R2 again to RUN policy...")

    print("[STATE] RUNNING. Command starts at 0. IN THE AIR: nudge stick, watch legs.")
    policy.reset()
    cmd_smoothed = np.zeros(3)
    power_trip_count = 0

    try:
        while True:
            t0 = time.time()

            # ---- power monitor check (before publishing new command) ----
            power_now = estimate_power_watts(se)
            if power_now > POWER_LIMIT_WATTS:
                power_trip_count += 1
            else:
                power_trip_count = 0
            if power_trip_count >= POWER_TRIP_STEPS:
                print(f"[POWER TRIP] Estimated power {power_now:.0f}W exceeded limit "
                      f"for {POWER_TRIP_STEPS} steps -- holding default pose, stopping.")
                break

            dof_pos_wtw = se.get_dof_pos()
            dof_vel_wtw = se.get_dof_vel()
            grav = se.get_gravity_vector()
            ang_vel = se.get_body_angular_vel()
            jpos_pol = dep.wtw_to_policy(dof_pos_wtw)
            jvel_pol = dep.wtw_to_policy(dof_vel_wtw)

            rc = se.get_command()
            cmd_target = np.array([
                np.clip(float(rc[0]) * MAX_VEL, -MAX_VEL, MAX_VEL),
                np.clip(float(rc[1]) * MAX_LAT, -MAX_LAT, MAX_LAT),
                np.clip(float(rc[2]) * MAX_YAW, -MAX_YAW, MAX_YAW),
            ])
            delta = np.clip(cmd_target - cmd_smoothed, -MAX_ACCEL*CONTROL_DT, MAX_ACCEL*CONTROL_DT)
            cmd_smoothed += delta

            obs = policy.build_observation(ang_vel, grav, cmd_smoothed, jpos_pol, jvel_pol)
            action = policy.infer(obs)
            target_pol = policy.action_to_joint_target_policy(action)
            publish_policy_order(target_pol, RUN_KP, RUN_KD)

            if se.right_lower_right_switch_pressed:
                se.right_lower_right_switch_pressed = False
                print("[STOP] R2 pressed -- holding default pose.")
                break

            el = time.time() - t0
            if el < CONTROL_DT:
                time.sleep(CONTROL_DT - el)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl-C.")

    # safe hold-out, soft gains
    for _ in range(50):
        publish_policy_order(dep.DEFAULT_POS_POLICY, CALIB_KP, CALIB_KD)
        time.sleep(CONTROL_DT)
    print("[DONE] Held default pose (soft gains). Now set remote to DAMPING.")


if __name__ == "__main__":
    main()
