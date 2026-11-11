> **Note on folder names:** this document was written against the original
> project layout (`1_isaac_training/`, `2_mujoco_sim2sim/`, `3_real_deployment/`).
> In this repository those folders are named `isaaclab/`, `mujoco_sim2sim/`,
> and `deployment/` respectively — same files, same roles, just renamed for
> clarity. Commands and paths below are otherwise exactly as originally written.

# Go1 Sim-to-Real Workflow: Isaac Lab → MuJoCo → Real Robot
### Complete step-by-step record, from training to hardware deployment

This document records the full pipeline used to train a walking policy for the
Unitree Go1 in NVIDIA Isaac Lab, validate it in MuJoCo through the exact code
path that runs on the robot, and deploy it on the physical Go1. Every code file
referenced here is in this same folder.

---

## Folder contents

**1_isaac_training/**   — Isaac Lab environment, rewards, PPO config, train/play scripts
- `go1_walk_env_cfg.py` — environment: observations, rewards, curriculum, domain randomization
- `rewards.py`          — custom reward terms + the command curriculum function
- `agents/`             — RSL-RL PPO hyperparameters
- `train.py`            — training entry point
- `play.py`             — runs a trained policy, exports ONNX + TorchScript
- `play_and_log.py`     — runs a policy and logs a trajectory CSV for analysis

**2_mujoco_sim2sim/**   — independent-simulator validation
- `go1_deploy_common.py`— SHARED module: joint permutations, observation builder, numpy MLP
- `deploy_sim_test.py`  — forward-speed sweep through the deployment code path
- `deploy_sim_viewer.py`— visual MuJoCo playback
- `mujoco_run.py`       — runner / six-direction test

**3_real_deployment/**  — on-robot execution
- `deploy_go1_isaaclab.py` — robot script: LCM I/O, R2 state machine, safety protocol
- `go1_deploy_common.py`   — IDENTICAL shared module (same file as in MuJoCo — this is the key design choice)

---

## The central design principle

The observation builder, joint-ordering permutations, and network forward pass
live in ONE file — `go1_deploy_common.py` — imported unchanged by both the MuJoCo
test and the robot script. A MuJoCo result is therefore genuine evidence about
the robot, because the identical code produces it. This single decision is what
makes sim-to-sim validation meaningful rather than a parallel reimplementation.

---

## STEP 1 — Train the walking policy in Isaac Lab

**Goal:** a flat-ground velocity-tracking policy using only observations the real
robot can measure.

**Observation (45-dim, single frame, NO base linear velocity):**
base angular velocity (×0.2), projected gravity, velocity command,
joint positions relative to default, joint velocities (×0.05), previous action.
Base linear velocity is deliberately excluded — the Go1 cannot measure it, so a
policy depending on it would not be deployable.

**Action:** target = default_pose + 0.25 × action, tracked by a PD controller
(Kp = 20, Kd = 0.5). Control at 50 Hz (decimation 4, sim dt 0.005).

**Reward (proven recipe, in go1_walk_env_cfg.py):**
linear-velocity tracking (2.0), angular-velocity tracking (0.75, std 0.5),
plus regularization: z-velocity, roll/pitch, torque, action rate, joint
acceleration, joint limits, flat orientation, feet air time.

**Curriculum:** commands start at ±0.1 m/s and ramp all axes together toward
the terminal range. MUST run uninterrupted — the curriculum reads a global step
counter that only advances correctly in one continuous run.

**Command:**
    cd isaac_project/go1_walk
    python scripts/rsl_rl/train.py --task Template-Go1-Walk-v0 --num_envs 4096 --headless | tee train.log

**Healthy convergence:** linear-tracking reward > ~1.0, fall rate < 0.05,
mean reward strongly positive, curriculum reaching its ceiling.

---

## STEP 2 — Export the trained policy

    python scripts/rsl_rl/play.py --task Template-Go1-Walk-Play-v0 --num_envs 32
    # writes exported/policy.onnx and exported/policy.pt next to the checkpoint

Always confirm the NEWEST export by timestamp (use `ls -dt`), never a plain `ls`.

---

## STEP 3 — Make the policy framework-independent (weights + numpy)

The robot runs Python 3.6 / PyTorch 1.4 and CANNOT load a modern TorchScript
file. The robust solution: extract the weights to a plain .npz and run the
forward pass in pure numpy.

    import torch, numpy as np
    pt = torch.jit.load('exported/policy.pt'); pt.eval()
    w = {n: p.detach().cpu().numpy() for n, p in pt.named_parameters()}
    np.savez('policy_weights.npz', **w)

The forward pass (in go1_deploy_common.py) is three ELU-activated layers plus a
linear output, written with np.dot (NOT the @ operator, which is Python 3.5+
only). This removes every framework/version dependency. Verified: the numpy
output is bit-for-bit identical across the training workstation (Py3.11/torch2.7),
the robot (Py3.6/numpy1.13), and a laptop (Py3.13/numpy2.5).

---

## STEP 4 — Validate in MuJoCo through the deployment code path

This is the gate that must pass before the robot is powered.

    cd mujoco_sim2sim
    python3 deploy_sim_test.py

**Forward speed sweep — a validated policy looks like (Policy A):**
    cmd 0.1 → 0.007    cmd 0.3 → 0.304    cmd 0.5 → 0.465
    cmd 0.7 → 0.576    cmd 1.0 → 0.679
Monotonic, trunk height ~0.33–0.35 m, projected gravity z ~ −1.0 throughout.
A consistent UNDERSHOOT is normal and safe. A DEAD ZONE (commands producing zero
motion) is NOT acceptable and indicates a curriculum problem.

Also run a six-direction check: forward, backward, both strafes, both turns —
confirm each responds with the correct sign (measure yaw rate explicitly).

---

## STEP 5 — Joint-order verification

Three joint conventions are in play and every value crossing between them must
be permuted:
  - Raw hardware / MuJoCo : FR, FL, RR, RL  (hip, thigh, calf each)
  - Isaac Lab policy      : all hips, all thighs, all calves
  - Walk-These-Ways       : FL, FR, RL, RR

In go1_deploy_common.py the permutations are built BY JOINT NAME and asserted to
compose with their inverses to the identity, so an incorrect mapping fails
immediately rather than silently mis-routing a command to the wrong motor.
The raw ordering was cross-checked against Unitree's own SDK header.

---

## STEP 6 — Copy files to the robot

Three files must be inside the folder the WTW Docker container mounts (~/go1_gym):
  - go1_deploy_common.py    (shared module)
  - deploy_go1_isaaclab.py  (run script)
  - policy_weights.npz      (network weights)

    scp go1_deploy_common.py deploy_go1_isaaclab.py policy_weights.npz unitree@192.168.123.15:~/go1_gym/

Verify in-container (no motors) that the pipeline loads and the printed action
matches the workstation bit-for-bit.

---

## STEP 7 — Suspended hardware test  (ROBOT HUNG UP)

Feet off the ground, remote in hand ready for damping.

**Terminal 1 (host)** — low-level motor bridge:
    cd ~/go1_gym/go1_gym_deploy/autostart
    ./start_unitree_sdk.sh          # press Enter only if the robot is suspended

**Terminal 2 (inside container)** — the policy:
    python3 deploy_go1_isaaclab.py

**R2 state machine:**
  - Press R2 (1st) → slow calibrated ramp to standing (soft gains Kp=5, Kd=1)
  - Press R2 (2nd) → policy runs, command starts at zero
  - Nudge left stick forward → legs make a coordinated walking rhythm
  - Press R2 (again) → emergency stop, holds default pose and exits

**Safety features (from Unitree's own reference examples):**
  - Soft calibration gains during the ramp, normal gains (Kp=20, Kd=0.5) only when running
  - Constant gravity-compensation feed-forward torque on the hips (±0.65 N·m)
  - Power monitor: trips to a safe hold if estimated mechanical power stays high

Watch for: correct joint direction, no violent motion, clean start/stop.
Any wrong-direction or jerky leg → remote to damping, Ctrl-C, re-check the
joint mapping in MuJoCo before retrying.

---

## STEP 8 — Ground test and speed ladder

Only after the suspended test looks correct in every commanded direction:
  - Set down with command at zero; confirm a stable weight-bearing stand.
  - Start at 0.5 m/s; confirm a straight, stable walk.
  - Climb 0.5 → 0.7 → 1.0 m/s, a stable minute at each; then backward and turning.
  - Slow to a stop with the stick (rate-limited), not by yanking.
  - Safe stop: remote → damping FIRST, then Ctrl-C both terminals.

---

## RESULTS: the three policies

| ID | Run | MuJoCo sweep (0.1/0.3/0.5/0.7/1.0) | Hardware |
|----|-----|------------------------------------|----------|
| **A** | 2026-07-10_12-53-01 | 0.007/0.304/0.465/0.576/0.679 — clean, monotonic | Deployed; never fell |
| **B** | 2026-07-16_15-29-48 | 5 of 6 directions; weak turn-left | Fell twice on ground |
| **C** | 2026-07-17_12-24-06 | −0.007/−0.001/0.103/0.770/1.209 — dead zone + overshoot | Not ground-tested |

**BEST POLICY: A (2026-07-10_12-53-01).**
It is the only policy that behaves correctly on hardware — smooth tracking, no
dead zone, no overshoot — and it was the one deployed.

**KEY FINDING:** Policy C had the BEST training-time metrics (highest tracking
reward, lowest fall rate) yet the WORST deployment behaviour. Training-simulator
metrics did NOT predict transfer quality — most likely because the longer-trained,
more heavily reward-shaped policy specialized to the training simulator's
neural-network actuator model, which MuJoCo (and the real robot) do not reproduce.

---

## Common failure modes and fixes

| Symptom | Cause / fix |
|---------|-------------|
| `ModuleNotFoundError: lcm` on host | lcm is only in the container; run the script inside the WTW Docker container |
| TorchScript load fails (file version) | Robot torch too old; use weights + numpy (Step 3) |
| SyntaxError near `@` under Python 2 | Use np.dot() instead of the @ operator |
| Robot frozen but stable, ignores commands | Policy doesn't transfer / has a dead zone; re-validate in MuJoCo, use a command curriculum |
| Robot stops on ground contact | Usually an R2 press (emergency stop), not a fault; use the stick to command |
| Legs move violently in the air | Joint-order mismatch; damping immediately, re-check name-based permutations |


