# Go1 Sim-to-Real Walking: Isaac Lab → MuJoCo → Hardware

![Overview: training in Isaac Lab, validation in MuJoCo, deployment on the real Go1](docs/images/overview.png)

**▶ [Watch the real-robot video](docs/results/videos/real_robot_walking.mp4)** · [Training diagram](#training-in-isaac-lab) · [Steps to deploy](#steps-to-deploy-on-the-real-go1) · [Graphs and all videos](docs/GALLERY.md) · [Full project report](docs/WORKFLOW_REPORT.md) · [Policy table](docs/POLICY_REFERENCE.md)

Sim-to-real transfer of a deep-RL walking controller for the **Unitree Go1** quadruped.
A policy is trained with PPO in **NVIDIA Isaac Lab**, gated through an independent **MuJoCo**
sim-to-sim check, and deployed on the real robot with a **pure-numpy** forward pass
(no PyTorch / ONNX on the robot).

Developed during an M2 EEEA (MMVAI) internship at the IBISC laboratory,
Université d'Évry Paris-Saclay, 2026.

## Highlights

- **Deployable observation (45-D, single frame, no base linear velocity)** — only IMU and joint encoders, because the real Go1 cannot measure its own linear velocity.
- **One shared deployment module** (`go1_deploy_common.py`) used unchanged by both the MuJoCo validation and the real robot — a MuJoCo pass is therefore genuine evidence about the robot, not a parallel reimplementation.
- **Framework-independent inference.** The actor MLP is reduced to plain arrays in an `.npz` and evaluated with numpy, bypassing the robot's Python 3.6 / PyTorch 1.4 stack (where ONNX Runtime couldn't be installed and modern TorchScript exports were unreadable).
- **Verified joint orderings** (Isaac Lab ↔ Unitree raw ↔ Walk-These-Ways), with every permutation asserted to compose with its inverse to the identity.
- **Empirical finding:** the policy with the best Isaac Lab training metrics (Policy C) transferred worst in MuJoCo and was never ground-tested; training-simulator metrics did not predict transfer quality.
- **Verified reproducible end-to-end**: `mujoco_sim2sim/deploy_sim_test.py`, run against the actual exported weights in this repo, reproduces the thesis's reported Policy A sweep exactly (see below).

## Pipeline

```mermaid
flowchart LR
    A[Isaac Lab<br/>PPO training] --> B[Export actor<br/>policy.pt / policy.onnx]
    B --> C[Reduce to weights<br/>policy_weights.npz]
    C --> D[MuJoCo sim-to-sim<br/>via go1_deploy_common]
    D -->|gate: velocity sweep<br/>+ six directions| E[Robot, suspended]
    E --> F[Robot, on ground]
    D -. same module .-> F
```

## Repository layout

```
.
├── isaaclab/                   # Isaac Lab task: env/reward/curriculum code, train/play scripts
│   └── run_configs/            # exact recorded config for every training run (26), incl. A/B/C
├── mujoco_sim2sim/
│   ├── go1_deploy_common.py    # shared obs / joint maps / numpy MLP — single source of truth
│   ├── deploy_sim_test.py      # forward-speed sweep through the deployment code path
│   ├── deploy_sim_viewer.py    # interactive MuJoCo viewer, same code path
│   ├── mujoco_run.py           # ONNX-based runner (interactive + scripted trajectory logging)
│   └── mujoco_menagerie/unitree_go1/   # Go1 MJCF model (Unitree, BSD-3-Clause)
├── deployment/
│   └── deploy_go1_isaaclab.py  # real-robot runner: LCM I/O, R2 state machine, full safety protocol
├── exported_policies/
│   ├── go1_deploy_proven/      # Policy A — the one actually deployed (.npz, .pt, .onnx)
│   ├── policy_B_5of6/          # Policy B weights, kept for comparison (fell on hardware — see below)
│   └── policy_C_dead_zone/     # Policy C weights, kept for comparison (best training metrics, worst transfer)
├── docs/
│   ├── images/
│   │   ├── overview.png                    # Isaac Lab → MuJoCo → Go1 picture at the top of this README
│   │   └── training_isaaclab_rl_loop.png   # RL-loop diagram used in "Training in Isaac Lab"
│   ├── WORKFLOW_REPORT.md      # full step-by-step project log, written during the internship
│   ├── POLICY_REFERENCE.md     # A/B/C table and training-log mapping
│   ├── GALLERY.md              # graphs, videos and photos
│   ├── thesis_numerical_results.txt
│   └── results/                # plots, trajectory CSVs, training logs, and videos for all three policies
└── tests/                      # hardware-free smoke tests, incl. loading the real deployed weights
```

## Policy interface

| Item | Value |
|---|---|
| Control rate | 50 Hz (physics dt 5 ms, decimation 4) |
| Network | MLP 45 → 512 → 256 → 128 → 12, ELU |
| Action | `q_target = q_default + 0.25 · a` (12 joint positions) |
| PD gains (running) | Kp = 20, Kd = 0.5 |
| PD gains (calibration ramp) | Kp = 5, Kd = 1 (soft, matches Unitree's own `example_position.cpp`) |

Observation (45-D), in this order:

| Component | Dim | Scale |
|---|---|---|
| Base angular velocity | 3 | 0.2 |
| Projected gravity | 3 | 1.0 |
| Velocity command (vx, vy, yaw rate) | 3 | 1.0 |
| Joint positions − default | 12 | 1.0 |
| Joint velocities | 12 | 0.05 |
| Previous action | 12 | 1.0 |

Joint orders (all built **by joint name** in `go1_deploy_common.py`, with every permutation asserted against its inverse):

- **Policy (Isaac Lab):** `FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, …, RR_calf` (grouped by joint type)
- **Raw hardware (Unitree SDK / MuJoCo):** `FR, FL, RR, RL`, each `(hip, thigh, calf)` — cross-checked against Unitree's own SDK header
- **Walk-These-Ways:** `FL, FR, RL, RR`, each `(hip, thigh, calf)`

Default standing pose (policy order): hips `[0.1, -0.1, 0.1, -0.1]`, thighs `[0.8, 0.8, 1.0, 1.0]`, calves `-1.5`.

## Results: three policies, one surprising winner

All three were evaluated through the **identical** deployment code path in MuJoCo (`deploy_sim_test.py` / `mujoco_run.py`), so differences reflect the policies, not the evaluation code.

| Commanded vx (m/s) | Policy A | Policy B | Policy C |
|---|---|---|---|
| 0.1 | 0.007 | ≈ 0 | −0.007 |
| 0.3 | 0.304 | 0.31 | −0.001 |
| 0.5 | 0.465 | 0.47 | 0.103 |
| 0.7 | 0.576 | 0.55 | 0.770 |
| 1.0 | 0.679 | 0.74 | 1.209 |

- **Policy A** (`exported_policies/go1_deploy_proven/`) — clean, monotonic tracking with the usual undershoot at higher speeds. **This is the policy that was deployed on hardware and never fell.**
- **Policy B** (`exported_policies/policy_B_5of6/`) — walks well in 5 of 6 commanded directions but turns weakly in one; **fell twice during ground testing**.
- **Policy C** (`exported_policies/policy_C_dead_zone/`) — had the *best* Isaac Lab training metrics (tracking reward 1.87, fall rate 0.0002) of the three, yet the *worst* MuJoCo transfer: a dead zone below 0.5 m/s and a high-speed overshoot. Never ground-tested, for this reason.

| | Forward tracking @ 0.8 m/s | Response by commanded direction |
|---|---|---|
| **A** | ![](docs/results/policy_a_velocity_forward.png) | ![](docs/results/policy_a_six_directions.png) |
| **B** | ![](docs/results/policy_b_velocity_forward.png) | ![](docs/results/policy_b_six_directions.png) |
| **C** | ![](docs/results/policy_c_velocity_forward.png) | ![](docs/results/policy_c_six_directions.png) |

Isaac Lab training curves (mean reward, tracking rewards, fall rate, command curriculum) for one run:

![Training curves](docs/results/plots/training_train_final.png)

More: all training curves, trajectory plots and the video list are in the **[gallery](docs/GALLERY.md)**. Raw trajectory CSVs are in [`docs/results/trajectories/`](docs/results/trajectories/), full training stdout in [`docs/results/training_logs/`](docs/results/training_logs/), and the clips in [`docs/results/videos/`](docs/results/videos/).

## Conclusion

A deployable walking controller for the Unitree Go1 was trained in Isaac Lab, validated in MuJoCo through the exact code path that runs on the robot, and deployed on the physical robot. Forward and backward walking transferred; turning and low-speed tracking were weaker on hardware.

1. **Training metrics did not predict transfer quality.** Policy C had the best training-time metrics (tracking reward 1.87, fall rate 0.0002) but the worst behaviour after transfer, with a dead zone below 0.5 m/s and a high-speed overshoot. Policy A, with more modest training metrics, transferred cleanly and was the one deployed. The most likely cause is that the longer-trained, more heavily shaped policy specialised to the Go1's neural-network actuator model in Isaac Lab, which MuJoCo does not reproduce.
2. **Independent validation is part of the method.** Requiring every policy to pass a MuJoCo gate through the *same* deployment module that runs on the robot exposed the Policy C failure before it reached hardware.
3. **Adding reward terms is riskier than tuning proven ones.** The `feet_clearance` term added to Policy C competed with the other terms and did not improve the measured foot clearance.
4. **The software gap is a real part of sim-to-real.** The robot's Python 3.6 / PyTorch 1.4 stack could not load modern ONNX or TorchScript exports. Reducing the policy to plain weights and a numpy forward pass removed that dependency.

**Future work:** an actuator-aware transfer (identify or randomise the actuator model, for example with a residual actuator model fitted to real Go1 trajectories); policy selection by agreement across independent simulators instead of training return; better turning and low-speed behaviour; and moving the simulated backflip toward hardware.

## ⚠️ A note on reproducing Policy A exactly

`isaaclab/go1_walk_env_cfg.py` is the **current** state of the training source — but it is not a byte-exact recipe for any single one of the three policies above. Specifically, it includes a `feet_clearance` reward term that **Policy C's** run used (and which the results above show *hurt* transfer) but that **neither Policy A nor Policy B's actual training runs included**, confirmed against their individually recorded configs.

For an exact, authoritative reproduction of any specific run — including the deployed Policy A — use the matching file in [`isaaclab/run_configs/`](isaaclab/run_configs/) rather than the current `.py` source:

| Policy | Run config |
|---|---|
| A (deployed) | `isaaclab/run_configs/2026-07-10_12-53-01_env.yaml` |
| B | `isaaclab/run_configs/2026-07-16_15-29-48_env.yaml` |
| C | `isaaclab/run_configs/2026-07-17_12-24-06_env.yaml` |

All 26 recorded run configs from the project are included in that folder for full transparency.

## Quick start

```bash
git clone https://github.com/Mubeen220/go1-sim2real-walking.git
cd go1-sim2real-walking
pip install -r requirements.txt
pytest tests/                                    # joint maps, observation layout, MLP forward pass

cd mujoco_sim2sim
python3 deploy_sim_test.py                        # reproduces the Policy A sweep above, exactly
```

## Training in Isaac Lab

Training is a standard RL loop: the Go1 in **Isaac Lab** (thousands of parallel environments, with terrain and domain randomisation) receives actions, returns observations (proprioception, IMU, joint states), and a scalar **reward** built from weighted terms (velocity tracking, stability, energy, collisions). **PPO (RSL-RL)** uses that reward to update the policy, and the updated policy drives the next round of simulation.

![The RL loop used for training in Isaac Lab: Go1 → Isaac Lab environment → reward function → PPO (RSL-RL) → updated policy → back to the robot](docs/images/training_isaaclab_rl_loop.png)

*Diagram: the robot/simulated robot acts, Isaac Lab simulates physics, the reward function scores the result, PPO updates the network, and the loop repeats. Only the final actor network is exported and run on the real robot.*

See [`isaaclab/README.md`](isaaclab/README.md) and [`docs/WORKFLOW_REPORT.md`](docs/WORKFLOW_REPORT.md) (Steps 1–3) for the full walkthrough (environment setup, training, exporting, and reducing to numpy weights). Short version:

```bash
# inside an Isaac Lab installation, with isaaclab/ added as a task source
python scripts/rsl_rl/train.py --task Template-Go1-Walk-v0 --num_envs 4096 --headless
python scripts/rsl_rl/play.py --task Template-Go1-Walk-Play-v0 --num_envs 32   # exports policy.onnx / policy.pt
```

The command curriculum reads a global step counter, so training **must run uninterrupted in one continuous run**. A healthy run reaches a linear-velocity tracking reward above about 1.0 and a fall rate below 0.05.

## Steps to deploy on the real Go1

> ⚠️ **Safety first.** The first runs are always with the robot **suspended (feet off the ground)** and the remote in hand, ready to put the robot in damping. Do not put the robot on the ground until the leg motion is correct in the air. `MAX_VEL` is capped at the validated 1.0 m/s.

The full record, with the reasoning behind each step, is in [`docs/WORKFLOW_REPORT.md`](docs/WORKFLOW_REPORT.md) (Steps 3–8). The short checklist:

**1. Export the policy as plain weights** (once, on the training machine)

```python
import torch, numpy as np
pt = torch.jit.load('exported/policy.pt'); pt.eval()
np.savez('policy_weights.npz', **{n: p.detach().cpu().numpy() for n, p in pt.named_parameters()})
```

The robot runs Python 3.6 / PyTorch 1.4, so it cannot load a modern TorchScript or ONNX file. The numpy forward pass in `go1_deploy_common.py` needs only numpy. The deployed Policy A weights are already in `exported_policies/go1_deploy_proven/policy_weights.npz`.

**2. Validate in MuJoCo first (the gate)**

```bash
cd mujoco_sim2sim
python3 deploy_sim_test.py
```

This runs through the same `go1_deploy_common.py` that runs on the robot. A good policy shows a monotonic forward sweep (Policy A: `0.007 / 0.304 / 0.465 / 0.576 / 0.679` for commands `0.1 / 0.3 / 0.5 / 0.7 / 1.0`). A consistent undershoot is normal; a **dead zone** (commands that produce no motion) is not acceptable. Also check the six directions (forward, backward, both strafes, both turns) for the correct sign. Do not go to the robot if this fails.

**3. Copy three files to the robot** (flat into the folder the Walk-These-Ways container mounts)

```bash
scp mujoco_sim2sim/go1_deploy_common.py deployment/deploy_go1_isaaclab.py \
    exported_policies/go1_deploy_proven/policy_weights.npz unitree@192.168.123.15:~/go1_gym/
```

The robot needs the [Walk-These-Ways](https://github.com/Improbable-AI/walk-these-ways) `go1_gym_deploy` package (state estimator and LCM message types). Check that the file paths at the top of `deploy_go1_isaaclab.py` (`sys.path.insert(...)` and `POLICY_PATH`) point to where you put the files.

**4. Suspend the robot and start the low-level bridge** (terminal 1, on the robot host)

```bash
cd ~/go1_gym/go1_gym_deploy/autostart
./start_unitree_sdk.sh      # press Enter only when the robot is hanging in the air
```

**5. Run the policy** (terminal 2, **inside the Walk-These-Ways Docker container**; `lcm` is only installed there)

```bash
python3 deploy_go1_isaaclab.py
```

**6. Step through the R2 state machine (robot still in the air)**

| Press | What happens |
|---|---|
| **R2** (1st) | Slow 2.5 s ramp to the standing pose with soft gains (Kp 5, Kd 1) |
| **R2** (2nd) | Policy runs at Kp 20 / Kd 0.5; the command starts at zero |
| Nudge the **left stick forward** | Legs should make a coordinated walking rhythm |
| **R2** again, **Ctrl-C**, or an automatic power-monitor trip | Safe hold at the default pose |

Check that every joint moves in the correct direction, with no violent or jerky motion and a clean start and stop. If a leg moves the wrong way or jerks: **remote to damping, then Ctrl-C**, and re-check the joint mapping in MuJoCo before retrying.

**7. Ground test and speed ladder** (only after step 6 is correct in every direction)

1. Set the robot down with the command at zero and confirm a stable weight-bearing stand.
2. Start at **0.5 m/s** and confirm a straight, stable walk.
3. Climb **0.5 → 0.7 → 1.0 m/s**, a stable minute at each step; then test backward and turning.
4. Slow to a stop with the stick (the command is rate-limited), do not yank it.

**8. Safe stop**: put the **remote in damping first**, then press Ctrl-C in both terminals.

Safety features built into the script: soft calibration gains during the ramp, ±0.65 N·m gravity-compensation feed-forward on the hips, an approximate power monitor (about 150 W, trips after 5 steps) that falls back to a safe hold, and rate-limited commands.

**Common problems**

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: lcm` on the host | `lcm` is only in the container; run the script inside the Walk-These-Ways Docker container |
| TorchScript load fails (file version) | The robot's torch is too old; use the `.npz` weights and the numpy forward pass (step 1) |
| Robot frozen but stable, ignores commands | The policy does not transfer / has a dead zone; re-validate in MuJoCo |
| Robot stops on ground contact | Usually an R2 press (emergency stop), not a fault; use the stick to command |
| Legs move violently in the air | Joint-order mismatch; damping immediately, re-check the name-based permutations |

## Limitations

- Forward/backward walking transferred cleanly; turning and low-speed tracking were weaker on hardware.
- Policy B fell twice during ground testing; Policy C was never ground-tested given its MuJoCo behaviour.
- Results come from a single robot and a limited number of hardware sessions; surface friction affects transfer and is only approximated in simulation.
- The Go1's neural-network actuator model likely lets policies over-specialize to the training simulator — the most probable explanation for Policy C's metrics/transfer mismatch.
- An earlier development checkpoint (a 48-dimensional observation that *did* include base linear velocity, from a different training stage) exists in the project history but is intentionally **not** included in this repository, since it contradicts the deployable-observation design this project is built around.
- No hardware backflip; that was explored in simulation only and is not part of this repo.

## Acknowledgements

- [Walk These Ways](https://github.com/Improbable-AI/walk-these-ways) (Margolis & Agrawal) for the Go1 LCM deployment transport.
- [legged_gym](https://github.com/leggedrobotics/legged_gym) / [rsl_rl](https://github.com/leggedrobotics/rsl_rl) lineage and NVIDIA Isaac Lab.
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)'s `unitree_go1` model (Unitree Robotics, BSD-3-Clause — included in `mujoco_sim2sim/mujoco_menagerie/unitree_go1/`).
- IBISC laboratory, Université d'Évry Paris-Saclay.

## License

TODO: choose a license for your own code (check with the laboratory first). The bundled `mujoco_menagerie/unitree_go1` model retains its original BSD-3-Clause license (see its own `LICENSE` file).
