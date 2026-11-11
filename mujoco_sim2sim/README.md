# MuJoCo sim-to-sim validation

`go1_deploy_common.py` is the **single source of truth** for deployment: observation
construction, joint permutations, scaling constants, PD gains, and the pure-numpy MLP.
It is imported unchanged by the robot script in `../deployment/`, which is the whole
point — a result here is genuine evidence about hardware behaviour, not a parallel
reimplementation that happens to agree.

## Files

- `go1_deploy_common.py` — shared module (see top-level README for the full interface).
- `deploy_sim_test.py` — the actual validation gate: runs a forward-speed sweep through `go1_deploy_common`'s numpy policy. **Verified in this repo to reproduce the thesis's reported Policy A numbers exactly** (`+0.007, +0.304, +0.465, +0.576, +0.679` for commands `0.1–1.0` m/s).
- `deploy_sim_viewer.py` — interactive MuJoCo viewer running the same numpy code path (arrow keys / A/D to command, Space to stop, R to reset). Click the viewer window first so it captures key presses.
- `mujoco_run.py` — a second, independent runner that loads `policy.onnx` via ONNX Runtime instead of the numpy path, for an extra cross-check. Supports `--mode interactive` (viewer + keyboard) and `--mode scripted` (fixed command sequence, logs a trajectory CSV).
- `mujoco_menagerie/unitree_go1/` — the Go1 MJCF model (Unitree Robotics, BSD-3-Clause), vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Running

```bash
pip install mujoco onnxruntime
python3 deploy_sim_test.py          # speed sweep, prints achieved vx at each command
python3 mujoco_run.py --mode scripted --log_csv out.csv
python3 deploy_sim_viewer.py        # interactive (needs a display)
```

All three default to `../exported_policies/go1_deploy_proven/` for weights, and to
`mujoco_menagerie/unitree_go1/scene.xml` (relative to this folder) for the model.
Override the policy path with the `GO1_POLICY_WEIGHTS` env var (numpy scripts) or
`GO1_POLICY_ONNX` (`mujoco_run.py`).

## What "pass" looks like

A validated policy shows a **monotonic** speed sweep with the usual undershoot at
higher commands (Policy A). A **dead zone** (near-zero achieved velocity across a
range of nonzero commands) is not acceptable and indicates a transfer problem — see
Policy C in the top-level README for exactly this failure mode, caught here before
it ever reached hardware.

Also run a six-direction check (forward, backward, both strafes, both turns) and
confirm each responds with the correct sign — this is what produced the
`*_six_directions.png` plots in `../docs/results/`.
