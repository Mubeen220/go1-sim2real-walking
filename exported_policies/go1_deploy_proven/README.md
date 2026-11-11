# Policy A — the deployed policy

This is the policy that was actually run on the physical Go1, and never fell.
Trained run: `isaaclab/run_configs/2026-07-10_12-53-01_env.yaml`.

| File | What it is |
|---|---|
| `policy_weights.npz` | Framework-independent numpy weights, consumed by `go1_deploy_common.Go1Policy`. **This is what the deployment scripts use.** |
| `policy.pt` | TorchScript export (intermediate artifact on the way to the `.npz` — not loadable on the robot's old PyTorch, kept for reference). |
| `policy.onnx` | ONNX export, used by `mujoco_sim2sim/mujoco_run.py` as an independent cross-check of the numpy path. |

All three contain the same actor MLP: 45 → 512 → 256 → 128 → 12, ELU activations.
`policy_weights.npz` holds the keys `mlp.{0,2,4,6}.{weight,bias}`.

MuJoCo sweep (via the deployment code path, `../../mujoco_sim2sim/deploy_sim_test.py`):

```
cmd=+0.10  achieved_vx=+0.007
cmd=+0.30  achieved_vx=+0.304
cmd=+0.50  achieved_vx=+0.465
cmd=+0.70  achieved_vx=+0.576
cmd=+1.00  achieved_vx=+0.679
```
