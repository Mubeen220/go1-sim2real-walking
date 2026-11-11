# Isaac Lab training

The task that trains the Go1 walking policy. Built on NVIDIA Isaac Lab's
manager-based `LocomotionVelocityRoughEnvCfg`.

## Files

- `go1_walk_env_cfg.py` — the environment: observations, rewards, curriculum, domain randomization. **See the top-level README's "reproducing Policy A exactly" note** — this file's current state matches Policy C's recipe (it includes the `feet_clearance` term), not Policy A's or B's. Use `run_configs/` for an exact per-run reproduction.
- `rewards.py` — custom reward terms (`feet_swing_clearance`, `energy`) and the command curriculum function `lin_vel_cmd_levels`.
- `__init__.py` — registers the task's MDP module.
- `train.py` — RSL-RL training entry point (standard Isaac Lab script).
- `play.py` — runs a trained checkpoint and exports `policy.onnx` + `policy.pt`.
- `play_and_log.py` — runs a policy and logs a trajectory CSV for analysis (used to produce the files in `docs/results/trajectories/`).
- `cli_args.py`, `list_envs.py`, `random_agent.py`, `zero_agent.py` — standard Isaac Lab RSL-RL utility scripts (list registered tasks, sanity-check an environment with random/zero actions).
- `run_configs/` — the exact, serialized environment config recorded at the start of every training run in this project (26 total). This is the authoritative record of what each specific policy was actually trained with — more reliable than the current `.py` source for reproducing a specific result.

## Observation contract (must match `mujoco_sim2sim/go1_deploy_common.py` exactly)

45-dimensional, single frame, no history:

```
self.observations.policy.base_lin_vel = None   # excluded: unmeasurable on hardware
self.observations.policy.history_length = 0
self.observations.policy.base_ang_vel.scale = 0.2
self.observations.policy.joint_vel.scale = 0.05
```

## Training

```bash
# from inside an Isaac Lab installation, with this folder added as a task source
python scripts/rsl_rl/train.py --task Template-Go1-Walk-v0 --num_envs 4096 --headless | tee train.log
```

Healthy convergence looks like: linear-tracking reward > ~1.0, fall rate < 0.05, mean reward strongly positive, curriculum reaching its ceiling. The command curriculum (`lin_vel_cmd_levels` in `rewards.py`) ramps forward/lateral/yaw commands together from ±0.1 up to (1.5, 0.7, 1.0) m/s and rad/s over the first ~20000 environment steps, then holds — **it must run uninterrupted**, since it reads a global step counter that resets on a restarted run.

## Exporting a trained policy

```bash
python scripts/rsl_rl/play.py --task Template-Go1-Walk-Play-v0 --num_envs 32
# writes exported/policy.onnx and exported/policy.pt next to the checkpoint
```

Always confirm you're using the *newest* export by timestamp (`ls -dt`), not a plain `ls`.

To get the framework-independent numpy weights consumed by `go1_deploy_common.Go1Policy` (see `mujoco_sim2sim/README.md`):

```python
import torch, numpy as np
pt = torch.jit.load('exported/policy.pt'); pt.eval()
w = {n: p.detach().cpu().numpy() for n, p in pt.named_parameters()}
np.savez('policy_weights.npz', **w)
```

Full details, including the reasoning behind each step: [`../docs/WORKFLOW_REPORT.md`](../docs/WORKFLOW_REPORT.md).
