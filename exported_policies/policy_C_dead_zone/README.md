# Policy C — best training metrics, worst transfer, not deployed

Trained run: `isaaclab/run_configs/2026-07-17_12-24-06_env.yaml` — includes an
added `feet_clearance` reward term (target height 8cm) that neither Policy A
nor Policy B's runs used, and the longest training of the three.

By Isaac Lab training-time metrics, this was the **best** policy: highest
linear-velocity tracking reward, lowest fall rate. Through the deployment
code path in MuJoCo it was the **worst**: a dead zone below 0.5 m/s and a
high-speed overshoot. Directly measured swing-foot clearance was also
*lower* and more uneven than baseline — the opposite of what the added
reward term was meant to achieve. Never ground-tested, given this.

This is the central empirical finding of the project: training-simulator
metrics did not predict transfer quality.

MuJoCo sweep (via `../../mujoco_sim2sim/deploy_sim_test.py`):

```
cmd=+0.10  achieved_vx=-0.007
cmd=+0.30  achieved_vx=-0.001
cmd=+0.50  achieved_vx=+0.103
cmd=+0.70  achieved_vx=+0.770
cmd=+1.00  achieved_vx=+1.209
```

Verified in this repo: loading `policy_weights.npz` here through
`deploy_sim_test.py` (via `GO1_POLICY_WEIGHTS`) reproduces these numbers
exactly.
