# Policy reference

| ID | Training run | Config summary | MuJoCo sweep (cmd 0.1 / 0.3 / 0.5 / 0.7 / 1.0) | Hardware result |
|----|--------------|----------------|------------------------------------------------|-----------------|
| **A** | 2026-07-10_12-53-01 | Proven Unitree recipe, curriculum to 1.0 m/s | 0.007 / 0.304 / 0.465 / 0.576 / 0.679 (clean, monotonic) | Suspended test OK; deployed |
| **B** | 2026-07-16_15-29-48 | Proven rewards + all-axis curriculum | Works forward / back / strafe / turn-right; turn-left weak | Suspended OK; **fell twice on ground** |
| **C** | 2026-07-17_12-24-06 | + `feet_clearance` reward, 4500 iterations, domain randomization | −0.007 / −0.001 / 0.103 / 0.770 / 1.209 (**dead zone 0.1–0.5, overshoots at 1.0**) | Not ground-tested |

## Key finding

Policy **C** had the best Isaac Lab training metrics (tracking reward 1.87, fall rate 0.0002,
vx error 0.024) but the **worst** deployment behaviour (dead zone + overshoot). Policy **A** had
more modest training metrics but transferred cleanly.

**Training-simulator metrics did not predict transfer quality.**

## Training log → policy mapping

| Log (`docs/results/training_logs/`) | What it was |
|---|---|
| `train_final.log` | Contributed to the A-family runs |
| `train_6dir.log` | Policy B (all-direction curriculum) |
| `train_clearance_fix.log` | Policy C (`feet_clearance` experiment) |
| `train_enhanced.log` | Failed run: over-penalised, did not walk |
| `train_stdfix.log` | Failed run: tracking std too tight, gait collapse |
| `train_alldir.log` | No-curriculum experiment |
| `train_resume.log` | Interrupted / resumed run (curriculum counter reset bug) |
