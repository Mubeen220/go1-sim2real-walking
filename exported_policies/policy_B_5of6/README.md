# Policy B — kept for comparison, not deployed

Trained run: `isaaclab/run_configs/2026-07-16_15-29-48_env.yaml`.

Walks well in 5 of its 6 commanded directions (forward, backward, both
strafes, one turn direction) but turns weakly the other way — an asymmetry
invisible in its aggregate training reward. **Fell twice during ground
testing on hardware** and was not retained as the deployed policy.

Kept here, alongside Policy A, so the three-policy comparison in the
top-level README is fully reproducible — only `policy_weights.npz` (numpy
format) was retained for this one; `.pt`/`.onnx` exports were not kept.
