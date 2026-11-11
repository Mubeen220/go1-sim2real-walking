"""Smoke tests for the shared deployment module (no robot, no simulator needed).

Run:  pytest tests/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mujoco_sim2sim"))
import go1_deploy_common as dep  # noqa: E402


def _random_weights(path, seed=0):
    rng = np.random.RandomState(seed)
    dims = [45, 512, 256, 128, 12]
    arrays = {}
    for i, idx in enumerate([0, 2, 4, 6]):
        arrays["mlp.%d.weight" % idx] = rng.randn(dims[i + 1], dims[i]).astype(np.float32) * 0.1
        arrays["mlp.%d.bias" % idx] = np.zeros(dims[i + 1], dtype=np.float32)
    np.savez(path, **arrays)


def test_permutations_are_bijections():
    ident = np.arange(12)
    assert np.array_equal(dep.HW_TO_POLICY[dep.POLICY_TO_HW], ident)
    assert np.array_equal(dep.POLICY_TO_HW[dep.HW_TO_POLICY], ident)
    assert np.array_equal(dep.WTW_TO_POLICY[dep.POLICY_TO_WTW], ident)


def test_roundtrip_raw_and_wtw():
    v = np.arange(12, dtype=float)
    assert np.array_equal(dep.policy_to_raw(dep.raw_to_policy(v)), v)
    assert np.array_equal(dep.policy_to_wtw(dep.wtw_to_policy(v)), v)


def test_named_joint_mapping():
    raw = np.arange(12)
    assert dep.raw_to_policy(raw)[0] == 3
    wtw = np.arange(12)
    assert dep.wtw_to_policy(wtw)[0] == 0
    assert dep.raw_to_policy(raw)[4] == 4
    assert dep.wtw_to_policy(wtw)[4] == 1


def test_observation_layout_and_scaling(tmp_path):
    w = str(tmp_path / "w.npz")
    _random_weights(w)
    pol = dep.Go1Policy(w)
    obs = pol.build_observation(
        base_ang_vel=[1, 1, 1],
        projected_gravity=[0, 0, -1],
        velocity_command=[0.5, 0.0, 0.0],
        joint_pos_policy=dep.DEFAULT_POS_POLICY,
        joint_vel_policy=np.ones(12),
    )
    assert obs.shape == (45,)
    assert np.allclose(obs[0:3], dep.ANG_VEL_SCALE)
    assert np.allclose(obs[3:6], [0, 0, -1])
    assert np.allclose(obs[6:9], [0.5, 0, 0])
    assert np.allclose(obs[9:21], 0.0)
    assert np.allclose(obs[21:33], dep.JOINT_VEL_SCALE)
    assert np.allclose(obs[33:45], 0.0)


def test_infer_and_action_feedback(tmp_path):
    w = str(tmp_path / "w.npz")
    _random_weights(w)
    pol = dep.Go1Policy(w)
    obs = np.zeros(45, dtype=np.float32)
    a = pol.infer(obs)
    assert a.shape == (12,) and np.all(np.isfinite(a))
    assert np.array_equal(pol.last_action, a.astype(np.float32))
    tgt = pol.action_to_joint_target_policy(a)
    assert np.allclose(tgt, dep.DEFAULT_POS_POLICY + dep.ACTION_SCALE * a)
    pol.reset()
    assert np.all(pol.last_action == 0)


def test_real_policy_weights_load_and_match_thesis_sweep():
    """Loads the actual deployed Policy A weights and checks the forward
    pass runs (full numeric sweep is covered by mujoco_sim2sim/deploy_sim_test.py,
    which reproduces the thesis's reported values exactly)."""
    weights = os.path.join(os.path.dirname(__file__), "..", "exported_policies",
                            "go1_deploy_proven", "policy_weights.npz")
    if not os.path.exists(weights):
        import pytest
        pytest.skip("policy_weights.npz not present")
    pol = dep.Go1Policy(weights)
    obs = np.zeros(45, dtype=np.float32)
    a = pol.infer(obs)
    assert a.shape == (12,) and np.all(np.isfinite(a))


def test_all_three_policy_weights_load():
    """Policies A, B and C should all load and produce a finite 12-dim action.
    Their actual walking behaviour (clean/weak-turn/dead-zone) only shows up
    over many control steps in mujoco_sim2sim/deploy_sim_test.py -- this just
    confirms each .npz is well-formed and compatible with Go1Policy."""
    import pytest
    base = os.path.join(os.path.dirname(__file__), "..", "exported_policies")
    for name, rel in [
        ("A", os.path.join("go1_deploy_proven", "policy_weights.npz")),
        ("B", os.path.join("policy_B_5of6", "policy_weights.npz")),
        ("C", os.path.join("policy_C_dead_zone", "policy_weights.npz")),
    ]:
        path = os.path.join(base, rel)
        if not os.path.exists(path):
            pytest.skip(f"Policy {name} weights not present at {rel}")
            continue
        pol = dep.Go1Policy(path)
        a = pol.infer(np.zeros(45, dtype=np.float32))
        assert a.shape == (12,) and np.all(np.isfinite(a)), f"Policy {name} produced bad output"
