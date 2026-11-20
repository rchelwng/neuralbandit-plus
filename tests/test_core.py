import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbplus import algos, data, metrics  # noqa: E402
from nbplus.committee import _forward, _backprop, _exp3_update, _sigmoid  # noqa: E402
from nbplus.runner import run_config  # noqa: E402

SYN = data.make_synthetic(20000, seed=3)


def _net(seed=0, C=6, D=20, K=2):
    rng = np.random.default_rng(seed)
    W1 = rng.uniform(-0.5, 0.5, (1, K, C, D + 1))
    W2 = rng.uniform(-0.5, 0.5, (1, K, C + 1))
    return W1, W2, np.zeros((1, K, C)), np.array([1, 4, 7, 12], np.int64)


def _loss(W1, W2, xi, y, C, D):
    H = np.zeros((1, 2, C))
    o = _sigmoid(_forward(W1, W2, 0, 1, C, C, D, xi, len(xi), H))
    return 0.5 * (y - o) ** 2


def test_backprop_is_gradient_step():
    """Eq. 1 step (g = 1, no dropout) equals minus the gradient of the quadratic loss."""
    C, D, y = 6, 20, 1.0
    W1, W2, H, xi = _net(C=C, D=D)
    a1, a2 = W1.copy(), W2.copy()
    _forward(a1, a2, 0, 1, C, C, D, xi, len(xi), H)
    _backprop(a1, a2, H, 0, 1, C, C, D, xi, len(xi), y, 1.0, False, 0.0, np.zeros(C))
    eps = 1e-6
    for (arr, step, pos) in [(W1, a1 - W1, (0, 1, 2, 4)), (W1, a1 - W1, (0, 1, 0, D)), (W2, a2 - W2, (0, 1, 3)),
                             (W2, a2 - W2, (0, 1, C))]:
        p, m = arr.copy(), arr.copy()
        p[pos] += eps
        m[pos] -= eps
        if arr is W1:
            g = (_loss(p, W2, xi, y, C, D) - _loss(m, W2, xi, y, C, D)) / (2 * eps)
        else:
            g = (_loss(W1, p, xi, y, C, D) - _loss(W1, m, xi, y, C, D)) / (2 * eps)
        assert np.isclose(step[pos], -g, rtol=1e-4, atol=1e-9)
    # inactive inputs and the other arm are untouched
    assert np.all(a1[0, 1, :, 0] == W1[0, 1, :, 0]) and np.all(a1[0, 0] == W1[0, 0])


def test_exp3_normalised_and_exp3s_keeps_floor():
    M = 5
    w0, w1 = np.ones(M) / M, np.ones(M) / M
    for _ in range(3000):
        _exp3_update(w0, 0, 1.0 / 0.2, 0.1, M, 0.0)
        _exp3_update(w1, 0, 1.0 / 0.2, 0.1, M, 1e-3)
    assert np.isclose(w0.sum(), 1) and np.isclose(w1.sum(), 1)
    assert w0[1] < 1e-12                      # EXP3 locks in
    assert w1[1] > 0.5 * np.e * 1e-3 / M      # EXP3.S keeps every model reachable (shifting experts)


def test_determinism_and_paired_streams():
    s = dict(algo="nb2", data="synthetic_small", T=20000, period=10000, hidden=[5, 25], lr=[0.1, 0.1])
    a, b = algos.run(s, 3, SYN), algos.run(s, 3, SYN)
    c = algos.run(s, 4, SYN)
    assert np.array_equal(a["rew"], b["rew"]) and not np.array_equal(a["rew"], c["rew"])
    assert data.stream_order(100, 5)[1] == data.stream_order(100, 5)[1]
    assert np.array_equal(data.stream_order(100, 5)[0], data.stream_order(100, 5)[0])


def test_covtype_binarisation():
    rng = np.random.default_rng(0)
    n = 500
    X = np.zeros((n, 54))
    X[:, :10] = rng.normal(size=(n, 10))
    X[np.arange(n), 10 + rng.integers(0, 4, n)] = 1
    X[np.arange(n), 14 + rng.integers(0, 40, n)] = 1
    idx, D = data.covtype_to_idx(X)
    assert D == 94 and idx.shape == (n, 12) and idx.min() >= 0 and idx.max() < 94
    for j in range(10):  # each continuous column -> 5 equal-frequency bins, all used
        assert set(np.unique(idx[:, j] - 5 * j)) == set(range(5))


def test_recovery_metric():
    block, period, T = 1000, 100_000, 300_000
    acc = np.full(T // block, 0.8)
    acc[100:130] = 0.2       # first drift: recovers 30k rounds after the drift
    acc[200:300] = 0.2       # second drift: never recovers -> censored
    rec = metrics.recovery_times(acc * block, block, period, T)
    assert rec[0] is not None and 30_000 <= rec[0] <= 50_000 and rec[1] is None


def test_baselines_beat_random():
    for algo, p, floor in [("banditron", 0.02, 0.20), ("linucb", 1.0, 0.30), ("lints", 0.1, 0.30)]:
        r = algos.run(dict(algo=algo, data="synthetic_small", T=30000, drift="none", param=p), 0, SYN)
        assert r["rew"].sum() / 30000 > floor, algo  # random ~ 1/7; Banditron learns slowly (paper: 57%)


def test_every_committee_variant_runs_and_learns():
    for extra in [dict(algo="nb1"), dict(algo="nb2"), dict(algo="nb3"), dict(algo="nb2", explore="ts"),
                  dict(algo="nb3", explore="ts"), dict(algo="nb2", alpha=1e-5), dict(algo="nb3", prop="behavior")]:
        r = algos.run(dict(data="synthetic_small", T=40000, drift="none", **extra), 1, SYN)
        assert r["rew"][-10:].sum() / 10000 > 0.25, extra


@pytest.mark.slow
def test_nb3_played_arm_reading_collapses():
    """Updating only the played arm's EXP3 lets over-predicting models take over (the bug in v1)."""
    d = data.make_synthetic(100000)
    s = dict(algo="nb3", data="synthetic", T=150000, drift="none")
    good = algos.run(s, 0, d)["rew"][-30:].sum()
    bad = algos.run(dict(s, nb3_reward="y"), 0, d)["rew"][-30:].sum()
    assert good > bad + 0.1 * 30000


def test_runner_resumes_and_refuses_mixed_settings(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"defaults": {"data": "synthetic_small", "T": 3000, "period": 1000, "seeds": "0-1"},
                               "runs": [{"name": "b", "algo": "banditron", "param": 0.05}]}))
    run_config(str(cfg), results=str(tmp_path / "res"))
    assert len(os.listdir(tmp_path / "res" / "synthetic_small" / "b")) == 3
    run_config(str(cfg), results=str(tmp_path / "res"))  # resume: nothing to do
    with pytest.raises(SystemExit):
        run_config(str(cfg), results=str(tmp_path / "res"), overrides=["T=4000"])
