"""Run one (method, seed) from a flat settings dict. This is the only entry point the scripts use."""
import time
import numpy as np
from . import data as data_mod
from .committee import run_committee, default_grid
from .baselines import run_banditron, run_linear, run_random

DEFAULTS = dict(
    T=2_500_000, period=500_000, drift="abrupt", block=1000,
    # NeuralBandit (paper values)
    gamma=0.005, gamma_model=0.1, hidden=None, lr=None,
    # interpretation choices (paper silent)
    ips_clip=100.0, prop="auto", nb3_reward="all",
    # extensions
    explore="eps", p_drop=0.1, min_h_drop=25, S=4, alpha=0.0,
    # baselines: Banditron gamma / LinUCB alpha / LinTS v
    param=None,
)
ALGOS = ("random", "banditron", "linucb", "lints", "nb1", "nb2", "nb3")
# settings that change results (used to refuse mixing results produced with different settings)
_IGNORED = {"name", "note", "label", "seeds"}


def resolve(settings):
    s = dict(DEFAULTS)
    s.update(settings)
    if s["algo"] not in ALGOS:
        raise ValueError(f"unknown algo {s['algo']}")
    if s["drift"] not in ("abrupt", "none"):
        raise ValueError("drift must be 'abrupt' or 'none'")
    if s["algo"] in ("nb2", "nb3") and s["hidden"] is None:
        h, l = default_grid()
        s["hidden"], s["lr"] = h.tolist(), l.tolist()
    if s["algo"] == "nb1" and s["hidden"] is None:
        s["hidden"], s["lr"] = [25], [0.1]
    if s["prop"] == "auto":  # each model trains as its own NB1 instance (own propensity), NB2 and NB3 alike
        s["prop"] = "own"
    if s["algo"] in ("banditron", "linucb", "lints") and s["param"] is None:
        raise ValueError(f"{s['algo']} needs 'param' (run the tuning stage first)")
    s["ips_clip"] = float(s["ips_clip"])
    return s


def fingerprint(settings):
    s = resolve(settings)
    keep = {k: v for k, v in s.items() if k not in _IGNORED and k != "seed"}
    if s["algo"] in ("random", "banditron", "linucb", "lints"):
        for k in ("gamma", "gamma_model", "hidden", "lr", "ips_clip", "prop", "nb3_reward", "explore",
                  "p_drop", "min_h_drop", "S", "alpha"):
            keep.pop(k, None)
    if s["algo"] not in ("banditron", "linucb", "lints"):
        keep.pop("param", None)
    return keep


def run(settings, seed, dataset=None):
    s = resolve(settings)
    d = dataset if dataset is not None else data_mod.load(s["data"])
    idx, y, D, K = d["idx"], d["y"], d["D"], d["K"]
    order, start = data_mod.stream_order(len(y), seed)
    T, block = int(s["T"]), int(s["block"])
    period = int(s["period"]) if s["drift"] == "abrupt" else 0
    aseed = 7919 * seed + 17
    t0 = time.time()
    sel = corr = w = None
    a = s["algo"]
    if a == "random":
        rew = run_random(y, order, start, T, period, K, block, aseed)
    elif a == "banditron":
        rew = run_banditron(idx, y, order, start, T, period, K, D, float(s["param"]), block, aseed)
    elif a in ("linucb", "lints"):
        rew = run_linear(idx, y, order, start, T, period, K, D, float(s["param"]), 1 if a == "lints" else 0,
                         block, aseed)
    else:
        rew, sel, corr, w = run_committee(
            idx, y, order, start, T, period, K, D,
            np.array(s["hidden"], np.int64), np.array(s["lr"], np.float64),
            2 if a == "nb3" else 1, 1 if s["explore"] == "ts" else 0,
            float(s["gamma"]), float(s["gamma_model"]), float(s["alpha"]), s["ips_clip"],
            float(s["p_drop"]), int(s["min_h_drop"]), int(s["S"]),
            0 if s["prop"] == "own" else 1, {"y": 0, "pred": 1, "all": 2}[s["nb3_reward"]], block, aseed)
    out = dict(rew=np.asarray(rew), block=block, T=T, period=period, K=K, runtime=time.time() - t0, seed=seed)
    if sel is not None:
        out.update(sel=sel, corr=corr, w=w)
    return out
