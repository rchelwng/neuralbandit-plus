"""Time a short run of each method on your machine and print the estimated core-hours of every stage.

  python scripts/benchmark.py            # uses data/covtype.npz if prepared, else the synthetic stream
"""
import json
import os
import time
import _path  # noqa: F401
from nbplus import algos, data
from nbplus.runner import ROOT, parse_seeds

if __name__ == "__main__":
    name = "covtype" if os.path.exists(os.path.join(data.DATA_DIR, "covtype.npz")) else "synthetic"
    d = data.load(name)
    T = 100_000
    rate = {}
    for key, s in {"nb": dict(algo="nb2"), "nb_ts": dict(algo="nb2", explore="ts"), "linear": dict(algo="linucb", param=1.0),
                   "banditron": dict(algo="banditron", param=0.02)}.items():
        algos.run(dict(s, data=name, T=2000), 0, d)  # compile
        t = time.time()
        algos.run(dict(s, data=name, T=T), 0, d)
        rate[key] = (time.time() - t) / T
        print(f"{key:10s} {rate[key] * 2.5e6 / 60:6.1f} min per 2.5M-round run  ({name})")
    for cfg in ["tuning", "reproduction", "extensions", "ablations", "extra_datasets"]:
        c = json.load(open(os.path.join(ROOT, "configs", f"{cfg}.json")))
        runs = [dict(c.get("defaults", {}), **r) for r in c["runs"]]
        sec = 0.0
        for s in runs:
            k = {"random": None, "banditron": "banditron", "linucb": "linear", "lints": "linear"}.get(s["algo"], "nb")
            if k == "nb" and s.get("explore") == "ts":
                k = "nb_ts"
            if k:
                sec += rate[k] * float(s["T"]) * len(parse_seeds(s.get("seeds", "0-9")))
        note = " per dataset (letter, K=26, is ~3.5x slower)" if cfg == "extra_datasets" else ""
        print(f"{cfg:15s} ~{sec / 3600:5.1f} core-hours{note}  -> divide by --workers")
