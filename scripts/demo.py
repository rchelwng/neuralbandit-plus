"""Live demo for the talk (~2 min): NB2, NB2+EXP3.S, NB2+MCD-TS and LinUCB on a drifting stream, then a plot.

  python scripts/demo.py               # Covertype if prepared, else synthetic
  python scripts/demo.py --T 600000 --period 200000
"""
import argparse
import os
import time
import numpy as np
import _path  # noqa: F401
from nbplus import algos, data, metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=400000)
    ap.add_argument("--period", type=int, default=200000)
    a = ap.parse_args()
    name = "covtype" if os.path.exists(os.path.join(data.DATA_DIR, "covtype.npz")) else "synthetic"
    d = data.load(name)
    runs = [("LinUCB", dict(algo="linucb", param=1.0)), ("NB2", dict(algo="nb2")),
            ("NB2+EXP3.S", dict(algo="nb2", alpha=1.0 / a.T)), ("NB2+MCD-TS", dict(algo="nb2", explore="ts"))]
    out = {}
    for lab, s in runs:
        s = dict(s, data=name, T=a.T, period=a.period)
        algos.run(dict(s, T=2000), 0, d)
        t = time.time()
        r = algos.run(s, 0, d)
        out[lab] = r["rew"]
        print(f"{lab:12s} accuracy {r['rew'].sum() / a.T:.3f}   oracle regret {a.T - r['rew'].sum():>8,d}   "
              f"{time.time() - t:5.1f}s", flush=True)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for lab, rew in out.items():
        t = np.arange(1, len(rew) + 1) * 1000
        ax[0].plot(t, metrics.cum_regret(rew, 1000, 1.0), label=lab)
        ax[1].plot(t, metrics.moving_acc(rew, 1000, 20000), label=lab)
    for x in ax:
        for td in range(a.period, a.T, a.period):
            x.axvline(td, color="grey", ls=":")
        x.set_xlabel("round")
    ax[0].set_ylabel("cumulative regret (oracle)")
    ax[1].set_ylabel("accuracy (20k moving avg.)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig("demo.png", dpi=120)
    print("saved demo.png")
