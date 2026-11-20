"""Offline reference accuracy rho for the regret baseline (paper: an offline classifier at 93%).

Trains a full-information MLP on 80% of the binarised data and reports test accuracy on the other 20%.
Writes data/<name>_reference.json, which scripts/report.py reads. Regret against the oracle (rho = 1) is
reported as well, so the conclusions do not hinge on this number.

  python scripts/make_reference.py covtype
"""
import argparse
import json
import os
import numpy as np
import _path  # noqa: F401
from nbplus import data

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-train", type=int, default=400000)
    a = ap.parse_args()
    from sklearn.neural_network import MLPClassifier
    from scipy import sparse
    for name in a.names:
        d = data.load(name)
        n, nnz = d["idx"].shape
        X = sparse.csr_matrix((np.ones(n * nnz), d["idx"].ravel(), np.arange(0, n * nnz + 1, nnz)), shape=(n, d["D"]))
        rng = np.random.default_rng(0)
        perm = rng.permutation(n)
        cut = int(0.8 * n)
        tr, te = perm[:cut][: a.max_train], perm[cut:]
        clf = MLPClassifier(hidden_layer_sizes=(a.hidden,), early_stopping=True, max_iter=200, random_state=0)
        clf.fit(X[tr], d["y"][tr])
        rho = float((clf.predict(X[te]) == d["y"][te]).mean())
        out = os.path.join(data.DATA_DIR, f"{name}_reference.json")
        with open(out, "w") as f:
            json.dump({"rho": rho, "model": f"MLP({a.hidden}) on binarised features, 80/20 split"}, f)
        print(f"{name}: offline reference accuracy rho = {rho:.4f} -> {out}")
