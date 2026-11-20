"""Download and binarise datasets (needs internet once).

  python scripts/prepare_data.py covtype            # Forest Cover Type -> data/covtype.npz (94 binary features)
  python scripts/prepare_data.py pendigits letter shuttle
"""
import argparse
import numpy as np
import _path  # noqa: F401
from nbplus import data

OPENML = {"pendigits": ("pendigits", 1), "letter": ("letter", 1), "shuttle": ("shuttle", 1)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    for name in ap.parse_args().names:
        if name == "covtype":
            from sklearn.datasets import fetch_covtype
            X, y = fetch_covtype(return_X_y=True)
            idx, D = data.covtype_to_idx(X)
        elif name in OPENML:
            from sklearn.datasets import fetch_openml
            oname, ver = OPENML[name]
            Xdf, y = fetch_openml(oname, version=ver, return_X_y=True, as_frame=True, parser="auto")
            X = Xdf.apply(lambda c: c.astype("category").cat.codes if c.dtype.name in ("category", "object") else c)
            X = X.to_numpy(dtype=float)
            y = np.asarray(y).astype(str)
            idx, D = data.continuous_to_idx(X)
        else:
            raise SystemExit(f"unknown dataset {name}; choose covtype, {', '.join(OPENML)}")
        path = data.save(name, idx, y, D)
        d = data.load(name)
        counts = np.bincount(d["y"]) / len(d["y"])
        print(f"{name}: N={len(d['y'])}, D={D} binary features ({idx.shape[1]} active per row), K={d['K']}, "
              f"majority class {counts.max():.3f} -> {path}")
