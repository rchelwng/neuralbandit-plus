"""Datasets as sparse binary contexts.

Every dataset is stored as ``idx`` (N x nnz int32): the indices of the active binary features of each row.
Every row has exactly ``nnz`` active features (one per discretised column / one-hot group), so the
networks only touch ``nnz`` input weights per hidden unit.

Covertype follows the original paper: each of the 10 continuous columns is discretised into 5
equal-frequency bins (one-hot), the 4 wilderness and 40 soil indicators are kept, giving 94 binary
features with 12 active per row. Other UCI/OpenML datasets use the same rule (5 equal-frequency
bins per continuous column).
"""
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_BINS = 5


def equal_frequency_bins(col, n_bins=N_BINS):
    """Bin index 0..n_bins-1 using quantile edges computed on the whole column (as in the paper)."""
    edges = np.quantile(col, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, col, side="right").astype(np.int32)


def continuous_to_idx(Xc, offset=0, n_bins=N_BINS):
    """One active index per continuous column."""
    cols = [offset + j * n_bins + equal_frequency_bins(Xc[:, j], n_bins) for j in range(Xc.shape[1])]
    return np.stack(cols, axis=1).astype(np.int32), Xc.shape[1] * n_bins


def covtype_to_idx(X):
    """Raw 54-column Covertype -> (idx [N,12], D=94)."""
    X = np.asarray(X)
    idx_c, d_c = continuous_to_idx(X[:, :10].astype(float))
    wild = np.argmax(X[:, 10:14], axis=1) + d_c
    soil = np.argmax(X[:, 14:54], axis=1) + d_c + 4
    assert np.all(X[:, 10:14].sum(1) == 1) and np.all(X[:, 14:54].sum(1) == 1), "one-hot groups expected"
    idx = np.concatenate([idx_c, wild[:, None], soil[:, None]], axis=1).astype(np.int32)
    return idx, d_c + 4 + 40


def save(name, idx, y, D, meta=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    classes, y = np.unique(y, return_inverse=True)
    path = os.path.join(DATA_DIR, f"{name}.npz")
    np.savez_compressed(path, idx=idx.astype(np.int32), y=y.astype(np.int32), D=np.int64(D),
                        K=np.int64(len(classes)), classes=classes.astype(str))
    return path


def load(name):
    """Return dict(idx, y, D, K). ``synthetic`` / ``synthetic_small`` are generated on the fly."""
    if name.startswith("synthetic"):
        n = 20000 if name == "synthetic_small" else 100000
        return make_synthetic(n, seed=12345)
    path = os.path.join(DATA_DIR, f"{name}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run  python scripts/prepare_data.py {name}")
    z = np.load(path)
    return dict(idx=z["idx"], y=z["y"], D=int(z["D"]), K=int(z["K"]))


def make_synthetic(n=100000, seed=12345, noise=0.1):
    """Covertype-shaped stream: 94 binary features, 12 active (10 ordinal 5-level groups, a 4-group, a
    40-group), 7 imbalanced classes from a random one-hidden-layer teacher, ``noise`` label noise."""
    rng = np.random.default_rng(seed)
    cols = [j * 5 + rng.integers(0, 5, n) for j in range(10)]
    cols.append(50 + rng.integers(0, 4, n))
    cols.append(54 + rng.choice(40, n, p=rng.dirichlet(np.ones(40) * 0.5)))
    idx = np.stack(cols, 1).astype(np.int32)
    D, K, H = 94, 7, 24
    V = rng.normal(0, 1.0, (D, H))
    U = rng.normal(0, 1.0, (H, K))
    hidden = np.tanh(V[idx].sum(1))
    logits = hidden @ U + np.log(np.array([0.36, 0.49, 0.06, 0.005, 0.016, 0.03, 0.035]))[None, :] * 0.6
    y = logits.argmax(1)
    flip = rng.random(n) < noise
    y[flip] = rng.integers(0, K, flip.sum())
    return dict(idx=idx, y=y.astype(np.int32), D=D, K=K)


def stream_order(n, seed):
    """Per-seed shuffled order and random start (paper: shuffled data replayed in a loop, random start).
    Depends only on the seed, so every method sees the same stream for a given seed (paired comparisons)."""
    rng = np.random.default_rng(1_000_003 + seed)
    return rng.permutation(n).astype(np.int64), int(rng.integers(n))
