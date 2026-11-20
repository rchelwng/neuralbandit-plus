"""Metrics from per-block reward counts.

* cumulative regret R(t) = rho * t - sum_{s<=t} r_s, rho = offline reference accuracy (paper: 93% classifier);
  we also report oracle regret (rho = 1), which needs no reference model.
* recovery time after a drift at t_d: pre-drift level L = mean accuracy over the ``pre`` rounds before t_d;
  recovery is the first round t > t_d (with a full trailing window of ``win`` rounds after t_d) at which the
  trailing-window accuracy reaches ``frac`` * L. Censored if this does not happen before the next drift / T.
* final accuracy = accuracy over the last 100,000 rounds (the paper's stationary number).
"""
import numpy as np


def cum_regret(rew, block, rho):
    t = np.arange(1, len(rew) + 1) * block
    return rho * t - np.cumsum(rew)


def final_regret(rew, block, rho, T=None):
    T = T if T is not None else len(rew) * block
    return rho * T - rew.sum()


def last_acc(rew, block, last=100_000):
    nb = max(1, last // block)
    return rew[-nb:].sum() / (nb * block)


def moving_acc(rew, block, win=50_000):
    nb = max(1, win // block)
    c = np.convolve(rew, np.ones(nb), mode="full")[: len(rew)]
    denom = np.minimum(np.arange(1, len(rew) + 1), nb) * block
    return c / denom


def recovery_times(rew, block, period, T, pre=50_000, win=20_000, frac=0.95):
    """List of (recovery_rounds or None if censored) for every drift."""
    if period <= 0:
        return []
    acc = rew / block
    out = []
    npre, nwin = pre // block, win // block
    for td in range(period, T, period):
        bd = td // block
        if bd - npre < 0:
            continue
        level = acc[bd - npre:bd].mean()
        end = min(T, td + period) // block
        rec = None
        for b in range(bd + nwin, end + 1):
            if acc[b - nwin:b].mean() >= frac * level:
                rec = b * block - td
                break
        out.append(rec)
    return out


def recovery_summary(list_of_lists):
    """Mean recovery over uncensored drifts, number censored, number of drifts."""
    flat = [r for l in list_of_lists for r in l]
    ok = [r for r in flat if r is not None]
    return (float(np.mean(ok)) if ok else float("nan")), len(flat) - len(ok), len(flat)




def paired_stats(a, b, n_boot=20000, seed=0):
    """Paired difference b - a over seeds: mean, 95% bootstrap CI, exact Wilcoxon p (two-sided)."""
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        p = float(wilcoxon(a, b).pvalue) if len(d) >= 2 and np.any(d != 0) else float("nan")
    except ValueError:
        p = float("nan")
    return float(d.mean()), float(lo), float(hi), p


def bootstrap_ci(x, n_boot=20000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    boots = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
