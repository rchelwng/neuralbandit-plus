"""Baselines of the original comparison: Banditron, LinUCB, linear Thompson sampling (CTS), uniform random.

All use the same sparse binary context (plus a bias feature) and the same stream as the committee.
* Banditron (Kakade et al. 2008): multiclass perceptron, eps-greedy with gamma, unbiased update
  U = x (1[y_tilde correct] / P(y_tilde) e_{y_tilde} - e_{y_hat}).
* LinUCB (Li et al. 2010), disjoint model: ridge regression per arm (A = I + sum x x^T), score
  theta^T x + alpha sqrt(x^T A^-1 x). A^-1 is kept with Sherman-Morrison.
* CTS / LinTS (Agrawal & Goyal 2013): theta_tilde ~ N(theta, v^2 A^-1) per arm. Only theta_tilde^T x is
  needed, and it is exactly N(theta^T x, v^2 x^T A^-1 x), so we sample that scalar directly.
"""
import math
import numpy as np
from .jit import njit


@njit
def run_banditron(idx, labels, order, start, T, period, K, D, gamma, block, seed):
    np.random.seed(seed)
    nnz = idx.shape[1]
    N = order.shape[0]
    W = np.zeros((K, D + 1))
    nblk = (T + block - 1) // block
    rew = np.zeros(nblk, np.int64)
    for t in range(T):
        row = order[(start + t) % N]
        xi = idx[row]
        lab = labels[row]
        if period > 0:
            lab = (lab + t // period) % K
        best = -1e300
        yh = 0
        off = np.random.randint(K)  # random tie-break (all-zero weights at start)
        for kk in range(K):
            k = (kk + off) % K
            s = W[k, D]
            for j in range(nnz):
                s += W[k, xi[j]]
            if s > best:
                best = s
                yh = k
        kp = np.random.randint(K) if np.random.random() < gamma else yh
        P = (1.0 - gamma) * (1.0 if kp == yh else 0.0) + gamma / K
        for j in range(nnz):
            W[yh, xi[j]] -= 1.0
        W[yh, D] -= 1.0
        if kp == lab:
            rew[t // block] += 1
            for j in range(nnz):
                W[kp, xi[j]] += 1.0 / P
            W[kp, D] += 1.0 / P
    return rew


@njit
def run_linear(idx, labels, order, start, T, period, K, D, param, use_ts, block, seed):
    """use_ts = 0: LinUCB with alpha = param; use_ts = 1: LinTS with v = param."""
    np.random.seed(seed)
    nnz = idx.shape[1]
    N = order.shape[0]
    d = D + 1
    Ainv = np.zeros((K, d, d))
    for k in range(K):
        for i in range(d):
            Ainv[k, i, i] = 1.0
    bvec = np.zeros((K, d))
    theta = np.zeros((K, d))
    act = np.zeros(nnz + 1, np.int64)
    u = np.zeros(d)
    nblk = (T + block - 1) // block
    rew = np.zeros(nblk, np.int64)
    for t in range(T):
        row = order[(start + t) % N]
        xi = idx[row]
        lab = labels[row]
        if period > 0:
            lab = (lab + t // period) % K
        for j in range(nnz):
            act[j] = xi[j]
        act[nnz] = D
        best = -1e300
        kp = 0
        off = np.random.randint(K)
        for kk in range(K):
            k = (kk + off) % K
            mu = 0.0
            q = 0.0
            for a in range(nnz + 1):
                mu += theta[k, act[a]]
                for c in range(nnz + 1):
                    q += Ainv[k, act[a], act[c]]
            sd = math.sqrt(q if q > 0.0 else 0.0)
            if use_ts == 1:
                s = mu + param * sd * np.random.standard_normal()
            else:
                s = mu + param * sd
            if s > best:
                best = s
                kp = k
        r = 1.0 if kp == lab else 0.0
        if r > 0.0:
            rew[t // block] += 1
        # Sherman-Morrison on arm kp: u = A^-1 x
        q = 0.0
        for i in range(d):
            s = 0.0
            for a in range(nnz + 1):
                s += Ainv[kp, i, act[a]]
            u[i] = s
        for a in range(nnz + 1):
            q += u[act[a]]
        den = 1.0 + q
        for i in range(d):
            ui = u[i] / den
            if ui != 0.0:
                for c in range(d):
                    Ainv[kp, i, c] -= ui * u[c]
        if r > 0.0:
            for a in range(nnz + 1):
                bvec[kp, act[a]] += 1.0
        for i in range(d):  # theta = A^-1 b (A^-1 changed even when r = 0)
            s = 0.0
            for c in range(d):
                s += Ainv[kp, i, c] * bvec[kp, c]
            theta[kp, i] = s
    return rew


@njit
def run_random(labels, order, start, T, period, K, block, seed):
    np.random.seed(seed)
    N = order.shape[0]
    nblk = (T + block - 1) // block
    rew = np.zeros(nblk, np.int64)
    for t in range(T):
        lab = labels[order[(start + t) % N]]
        if period > 0:
            lab = (lab + t // period) % K
        if np.random.randint(K) == lab:
            rew[t // block] += 1
    return rew
