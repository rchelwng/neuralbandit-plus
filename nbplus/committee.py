"""NeuralBandit 1/2/3 (Allesiardo, Feraud, Bouneffouf 2014) and our two extensions, in one compiled loop.

Notation follows the paper.
* One MLP per arm k (one hidden layer of C sigmoid units, sigmoid output N^k(x) ~ P(reward=1 | x, k)),
  quadratic loss, weights ~ U(-0.5, 0.5), biases included.
* NB1 (Alg. 1): k_hat = argmax_k N^k(x); P(k) = (1-gamma) 1[k=k_hat] + gamma/K; play k ~ P; update the
  played arm's network with the importance-weighted backprop step of Eq. 1, step = lambda * delta * x / P(k).
* NB2 (Alg. 2): M NB1 instances ("models", different C and lambda); EXP3 over models (Eqs. 2-3,
  gamma_model); the drawn model chooses the arm; every model's network for the played arm is updated with
  Eq. 1; the EXP3 reward is the observed reward y.
* NB3 (Alg. 3): one EXP3 per arm; for every arm k a model m^k is drawn and scores the arm with N^{m^k,k}(x);
  eps-greedy over the scores; every model's network for the played arm is updated with Eq. 1; only the played
  arm's EXP3 is updated (the only one with feedback).

Choices the paper leaves open (all switchable, see README "Interpretation choices"):
* ``prop``: propensity used in Eq. 1. 0 = each model's own NB1 propensity (literal Alg. 2, each model is an
  NB1 instance), 1 = propensity of the policy that actually acted ("behaviour"; literal Alg. 3, where the
  P_t of step 5 is the committee's).
* ``ips_clip``: cap on 1/P (the paper has none; 1/P reaches K/gamma = 1400).
* ``nb3_reward``: which EXP3s are updated in NB3, and with what. 2 (default) = every arm's EXP3 gets the
  round's reward y for the model it drew (Alg. 3 step 8: "update each arm's EXP3"); the K selectors then
  share credit for the committee's decision. 0 = only the played arm's EXP3, reward y. 1 = only the played
  arm's EXP3, reward 1-(y-N)^2. Options 0 and 1 are biased: an arm's EXP3 only gets feedback when its arm is
  played, so a model that over-predicts its arm gets that arm played more often and collects more reward;
  NB3 then locks onto over-predicting models and plays one arm almost always (see tests/test_nb3.py).

Extensions
* H1, ``explore=1``: Thompson-style arm choice by Monte-Carlo dropout. Each forward pass draws a dropout
  mask on the hidden units (rate p_drop, inverted scaling, only networks with C >= min_h_drop); the arm is
  argmax of the sampled outputs, mixed with gamma-uniform exploration. The propensity needed by Eq. 1 is
  estimated from S extra dropout passes. Training uses dropout (fresh mask per update), as in MC dropout.
  Hidden activations are computed once per round; only the output layer is resampled, so S passes are cheap.
* H2, ``alpha>0``: EXP3.S (Auer et al. 2002) instead of EXP3:
  w_i <- w_i exp(gamma_model x_hat_i / M) + (e alpha / M) sum_j w_j.

Logged per block of ``block`` rounds: reward count, model-selection counts, and (simulation-only diagnostic)
how often each model's own greedy arm was correct, which gives the hindsight-best-single-model curve.
"""
import math
import numpy as np
from .jit import njit


@njit
def _sigmoid(z):
    if z > 35.0:
        z = 35.0
    elif z < -35.0:
        z = -35.0
    return 1.0 / (1.0 + math.exp(-z))


@njit
def _sample(p, n):
    u = np.random.random()
    c = 0.0
    for i in range(n):
        c += p[i]
        if u < c:
            return i
    return n - 1


@njit
def _forward(W1, W2, m, k, C, Cmax, D, xi, nnz, H):
    """Hidden activations into H[m,k,:C]; returns output pre-activation (no dropout)."""
    z = W2[m, k, Cmax]
    for c in range(C):
        a = W1[m, k, c, D]
        for j in range(nnz):
            a += W1[m, k, c, xi[j]]
        h = _sigmoid(a)
        H[m, k, c] = h
        z += W2[m, k, c] * h
    return z


@njit
def _dropout_out(W2, H, m, k, C, Cmax, p_drop):
    """Output with a fresh dropout mask on the hidden units (inverted scaling)."""
    z = W2[m, k, Cmax]
    s = 1.0 / (1.0 - p_drop)
    for c in range(C):
        if np.random.random() >= p_drop:
            z += W2[m, k, c] * H[m, k, c] * s
    return _sigmoid(z)


@njit
def _backprop(W1, W2, H, m, k, C, Cmax, D, xi, nnz, y, g, use_drop, p_drop, mbuf):
    """One importance-weighted step of Eq. 1 on network (m, k): W += g * delta * x, g = lambda * w_ips."""
    z = W2[m, k, Cmax]
    s = 1.0 / (1.0 - p_drop) if use_drop else 1.0
    for c in range(C):
        if use_drop:
            mbuf[c] = s if np.random.random() >= p_drop else 0.0
        else:
            mbuf[c] = 1.0
        z += W2[m, k, c] * H[m, k, c] * mbuf[c]
    o = _sigmoid(z)
    d_o = (y - o) * o * (1.0 - o)
    if d_o == 0.0:
        return
    for c in range(C):
        if mbuf[c] == 0.0:
            continue
        h = H[m, k, c]
        dh = d_o * W2[m, k, c] * mbuf[c] * h * (1.0 - h)
        W2[m, k, c] += g * d_o * h * mbuf[c]
        step = g * dh
        for j in range(nnz):
            W1[m, k, c, xi[j]] += step
        W1[m, k, c, D] += step
    W2[m, k, Cmax] += g * d_o


@njit
def _exp3_update(w, i, xhat, gm, M, alpha):
    """Normalised EXP3 / EXP3.S update of weight vector w (sum 1 on entry and exit)."""
    w[i] *= math.exp(gm * xhat / M)
    if alpha > 0.0:
        add = math.e * alpha / M  # times sum_j w_j(t) = 1
        for j in range(M):
            w[j] += add
    tot = 0.0
    for j in range(M):
        tot += w[j]
    for j in range(M):
        w[j] /= tot


@njit
def run_committee(idx, labels, order, start, T, period, K, D, hid, lr, mode, explore, gamma, gamma_model,
                  alpha, ips_clip, p_drop, min_h_drop, S, prop, nb3_reward, block, seed):
    """mode: 1 = NB1/NB2 (EXP3 over models; NB1 is M=1), 2 = NB3 (EXP3 per arm)."""
    np.random.seed(seed)
    M = hid.shape[0]
    Cmax = 1
    for m in range(M):
        if hid[m] > Cmax:
            Cmax = hid[m]
    nnz = idx.shape[1]
    N = order.shape[0]
    W1 = np.random.random((M, K, Cmax, D + 1)) - 0.5
    W2 = np.random.random((M, K, Cmax + 1)) - 0.5
    H = np.zeros((M, K, Cmax))
    out0 = np.zeros((M, K))
    greedy = np.zeros(M, np.int64)
    nblk = (T + block - 1) // block
    rew = np.zeros(nblk, np.int64)
    sel = np.zeros((nblk, M), np.int64)
    corr = np.zeros((nblk, M), np.int64)
    nw = K if mode == 2 else 1
    w = np.ones((nw, M)) / M
    pm = np.zeros((nw, M))
    Pown = np.zeros(M)                    # each model's own NB1 propensity of the played arm
    counts = np.zeros((M, K))
    cnt3 = np.zeros(K)
    first = np.zeros(M, np.int64)
    msel_k = np.zeros(K, np.int64)
    score = np.zeros(K)
    mbuf = np.zeros(Cmax)
    dscale = 1.0 / (1.0 - p_drop)
    for t in range(T):
        row = order[(start + t) % N]
        xi = idx[row]
        lab = labels[row]
        if period > 0:
            lab = (lab + t // period) % K
        b = t // block
        msel = 0
        a0 = 0
        # ---- forward pass of every network (deterministic) ----
        for m in range(M):
            best = -1.0
            for k in range(K):
                o = _sigmoid(_forward(W1, W2, m, k, hid[m], Cmax, D, xi, nnz, H))
                out0[m, k] = o
                if o > best:
                    best = o
                    greedy[m] = k
            if greedy[m] == lab:
                corr[b, m] += 1
        # ---- model-selection probabilities (Eq. 2) ----
        for r in range(nw):
            for m in range(M):
                pm[r, m] = (1.0 - gamma_model) * w[r, m] + gamma_model / M
        if mode == 1:
            msel = _sample(pm[0], M)
            if explore == 0:
                a0 = greedy[msel]
            else:
                for m in range(M):
                    use = hid[m] >= min_h_drop
                    for k in range(K):
                        counts[m, k] = 0.0
                    if not use:
                        counts[m, greedy[m]] = S + 1.0
                        first[m] = greedy[m]
                        continue
                    for s in range(S + 1):
                        # one dropout mask per model and pass, applied to every arm's network
                        for c in range(hid[m]):
                            mbuf[c] = dscale if np.random.random() >= p_drop else 0.0
                        best = -1.0
                        arg = 0
                        for k in range(K):
                            z = W2[m, k, Cmax]
                            for c in range(hid[m]):
                                z += W2[m, k, c] * H[m, k, c] * mbuf[c]
                            if z > best:
                                best = z
                                arg = k
                        counts[m, arg] += 1.0
                        if s == 0:
                            first[m] = arg
                a0 = first[msel]
            kp = np.random.randint(K) if np.random.random() < gamma else a0
            pb = 0.0
            for m in range(M):
                if explore == 0:
                    Pown[m] = (1.0 - gamma) * (1.0 if greedy[m] == kp else 0.0) + gamma / K
                else:
                    Pown[m] = (1.0 - gamma) * counts[m, kp] / (S + 1) + gamma / K
                pb += pm[0, m] * Pown[m]
            y = 1.0 if kp == lab else 0.0
            sel[b, msel] += 1
        else:
            for k in range(K):
                msel_k[k] = _sample(pm[k], M)
            for k in range(K):
                cnt3[k] = 0.0
            for s in range(S + 1 if explore == 1 else 1):
                best = -1.0
                arg = 0
                for k in range(K):
                    mk = msel_k[k] if s == 0 else _sample(pm[k], M)
                    if explore == 1 and hid[mk] >= min_h_drop:
                        o = _dropout_out(W2, H, mk, k, hid[mk], Cmax, p_drop)
                    else:
                        o = out0[mk, k]
                    if o > best:
                        best = o
                        arg = k
                cnt3[arg] += 1.0
                if s == 0:
                    a0 = arg
            kp = np.random.randint(K) if np.random.random() < gamma else a0
            if explore == 0:
                pb = (1.0 - gamma) * (1.0 if kp == a0 else 0.0) + gamma / K
            else:
                pb = (1.0 - gamma) * cnt3[kp] / (S + 1) + gamma / K
            for m in range(M):
                Pown[m] = (1.0 - gamma) * (1.0 if greedy[m] == kp else 0.0) + gamma / K
            y = 1.0 if kp == lab else 0.0
            sel[b, msel_k[kp]] += 1
        if y > 0.0:
            rew[b] += 1
        # ---- Eq. 1: update every model's network of the played arm ----
        for m in range(M):
            P = Pown[m] if prop == 0 else pb
            wi = 1.0 / P
            if wi > ips_clip:
                wi = ips_clip
            use = explore == 1 and hid[m] >= min_h_drop
            _backprop(W1, W2, H, m, kp, hid[m], Cmax, D, xi, nnz, y, lr[m] * wi, use, p_drop, mbuf)
        # ---- Eq. 3 (EXP3) or EXP3.S ----
        if mode == 1:
            _exp3_update(w[0], msel, y / pm[0, msel], gamma_model, M, alpha)
        elif nb3_reward == 2:
            # literal Alg. 3 step 8: every arm's EXP3 is updated with the round's reward y for the model it drew
            for k in range(K):
                ms = msel_k[k]
                _exp3_update(w[k], ms, y / pm[k, ms], gamma_model, M, alpha)
        else:
            ms = msel_k[kp]
            if nb3_reward == 0:
                r = y
            else:
                d = y - out0[ms, kp]
                r = 1.0 - d * d
            _exp3_update(w[kp], ms, r / pm[kp, ms], gamma_model, M, alpha)
    return rew, sel, corr, w


def default_grid():
    """The paper's committee: hidden sizes {1,5,25,50,100} x learning rates {0.01,0.1,1} = 15 models."""
    hid, lr = [], []
    for h in (1, 5, 25, 50, 100):
        for l in (0.01, 0.1, 1.0):
            hid.append(h)
            lr.append(l)
    return np.array(hid, np.int64), np.array(lr, np.float64)
