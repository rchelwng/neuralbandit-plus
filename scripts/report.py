"""Build every table, figure and number for the paper from results/. Never type result numbers by hand.

  python scripts/report.py                      # Covertype (reproduction, extensions, ablations) + any extra datasets
  python scripts/report.py --results results_quick --out paper_quick

Writes  <out>/tables/*.tex, <out>/figures/*.pdf, <out>/generated/macros.tex  and  results/REPORT.md
(markdown with every table and every paired comparison, for checking the paper text and for the slides).

Metrics: cumulative regret rho*t - sum r (rho from data/<data>_reference.json; oracle rho=1 if missing),
accuracy over the last 100k rounds, recovery time after each drift (restricted mean: a drift that is not
recovered before the next one counts as the full period, a lower bound; the number of such censored drifts is
shown), runtime. Paired comparisons over seeds: mean difference, 95% bootstrap CI, exact Wilcoxon p.
"""
import argparse
import json
import os
import numpy as np
import _path  # noqa: F401
from nbplus import metrics
from nbplus.runner import ROOT, load_results

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABEL = {"random": "Random", "banditron": "Banditron", "linucb": "LinUCB", "lints": "CTS (LinTS)",
         "nb2": "NB2", "nb3": "NB3",
         "nb2_ts": "NB2+MCD-TS", "nb2_exp3s": "NB2+EXP3.S", "nb2_ts_exp3s": "NB2+MCD-TS+EXP3.S",
         "nb3_ts": "NB3+MCD-TS", "nb3_exp3s": "NB3+EXP3.S", "nb3_ts_exp3s": "NB3+MCD-TS+EXP3.S",
         "nb2_exp3s_a1e-6": r"NB2+EXP3.S, $\alpha=10^{-6}$", "nb2_exp3s_a1e-5": r"NB2+EXP3.S, $\alpha=10^{-5}$",
         "nb2_exp3s_a1e-4": r"NB2+EXP3.S, $\alpha=10^{-4}$", "nb2_ts_p0.05": "NB2+MCD-TS, $p=0.05$",
         "nb2_ts_p0.2": "NB2+MCD-TS, $p=0.2$", "nb2_ts_S1": "NB2+MCD-TS, $S=1$", "nb2_ts_S8": "NB2+MCD-TS, $S=8$",
         "nb2_behavior": "NB2, behaviour propensity", "nb3_behavior": "NB3, behaviour propensity",
         "nb3_played_y": "NB3, played-arm EXP3 only", "nb2_clip10": "NB2, clip 10", "nb2_noclip": "NB2, no clip"}
PAPER_STAT = {"banditron": 57, "linucb": 72, "lints": 73, "nb3": 73, "nb2": 76}  # Fig. 1 (left), last 100k
COLORS = {"random": "#9e9e9e", "banditron": "#e08a2c", "linucb": "#3b7dd8", "lints": "#7b5cc4", "nb2": "#1a1a1a",
          "nb3": "#c0392b", "nb2_ts": "#16a085", "nb2_exp3s": "#d35400", "nb2_ts_exp3s": "#8e44ad",
          "nb3_ts": "#27ae60", "nb3_exp3s": "#e67e22", "nb3_ts_exp3s": "#9b59b6"}


def texname(n):
    """LaTeX command names may only contain letters: nb2_ts_exp3s -> NbtwoTsExpthrees."""
    words = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
             "8": "eight", "9": "nine"}
    out = "".join(words.get(c, c) for c in n)
    return "".join(w.capitalize() for w in out.replace("-", "_").replace(".", "_").split("_") if w.isalpha())


def lab(n):
    if n.endswith("_stat"):
        n = n[:-5]
    return LABEL.get(n, n)


def plain(s):
    return s.replace("$", "").replace("\\alpha", "alpha").replace("^{-", "e-").replace("}", "")


def rho_of(data):
    p = os.path.join(ROOT, "data", f"{data}_reference.json")
    if os.path.exists(p):
        return float(json.load(open(p))["rho"]), "reference"
    return 1.0, "oracle"


class Run:
    def __init__(self, results, data, name, rho):
        self.name, self.data = name, data
        self.r = load_results(results, data, name)
        self.ok = self.r is not None
        if not self.ok:
            return
        self.seeds = sorted(self.r)
        v0 = self.r[self.seeds[0]]
        self.block, self.T, self.period = int(v0["block"]), int(v0["T"]), int(v0["period"])
        self.rew = {s: self.r[s]["rew"].astype(float) for s in self.seeds}
        self.final = {s: metrics.final_regret(self.rew[s], self.block, rho, self.T) for s in self.seeds}
        self.acc = {s: metrics.last_acc(self.rew[s], self.block) for s in self.seeds}
        self.runtime = {s: float(self.r[s]["runtime"]) for s in self.seeds}
        self.rec = {s: metrics.recovery_times(self.rew[s], self.block, self.period, self.T) for s in self.seeds}
        self.rec_mean, self.censored, self.ndrift = metrics.recovery_summary(list(self.rec.values()))
        self.rec_rmst = {s: (np.mean([x if x is not None else self.period for x in self.rec[s]]) if self.rec[s] else np.nan)
                         for s in self.seeds}
        self.curve = np.array([metrics.cum_regret(self.rew[s], self.block, rho) for s in self.seeds])


def fmt_k(x):
    return f"{x / 1000:.1f}"


def ci_str(vals, f=fmt_k):
    m, lo, hi = metrics.bootstrap_ci(list(vals))
    return f"{f(m)}\\,[{f(lo)}, {f(hi)}]", (m, lo, hi)


def paired(a, b, attr):
    s = sorted(set(a.seeds) & set(b.seeds))
    if len(s) < 2:
        return None
    x = [getattr(a, attr)[i] for i in s]
    y = [getattr(b, attr)[i] for i in s]
    return metrics.paired_stats(x, y) + (len(s),)


def sentence(a, b, attr, what, unit=""):
    """Rule-based sentence: 'lower/higher' only if the CI excludes 0 and n >= 6, else descriptive."""
    st = paired(a, b, attr)
    if st is None:
        return None
    d, lo, hi, p, n = st
    if not np.isfinite(d):
        return None
    scale = 1000.0
    rel = 100.0 * d / np.mean([getattr(a, attr)[i] for i in sorted(set(a.seeds) & set(b.seeds))])
    if n < 6:
        s = f"{lab(b.name)} vs {lab(a.name)}, {what} (descriptive only, fewer than 6 seeds)"
    elif lo > 0 or hi < 0:
        verdict = "higher" if d > 0 else "lower"
        s = f"{lab(b.name)} had {verdict} {what} than {lab(a.name)}"
    else:
        s = f"{lab(b.name)} showed no clear difference in {what} from {lab(a.name)}"
    return (f"{plain(s)} (mean difference {d / scale:+.1f}k{unit} ({rel:+.1f}%), 95% bootstrap CI [{lo / scale:.1f}k, "
            f"{hi / scale:.1f}k], Wilcoxon p = {p:.3f}, n = {n} seeds).")


def table(runs, ref, path, stationary=False, runtime=True):
    """LaTeX + markdown table. Columns: final regret (x10^3), last-100k accuracy, recovery, runtime, p vs ref."""
    head = ["Method", r"Final regret ($\times10^3$)", r"Acc.\ last 100k (\%)"]
    if not stationary:
        head.append("Recovery (k rounds)")
    head += (["Runtime (min)"] if runtime else []) + [f"$p$ vs.\\ {lab(ref.name)}" if ref is not None and ref.ok else "$p$"]
    tex = ["% generated by scripts/report.py: do not edit by hand", r"\begin{tabular}{l" + "c" * (len(head) - 1) + "}",
           r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    md = ["| " + " | ".join(plain(h).replace("\\", "") for h in head) + " |", "|" + "---|" * len(head)]
    for r in runs:
        if not r.ok:
            continue
        reg, _ = ci_str(r.final.values())
        acc, _ = ci_str([100 * v for v in r.acc.values()], f=lambda x: f"{x:.1f}")
        row = [lab(r.name) + (f" ($n={len(r.seeds)}$)" if len(r.seeds) < 10 else ""), reg, acc]
        if not stationary:
            row.append(f"{np.nanmean(list(r.rec_rmst.values())) / 1000:.0f} ({r.censored}/{r.ndrift} cens.)")
        if runtime:
            rt = np.mean(list(r.runtime.values())) / 60
            row.append(f"{rt:.1f}" if rt >= 0.1 else "$<$0.1")
        if ref is not None and ref.ok and r is not ref:
            st = paired(ref, r, "final")
            row.append(f"{st[3]:.3f}" if st else "--")
        else:
            row.append("--")
        tex.append(" & ".join(row) + r" \\")
        md.append("| " + " | ".join(plain(c).replace("\\,", " ") for c in row) + " |")
    tex += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(tex) + "\n")
    return md


def regret_plot(ax, runs, title, oracle_note=""):
    for r in runs:
        if not r.ok:
            continue
        t = np.arange(1, r.curve.shape[1] + 1) * r.block / 1e6
        m = r.curve.mean(0) / 1e3
        se = r.curve.std(0, ddof=1) / np.sqrt(len(r.seeds)) / 1e3 if len(r.seeds) > 1 else 0 * m
        base = r.name.replace("_stat", "")
        ax.plot(t, m, label=plain(lab(base)), color=COLORS.get(base), lw=1.6)
        ax.fill_between(t, m - 1.96 * se, m + 1.96 * se, color=COLORS.get(base), alpha=0.18, lw=0)
        if r.period > 0:
            for td in range(r.period, r.T, r.period):
                ax.axvline(td / 1e6, color="#bbbbbb", ls=":", lw=0.8)
    ax.set_xlabel("round (millions)")
    ax.set_ylabel("cumulative regret (thousands)" + oracle_note)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def save_fig(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", name + ".pdf"))
    fig.savefig(os.path.join(out, "figures", name + ".png"), dpi=130)
    plt.close(fig)


def diag_plot(r, out, name):
    """Realised policy vs hindsight-best single model, per drift segment (simulation-only diagnostic)."""
    if not r.ok or "corr" not in r.r[r.seeds[0]]:
        return None
    acc_pol, acc_best = [], []
    for s in r.seeds:
        rew, corr = r.rew[s], r.r[s]["corr"].astype(float)
        seg = (r.period // r.block) if r.period > 0 else len(rew)
        best = np.zeros_like(rew)
        for a in range(0, len(rew), seg):
            j = corr[a:a + seg].sum(0).argmax()
            best[a:a + seg] = corr[a:a + seg, j]
        acc_pol.append(metrics.moving_acc(rew, r.block))
        acc_best.append(metrics.moving_acc(best, r.block))
    t = np.arange(1, len(acc_pol[0]) + 1) * r.block / 1e6
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(t, np.mean(acc_best, 0), color="#888888", lw=1.4, label="best single model in hindsight (per segment)")
    ax.plot(t, np.mean(acc_pol, 0), color=COLORS.get(r.name, "k"), lw=1.4, label=f"realised {plain(lab(r.name))} policy")
    if r.period > 0:
        for td in range(r.period, r.T, r.period):
            ax.axvline(td / 1e6, color="#bbbbbb", ls=":", lw=0.8)
    ax.set_xlabel("round (millions)")
    ax.set_ylabel("accuracy (50k moving avg.)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)
    save_fig(fig, out, name)
    return float(np.mean([a.mean() for a in acc_best])), float(np.mean([a.mean() for a in acc_pol]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper"))
    ap.add_argument("--data", default="covtype")
    a = ap.parse_args()
    md = ["# Results report (generated by scripts/report.py)", ""]
    macros = ["% generated by scripts/report.py"]
    data = a.data
    rho, kind = rho_of(data)
    md.append(f"Dataset `{data}`; regret baseline rho = {rho:.4f} ({kind}).")
    macros.append(f"\\newcommand{{\\RefAcc}}{{{rho:.3f}}}")
    R = lambda n, d=data, rr=rho: Run(a.results, d, n, rr)  # noqa: E731

    # ---------------- Stage 1: reproduction (Fig. 1) ----------------
    base = ["banditron", "linucb", "lints", "nb3", "nb2"]
    stat = [R(n + "_stat") for n in base]
    drift = [R(n) for n in base]
    if any(r.ok for r in stat + drift):
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
        regret_plot(axs[0], stat, "stationary")
        per = next((r.period for r in drift if r.ok), 0)
        regret_plot(axs[1], drift, f"abrupt drift every {per // 1000}k rounds")
        axs[1].set_ylabel("")
        axs[0].legend(fontsize=7, frameon=False)
        save_fig(fig, a.out, "fig1_reproduction")
        md += ["", "## Reproduction (paper Fig. 1)", "", "Stationary, accuracy over the last 100k rounds:", "",
               "| Method | Paper (%) | Ours (%) [95% CI] | Delta (pp) | seeds |", "|---|---|---|---|---|"]
        tex = ["% generated by scripts/report.py", r"\begin{tabular}{lccc}", r"\toprule",
               r"Method & Paper (\%) & Ours (\%) & $\Delta$ (pp) \\", r"\midrule"]
        for r in stat:
            if not r.ok:
                continue
            b = r.name.replace("_stat", "")
            m, lo, hi = metrics.bootstrap_ci([100 * v for v in r.acc.values()])
            pv = PAPER_STAT[b]
            tex.append(f"{lab(b)} & {pv} & {m:.1f}\\,[{lo:.1f}, {hi:.1f}] & {m - pv:+.1f} \\\\")
            md.append(f"| {plain(lab(b))} | {pv} | {m:.1f} [{lo:.1f}, {hi:.1f}] | {m - pv:+.1f} | {len(r.seeds)} |")
            macros.append(f"\\newcommand{{\\Stat{texname(b)}}}{{{m:.1f}}}")
        tex += [r"\bottomrule", r"\end{tabular}"]
        os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
        open(os.path.join(a.out, "tables", "tab_repro.tex"), "w").write("\n".join(tex) + "\n")
        md += ["", "Stationary (all metrics):", ""] + table(stat, stat[-1], os.path.join(a.out, "tables", "tab_stationary.tex"), True)
        md += ["", "Abrupt drift (cumulative regret, recovery, runtime; p = paired Wilcoxon vs NB2):", ""]
        md += table(drift, drift[-1], os.path.join(a.out, "tables", "tab_drift.tex"))
        nb2 = drift[-1]
        md += ["", "Paired comparisons (drift, final regret):", ""]
        for r in drift[:-1]:
            s = sentence(nb2, r, "final", "final cumulative regret") if r.ok and nb2.ok else None
            if s:
                md.append("* " + s)

    # ---------------- Stage 2: extensions (2x2 on NB2 and NB3) ----------------
    lin = R("linucb")
    for b in ("nb2", "nb3"):
        runs = [R(b), R(f"{b}_ts"), R(f"{b}_exp3s"), R(f"{b}_ts_exp3s")]
        if sum(r.ok for r in runs) < 2:
            continue
        ref = runs[0]
        md += ["", f"## Extensions on {b.upper()} (2x2: exploration x selector), abrupt drift", ""]
        md += table(runs + [lin], ref, os.path.join(a.out, "tables", f"tab_ext_{b}.tex"))
        md += ["", "Paired comparisons:", ""]
        for r in runs[1:]:
            if r.ok and ref.ok:
                for attr, what, unit in (("final", "final cumulative regret", ""), ("rec_rmst", "recovery time", " rounds")):
                    s = sentence(ref, r, attr, what, unit)
                    if s:
                        md.append("* " + s)
        for r in runs:
            if r.ok and lin.ok:
                s = sentence(lin, r, "final", "final cumulative regret")
                if s:
                    md.append("* " + s)
        # interaction contrast (TS+S - TS) - (S - base) on final regret
        if all(r.ok for r in runs):
            s = sorted(set.intersection(*[set(r.seeds) for r in runs]))
            ic = [(runs[3].final[i] - runs[1].final[i]) - (runs[2].final[i] - runs[0].final[i]) for i in s]
            m, lo, hi = metrics.bootstrap_ci(ic)
            md.append(f"* Interaction contrast (TS+EXP3.S - TS) - (EXP3.S - {b.upper()}) on final regret: "
                      f"{m / 1000:+.1f}k, 95% CI [{lo / 1000:.1f}k, {hi / 1000:.1f}k], n = {len(s)}.")
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        regret_plot(ax, runs + [lin], f"{b.upper()} extensions, abrupt drift")
        ax.legend(fontsize=7, frameon=False)
        save_fig(fig, a.out, f"fig_ext_{b}")
        for r in runs:
            if r.ok:
                macros.append(f"\\newcommand{{\\Reg{texname(r.name)}}}{{{np.mean(list(r.final.values())) / 1000:.1f}}}")

    # ---------------- diagnostic ----------------
    for b in ("nb2", "nb2_exp3s"):
        res = diag_plot(R(b), a.out, f"fig_diag_{b}")
        if res:
            md.append(f"\nDiagnostic ({lab(b)}): mean accuracy of the hindsight-best single model per segment {res[0]:.3f} "
                      f"vs realised policy {res[1]:.3f}.")

    # ---------------- Stage 3: ablations ----------------
    abl_names = ["nb2", "nb2_exp3s", "nb2_exp3s_a1e-6", "nb2_exp3s_a1e-5", "nb2_exp3s_a1e-4", "nb2_ts", "nb2_ts_p0.05",
                 "nb2_ts_p0.2", "nb2_ts_S1", "nb2_ts_S8", "nb2_behavior", "nb2_clip10", "nb2_noclip", "nb3", "nb3_behavior",
                 "nb3_played_y"]
    abl = [R(n) for n in abl_names]
    real_abl = ("nb2_exp3s_a1e-6", "nb2_exp3s_a1e-5", "nb2_exp3s_a1e-4", "nb2_ts_p0.05", "nb2_ts_p0.2", "nb2_ts_S1",
                "nb2_ts_S8", "nb2_behavior", "nb2_clip10", "nb2_noclip", "nb3_behavior", "nb3_played_y")
    if any(r.ok for r in abl if r.name in real_abl):
        seeds = None
        for r in abl:
            if r.ok and len(r.seeds) >= 2 and r.name in ("nb2_exp3s_a1e-5", "nb2_ts_p0.05", "nb2_behavior", "nb3_behavior"):
                seeds = set(r.seeds) if seeds is None else seeds & set(r.seeds)
        for r in abl:  # restrict to shared seeds so rows are comparable
            if r.ok and seeds:
                keep = [s for s in r.seeds if s in seeds] or r.seeds
                for attr in ("final", "acc", "runtime", "rec", "rec_rmst", "rew"):
                    setattr(r, attr, {s: getattr(r, attr)[s] for s in keep})
                r.curve = r.curve[[r.seeds.index(s) for s in keep]]
                r.seeds = keep
                r.rec_mean, r.censored, r.ndrift = metrics.recovery_summary(list(r.rec.values()))
        md += ["", f"## Ablations (shared seeds {sorted(seeds) if seeds else '?'}; descriptive)", ""]
        md += table(abl, abl[0], os.path.join(a.out, "tables", "tab_ablations.tex"), runtime=False)

    # ---------------- Stage 4: extra datasets ----------------
    extra = {}
    for d in ("pendigits", "letter", "shuttle"):
        rr, kd = rho_of(d)
        runs = [Run(a.results, d, n, rr) for n in ["random", "banditron", "linucb", "lints", "nb3", "nb2", "nb2_ts",
                                                   "nb2_exp3s", "nb2_ts_exp3s", "nb3_ts", "nb3_exp3s", "nb3_ts_exp3s"]]
        if not any(r.ok for r in runs):
            continue
        ref = runs[5]
        md += ["", f"## {d} (rho = {rr:.4f}, {kd})", ""]
        md += table(runs, ref, os.path.join(a.out, "tables", f"tab_{d}.tex"))
        md += [""]
        for r in runs:
            if r.ok and ref.ok and r is not ref and r.name != "random":
                s = sentence(ref, r, "final", "final cumulative regret")
                if s:
                    md.append("* " + s)
        byname = {r.name: r for r in runs}
        md += ["", "Each variant against its own committee:", ""]
        for b in ("nb2", "nb3"):
            for v in ("_ts", "_exp3s", "_ts_exp3s"):
                x, y = byname.get(b), byname.get(b + v)
                if x is not None and y is not None and x.ok and y.ok:
                    for attr, what, unit in (("final", "final cumulative regret", ""), ("rec_rmst", "recovery time", " rounds")):
                        s = sentence(x, y, attr, what, unit)
                        if s:
                            md.append("* " + s)
        extra[d] = byname
        fig, ax = plt.subplots(figsize=(6.5, 3.0))
        regret_plot(ax, [r for r in runs if r.name in ("banditron", "linucb", "lints", "nb2", "nb3", "nb2_exp3s",
                                                        "nb2_ts")], f"{d}, abrupt drift")
        ax.legend(fontsize=7, frameon=False)
        save_fig(fig, a.out, f"fig_{d}")

    if extra:  # one compact table for the paper: final regret and last-100k accuracy per dataset
        order = ["banditron", "linucb", "lints", "nb2", "nb2_ts", "nb2_exp3s", "nb2_ts_exp3s", "nb3", "nb3_ts", "nb3_exp3s",
                 "nb3_ts_exp3s"]
        ds = list(extra)
        tex = ["% generated by scripts/report.py: do not edit by hand", r"\begin{tabular}{l" + "cc" * len(ds) + "}",
               r"\toprule", " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{d.capitalize()}}}" for d in ds) + r" \\",
               " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ds))),
               "Method" + " & Regret ($\\times10^3$) & Acc.\\ (\\%)" * len(ds) + r" \\", r"\midrule"]
        for n in order:
            cells = []
            for d in ds:
                r = extra[d].get(n)
                if r is None or not r.ok:
                    cells += ["--", "--"]
                    continue
                reg, _ = ci_str(r.final.values())
                cells += [reg, f"{100 * np.mean(list(r.acc.values())):.1f}"]
            tex.append(lab(n) + " & " + " & ".join(cells) + r" \\")
            if n in ("lints", "nb2_ts_exp3s"):
                tex.append(r"\midrule")
        tex += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(a.out, "tables", "tab_extra.tex"), "w").write("\n".join(tex) + "\n")

    os.makedirs(os.path.join(a.out, "generated"), exist_ok=True)
    open(os.path.join(a.out, "generated", "macros.tex"), "w").write("\n".join(macros) + "\n")
    os.makedirs(a.results, exist_ok=True)
    rep = os.path.join(a.results, "REPORT.md")
    open(rep, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {rep}, {a.out}/tables, {a.out}/figures")


if __name__ == "__main__":
    main()
