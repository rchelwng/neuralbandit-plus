"""Run the whole study in order. Safe to stop and restart at any time (finished seeds are skipped).

  python scripts/run_all.py --workers 4                 # Covertype: tuning -> reproduction -> extensions
  python scripts/run_all.py --workers 4 --ablations     # + ablations (optional, ~5 seeds)
  python scripts/run_all.py --workers 4 --extra pendigits letter shuttle
  python scripts/run_all.py --quick                     # 5-minute smoke test on synthetic data (results_quick/)
"""
import argparse
import os
import subprocess
import sys
import _path  # noqa: F401
from nbplus.runner import ROOT

PY = sys.executable


def sh(*args):
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([PY, *args], check=True, cwd=ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    w = ["--workers", str(a.workers)]
    if a.quick:
        q = ["--data", "synthetic", "--results", "results_quick", "--set", "T=200000", "period=50000"]
        sh("scripts/run.py", "configs/tuning.json", *q, "--seeds", "100", *w)
        sh("scripts/select_tuned.py", "--results", "results_quick")
        sh("scripts/run.py", "configs/reproduction.json", "configs/extensions.json", *q, "--seeds", "0-1", *w)
        sh("scripts/report.py", "--results", "results_quick", "--data", "synthetic", "--out", "paper_quick")
        sys.exit(0)
    if not os.path.exists(os.path.join(ROOT, "data", "covtype.npz")):
        sh("scripts/prepare_data.py", "covtype")
    if not os.path.exists(os.path.join(ROOT, "data", "covtype_reference.json")):
        sh("scripts/make_reference.py", "covtype")
    sh("scripts/run.py", "configs/tuning.json", *w)
    sh("scripts/select_tuned.py")
    sh("scripts/run.py", "configs/reproduction.json", "configs/extensions.json", *w)
    if a.ablations:
        sh("scripts/run.py", "configs/ablations.json", *w)
    for d in a.extra:
        if not os.path.exists(os.path.join(ROOT, "data", f"{d}.npz")):
            sh("scripts/prepare_data.py", d)
        if not os.path.exists(os.path.join(ROOT, "data", f"{d}_reference.json")):
            sh("scripts/make_reference.py", d)
        sh("scripts/run.py", "configs/tuning.json", "--data", d, "--set", "T=400000", "period=200000", *w)
        sh("scripts/select_tuned.py")
        sh("scripts/run.py", "configs/extra_datasets.json", "--data", d, *w)
    sh("scripts/report.py")
