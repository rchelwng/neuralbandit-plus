"""Run every (run, seed) of a config.  Re-running the same command resumes (finished seeds are skipped).

  python scripts/run.py configs/reproduction.json --workers 4
  python scripts/run.py configs/extensions.json --workers 4 --seeds 0-4      # faster first pass
  python scripts/run.py configs/reproduction.json --dry-run                   # list what would run
  python scripts/run.py configs/reproduction.json --set T=200000 period=50000 --seeds 0-1 --results results_quick
"""
import argparse
import _path  # noqa: F401
from nbplus.runner import run_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="+")
    ap.add_argument("--seeds", default=None, help="e.g. 0-9 (default: the config's seeds)")
    ap.add_argument("--only", default=None, help="comma-separated run names")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (use your physical core count)")
    ap.add_argument("--set", nargs="*", default=[], help="override settings, e.g. T=200000 period=50000")
    ap.add_argument("--data", default=None, help="override the dataset of every run")
    ap.add_argument("--results", default=None, help="results folder (default: results/)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for c in a.config:
        run_config(c, a.seeds, a.only.split(",") if a.only else None, a.workers, a.set, a.results, a.dry_run, a.data)
