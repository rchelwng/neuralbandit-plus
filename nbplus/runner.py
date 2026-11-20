"""Config-driven experiment runner: resumable, parallel, refuses to mix results from different settings.

results/<data>/<run name>/settings.json   resolved settings (everything that changes the result)
results/<data>/<run name>/seed_<s>.npz     per-block rewards (+ model-selection logs for the committees)
"""
import json
import os
import time
import multiprocessing as mp
import numpy as np

from . import algos

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNED = os.path.join(ROOT, "configs", "tuned.json")


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def _coerce(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def load_config(path, overrides=None, data=None):
    with open(path) as f:
        cfg = json.load(f)
    tuned = None
    runs = []
    for r in cfg["runs"]:
        s = dict(cfg.get("defaults", {}))
        s.update(r)
        if data:
            s["data"] = data
        for kv in overrides or []:
            k, v = kv.split("=", 1)
            s[k] = _coerce(v)
        if s.get("param") == "tuned":
            if tuned is None:
                if not os.path.exists(TUNED):
                    raise SystemExit("configs/tuned.json missing: run the tuning stage first "
                                     "(python scripts/run.py configs/tuning.json, then python scripts/select_tuned.py)")
                with open(TUNED) as f:
                    tuned = json.load(f)
            try:
                s["param"] = tuned[s["data"]][s["algo"]]
            except KeyError:
                raise SystemExit(f"no tuned value for {s['algo']} on {s['data']} in configs/tuned.json")
        runs.append(s)
    return cfg, runs


def run_dir(results, s):
    return os.path.join(results, s["data"], s["name"])


def _check_settings(d, fp):
    path = os.path.join(d, "settings.json")
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
        if old != fp:
            diff = {k: (old.get(k), fp.get(k)) for k in set(old) | set(fp) if old.get(k) != fp.get(k)}
            raise SystemExit(f"{d} was produced with different settings {diff}.\n"
                             f"Delete that folder to rerun, or use --results <other folder>.")
    else:
        os.makedirs(d, exist_ok=True)
        with open(path, "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)


def _job(args):
    s, seed, d = args
    out = os.path.join(d, f"seed_{seed}.npz")
    if os.path.exists(out):
        return s["name"], seed, None
    res = algos.run(s, seed)
    tmp = out + ".tmp.npz"
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in res.items()})
    os.replace(tmp, out)
    return s["name"], seed, res["runtime"]


def _cost(s):
    c = {"random": 0.01, "banditron": 0.02, "linucb": 0.3, "lints": 0.3, "nb1": 0.2}.get(s["algo"], 1.0)
    if s.get("explore") == "ts":
        c *= 1.4
    return c * float(s.get("T", algos.DEFAULTS["T"]))


def run_config(path, seeds=None, only=None, workers=1, overrides=None, results=None, dry_run=False, data=None):
    results = results or os.path.join(ROOT, "results")
    cfg, runs = load_config(path, overrides, data)
    jobs = []
    for s in runs:
        if only and s["name"] not in only:
            continue
        d = run_dir(results, s)
        fp = algos.fingerprint(s)
        if not dry_run:
            _check_settings(d, fp)
        for seed in parse_seeds(seeds if seeds is not None else s.get("seeds", "0-9")):
            done = os.path.exists(os.path.join(d, f"seed_{seed}.npz"))
            if not done:
                jobs.append((s, seed, d))
    jobs.sort(key=lambda j: -_cost(j[0]))
    print(f"{path}: {len(jobs)} jobs to run ({workers} workers)")
    if dry_run:
        for s, seed, d in jobs:
            print(f"  {s['data']}/{s['name']} seed {seed}")
        return
    t0 = time.time()
    if workers <= 1:
        it = map(_job, jobs)
    else:
        pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=50)
        it = pool.imap_unordered(_job, jobs)
    for i, (name, seed, rt) in enumerate(it, 1):
        if rt is not None:
            print(f"[{i}/{len(jobs)}] {name} seed {seed}: {rt / 60:.1f} min  (elapsed {(time.time() - t0) / 60:.1f} min)",
                  flush=True)
    if workers > 1:
        pool.close()
        pool.join()


def load_results(results, data, name):
    d = os.path.join(results, data, name)
    if not os.path.isdir(d):
        return None
    out = {}
    for f in sorted(os.listdir(d)):
        if f.startswith("seed_") and f.endswith(".npz") and ".tmp" not in f:
            z = np.load(os.path.join(d, f))
            out[int(f[5:-4])] = {k: z[k] for k in z.files}
    return out or None
