# NeuralBandit++: reproducing *A Neural Networks Committee for the Contextual Bandit Problem* and extending it for concept drift

Allesiardo, Féraud, Bouneffouf, ICONIP 2014 ([arXiv 1409.8191](https://arxiv.org/abs/1409.8191)).
We reproduce NeuralBandit2/3 (NB2, NB3) against Banditron, LinUCB and contextual Thompson sampling (CTS) on Forest
Cover Type, with and without the authors' cyclic drift, over 10 seeds, then test two extensions:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rchelwng/neuralbandit-plus/blob/main/neuralbandit_drift_demo.ipynb)
**Live demo:** the notebook above runs LinUCB, NB2, NB2+EXP3.S and NB2+MC dropout on Covertype with one drift (≈ 5 min, nothing to install).

* **H1** Thompson-style arm selection by **Monte-Carlo dropout** (uncertainty-aware exploration) → lower cumulative regret?
* **H2** **EXP3.S** instead of EXP3 for model selection (shifting best model) → faster post-drift recovery?

Both are evaluated in a 2×2 factorial design on NB2 and NB3, plus two extra UCI datasets (Pendigits, Shuttle). Metrics: cumulative regret,
recovery time after each drift, runtime. Everything runs on a laptop CPU (NumPy + Numba).

## Results in brief
Full numbers, CIs and paired tests: `results/REPORT.md`; write-up: the paper in `paper/`.
* **Reproduction:** stationary accuracies within 1.1 points of the paper for NB2, NB3, LinUCB and CTS (Banditron +2.6),
  same ranking. Under drift every committee variant has 36–45% lower cumulative regret than LinUCB on Covertype.
* **NB3:** the paper does not say which selectors learn each round. Updating only the played arm's EXP3 collapses to
  3.7% accuracy under drift; updating every arm's EXP3 gives 61.1% (73.0% without drift, paper 73%).
* **H1 (MC dropout):** not supported on Covertype (NB2 regret +6%, recovery +51%).
* **H2 (EXP3.S):** partly supported: NB2 regret −4% on Covertype (lowest of any method), NB3 recovery −29%; lower average
  regret for both committees on Pendigits and Shuttle (up to −39%).
* **Diagnostic:** the best single model also drops at every drift, so recovery is limited mainly by the base networks,
  not the selector.
* **Extra datasets:** Banditron is best on both; on Pendigits LinUCB also beats the committees, which mostly fail to
  recover after drifts (likely saturating sigmoid outputs under the quadratic loss).

## Quick start
```
pip install -r requirements.txt
python -m pytest tests -q                 # 9 tests, ~1.5 min
python scripts/run_all.py --quick         # ~5 min end-to-end smoke test on synthetic data -> results_quick/, paper_quick/
python scripts/prepare_data.py covtype    # downloads Covertype once -> data/covtype.npz (94 binary features)
python scripts/benchmark.py               # times your laptop, prints core-hours per stage
```

## The real runs (Covertype)
One command runs everything in order and can be stopped and restarted at any time (finished seeds are skipped):
```
python scripts/run_all.py --workers 4
```
`--workers` = number of physical cores. Or step by step:

| Step | Command | What it does |
|---|---|---|
| 0 | `python scripts/prepare_data.py covtype` and `python scripts/make_reference.py covtype` | data + offline reference accuracy ρ (paper's "93% classifier") |
| 1 | `python scripts/run.py configs/tuning.json --workers 4` then `python scripts/select_tuned.py` | tunes Banditron γ, LinUCB α, CTS v on seeds 100–101 → `configs/tuned.json`; warns if a winner is at a grid edge |
| 2 | `python scripts/run.py configs/reproduction.json --workers 4` | paper Fig. 1: stationary + drift, 6 methods, **10 seeds** |
| 3 | `python scripts/run.py configs/extensions.json --workers 4` | H1/H2 2×2 on NB2 and NB3, **10 seeds** |
| 4 (optional) | `python scripts/run.py configs/ablations.json --workers 4` | α, dropout rate, S, propensity, clip, NB3 reading; 5 seeds |
| 5 | `python scripts/run_all.py --workers 4 --extra pendigits shuttle` | extra datasets (tunes baselines per dataset, 10 seeds) |
| 6 | `python scripts/report.py` | all tables/figures → `paper/tables`, `paper/figures`; every number + paired test → `results/REPORT.md` |

Useful flags: `--dry-run` (list jobs), `--seeds 0-4` (faster first pass; rerun later with the default to fill in
seeds 5–9), `--only nb2,nb3`, `--set T=200000 period=50000 --results results_test` (quick experiments in a separate
folder). If a results folder was produced with different settings the runner refuses to mix them: delete that folder.

**Time** (from `benchmark.py` on the build machine; yours will differ): one NB run of 2.5M rounds ≈ 6 min on one core
(≈ 8 min with MC dropout); linear baselines ≈ 1 min; Banditron seconds. Reproduction + extensions ≈ 12 core-hours ≈
3 h with 4 workers. Ablations ≈ 7 core-hours. Each extra dataset ≈ 4 core-hours.

## Paper
`paper/main.tex` (ACM `sigconf`, 7 pages) inputs the generated tables and figures from `report.py`; compile the whole
`paper/` folder on Overleaf. Every number in the text comes from `results/REPORT.md` (95% CI and Wilcoxon p).

## What the code implements (faithful to the paper)
* **NB1** (Alg. 1): one 1-hidden-layer sigmoid MLP per arm, quadratic loss, ε-greedy with γ = 0.005, importance-weighted
  backprop (Eq. 1). **NB2** (Alg. 2): 15 models (hidden {1,5,25,50,100} × λ {0.01,0.1,1}), one EXP3 (γ_model = 0.1)
  over models; every model's network of the played arm is updated. **NB3** (Alg. 3): one EXP3 per arm.
* **Baselines:** Banditron, LinUCB (disjoint), CTS = linear Thompson sampling, uniform random.
* **Data:** Covertype → 10 continuous columns × 5 equal-frequency bins + 4 wilderness + 40 soil = 94 binary features;
  shuffled per seed, random start, replayed in a loop; drift = labels shifted by one class every 500k rounds (cumulative).

## Interpretation choices (the paper is silent; all switchable, all in the paper's appendix)
1. **NB3 selector feedback** (`nb3_reward`, default `all`): every arm's EXP3 is updated with the round's reward for the
   model it drew (our reading of Alg. 3's last step, "update of each EXP3 weight vectors"). Updating only the played arm's EXP3 (`y`, our v1 reading)
   is self-reinforcing: a model that over-predicts its arm gets that arm played more and collects more reward, NB3 locks
   onto over-predicting models and ends below random play. This is why v1's NB3 got 48% instead of 73%. The ablation
   `nb3_played_y` and `tests/test_core.py::test_nb3_played_arm_reading_collapses` document it.
2. **Propensity in Eq. 1** (`prop`, default `own`): each model is an NB1 instance and divides by its own P(k);
   `behavior` (the committee's P) is an ablation.
3. **Importance-weight clip** (`ips_clip`, default 100): 1/P reaches K/γ = 1400; unclipped and clip 10 are ablations.
4. Biases included, weights U(−0.5, 0.5), sigmoid output, δ = (y − N)N(1 − N).
5. Baseline parameters are not given in the paper: tuned on disjoint seeds with equal 8-value grids.
6. Regret = ρt − cumulative reward with ρ from `make_reference.py` (MLP on the binary features, 80/20 split).
7. Recovery time: first round after a drift at which the 20k-round moving accuracy reaches 95% of the mean accuracy over
   the 50k rounds before the drift; not recovered before the next drift = censored (counted as the full period in the
   mean, and the count of censored drifts is reported).
8. MC dropout: p = 0.1 on networks with ≥ 25 hidden units, one mask per model and pass shared by its K arm networks,
   play probability estimated from S = 4 extra passes, training with dropout.
9. EXP3.S α = 1/T (pre-declared); 1e-6, 1e-5, 1e-4 are ablations.

## Status
| Course requirement | Where | Status |
|---|---|---|
| Reproduce the paper | `nbplus/`, `configs/reproduction.json`, Fig. 1 + Table 2 of the paper | Done on Covertype, 10 seeds |
| Find a gap | paper Sec. 4 (G1–G4) + the NB3 under-specification | Done |
| DL extension | H1 MC-dropout TS, H2 EXP3.S | Done, 2×2 on NB2 and NB3 |
| Strong baselines | Banditron, LinUCB, CTS (the paper's three) | Done. NeuralUCB/NeuralTS not run: limitation in the paper |
| Ablations / analysis | `configs/ablations.json` (5 seeds), oracle diagnostic | Done |
| Rigorous evaluation | 10 seeds, paired bootstrap CIs, Wilcoxon, recovery time, runtime | Done (`results/REPORT.md`) |
| Extra datasets | Pendigits, Shuttle | Done |
| Reproducible code | seeds, tests, configs, one-command pipeline, Colab demo | Done |
| Paper (ACM WWW, 6–8 p.) | `paper/main.tex` | Done (7 pages) |
| Presentation | slides + `neuralbandit_drift_demo.ipynb` / `scripts/demo.py` | Done |

Also checked on synthetic data: backprop = finite-difference gradient, EXP3/EXP3.S, determinism per seed, paired
streams, LinUCB vs an explicit-inverse implementation, recovery metric, runner resume/refusal, the NB3 collapse and its fix.

## Layout
```
nbplus/   committee.py (NB1/2/3, EXP3/EXP3.S, MC dropout)  baselines.py  data.py  metrics.py  algos.py  runner.py
scripts/  run_all  run  prepare_data  make_reference  select_tuned  benchmark  report  demo
neuralbandit_drift_demo.ipynb   Colab demo (clones this repo, prepares Covertype, runs scripts/demo.py)
configs/  tuning  reproduction  extensions  ablations  extra_datasets   (+ tuned.json, written by select_tuned)
paper/    main.tex refs.bib (+ generated tables/ figures/ generated/)
tests/    test_core.py
```
