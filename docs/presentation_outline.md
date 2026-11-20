# Final presentation (Dec 14) — outline, ~12 slides

1. **Problem & motivation** — contextual bandits in recommendation/ads; behaviour drifts.
2. **Original paper** — NB1 (one MLP per arm, ε-greedy, importance-weighted backprop); NB2 (EXP3 over 15 models);
   NB3 (one EXP3 per arm). Their Fig. 1.
3. **Reproduction protocol** — Covertype → 94 binary features, cyclic class shift every 500k, 10 seeds, tuned baselines.
4. **Reproduction results** — `fig1_reproduction.png`, `tab_repro` (ours vs paper).
5. **Reproducibility finding** — NB3's per-arm selector feedback is unspecified; played-arm-only reading collapses
   (over-predicting models win); "update every arm's EXP3" works. One chart: `nb3_played_y` vs `nb3`.
6. **Gaps** — G1 fixed ε, G2 EXP3 tracks a fixed best model, G3 one dataset, G4 is it the networks or the selector?
7. **Method H1** — MC-dropout Thompson sampling; propensity from S extra passes; cheap because only the output layer is resampled.
8. **Method H2** — EXP3.S equation; α = 1/T pre-declared.
9. **Results** — 2×2 table (`tab_ext_nb2`), regret curves vs LinUCB (`fig_ext_nb2.png`).
10. **Analysis** — diagnostic (`fig_diag_nb2.png`), ablations, extra datasets.
11. **Live demo** — `python scripts/demo.py` (~2 min), or walk through `nbplus/committee.py` (Eq. 1, EXP3.S update).
12. **Limitations & future work; Q&A**.
