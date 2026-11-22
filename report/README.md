# report

[REPORT.md](REPORT.md) is the technical report. The same text is submitted as a PDF in the ZIP archive.

| Path | What it holds |
|---|---|
| `REPORT.md` | The report: task, policy, training, evaluation, contact checks, limits and an appendix on other algorithms and devices. |
| `figures/` | All figure files. The report uses `fig1`, `fig5`, `fig10` and `fig9` as Figures 1–4, and `fig7` and `fig6` as Figures 5 and 6 in the appendix. |
| `data/seed_results.csv` | Results of the five-seed study behind the seed figure. |
| `make_figures.py` | Reruns the saved policy to draw figures 1–5. |
| `plot_benchmarks.py` | Draws figures 6–10 from the saved CSV and text results, without running the simulation. |

To redraw the figures from the repository root:

```bash
MUJOCO_GL=egl python -m report.make_figures
python -m report.plot_benchmarks
```

Figures 2, 3, 4 and 8 are not in the report. They show coverage at each training checkpoint, scanner depth and coverage over time, targets per dropout condition, and the six-joint stress test. I kept them for reference.
