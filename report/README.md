# report

The technical report is in two formats with the same text: [REPORT.md](REPORT.md) for reading on GitHub and [HC20DT_technical_report.pdf](HC20DT_technical_report.pdf) for print.

| Path | What it holds |
|---|---|
| `REPORT.md` | The report: task, policy, training, evaluation, contact checks, limits, an appendix on other algorithms and devices, and a demo and discussion section. |
| `HC20DT_technical_report.pdf` | The same report as a seven-page PDF. |
| `figures/` | All figure files. The report uses `fig1`, `fig5`, `fig10` and `fig9` as Figures 1–4, `fig7` and `fig6` as Figures 5 and 6 in the appendix, and `fig11` as Figure 7. |
| `data/seed_results.csv` | Results of the five-seed study behind the seed figure. |
| `make_figures.py` | Reruns the saved policy to draw figures 1–5. |
| `plot_benchmarks.py` | Draws figures 6–10 from the saved CSV and text results, without running the simulation. |
| `demo_frames.py` | Takes nine frames from `demo/inspection.mp4` with FFmpeg and draws figure 11. |

To redraw the figures from the repository root:

```bash
MUJOCO_GL=egl python -m report.make_figures
python -m report.plot_benchmarks
python -m report.demo_frames
```

Figures 2, 3, 4 and 8 are not in the report. They show coverage at each training checkpoint, scanner depth and coverage over time, targets per dropout condition, and the six-joint stress test. I kept them for reference.
