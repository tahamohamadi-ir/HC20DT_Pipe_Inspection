# evaluation

This folder holds the fixed test starts, the scripts that run the saved policy on them, and every result file quoted in the report.

| File | What it holds |
|---|---|
| `starts.csv` | The ten official starts: the nominal pose and nine `joint_1` and `joint_2` offsets between −1° and +1°, fixed before evaluation. |
| `run.py` | Runs the saved policy on those starts and writes `results.csv`. |
| `results.csv` | One row per official run: coverage at each depth, collision, withdrawal, number of decisions and outcome. |
| `dropout_scenarios.csv` | Seven scan conditions. Hits are withheld for 1 or 3 seconds starting at 4, 12 or 20 s, plus one run with normal readings. |
| `dropout.py` | Runs four controllers under each condition and prints one CSV row per episode. |
| `dropout_results.csv` | The 280 episodes from `dropout.py`, including the action sequence of each run. |
| `hardware_seed0_4096/` | The CPU and CUDA comparison: `summary.csv` per controller and device, plus one row per episode in the three `*_episodes.csv` files. |
| `six_joint_starts_results.txt` | Output of `experiments/check_six_joint_starts.py`, including the bodies in contact for each failure. |

To repeat the main results from the repository root:

```bash
python -m evaluation.run
python -m evaluation.dropout > evaluation/dropout_results.csv
```

The tables and what they show are in the [main README](../README.md#evaluate-the-saved-policy) and the [report](../report/REPORT.md).
