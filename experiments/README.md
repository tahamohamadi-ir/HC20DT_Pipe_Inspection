# experiments

These are the smaller checks I ran while building the project. Some test one piece of the model in a stripped-down scene. Others were first looks at the trained policy that later became the scripts in `evaluation/`. I kept them so the reasoning behind the final setup can be traced.

| File | What it checks |
|---|---|
| `ray_occlusion.py` | The first ray test: an obstacle in front of a wall must block the ray, and moving it aside must expose the wall. |
| `ray_occlusion.launch.py` | Runs the same test from ROS 2 to confirm the ROS and MuJoCo setup. |
| `collar_contact.py` | Checks the 32-box collar and pipe wall against hand-calculated contact heights. `collar_contact.md` lists the cases and results. |
| `wrist_action.py` | Turns the wrist by ±22.5° at the 300 mm band and checks that the ray lands in sectors 1 and 8 without contact. |
| `compare_inspection_paths.py` | Compares the saved policy with the scripted path, a shorter scripted path and random actions on the ten starts. |
| `check_dropout_official_starts.py` | Withholds scan hits during the second decision, for the policy and for a replay of its nominal actions. |
| `check_coverage_exit.py` | Withdraws as soon as all 24 targets are covered, with and without that lost decision. |
| `check_six_joint_starts.py` | The stress test with random offsets on all six starting joints. Its output is in `evaluation/six_joint_starts_results.txt`. |

Run them from the repository root, for example `python -m experiments.wrist_action`. The last four also load `models/hc20dt_ppo_4096.zip`.

`compare_inspection_paths.py`, `check_dropout_official_starts.py` and `check_coverage_exit.py` print their results to the terminal and save nothing. Their findings led to the fixed scan conditions in `evaluation/dropout.py` and to the staged withdrawal compared in the report.
