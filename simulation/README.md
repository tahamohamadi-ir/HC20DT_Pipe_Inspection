# simulation

This folder builds the MuJoCo scene, runs the robot and checks coverage and contact. Everything else in the project imports from here.

| File | What it does |
|---|---|
| `robot.py` | Loads the Yaskawa HC20DT description, mounts the scanner on the flange and adds the scan origin at (475, 0, 50) mm. Running it checks the emitter position and ray direction. |
| `scene.py` | Adds the pipe and entrance collar at `(1.255, 0, 1.200)` m. Each ring uses 32 boxes for contact and the supplied mesh for rendering. `set_scan_pose` sets the joints that keep the scanner level for a given shoulder and wrist angle. |
| `coverage.py` | Casts the scan ray, ignores the scanner itself, and turns a valid first hit into a depth band and sector. |
| `reference.py` | The scripted inspection used as a baseline. It also holds the joint controller (`step_toward`) and the full-withdrawal check (`fully_outside`). |
| `inspection_env.py` | One episode of the task: the eight actions, the reward, contact checks at every physics step and the end conditions. |
| `gym_env.py` | Wraps the episode as a Gymnasium environment with the 31-value observation used for training. |
| `run_policy.py` | The ROS 2 node started by `launch/inspection.launch.py`. It runs the saved policy and publishes `/joint_states` and `/scanner_pose`. |

Most files run their own checks when started from the repository root, for example `python -m simulation.scene`. The [main README](../README.md) lists the commands and what they print.
