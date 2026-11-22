# HC20DT Pipe Inspection

This project uses ROS 2 Jazzy and MuJoCo to simulate a Yaskawa Motoman HC20DT inspecting the inside of a pipe. It includes the robot model, a flange-mounted scanner, collision and ray checks, and a scripted reference inspection.

## Local setup

I ran the project on Ubuntu 24.04.5 LTS under WSL 2 with ROS 2 Jazzy already installed. The commands below use `git`, `colcon`, `xacro`, and the system Python with `venv` support.

### Create the Python environment

```bash
/usr/bin/python3 -m venv --system-site-packages ~/.venvs/hc20dt
source /opt/ros/jazzy/setup.bash
source ~/.venvs/hc20dt/bin/activate
python -m pip install -r requirements.txt
```

I used the system Python so the environment could load the ROS Python bindings. To check both imports in the same session, run

```bash
python -c "import os, rclpy, mujoco; print(os.environ['ROS_DISTRO'], mujoco.__version__)"
```

On my installation this printed `jazzy 4.0.1`.

### Build the Yaskawa model packages

I built Yaskawa's robot description in a separate ROS workspace. The checkout below uses the revision I tested.

```bash
source /opt/ros/jazzy/setup.bash
mkdir -p ~/hc20dt_ws/src
cd ~/hc20dt_ws/src

git clone --filter=blob:none --no-checkout https://github.com/Yaskawa-Global/motoman_ros2_support_packages.git
cd motoman_ros2_support_packages
git sparse-checkout set motoman_hc20_support motoman_resources
git checkout --detach a77f186048bd3c35fc3bb5142f9d9270dbebe69f

cd ~/hc20dt_ws
colcon build --packages-select motoman_resources motoman_hc20_support
```

### Start a new terminal

Open a WSL terminal in the `HC20DT_Pipe_Inspection` repository root. Run these three lines once in that terminal before starting the simulation.

```bash
source /opt/ros/jazzy/setup.bash
source ~/hc20dt_ws/install/local_setup.bash
source ~/.venvs/hc20dt/bin/activate
```

To check that ROS can find the robot description, run

```bash
ros2 pkg prefix --share motoman_hc20_support
```

This should print the package's `share/motoman_hc20_support` directory under `~/hc20dt_ws/install`.

## Reference inspection

```bash
python -m simulation.reference
```

The scripted controller starts with the scanner fully outside the entrance, inserts it, sweeps the wrist at three depths, and withdraws through the intermediate depths. The script checks for contact at every physics step. It checks withdrawal using the full scanner body, including its orientation.

In the nominal run, MuJoCo reported zero contacts. The final output was

```text
Coverage at 100, 200, 300 mm: [8, 8, 8]
Fully withdrawn: True
```

This run provides a baseline for evaluating a learned policy. The controller applies joint torques directly in MuJoCo without hardware torque limits.

## HC20DT model in RViz

To view the robot and move its six joints with sliders, run

```bash
ros2 launch motoman_hc20_support view_hc20dt.launch.xml
```

I checked the expanded URDF before mounting the scanner. It has six revolute joints. The fixed `flange` frame is 200 mm along the positive X axis of `link_6`. The robot description already includes this offset.

## Scanner mount in MuJoCo

```bash
python simulation/robot.py
```

At zero joint angles, the emitter was 475 mm forward and 50 mm above the flange. After I rotated `joint_6` by +90°, the ray pointed along +Y in the robot base frame. The script checks the emitter position and ray direction at both poses.

I excluded contact between `base_link` and `link_1` because their collision meshes overlap at the first joint. The zero pose then had no contacts. With `joint_2` at −60° and `joint_3` at 240°, MuJoCo detected contact between `base_link` and `link_5`.

## Pipe scene and contact checks

```bash
python -m simulation.scene
```

I placed the entrance centre at `(1.255, 0, 1.200)` m in `base_link`, with the pipe axis pointing along +X. At zero joint angles the emitter is at x = 1.555 m, which puts it 300 mm inside the pipe. The pipe and collar dimensions follow the supplied geometry. The compiled scene uses 32 box segments per ring for rendering and contact checks. This leaves the bore open for the scanner.

The scanner tip started 24.9 mm outside the entrance. MuJoCo reported zero contacts at the 101 sampled insertion poses, and the emitter reached the 100, 200 and 300 mm depth bands.

I checked two deliberate collisions. Turning `joint_1` by +4° from the final insertion pose brought the scanner into contact with the entrance collar. In a separate pose, with `joint_3` at −2° and `joint_5` at +15° and the other joints at zero, the scanner contacted the pipe wall 268.2–296.7 mm past the entrance. There was no collar contact in that pose.

## Scan coverage

```bash
python -m simulation.coverage
```

A target is one of eight 45° sectors in a depth band centred at 100, 200 or 300 mm. The depth tolerance is ±10 mm. A scan counts when the ray's first hit is the inner pipe wall and its range is between 75 and 175 mm, inclusive. The ray test excludes the scanner body. Sector numbering starts at +Z and follows the right-hand rule around +X.

I checked the centre of every sector at each depth. All 24 static poses returned the expected target without contact. Scanning the last pose a second time kept the count at 24.

## Ray occlusion experiment

```bash
ros2 launch experiments/ray_occlusion.launch.py
```

With the obstacle in the ray's path, the first hit was the obstacle at 100 mm. Moving it 30 mm sideways exposed the wall at 125 mm.

## PPO pilot

Run `python -m training.pilot` from the repository root. I trained PPO on CPU
for 128 decisions with seed 0. Training took 11.0 s, excluding validation.

Before training, the deterministic policy withdrew with zero coverage on all
five validation starts. After reloading the trained policy, coverage was
[4, 1, 0] at 100, 200 and 300 mm on all five starts. Each episode reached
the 16-decision limit. On the nominal start, the policy kept requesting the
same wrist position at 200 mm after its second decision.

To reproduce training, run python -m training.pilot followed by python -m training.checkpoints from the repository root. I selected the 4,096-decision checkpoint after it covered all 24 targets and withdrew without contact on all five validation starts.

## Hardware and training time

| Item | Value |
|---|---|
| CPU | Intel Core i7-13620H (10 cores, 16 threads) |
| RAM | 40 GB DDR5-4800 (32 GB + 8 GB); about 19 GB available to WSL 2 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 6 GB (not used; training ran on CPU) |
| Host OS | Windows 11 Enterprise, build 26300 |
| Linux environment | Ubuntu 24.04.5 LTS on WSL 2 (WSL 3.0.5, kernel 6.18.58.1) |
| ROS 2 | Jazzy |
| Python | 3.12.3 |
| Key libraries | MuJoCo 4.0.1, Gymnasium 1.2.2, Stable-Baselines3 2.7.1, PyTorch 2.15.0, NumPy 1.26.4 |

| Training stage | Decisions | Wall time (CPU) |
|---|---|---|
| Pilot (`python -m training.pilot`) | 0 → 128 | 11.0 s |
| Checkpoints (`python -m training.checkpoints`) | 128 → 1,024 | 84.7 s |
| | 1,024 → 2,048 | 104.7 s |
| | 2,048 → 4,096 | 260.6 s |
| **Total** | **0 → 4,096** | **461.0 s (7 min 41 s)** |

Times exclude validation episodes. PPO was configured with `device="cpu"`.
The selected policy is `models/hc20dt_ppo_4096.zip`.

## Run the saved policy with ROS 2

From the repository root, activate the project environment as described above and run:

```bash
ros2 launch launch/inspection.launch.py
```

The launch runs the saved PPO policy from the nominal start. During the
inspection, `/joint_states` publishes all six joint positions and velocities,
and `/scanner_pose` publishes the scanner body's pose in `base_link`.
The final log line reports the episode outcome and the number of targets covered.

## Evaluate the saved policy

From the repository root, with the project environment active, run

```bash
python -m evaluation.run
```

The script loads the saved policy, reads the ten starts in [evaluation/starts.csv](evaluation/starts.csv), and writes one row per episode to [evaluation/results.csv](evaluation/results.csv). Across these ten starts, the policy covered all 24 targets on every run and withdrew fully with zero collisions.

## View the inspection

Watch the [recorded inspection](demo/inspection.mp4), or run the policy live in WSL with graphical application support:

```bash
MUJOCO_GL=egl python -m demo.live
```

The cutaway hides the near side of the rings so the scanner and its ray remain visible. The window shows coverage at each depth. Press Q or Esc to close it during the run.

To record a new video at twice the simulation speed, run

```bash
MUJOCO_GL=egl python -m demo.record_demo
```

Both commands use OpenCV and EGL. Recording also requires FFmpeg with the libx264 encoder and replaces `demo/inspection.mp4`.