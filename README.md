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

I placed the entrance centre at `(1.255, 0, 1.200)` m in `base_link`, with the pipe axis pointing along +X. At zero joint angles the emitter is at x = 1.555 m, which puts it 300 mm inside the pipe. The scene uses the supplied pipe and collar meshes for display and 32 box segments per ring for contact checks. This leaves the bore open for the scanner.

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