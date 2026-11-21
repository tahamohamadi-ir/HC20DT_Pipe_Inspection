# HC20DT_Pipe_Inspection

## Local setup

I ran the project setup on Ubuntu 24.04.5 LTS in WSL 2. ROS 2 Jazzy was already installed.

```bash
/usr/bin/python3 -m venv --system-site-packages ~/.venvs/hc20dt
source /opt/ros/jazzy/setup.bash
source ~/.venvs/hc20dt/bin/activate
python -m pip install mujoco==4.0.1
```

I checked that Python could load both ROS 2 and MuJoCo in the same session:

```bash
python -c "import os, rclpy, mujoco; print(os.environ['ROS_DISTRO'], mujoco.__version__)"
```

Output: `jazzy 4.0.1`


## Ray occlusion experiment

With the environment above active, run this from the repository root:

```bash
ros2 launch experiments/ray_occlusion.launch.py
```

With the obstacle in the ray's path, the first hit is at 100 mm.
Moving the obstacle 30 mm sideways exposes the wall at 125 mm.

## HC20DT model in RViz

I used Yaskawa's HC20DT description and built the two ROS packages in a separate workspace. To reproduce the setup:

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
source install/local_setup.bash
```

To open the robot model and joint sliders:

```bash
ros2 launch motoman_hc20_support view_hc20dt.launch.xml
```

I checked the expanded URDF before mounting the scanner. It has six revolute joints. The `flange` frame is fixed 200 mm along the positive X axis of `link_6`. This offset is already included in the robot model.

## Scanner mount in MuJoCo

With the ROS workspace sourced and the Python environment active, run `python simulation/robot.py` from the repository root.

At zero joint angles, the emitter was 475 mm forward and 50 mm above the flange. After I rotated `joint_6` by +90°, the ray pointed along +Y in the robot base frame. The script checks both poses.


I excluded the base_link–`link_1` pair because their collision meshes overlap at the first joint. The zero pose then had no contacts. With joint_2 at −60° and joint_3 at 240°, MuJoCo still detected contact between base_link and link_5.