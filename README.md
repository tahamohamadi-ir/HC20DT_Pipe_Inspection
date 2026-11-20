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