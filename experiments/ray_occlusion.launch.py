from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    experiment = Path(__file__).with_name("ray_occlusion.py")
    return LaunchDescription([
        ExecuteProcess(cmd=["python", str(experiment)], output="screen")
    ])