from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    root = Path(__file__).resolve().parents[1]
    return LaunchDescription([
        ExecuteProcess(
            cmd=["python", "-m", "simulation.run_policy"],
            cwd=str(root),
            output="screen",
        )
    ])