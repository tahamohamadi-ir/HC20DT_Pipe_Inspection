import mujoco
import numpy as np

from simulation.coverage import scan_target
from simulation.reference import step_toward
from simulation.scene import build_scene, set_scan_pose


if __name__ == "__main__":
    model = build_scene()

    for wrist_deg, expected in [(-22.5, (300, 1)), (22.5, (300, 8))]:
        data = mujoco.MjData(model)
        command = mujoco.MjData(model)

        set_scan_pose(data, 0.0)
        mujoco.mj_forward(model, data)
        assert data.ncon == 0

        set_scan_pose(command, 0.0, np.deg2rad(wrist_deg))
        for _ in range(round(1.0 / model.opt.timestep)):
            step_toward(model, data, command)
            assert data.ncon == 0, (wrist_deg, data.time)

        actual_deg = np.rad2deg(data.joint("joint_6").qpos[0])
        target = scan_target(model, data)
        print(f"command={wrist_deg:+.1f} deg, wrist={actual_deg:+.2f} deg, target={target}")

        assert abs(actual_deg - wrist_deg) < 1.0, actual_deg
        assert target == expected, (expected, target)