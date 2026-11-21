import mujoco
import numpy as np

from simulation.coverage import scan_target
from simulation.scene import ENTRANCE, build_scene, set_scan_pose


def fully_outside(data):
    scanner = data.body("scanner")
    axis_x = scanner.xmat.reshape(3, 3)[0]

    # The scanner spans x=[0, 0.500] and y,z=[-0.050, 0.050] in its frame.
    front_x = scanner.xpos[0] + max(0.0, 0.500 * axis_x[0])
    front_x += 0.050 * (abs(axis_x[1]) + abs(axis_x[2]))
    return front_x < ENTRANCE[0]


def run_reference():
    model = build_scene()
    data = mujoco.MjData(model)
    reference = mujoco.MjData(model)

    set_scan_pose(data, np.deg2rad(-25))
    mujoco.mj_forward(model, data)
    assert fully_outside(data), "Scanner starts inside the entrance plane"
    assert data.ncon == 0, "Contact at the starting pose"

    # Shoulder sets insertion depth. The wrist sweeps each band in turn.
    waypoints_deg = [
        (-14, -165), (-14, 165),
        (-7, 165), (-7, -165),
        (0, -165), (0, 165),
        (-7, 165), (-14, 165), (-25, 165),
    ]
    covered = set()

    for shoulder, wrist in waypoints_deg:
        set_scan_pose(reference, np.deg2rad(shoulder), np.deg2rad(wrist))

        for _ in range(round(12 / model.opt.timestep)):
            error = reference.qpos - data.qpos
            speed = np.clip(5 * error, -0.8, 0.8)
            acceleration = 20 * (speed - data.qvel)
            mujoco.mj_mulM(model, data, data.qfrc_applied, acceleration)
            data.qfrc_applied[:] += data.qfrc_bias

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if data.ncon:
                raise RuntimeError(
                    f"Contact at t={data.time:.3f} s moving toward "
                    f"shoulder={shoulder}, wrist={wrist}"
                )

            target = scan_target(model, data)
            if target is not None:
                covered.add(target)

            position_error = np.max(np.abs(reference.qpos - data.qpos))
            if position_error < 0.002 and np.max(np.abs(data.qvel)) < 0.01:
                break
        else:
            raise RuntimeError(f"Waypoint timed out: {shoulder}, {wrist}")

    counts = [sum(band == depth for band, _ in covered) for depth in (100, 200, 300)]
    withdrawn = fully_outside(data)
    print("Coverage at 100, 200, 300 mm:", counts)
    print("Fully withdrawn:", bool(withdrawn))
    assert sum(counts) >= 20 and min(counts) >= 6 and withdrawn


if __name__ == "__main__":
    run_reference()