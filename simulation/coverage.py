import mujoco
import numpy as np

from simulation.scene import ENTRANCE, build_scene, set_scan_pose


# Group 2 contains the display meshes; group 3 contains the hollow wall segments.
RAY_GROUPS = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)


def scan_target(model, data):
    """Return (depth band in mm, sector 1–8), or None for an invalid scan."""
    site = data.site("scan_origin")
    direction = site.xmat.reshape(3, 3)[:, 2].copy()
    geom_id = np.array([-1], dtype=np.int32)
    normal = np.zeros(3)

    distance = mujoco.mj_ray(
        model, data, site.xpos, direction, RAY_GROUPS, True,
        model.body("scanner").id, geom_id, normal,
    )

    # Keep the inclusive endpoints stable under floating-point rounding.
    eps_m = 1e-9
    if not 0.075 - eps_m <= distance <= 0.175 + eps_m:
        return None

    geom = model.geom(int(geom_id[0]))
    if not geom.name.startswith("pipe_"):
        return None

    # Local +Z points outward on each wall box; its inner face points inward.
    outward = data.geom_xmat[geom.id].reshape(3, 3)[:, 2]
    if not np.allclose(normal, -outward, atol=1e-6):
        return None

    x, y, z = site.xpos + distance * direction - ENTRANCE
    for band_mm in (100, 200, 300):
        if abs(x - band_mm / 1000) <= 0.010 + eps_m:
            angle_deg = np.degrees(np.arctan2(-y, z)) % 360
            sector = int(angle_deg // 45) % 8 + 1
            return band_mm, sector

    return None


if __name__ == "__main__":
    model = build_scene()
    data = mujoco.MjData(model)

    set_scan_pose(data, np.deg2rad(-25))
    mujoco.mj_forward(model, data)
    assert scan_target(model, data) is None

    set_scan_pose(data, 0.0)
    mujoco.mj_forward(model, data)
    target = scan_target(model, data)
    print("Target at zero pose:", target)
    assert target == (300, 1), target

    covered = set()

    for shoulder_deg, expected_band in [(-14, 100), (-7, 200), (0, 300)]:
        for expected_sector in range(1, 9):
            centre_deg = 45 * (expected_sector - 0.5)

            # joint_6 turns about -X. Keep the wrist command within ±180°.
            wrist_deg = (180 - centre_deg) % 360 - 180
            set_scan_pose(data, np.deg2rad(shoulder_deg), np.deg2rad(wrist_deg))
            mujoco.mj_forward(model, data)

            expected = (expected_band, expected_sector)
            assert data.ncon == 0, f"Unexpected contact at {expected}"

            target = scan_target(model, data)
            assert target == expected, (expected, target)
            covered.add(target)

        print(f"{expected_band} mm: 8 sectors verified")

    assert len(covered) == 24

    # Scan the last pose again without moving the robot.
    repeated = scan_target(model, data)
    assert repeated == (300, 8), repeated
    covered.add(repeated)
    assert len(covered) == 24
    print("Unique targets after a repeated scan:", len(covered))