import mujoco
import numpy as np

scene = """
<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <replicate count="32" euler="11.25 0 0">
      <geom name="rim" type="box"
            pos="0.0125 0 0.150" size="0.0125 0.018 0.025"/>
    </replicate>

    <replicate count="32" euler="11.25 0 0">
      <geom name="pipe_wall" type="box"
            pos="0.200 0 0.1875" size="0.200 0.020 0.0125"/>
    </replicate>

    <body name="scanner">
      <freejoint/>
      <geom type="box" size="0.250 0.050 0.050" mass="1"/>
      <site name="scan_origin" pos="0.225 0 0.050"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(scene)
data = mujoco.MjData(model)

# Each case gives the label, scanner-centre x and z in metres, and expected contact.
cases = [
    ("fully outside", -0.260, 0.000, False),
    ("centred in opening", -0.200, 0.000, False),
    ("raised into rim", -0.200, 0.085, True),
    ("upper corner clear", -0.200, 0.060, False),
    ("upper corner intersects rim", -0.200, 0.070, True),
    ("inside pipe with clearance", 0.300, 0.110, False),
    ("pipe wall clear at 118 mm", 0.300, 0.118, False),
    ("pipe wall contact at 119 mm", 0.300, 0.119, True),
    ("upper corner intersects pipe wall", 0.300, 0.125, True),
]

for label, x, z, expected_contact in cases:
    data.qpos[:3] = (x, 0, z)
    mujoco.mj_forward(model, data)

    contact = data.ncon > 0
    print(f"{label}: contact={contact}, points={data.ncon}")
    assert contact == expected_contact, label

# Keep the scanner outside the pipe while testing rays from known positions.
data.qpos[:3] = (-0.260, 0.0, 0.0)
mujoco.mj_forward(model, data)

# Depth (m), ray angle (degrees), expected first surface, expected range (mm).
ray_cases = [
    (0.2000, 0.0, "pipe_wall", 125.0),
    (0.0125, 0.0, "rim", 75.0),
    (0.2000, 112.5, "pipe_wall", 125.0),
]

covered = set()

for x, ray_angle_deg, expected_surface, expected_mm in ray_cases:
    theta = np.radians(ray_angle_deg)
    direction = np.array([0.0, -np.sin(theta), np.cos(theta)])
    origin = np.array([x, 0.0, 0.0]) + 0.050 * direction
    geom_id = np.array([-1], dtype=np.int32)

    distance = mujoco.mj_ray(
        model, data, origin, direction, None, True, -1, geom_id, None
    )
    assert distance >= 0, f"No ray hit at x={x} m, angle={ray_angle_deg} deg"

    surface = model.geom(int(geom_id[0])).name
    distance_mm = distance * 1000
    print(f"x={x * 1000:g} mm, angle={ray_angle_deg:g} deg -> "
          f"{surface}, {distance_mm:.1f} mm")

    assert surface.startswith(expected_surface), surface
    assert np.isclose(distance_mm, expected_mm), distance_mm

    if not surface.startswith("pipe_wall"):
        continue
    if not 75 <= distance_mm <= 175:
        continue

    hit_mm = (origin + distance * direction) * 1000
    depth_mm, y_mm, z_mm = hit_mm

    for band_mm in (100, 200, 300):
        if abs(depth_mm - band_mm) <= 10:
            # Around +X, the angle starts at +Z and increases toward -Y.
            angle_deg = np.degrees(np.arctan2(-y_mm, z_mm)) % 360
            sector = int(angle_deg // 45) + 1
            covered.add((band_mm, sector))
            break

print("Covered targets:", sorted(covered))
assert covered == {(200, 1), (200, 3)}, covered

site_id = model.site("scan_origin").id
scanner_id = model.body("scanner").id

# Scanner centre at x = -25 mm puts the emitter at x = 200 mm.
data.qpos[:3] = (-0.025, 0.0, 0.0)

# Roll about +X, expected emitter position (m), expected world direction.
mounted_ray_cases = [
    (0.0,  (0.200,  0.000, 0.050), (0.0,  0.0, 1.0)),
    (90.0, (0.200, -0.050, 0.000), (0.0, -1.0, 0.0)),
]

for roll_deg, expected_origin, expected_direction in mounted_ray_cases:
    half_angle = np.radians(roll_deg) / 2
    data.qpos[3:7] = (np.cos(half_angle), np.sin(half_angle), 0.0, 0.0)
    mujoco.mj_forward(model, data)

    origin = data.site_xpos[site_id]
    rotation = data.site_xmat[site_id].reshape(3, 3)
    direction = rotation[:, 2].copy()  # Local +Z expressed in world coordinates.

    assert np.allclose(origin, expected_origin), (roll_deg, origin)
    assert np.allclose(direction, expected_direction), (roll_deg, direction)

    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model, data, origin, direction, None, True, scanner_id, geom_id, None
    )
    assert distance >= 0, f"No hit from the mounted scanner at {roll_deg:g} deg"

    surface = model.geom(int(geom_id[0])).name
    distance_mm = distance * 1000
    print(f"Mounted ray at {roll_deg:g} deg -> {surface}, {distance_mm:.1f} mm")

    assert surface.startswith("pipe_wall"), (roll_deg, surface)
    assert np.isclose(distance_mm, 125.0), (roll_deg, distance_mm)