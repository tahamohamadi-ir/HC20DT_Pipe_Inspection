import mujoco
import numpy as np


def measure_first_hit(obstacle_y):
    scene = f"""
    <mujoco>
      <worldbody>
        <geom name="wall" type="box"
              pos="0.200 0 0.180" size="0.050 0.050 0.005"/>
        <geom name="obstacle" type="box"
              pos="0.200 {obstacle_y} 0.155" size="0.010 0.010 0.005"/>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(scene)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    origin = np.array([0.200, 0.0, 0.050])
    direction = np.array([0.0, 0.0, 1.0])
    geom_id = np.array([-1], dtype=np.int32)

    distance = mujoco.mj_ray(
        model, data, origin, direction, None, True, -1, geom_id, None
    )
    assert distance >= 0, "Expected the ray to hit a surface"
    surface = model.geom(int(geom_id[0])).name
    return surface, distance * 1000


for obstacle_y, expected_surface, expected_mm in [
    (0.000, "obstacle", 100.0),
    (0.030, "wall", 125.0),
]:
    surface, distance_mm = measure_first_hit(obstacle_y)
    print(f"Y={obstacle_y * 1000:.0f} mm -> {surface}, {distance_mm:.1f} mm")
    assert surface == expected_surface, surface
    assert np.isclose(distance_mm, expected_mm), distance_mm