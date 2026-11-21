from pathlib import Path

import mujoco
import numpy as np

from simulation.robot import load_robot


ENTRANCE = np.array([1.255, 0.0, 1.200])
SEGMENTS = 32


def build_scene():
    spec = load_robot()
    fixture = spec.worldbody.add_body(name="pipe_fixture", pos=ENTRANCE)
    assets = Path(__file__).resolve().parents[1] / "assets"

    # Name, supplied mesh, inner radius, outer radius, axial length (m).
    rings = [
        ("pipe", "pipe_id350_od400_l400_mm.stl", 0.175, 0.200, 0.400),
        ("rim", "entrance_collar_id250_od350_l25_mm.stl", 0.125, 0.175, 0.025),
    ]

    for name, filename, inner, outer, length in rings:
        spec.add_mesh(name=name, file=str(assets / filename), scale=[0.001] * 3)
        fixture.add_geom(
            name=f"{name}_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=name,
            group=2,
            contype=0,
            conaffinity=0,
        )

        radius = (inner + outer) / 2
        half_width = outer * np.tan(np.pi / SEGMENTS)

        for i in range(SEGMENTS):
            angle = 2 * np.pi * i / SEGMENTS
            fixture.add_geom(
                name=f"{name}_{i}",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                group=3,
                pos=[length / 2, -radius * np.sin(angle), radius * np.cos(angle)],
                quat=[np.cos(angle / 2), np.sin(angle / 2), 0, 0],
                size=[length / 2, half_width, (outer - inner) / 2],
            )

    return spec.compile()


if __name__ == "__main__":
    model = build_scene()
    data = mujoco.MjData(model)
    bands_reached = set()

    for step, shoulder in enumerate(np.deg2rad(np.linspace(-25, 0, 101))):
        # Keep the scanner horizontal and its centre on the pipe axis.
        pitch = np.arcsin(0.820 * (np.cos(shoulder) - 1) / 0.880)
        data.joint("joint_2").qpos[0] = shoulder
        data.joint("joint_3").qpos[0] = shoulder - pitch
        data.joint("joint_5").qpos[0] = pitch
        mujoco.mj_forward(model, data)

        assert data.ncon == 0, (step, data.ncon)
        depth_mm = 1000 * (data.site("scan_origin").xpos[0] - ENTRANCE[0])

        if step == 0:
            # The tip is 25 mm ahead of the emitter along the pipe axis.
            tip_depth_mm = depth_mm + 25
            assert tip_depth_mm < 0, tip_depth_mm
            print(f"Starting tip depth: {tip_depth_mm:.1f} mm")

        for band in (100, 200, 300):
            if abs(depth_mm - band) <= 10:
                bands_reached.add(band)

    assert bands_reached == {100, 200, 300}, bands_reached
    print("Depth bands reached (mm):", sorted(bands_reached))

    data.joint("joint_1").qpos[0] = np.deg2rad(4)
    mujoco.mj_forward(model, data)

    scanner_geom = model.geom("scanner_collision").id
    scanner_contacts = set()
    for contact in data.contact:
        first, second = contact.geom
        if first == scanner_geom:
            scanner_contacts.add(model.geom(second).name)
        elif second == scanner_geom:
            scanner_contacts.add(model.geom(first).name)

    assert any(name.startswith("rim_") for name in scanner_contacts), scanner_contacts
    print("Scanner contact:", sorted(scanner_contacts))