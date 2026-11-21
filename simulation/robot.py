from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


def load_robot():
    package = "motoman_hc20_support"
    share = Path(subprocess.check_output(
        ["ros2", "pkg", "prefix", "--share", package], text=True
    ).strip())
    urdf = subprocess.check_output(
        ["xacro", str(share / "urdf" / "hc20dt.xacro")], text=True
    )
    robot = ET.fromstring(urdf)

    compiler = ET.SubElement(ET.SubElement(robot, "mujoco"), "compiler")
    compiler.attrib.update(discardvisual="true", fusestatic="false")

    # Use the manufacturer's collision meshes and keep the fixed flange frame.
    assets = {}
    for mesh in robot.findall("./link/collision/geometry/mesh"):
        relative = mesh.attrib["filename"].removeprefix(f"package://{package}/")
        path = share / relative
        assets[path.name] = path.read_bytes()
        mesh.set("filename", path.name)

    scanner_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "scanner_box_500x100x100_mm.stl"
    )
    assets[scanner_path.name] = scanner_path.read_bytes()

    scanner = ET.SubElement(robot, "link", name="scanner")
    collision = ET.SubElement(scanner, "collision", name="scanner_collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(
        geometry, "mesh", filename=scanner_path.name, scale="0.001 0.001 0.001"
    )

    mount = ET.SubElement(robot, "joint", name="scanner_mount", type="fixed")
    ET.SubElement(mount, "parent", link="flange")
    ET.SubElement(mount, "child", link="scanner")

    spec = mujoco.MjSpec.from_string(
        ET.tostring(robot, encoding="unicode"), assets=assets
    )
    spec.body("scanner").add_site(name="scan_origin", pos=[0.475, 0.0, 0.050])
    return spec


if __name__ == "__main__":
    model = load_robot().compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.nq == model.nv == 6
    assert model.body("base_link").parentid.item() == 0
    assert model.body("base_link").jntnum.item() == 0
    assert model.body("scanner").parentid.item() == model.body("flange").id

    site = data.site("scan_origin")
    assert np.allclose(data.body("flange").xpos, [1.080, 0.0, 1.200])
    assert np.allclose(site.xpos, [1.555, 0.0, 1.250])
    assert np.allclose(site.xmat.reshape(3, 3)[:, 2], [0.0, 0.0, 1.0])
    print("Flange at zero (m):", data.body("flange").xpos)
    print("Emitter at zero (m):", site.xpos)

    data.joint("joint_6").qpos[0] = np.pi / 2
    mujoco.mj_forward(model, data)

    direction = site.xmat.reshape(3, 3)[:, 2]
    assert np.allclose(site.xpos, [1.555, 0.050, 1.200])
    assert np.allclose(direction, [0.0, 1.0, 0.0])
    print("Emitter after joint_6 +90 deg (m):", site.xpos.round(6))
    print("Ray direction:", direction.round(6))