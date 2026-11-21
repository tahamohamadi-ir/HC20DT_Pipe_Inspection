import mujoco

scene = """
<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <replicate count="32" euler="11.25 0 0">
      <geom name="rim" type="box"
            pos="0.0125 0 0.150" size="0.0125 0.018 0.025"/>
    </replicate>

    <body name="scanner">
      <freejoint/>
      <geom type="box" size="0.250 0.050 0.050" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(scene)
data = mujoco.MjData(model)

# Positions refer to the scanner centre, in metres.
cases = [
    ("fully outside", -0.260, 0.000, False),
    ("centred in opening", -0.200, 0.000, False),
    ("raised into rim", -0.200, 0.085, True),
    ("upper corner clear", -0.200, 0.060, False),
    ("upper corner intersects rim", -0.200, 0.070, True),
]

for label, x, z, expected_contact in cases:
    data.qpos[:3] = (x, 0, z)
    mujoco.mj_forward(model, data)
    contact = data.ncon > 0
    print(f"{label}: contact={contact}, points={data.ncon}")
    assert contact == expected_contact, label