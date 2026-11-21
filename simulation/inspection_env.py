import mujoco
import numpy as np

from simulation.coverage import scan_target
from simulation.reference import fully_outside, step_toward
from simulation.scene import build_scene, set_scan_pose


SHOULDER_DEG = (-25, -14, -7, 0)
WRIST_DEG = (-165, 165)


class InspectionEnv:
    def __init__(self):
        self.model = build_scene()
        self.on_physics_step = None
        self.command = mujoco.MjData(self.model)

    def reset(self, *, base_offset_deg=0.0, shoulder_offset_deg=0.0):
        offsets = np.array([base_offset_deg, shoulder_offset_deg], dtype=float)
        if offsets.shape != (2,) or not np.all(np.abs(offsets) <= 1.0):
            raise ValueError("Start offsets must each be within [-1, 1] degrees")

        self.data = mujoco.MjData(self.model)
        set_scan_pose(self.data, np.deg2rad(SHOULDER_DEG[0]))
        for joint, offset in zip(("joint_1", "joint_2"), np.deg2rad(offsets)):
            self.data.joint(joint).qpos[0] += offset

        mujoco.mj_forward(self.model, self.data)
        if not fully_outside(self.data) or self.data.ncon:
            raise ValueError(f"Invalid starting pose for offsets {offsets.tolist()}")

        self.start_qpos_rad = self.data.qpos.copy()
        self.covered = set()
        self.remaining = 16
        self.done = False
        return self.observation()

    def observation(self):
        return self.data.qpos.copy(), frozenset(self.covered), self.remaining

    def step(self, level, side):
        if self.done:
            raise RuntimeError("Call reset before starting another episode")
        if level not in range(4) or side not in range(2):
            raise ValueError((level, side))

        shoulder, wrist = np.deg2rad([SHOULDER_DEG[level], WRIST_DEG[side]])
        set_scan_pose(self.command, shoulder, wrist)
        before = len(self.covered)
        self.remaining -= 1
        outcome = "moving"

        for _ in range(round(12 / self.model.opt.timestep)):
            step_toward(self.model, self.data, self.command)
            if self.on_physics_step is not None:
                self.on_physics_step()

            if self.data.ncon:
                outcome = "collision"
                break

            target = scan_target(self.model, self.data)
            if target is not None:
                self.covered.add(target)

            error = np.max(np.abs(self.command.qpos - self.data.qpos))
            if error < 0.002 and np.max(np.abs(self.data.qvel)) < 0.01:
                break
        else:
            outcome = "motion_timeout"

        if outcome == "moving" and level == 0:
            counts = [
                sum(b == band for b, _ in self.covered)
                for band in (100, 200, 300)
            ]
            passed = len(self.covered) >= 20 and min(counts) >= 6
            outcome = "success" if passed and fully_outside(self.data) else "incomplete"

        if outcome == "moving" and self.remaining == 0:
            outcome = "decision_limit"

        self.done = outcome != "moving"
        reward = len(self.covered) - before
        if self.done:
            reward += 25 if outcome == "success" else -25

        return self.observation(), reward, self.done, outcome


if __name__ == "__main__":
    env = InspectionEnv()
    env.reset()

    path = [
        (1, 0), (1, 1), (2, 1), (2, 0), (3, 0),
        (3, 1), (2, 1), (1, 1), (0, 1),
    ]
    total = 0
    for action in path:
        _, reward, done, outcome = env.step(*action)
        total += reward

    print(outcome, len(env.covered), total)
    assert done and outcome == "success" and len(env.covered) == 24 and total == 49

    env.reset()
    _, reward, done, outcome = env.step(0, 0)
    print(outcome, len(env.covered), reward)
    assert done and outcome == "incomplete" and reward == -25

    for base, shoulder in [(0.0, 0.0), (-1.0, 1.0), (1.0, -1.0)]:
        qpos, covered, remaining = env.reset(
            base_offset_deg=base, shoulder_offset_deg=shoulder
        )
        measured = np.rad2deg(qpos[:2])
        assert np.allclose(measured, [base, -25.0 + shoulder])
        assert not covered and remaining == 16
        print("Starting joint_1 and joint_2 (deg):", measured.round(2).tolist())