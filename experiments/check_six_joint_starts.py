"""Run the saved policy from starts with offsets on all six joints.

The official evaluation changes joint_1 and joint_2 by up to 1 degree. This check
draws 20 valid starts per range, with every joint offset uniformly at random.
A start is valid when the scanner is fully outside the entrance without contact.
"""

from collections import Counter

import mujoco
import numpy as np
from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym
from simulation.reference import fully_outside

RANGES_DEG = (1.0, 2.0, 3.0)
STARTS_PER_RANGE = 20


def run_from(env, policy, offsets_deg):
    sim = env.inspection
    env.reset(options={"offsets_deg": (0.0, 0.0)})
    sim.data.qpos[:] += np.deg2rad(offsets_deg)
    mujoco.mj_forward(sim.model, sim.data)
    if not fully_outside(sim.data) or sim.data.ncon:
        return None

    obs = env.encode(sim.observation())
    while True:
        action = int(policy.predict(obs, deterministic=True)[0])
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["outcome"], len(sim.covered), sim.data.time


def main():
    policy = PPO.load("models/hc20dt_ppo_4096.zip", device="cpu")
    env = PipeInspectionGym()

    for limit in RANGES_DEG:
        rng = np.random.default_rng(0)
        outcomes, failures, drawn = Counter(), [], 0
        while sum(outcomes.values()) < STARTS_PER_RANGE:
            offsets = rng.uniform(-limit, limit, 6)
            drawn += 1
            result = run_from(env, policy, offsets)
            if result is None:
                continue
            outcome, targets, sim_time = result
            outcomes[outcome] += 1
            if outcome != "success":
                failures.append((np.round(offsets, 2).tolist(), outcome, targets, round(sim_time, 2)))

        print(f"Offsets up to {limit:g} deg on all six joints: {dict(outcomes)} "
              f"({STARTS_PER_RANGE} valid of {drawn} drawn)", flush=True)
        for offsets, outcome, targets, sim_time in failures:
            print(f"  {outcome} at {sim_time} s with {targets} targets, offsets {offsets}", flush=True)


if __name__ == "__main__":
    main()
