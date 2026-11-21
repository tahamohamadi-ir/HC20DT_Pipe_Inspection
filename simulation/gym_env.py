import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.inspection_env import InspectionEnv


class PipeInspectionGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.inspection = InspectionEnv()
        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

    @staticmethod
    def encode(state):
        qpos, covered, remaining = state
        mask = [
            (depth, sector) in covered
            for depth in (100, 200, 300)
            for sector in range(1, 9)
        ]
        return np.asarray(
            [*(qpos / np.pi), *mask, remaining / 16],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if options and "offsets_deg" in options:
            base, shoulder = options["offsets_deg"]
        else:
            base, shoulder = self.np_random.uniform(-1.0, 1.0, size=2)

        state = self.inspection.reset(
            base_offset_deg=base,
            shoulder_offset_deg=shoulder,
        )
        info = {"offsets_deg": (float(base), float(shoulder))}
        return self.encode(state), info

    def step(self, action):
        level, side = divmod(int(action), 2)
        state, reward, done, outcome = self.inspection.step(level, side)
        info = {"outcome": outcome}
        return self.encode(state), float(reward), done, False, info


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = PipeInspectionGym()
    check_env(env)
    obs, info = env.reset(options={"offsets_deg": (0.0, 0.0)})
    print("Observation:", obs.shape, obs.dtype)
    print("Nominal offsets:", info["offsets_deg"])