from pathlib import Path
from time import perf_counter

from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym
from training.pilot import validate


if __name__ == "__main__":
    train_env = PipeInspectionGym()
    validation_env = PipeInspectionGym()
    policy = PPO.load("/tmp/hc20dt_ppo_pilot", env=train_env, device="cpu")
    assert policy.num_timesteps == 128, policy.num_timesteps

    for budget in (1024, 2048, 4096):
        extra = budget - policy.num_timesteps
        start = perf_counter()
        policy.learn(total_timesteps=extra, reset_num_timesteps=False)
        elapsed = perf_counter() - start

        path = Path("/tmp") / f"hc20dt_ppo_{budget}.zip"
        policy.save(path)
        print(f"Checkpoint {policy.num_timesteps}, additional training {elapsed:.1f} s")
        validate(PPO.load(path, device="cpu"), validation_env)