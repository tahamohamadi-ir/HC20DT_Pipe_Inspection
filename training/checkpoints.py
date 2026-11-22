from pathlib import Path

from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym
from training.pilot import learn_timed, validate


def run_checkpoints(device="cpu", folder=Path("/tmp"), callback=None):
    train_env = PipeInspectionGym()
    validation_env = PipeInspectionGym()
    policy = PPO.load(folder / "hc20dt_ppo_pilot.zip", env=train_env, device=device)
    assert policy.num_timesteps == 128, policy.num_timesteps

    total = 0.0
    for budget in (1024, 2048, 4096):
        extra = budget - policy.num_timesteps
        elapsed = learn_timed(policy, extra, callback, reset=False)
        total += elapsed

        path = folder / f"hc20dt_ppo_{budget}.zip"
        policy.save(path)
        print(f"Checkpoint {policy.num_timesteps}, additional training {elapsed:.1f} s")
        validate(PPO.load(path, device=device), validation_env)
    train_env.close()
    validation_env.close()
    return path, total


if __name__ == "__main__":
    run_checkpoints()
