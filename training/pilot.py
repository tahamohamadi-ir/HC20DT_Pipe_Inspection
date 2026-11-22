from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym


VALIDATION_STARTS = [
    (0.0, 0.0),
    (-0.5, -0.5),
    (-0.5, 0.5),
    (0.5, -0.5),
    (0.5, 0.5),
]


def validate(policy, env):
    for offsets in VALIDATION_STARTS:
        obs, _ = env.reset(options={"offsets_deg": offsets})
        while True:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        covered = env.inspection.covered
        counts = [
            sum(band == depth for band, _ in covered)
            for depth in (100, 200, 300)
        ]
        print(f"start={offsets} coverage={counts} outcome={info['outcome']}")


def learn_timed(policy, steps, callback=None, reset=True):
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    start = perf_counter()
    policy.learn(total_timesteps=steps, reset_num_timesteps=reset, callback=callback)
    if policy.device.type == "cuda":
        torch.cuda.synchronize(policy.device)
    return perf_counter() - start


def run_pilot(device="cpu", folder=Path("/tmp"), seed=0, callback=None):
    train_env = PipeInspectionGym()
    validation_env = PipeInspectionGym()
    policy = PPO(
        "MlpPolicy",
        train_env,
        n_steps=64,
        batch_size=32,
        gamma=1.0,
        device=device,
        seed=seed,
        verbose=1,
    )

    print("Before training")
    validate(policy, validation_env)

    elapsed = learn_timed(policy, 128, callback)
    print(f"Training time: {elapsed:.1f} s")

    path = folder / "hc20dt_ppo_pilot.zip"
    policy.save(path)
    saved_policy = PPO.load(path, device=device)
    print("After training, reloaded from disk")
    validate(saved_policy, validation_env)
    train_env.close()
    validation_env.close()
    return elapsed


if __name__ == "__main__":
    run_pilot()
