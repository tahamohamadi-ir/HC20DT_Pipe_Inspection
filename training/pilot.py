from time import perf_counter

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


if __name__ == "__main__":
    train_env = PipeInspectionGym()
    validation_env = PipeInspectionGym()
    policy = PPO(
        "MlpPolicy",
        train_env,
        n_steps=64,
        batch_size=32,
        gamma=1.0,
        device="cpu",
        seed=0,
        verbose=1,
    )

    print("Before training")
    validate(policy, validation_env)

    start = perf_counter()
    policy.learn(total_timesteps=128)
    print(f"Training time: {perf_counter() - start:.1f} s")

    policy.save("/tmp/hc20dt_ppo_pilot")
    saved_policy = PPO.load("/tmp/hc20dt_ppo_pilot", device="cpu")
    print("After training, reloaded from disk")
    validate(saved_policy, validation_env)