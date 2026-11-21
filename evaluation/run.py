import csv
from collections import Counter
from pathlib import Path

from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym
from simulation.reference import fully_outside


def evaluate_start(policy, env, start):
    offsets = (float(start["base_offset_deg"]), float(start["shoulder_offset_deg"]))
    obs, _ = env.reset(options={"offsets_deg": offsets})
    decisions = 0

    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        decisions += 1
        if terminated or truncated:
            break

    covered = env.inspection.covered
    counts = [sum(b == depth for b, _ in covered) for depth in (100, 200, 300)]
    outcome = info["outcome"]
    withdrawn = bool(fully_outside(env.inspection.data))

    return {
        **start,
        "sectors_100": counts[0],
        "sectors_200": counts[1],
        "sectors_300": counts[2],
        "total_targets": sum(counts),
        "collision": outcome == "collision",
        "fully_outside_at_end": withdrawn,
        "decisions": decisions,
        "outcome": outcome,
        "success": outcome == "success",
    }


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    policy = PPO.load(folder.parent / "models/hc20dt_ppo_4096.zip", device="cpu")
    env = PipeInspectionGym()

    with (folder / "starts.csv").open(newline="") as file:
        starts = list(csv.DictReader(file))
    assert len(starts) == 10

    rows = [evaluate_start(policy, env, start) for start in starts]

    with (folder / "results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    successes = sum(row["success"] for row in rows)
    print(f"Success: {successes}/{len(rows)} ({successes / len(rows):.0%})")
    print("Outcomes:", dict(Counter(row["outcome"] for row in rows)))