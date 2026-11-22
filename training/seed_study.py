"""Train PPO from scratch for several seeds, without the pilot restart.

Each seed trains continuously and is checked at fixed decision budgets on the
five validation starts and the ten official starts. The columns match
report/data/seed_results.csv, which holds the data behind Figure 3.
"""

import argparse
import csv
from pathlib import Path

from stable_baselines3 import PPO

from evaluation.run import evaluate_start
from simulation.gym_env import PipeInspectionGym
from training.pilot import VALIDATION_STARTS, learn_timed

ROOT = Path(__file__).resolve().parent.parent
BUDGETS = (128, 1024, 2048, 4096, 8192)
FIELDS = (
    "seed", "decisions", "training_s",
    "validation_success", "validation_min_targets",
    "official_success", "official_min_targets",
)


def check(policy, env, starts):
    rows = [evaluate_start(policy, env, start) for start in starts]
    passed = sum(row["success"] for row in rows)
    return f"{passed}/{len(rows)}", min(row["total_targets"] for row in rows)


def run_seed(seed, budgets, validation, official, device):
    train_env = PipeInspectionGym()
    check_env = PipeInspectionGym()
    policy = PPO(
        "MlpPolicy", train_env,
        n_steps=64, batch_size=32, gamma=1.0,
        device=device, seed=seed, verbose=0,
    )

    total = 0.0
    for budget in budgets:
        first = policy.num_timesteps == 0
        total += learn_timed(policy, budget - policy.num_timesteps, reset=first)
        val_passed, val_min = check(policy, check_env, validation)
        off_passed, off_min = check(policy, check_env, official)
        yield {
            "seed": seed,
            "decisions": budget,
            "training_s": round(total, 1),
            "validation_success": val_passed,
            "validation_min_targets": val_min,
            "official_success": off_passed,
            "official_min_targets": off_min,
        }
    train_env.close()
    check_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "report/data/seed_results_rerun.csv")
    args = parser.parse_args()

    validation = [
        {"base_offset_deg": base, "shoulder_offset_deg": shoulder}
        for base, shoulder in VALIDATION_STARTS
    ]
    with (ROOT / "evaluation/starts.csv").open(newline="") as file:
        official = list(csv.DictReader(file))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for seed in args.seeds:
            for row in run_seed(seed, args.budgets, validation, official, args.device):
                writer.writerow(row)
                file.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()