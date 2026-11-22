"""Compare controllers under fixed windows of missing scan measurements."""

import csv
import sys
from pathlib import Path
from time import monotonic

from stable_baselines3 import PPO

import simulation.inspection_env as inspection
from simulation.gym_env import PipeInspectionGym
from simulation.reference import fully_outside


BANDS = (100, 200, 300)
PPO_REPLAY = (2, 7, 2, 5, 2, 5, 2, 5, 1)
METHODS = ("ppo", "ppo_replay", "coverage_feedback", "ppo_staged_exit")


def depth_counts(env):
    return [sum(depth == band for depth, _ in env.inspection.covered) for band in BANDS]


def feedback_actions(env):
    """Sweep each band until complete or four commands have been spent there."""
    side = 0
    for level in range(1, 4):
        yield 2 * level + side
        for _ in range(3):
            if depth_counts(env)[level - 1] == 8:
                break
            side = 1 - side
            yield 2 * level + side

    for level in (2, 1, 0):
        yield 2 * level + side


def run_episode(policy, env, start, scenario, method):
    offsets = tuple(float(start[key]) for key in ("base_offset_deg", "shoulder_offset_deg"))
    obs, _ = env.reset(options={"offsets_deg": offsets})
    window_start = float(scenario["start_s"])
    window_end = window_start + float(scenario["duration_s"])
    tripped = set()
    actions = []
    scripted = feedback_actions(env) if method == "coverage_feedback" else iter(PPO_REPLAY)
    scan = inspection.scan_target

    def scan_with_dropout(model, data):
        target = scan(model, data)
        # Convert the clock to simulation steps to avoid drift at window boundaries.
        step = round(data.time / model.opt.timestep)
        time_s = step * model.opt.timestep
        if target is not None and window_start <= time_s < window_end:
            tripped.add(target)
            return None
        return target

    inspection.scan_target = scan_with_dropout
    try:
        while True:
            if method in ("coverage_feedback", "ppo_replay"):
                action = next(scripted)
            elif method == "ppo_staged_exit" and len(env.inspection.covered) == 24:
                action = actions[-1] - 2  # One shallower depth, with the same wrist side.
            else:
                action = int(policy.predict(obs, deterministic=True)[0])

            actions.append(action)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    finally:
        inspection.scan_target = scan

    counts = depth_counts(env)
    outcome = info["outcome"]
    return {
        **scenario,
        **start,
        "method": method,
        **{f"sectors_{band}": count for band, count in zip(BANDS, counts)},
        "total_targets": sum(counts),
        "success": outcome == "success",
        "collision": outcome == "collision",
        "fully_outside_at_end": bool(fully_outside(env.inspection.data)),
        "outcome": outcome,
        "sim_seconds": f"{env.inspection.data.time:.3f}",
        "unique_targets_suppressed": len(tripped),
        "actions": " ".join(map(str, actions)),
    }


def main():
    folder = Path("evaluation")
    with (folder / "starts.csv").open(newline="") as stream:
        starts = list(csv.DictReader(stream))
    with (folder / "dropout_scenarios.csv").open(newline="") as stream:
        scenarios = list(csv.DictReader(stream))

    total = len(scenarios) * len(METHODS) * len(starts)
    if total == 0:
        raise ValueError("At least one evaluation start and dropout scenario are required")
    started = monotonic()
    completed = successes = 0
    print(f"Loading policy and scene for {total} episodes...", file=sys.stderr, flush=True)
    policy = PPO.load("models/hc20dt_ppo_4096.zip", device="cpu")
    env = PipeInspectionGym()
    writer = None
    for scenario in scenarios:
        for method in METHODS:
            for start in starts:
                row = run_episode(policy, env, start, scenario, method)
                if writer is None:
                    writer = csv.DictWriter(sys.stdout, fieldnames=row.keys(), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(row)
                sys.stdout.flush()
                completed += 1
                successes += row["success"]
                print(
                    f"[{completed}/{total}] {scenario['scenario']} | {method} | "
                    f"start {start['episode']} | {row['outcome']} | "
                    f"{row['total_targets']}/24 targets | sim {row['sim_seconds']} s",
                    file=sys.stderr, flush=True,
                )

    print(
        f"Finished {completed} episodes in {monotonic() - started:.1f} s wall time "
        f"({successes} successful).", file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
