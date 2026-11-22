"""Compare inspection controllers on the ten fixed evaluation starts."""
import argparse
import csv
import hashlib
import sys
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from evaluation.dropout import run_episode
from simulation.gym_env import PipeInspectionGym
from training.checkpoints import run_checkpoints
from training.pilot import learn_timed, run_pilot


ROOT = Path(__file__).resolve().parents[1]
LEARNERS = {"ppo": PPO, "dqn": DQN, "a2c": A2C}
BASELINES = ("coverage_feedback", "ppo_replay")
METHODS = (*LEARNERS, "original_ppo", "saved_ppo", *BASELINES)


def log(message):
    print(message, file=sys.stderr, flush=True)


class Progress(BaseCallback):
    def __init__(self, label, budget):
        super().__init__()
        self.label = label
        self.budget = budget

    def _on_training_start(self):
        self.started = perf_counter()
        self.initial = self.model.num_timesteps

    def _on_step(self):
        count = self.num_timesteps
        if count == 1 or count % 128 == 0:
            elapsed = perf_counter() - self.started
            completed = count - self.initial
            eta = f"{elapsed * (self.budget - count) / completed:.0f}s" if completed >= 128 else "..."
            log(f"{self.label} train {count}/{self.budget} ({100*count/self.budget:.0f}%) "
                f"elapsed {elapsed:.0f}s | estimated remaining {eta}")
        return True


def train(method, args, folder):
    common = dict(seed=args.seed, device=args.device, gamma=1.0, verbose=0)
    # Refresh DQN's target network within the short inspection training budget.
    settings = {
        "ppo": dict(n_steps=64, batch_size=32),
        "a2c": dict(n_steps=64),
        "dqn": dict(batch_size=32, learning_starts=100, buffer_size=args.steps,
                    train_freq=4, target_update_interval=512),
    }
    env = PipeInspectionGym()
    try:
        model = LEARNERS[method]("MlpPolicy", env, **common, **settings[method])
        log(f"{method} seed={args.seed} device={model.device} settings={settings[method]}")
        seconds = learn_timed(model, args.steps, Progress(method, args.steps))
        assert model.num_timesteps == args.steps
        path = folder / f"{method}.zip"
        model.save(path)
        log(f"{method} training finished in {seconds:.1f}s")
        return path, seconds
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=[*LEARNERS, *BASELINES])
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-devices", nargs="+", choices=["cpu", "cuda"],
                        help="Evaluate identical weights on these devices (default: training device)")
    parser.add_argument("--out", type=Path, required=True, help="A new results directory")
    args = parser.parse_args()
    if args.steps < 128 or args.steps % 64:
        parser.error("--steps must be at least 128 and divisible by 64")
    if len(set(args.methods)) != len(args.methods):
        parser.error("Choose each method once")
    if "original_ppo" in args.methods and args.steps != 4096:
        parser.error("The original PPO recipe uses exactly 4096 decisions")
    eval_devices = args.eval_devices or [args.device]
    if len(set(eval_devices)) != len(eval_devices):
        parser.error("Choose each evaluation device once")
    trains = set(args.methods) - {*BASELINES, "saved_ppo"}
    if (args.device == "cuda" and trains) or ("cuda" in eval_devices and set(args.methods) - set(BASELINES)):
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable in this Python environment. Use --device cpu.")
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.set_num_threads(1)
    log(f"PyTorch {torch.__version__} | CUDA build {torch.version.cuda} | torch CPU threads=1")

    with (ROOT / "evaluation/starts.csv").open(newline="") as file:
        starts = list(csv.DictReader(file))
    if len(starts) != 10:
        raise ValueError(f"Expected 10 evaluation starts, found {len(starts)}")
    if args.out.exists():
        parser.error(f"Choose a new output directory: {args.out}")
    args.out.mkdir(parents=True)
    scenario = {"scenario": "no_dropout", "start_s": 0, "duration_s": 0}
    log(f"Results: {args.out.resolve()} | methods: {', '.join(args.methods)}")
    with (args.out / "episodes.csv").open("x", newline="") as file, (args.out / "summary.csv").open("x", newline="") as summary:
        writer = None
        totals = None
        for method in args.methods:
            log(f"Starting {method}")
            path, seconds = None, 0.0
            if method == "original_ppo":
                callback = Progress(method, args.steps)
                pilot_seconds = run_pilot(args.device, args.out, args.seed, callback)
                path, checkpoint_seconds = run_checkpoints(args.device, args.out, callback)
                seconds = pilot_seconds + checkpoint_seconds
            elif method == "saved_ppo":
                path = ROOT / "models/hc20dt_ppo_4096.zip"
            elif method in LEARNERS:
                path, seconds = train(method, args, args.out)
            metadata = dict(method=method, seed=args.seed if method in trains else "",
                            train_device=args.device if method in trains else "",
                            train_steps=args.steps if method in trains else 0,
                            train_seconds=round(seconds, 3),
                            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path else "")
            for device in (eval_devices if path else ["cpu"]):
                model = LEARNERS.get(method, PPO).load(path, device=device) if path else None
                env = PipeInspectionGym()
                successes = collisions = full = 0
                eval_seconds = 0.0
                try:
                    if model is not None:
                        obs, _ = env.reset(options={"offsets_deg": (0.0, 0.0)})
                        model.predict(obs, deterministic=True)
                        if device == "cuda":
                            torch.cuda.synchronize(model.device)
                    for start in starts:
                        began = perf_counter()
                        result = run_episode(model, env, start, scenario, "ppo" if path else method)
                        if device == "cuda":
                            torch.cuda.synchronize(model.device)
                        elapsed = perf_counter() - began
                        eval_seconds += elapsed
                        result.update(metadata, eval_device=device, episode_wall_seconds=round(elapsed, 4))
                        if writer is None:
                            writer = csv.DictWriter(file, fieldnames=list(result), lineterminator="\n")
                            writer.writeheader()
                        writer.writerow(result)
                        file.flush()
                        successes += result["success"]
                        collisions += result["collision"]
                        full += result["total_targets"] == 24
                        log(f"{method} [{device}] eval {start['episode']}/10 | {result['outcome']} | "
                            f"{result['total_targets']}/24 | sim {result['sim_seconds']}s | wall {elapsed:.2f}s")
                finally:
                    env.close()
                row = dict(metadata, eval_device=device, successes=successes, full_coverage=full,
                           collisions=collisions, eval_wall_seconds=round(eval_seconds, 4))
                if totals is None:
                    totals = csv.DictWriter(summary, fieldnames=list(row), lineterminator="\n")
                    totals.writeheader()
                totals.writerow(row)
                summary.flush()
                log(f"{method} [{device}] summary: success {successes}/10 | full coverage {full}/10 | "
                    f"collisions {collisions}/10 | train {seconds:.1f}s | eval {eval_seconds:.1f}s")


if __name__ == "__main__":
    main()
