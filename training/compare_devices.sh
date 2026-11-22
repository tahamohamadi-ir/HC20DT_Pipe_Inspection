#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p "$HOME/hc20dt_runs"
run_dir=$(mktemp -d "$HOME/hc20dt_runs/hardware-XXXXXX")
echo "Results and logs: $run_dir"
exec > >(tee "$run_dir/run.log") 2>&1
python - <<'PY'
import platform
import torch

print(f"Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA build {torch.version.cuda}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable. Check PyTorch and the WSL GPU driver before starting.")
print("GPU:", torch.cuda.get_device_name(0))
PY
lscpu | grep -E 'Model name|^CPU\(s\):' || true
git rev-parse HEAD

python -u -m training.compare_controllers \
    --methods saved_ppo coverage_feedback ppo_replay --eval-devices cpu cuda --out "$run_dir/reference"
for device in cpu cuda; do
    python -u -m training.compare_controllers \
        --methods original_ppo ppo dqn a2c --steps 4096 --seed 0 \
        --device "$device" --eval-devices cpu cuda --out "$run_dir/$device"
done

python - "$run_dir" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
with (root / "summary.csv").open("x", newline="") as dest:
    writer = None
    for block in ("reference", "cpu", "cuda"):
        with (root / block / "summary.csv").open(newline="") as source:
            rows = csv.DictReader(source)
            if writer is None:
                writer = csv.DictWriter(dest, rows.fieldnames, lineterminator="\n")
                writer.writeheader()
            writer.writerows(rows)
print("Completed:", root / "summary.csv")
PY
