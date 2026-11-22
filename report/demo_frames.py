"""Draw nine frames from the recorded demo as one figure for the report.

Needs FFmpeg and Pillow. Run from the repository root:

    python -m report.demo_frames
"""

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

VIDEO = Path("demo/inspection.mp4")
OUT = Path("report/figures/fig11_demo_frames.png")
# Video times in seconds. The video plays at twice the simulation speed.
TIMES_S = (0, 3, 6, 9, 12, 16, 20, 25, 30)
COLUMNS = 3
GAP = 8


def read_frame(time_s, folder):
    path = Path(folder) / f"frame_{time_s}.png"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(time_s), "-i", str(VIDEO),
         "-frames:v", "1", str(path)],
        check=True,
    )
    with Image.open(path) as image:
        return image.convert("RGB")


def main():
    with tempfile.TemporaryDirectory() as folder:
        frames = [read_frame(t, folder) for t in TIMES_S]

    width, height = frames[0].size
    rows = -(-len(frames) // COLUMNS)
    grid = Image.new(
        "RGB",
        (COLUMNS * width + (COLUMNS - 1) * GAP, rows * height + (rows - 1) * GAP),
        "white",
    )
    for i, frame in enumerate(frames):
        row, column = divmod(i, COLUMNS)
        grid.paste(frame, (column * (width + GAP), row * (height + GAP)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    grid.save(OUT, optimize=True)
    print(OUT, grid.size)


if __name__ == "__main__":
    main()
