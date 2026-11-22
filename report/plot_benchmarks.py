"""Render figures 6–10 from recorded benchmarks and the pipe cross-section geometry."""

import ast
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


REPORT = Path(__file__).resolve().parent
EVAL = REPORT.parent / "evaluation"
HARDWARE = EVAL / "hardware_seed0_4096"
OUT = REPORT / "figures"
BLUE, ORANGE, INK, PALE = "#2166a7", "#b75321", "#20364a", "#e5edf2"
METHODS = {
    "original_ppo": "PPO · original recipe",
    "ppo": "PPO · continuous",
    "dqn": "DQN",
    "a2c": "A2C",
}
DEVICES = ("cpu", "cuda")
plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.edgecolor": "#b9c7d1",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def frame(title, subtitle, size):
    fig = plt.figure(figsize=size)
    fig.text(0.045, 0.955, title, fontsize=17, weight="bold", va="top")
    fig.text(0.045, 0.875, subtitle, fontsize=10, va="top")
    return fig


def save(fig, name, note):
    fig.text(0.045, 0.035, note, fontsize=9, va="bottom")
    fig.savefig(OUT / name, dpi=200, facecolor="white")
    plt.close(fig)


def hardware(rows):
    keyed = {(r["method"], r["train_device"], r["eval_device"]): r for r in rows}
    fig = frame(
        "Training and deployment time on the same laptop",
        "Intel Core i7-13620H and RTX 3050 Laptop GPU   |   Seed 0, 4,096 training decisions",
        (10.7, 5.0),
    )
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.45, 1]})
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.24, top=0.70, wspace=0.54)
    y = np.arange(len(METHODS))
    for offset, device, color in [(-0.17, "cpu", BLUE), (0.17, "cuda", ORANGE)]:
        vals = [float(keyed[m, device, "cpu"]["train_seconds"]) / 60 for m in METHODS]
        bars = left.barh(
            y + offset, vals, height=0.30, color=color, label=device.upper()
        )
        left.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    left.set(
        yticks=y,
        yticklabels=METHODS.values(),
        xlabel="Training time (min)",
        xlim=(0, 10.4),
    )
    left.invert_yaxis()
    left.legend(
        frameon=False,
        ncols=2,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        borderaxespad=0,
    )
    right.set_title("Saved PPO · identical weights", fontsize=11, loc="left", pad=13)
    paired = [keyed["saved_ppo", "", d] for d in DEVICES]
    assert paired[0]["checkpoint_sha256"] == paired[1]["checkpoint_sha256"]
    bars = right.barh(
        [0, 1],
        [float(r["eval_wall_seconds"]) for r in paired],
        height=0.44,
        color=[BLUE, ORANGE],
    )
    right.bar_label(bars, fmt="%.2f s", padding=4)
    right.set(
        yticks=[0, 1],
        yticklabels=["CPU", "CUDA"],
        xlabel="Wall time for 10 inspections (s)",
        xlim=(0, 25),
    )
    right.invert_yaxis()
    assert all(int(r["successes"]) == 10 for r in paired)
    save(
        fig,
        "fig6_hardware.png",
        "One timing run per condition. Deployment time includes MuJoCo physics on CPU. Both devices completed all ten inspections.",
    )


def outcomes(episodes):
    groups = defaultdict(list)
    for row in episodes:
        if row["eval_device"] == "cpu":
            groups[row["method"], row["train_device"]].append(row)
    ordering = [(m, d) for m in METHODS for d in DEVICES]
    ordering += [(m, "") for m in ("saved_ppo", "coverage_feedback", "ppo_replay")]
    labels = [f"{METHODS[m]}  /  {d.upper()}" for m, d in ordering[:8]]
    labels += ["Saved PPO", "Scripted feedback", "Fixed replay"]
    matrix = []
    for key in ordering:
        rs = groups[key]
        assert len(rs) == 10 and len({r["episode"] for r in rs}) == 10
        counts = [
            np.mean([int(r[f"sectors_{band}"]) for r in rs]) for band in (100, 200, 300)
        ]
        counts.extend(
            sum(r[k] == "True" for r in rs) for k in ("fully_outside_at_end", "success")
        )
        matrix.append(counts)
    fig = frame(
        "Full coverage and successful withdrawal are separate outcomes",
        "Ten starts per row, evaluated on CPU   |   Training device is shown beside each learned controller",
        (10, 6.6),
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=0.255, right=0.96, bottom=0.13, top=0.74)
    denominators = np.array([8, 8, 8, 10, 10])
    scaled = np.array(matrix) / denominators
    ax.imshow(scaled, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(5), ["100 mm", "200 mm", "300 mm", "Withdrawn", "Successful"])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(labels)), labels)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            ax.text(
                j,
                i,
                f"{v:g}/{denominators[j]}",
                ha="center",
                va="center",
                color="white" if scaled[i, j] > 0.56 else INK,
            )
    ax.axhline(7.5, color="white", lw=3)
    ax.axvline(2.5, color="white", lw=3)
    ax.tick_params(length=0, pad=6)
    save(
        fig,
        "fig7_controller_outcomes.png",
        "Depth columns show mean sectors covered. Final columns count episodes. No scan dropout was applied.",
    )


def six_joint():
    lines = (EVAL / "six_joint_starts_results.txt").read_text().splitlines()
    cases = {}
    for line in lines:
        if line.startswith("Offsets"):
            match = re.fullmatch(
                r"Offsets up to ([\d.]+) deg on all six joints: (\{.*\}) \((\d+) valid of (\d+) drawn\)",
                line,
            )
            degree, outcomes, valid, drawn = match.groups()
            counts = ast.literal_eval(outcomes)
            times = []
            cases[float(degree)] = (counts, int(valid), int(drawn), times)
        elif line.strip():
            match = re.match(r"\s*collision at ([\d.]+) s with (\d+) targets", line)
            times.append((float(match[1]), int(match[2])))
    fig = frame(
        "Sensitivity to initial offsets on all six joints",
        "Saved PPO policy   |   20 valid starts per range; each start was fully outside and free of contact",
        (10.5, 5),
    )
    a, b = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.72, wspace=0.45)
    for i, (deg, (counts, valid, drawn, times)) in enumerate(cases.items()):
        n = counts.get("success", 0)
        assert n + len(times) == valid and len(times) == counts.get("collision", 0)
        a.barh(i, valid, color=PALE, height=0.52)
        a.barh(i, n, color=BLUE, height=0.52)
        a.text(
            n - 0.35,
            i,
            f"{n}/{valid}",
            ha="right",
            va="center",
            color="white",
            weight="bold",
        )
        a.text(20.8, i, f"{valid}/{drawn}", va="center", fontsize=9)
        for t, mult in Counter(t for t, _ in times).items():
            b.scatter(
                [1000 * t] * mult,
                i + (np.arange(mult) - (mult - 1) / 2) * 0.17,
                s=52,
                color=ORANGE,
                zorder=3,
            )
        assert all(targets == 0 for _, targets in times)
        if not times:
            b.text(100, i, "No collision", ha="center", va="center", color="#52677a")
    ys = np.arange(len(cases))
    for ax in [a, b]:
        ax.set_yticks(ys, [f"±{deg:g}°" for deg in cases])
        ax.set_ylim(len(cases) - 0.5, -0.5)
    a.set_title("Successful inspections", loc="left", fontsize=12)
    a.text(20.8, -0.53, "Valid/drawn", fontsize=8, va="bottom")
    a.set(xlim=(0, 26), xticks=[0, 5, 10, 15, 20], xlabel="Episodes")
    b.set_title("Time of first collision", loc="left", fontsize=12)
    b.set(
        xlim=(75, 175), xticks=[80, 100, 120, 140, 160], xlabel="Simulation time (ms)"
    )
    failures = sum(len(times) for _, _, _, times in cases.values())
    save(
        fig,
        "fig8_six_joint_starts.png",
        f"Each dot is one failed inspection. All {failures} collisions occurred before the first target was recorded.",
    )


def geometry():
    # Dimensions in mm from the supplied pipe and scanner. The boxes are tangent at radius R.
    radius, half, segments = 175, 50, 32
    circle_z = np.sqrt(radius**2 - half**2) - half
    angles = np.arange(segments) * 2 * np.pi / segments
    upper = angles[np.cos(angles) > 1e-8]
    facet_z = min((radius - half * np.abs(np.sin(upper))) / np.cos(upper)) - half
    fig = frame(
        "How high can the scanner rise before its upper corner touches?",
        "Pipe cross-section, centred scanner, fixed orientation   |   Dimensions in millimetres",
        (10.2, 5.7),
    )
    ax, formula = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.05, 1]})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.79, wspace=0.18)
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), radius, fill=False, color=BLUE, lw=2))
    ax.add_patch(
        Rectangle(
            (-half, circle_z - half),
            2 * half,
            2 * half,
            facecolor="#ffe6c8",
            edgecolor=ORANGE,
            lw=1.6,
        )
    )
    corner = (half, circle_z + half)
    ax.plot([0, half, half], [0, 0, corner[1]], "--", color="#8598a5")
    ax.plot([0, half], [0, corner[1]], color=BLUE, lw=1.6)
    ax.scatter(
        [0, 0, half],
        [0, circle_z, corner[1]],
        s=[15, 18, 35],
        color=[INK, ORANGE, ORANGE],
        zorder=4,
    )
    ax.annotate(
        "Upper corner",
        corner,
        (80, 173),
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=INK),
    )
    ax.text(half / 2, -13, "50", ha="center", fontsize=9)
    ax.text(half + 5, 35, "z + 50", fontsize=9)
    ax.text(
        1,
        47,
        "R = 175",
        rotation=np.degrees(np.arctan2(corner[1], half)),
        fontsize=9,
        color=BLUE,
    )
    ax.text(-46, 91, "Scanner", fontsize=9)
    ax.set(
        xlim=(-190, 195),
        ylim=(-40, 200),
        xlabel="Lateral position y (mm)",
        ylabel="Height z (mm)",
    )
    formula.axis("off")
    lines = [
        (0.94, "Circular opening", "bold", 12),
        (0.76, r"$50^2+(z+50)^2=175^2$", "normal", 15),
        (0.60, f"Centre height at contact     {circle_z:.2f} mm", "bold", 11),
        (0.40, "32 tangent faces", "bold", 12),
        (0.24, f"Geometric contact height    {facet_z:.2f} mm", "normal", 11),
        (0.08, "MuJoCo check     118 mm clear · 119 mm hit", "normal", 10),
    ]
    for y, t, w, size in lines:
        formula.text(0, y, t, transform=formula.transAxes, weight=w, fontsize=size)
    save(
        fig,
        "fig9_corner_clearance.png",
        "The rectangle is drawn at first contact with the ideal circle. The two MuJoCo poses bracket contact in the segmented model.",
    )


def seeds():
    rows = read_csv(REPORT / "data/seed_results.csv")
    by_key = {(int(r["seed"]), int(r["decisions"])): r for r in rows}
    ids = sorted({k[0] for k in by_key})
    budgets = sorted({k[1] for k in by_key})
    assert len(rows) == len(by_key) == len(ids) * len(budgets)
    passed = np.array(
        [
            [
                by_key[s, b]["validation_success"] == "5/5"
                and by_key[s, b]["official_success"] == "10/10"
                for b in budgets
            ]
            for s in ids
        ]
    )
    fig = frame(
        "Sensitivity of PPO training to the random seed",
        "Five independently initialised runs   |   Continuous training, checked at five decision budgets",
        (8.5, 4.7),
    )
    ax = fig.subplots()
    fig.subplots_adjust(left=0.12, right=0.95, top=0.72, bottom=0.24)
    ax.imshow(
        passed,
        cmap=matplotlib.colors.ListedColormap([PALE, BLUE]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for i, j in np.ndindex(passed.shape):
        ax.text(
            j,
            i,
            "Pass" if passed[i, j] else "Fail",
            va="center",
            ha="center",
            color="white" if passed[i, j] else INK,
        )
    ax.set_xticks(range(len(budgets)), [f"{b:,}" for b in budgets])
    ax.set_yticks(range(len(ids)), ids)
    ax.set(xlabel="Training decisions", ylabel="Seed")
    ax.tick_params(length=0, pad=5)
    save(
        fig,
        "fig10_training_seeds.png",
        "Pass requires success on all five validation starts and all ten official starts at that checkpoint.",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    hardware(read_csv(HARDWARE / "summary.csv"))
    episodes = [r for f in sorted(HARDWARE.glob("*_episodes.csv")) for r in read_csv(f)]
    outcomes(episodes)
    six_joint()
    geometry()
    seeds()
