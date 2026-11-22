"""Generate the technical report figures from the saved policy and committed results."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from stable_baselines3 import PPO

from demo.record_demo import draw_frame
from evaluation.dropout import run_episode
from simulation.gym_env import PipeInspectionGym
from simulation.scene import ENTRANCE

OUT = Path("report/figures")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
MISSED = "#f0efec"
SERIES = {"ppo": "#2a78d6", "coverage_feedback": "#eb6834", "ppo_staged_exit": "#1baf7a"}
LABELS = {
    "ppo": "PPO (saved policy)",
    "ppo_replay": "Fixed replay of PPO commands",
    "coverage_feedback": "Scripted, coverage feedback",
    "ppo_staged_exit": "PPO + staged exit rule",
}
BANDS = (100, 200, 300)
BAND_COLORS = ("#86b6ef", "#2a78d6", "#104281")  # ordinal blue: shallow to deep

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})

# Values printed by training/pilot.py and training/checkpoints.py (README, training time table).
CHECKPOINTS = {
    128: ((4, 1, 0), 11.0),
    1024: ((4, 0, 0), 84.7),
    2048: ((5, 2, 8), 104.7),
    4096: ((8, 8, 8), 260.6),
}


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def traced_episode(policy, env, start, scenario, method):
    """Run one episode and sample time, emitter depth and coverage every 50 ms."""
    trace = []
    sim = env.inspection

    def sample():
        t = sim.data.time
        if trace and t - trace[-1][0] < 0.05 - 1e-9:
            return
        depth = 1000 * (sim.data.site("scan_origin").xpos[0] - ENTRANCE[0])
        trace.append((t, depth, len(sim.covered)))

    sim.on_physics_step = sample
    try:
        row = run_episode(policy, env, start, scenario, method)
    finally:
        sim.on_physics_step = None
    return row, np.array(trace), set(sim.covered)


def label_line_end(ax, x, y, text, color, dy=0):
    ax.plot(x, y, "o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points",
                va="center", fontsize=9, color=INK)


def fig_scene(env, policy, nominal):
    """Two cutaway views: start pose outside the pipe, and scanning at 300 mm."""
    sim = env.inspection
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (1.20, 0, 1.17)
    camera.distance, camera.azimuth, camera.elevation = 1.05, 70, -14
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    options.sitegroup[:] = 0
    frames = {}
    with mujoco.Renderer(sim.model, height=480, width=640) as renderer:
        def grab():
            depth = 1000 * (sim.data.site("scan_origin").xpos[0] - ENTRANCE[0])
            if "start" not in frames:
                frames["start"] = draw_frame(renderer, env, camera, options)
            if "deep" not in frames and depth > 295 and len(sim.covered) >= 3:
                frames["deep"] = draw_frame(renderer, env, camera, options)

        sim.on_physics_step = grab
        obs, _ = env.reset(options={"offsets_deg": (0.0, 0.0)})
        grab()
        while True:
            obs, _, done, _, _ = env.step(int(policy.predict(obs, deterministic=True)[0]))
            if done:
                break
        sim.on_physics_step = None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    titles = ("(a) Start: scanner fully outside the entrance",
              "(b) Scanning at 300 mm; green line is the ray to the wall")
    for ax, key, title in zip(axes, ("start", "deep"), titles):
        ax.imshow(frames[key])
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis("off")
    fig.text(0.01, 0.02, "Cutaway: the near half of the pipe (blue) and entrance collar (gold) is hidden. "
             "Scanner in orange.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT / "fig1_scene.png", dpi=200)
    plt.close(fig)


def fig_training():
    """Coverage per depth band on the nominal validation start at each checkpoint."""
    budgets = list(CHECKPOINTS)
    x = np.arange(len(budgets))
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (band, color) in enumerate(zip(BANDS, BAND_COLORS)):
        values = [CHECKPOINTS[b][0][i] for b in budgets]
        ax.bar(x + (i - 1) * width, values, width, color=color,
               edgecolor=SURFACE, linewidth=2, label=f"{band} mm", zorder=3)
    cumulative = np.cumsum([CHECKPOINTS[b][1] for b in budgets])
    ax.set_xticks(x, [f"{b:,} decisions\n{c:.0f} s total" for b, c in zip(budgets, cumulative)])
    ax.set_ylim(0, 9.2)
    ax.set_yticks(range(0, 9, 2))
    ax.set_ylabel("Sectors covered (of 8)")
    ax.axhline(6, color=INK_2, lw=1, ls="--", zorder=2)
    ax.text(1.5, 6.1, "success needs at least 6 per band", va="bottom", ha="center",
            fontsize=8, color=INK_2)
    ax.grid(axis="x", visible=False)
    ax.legend(title="Depth band", frameon=False, ncol=3, loc="upper left")
    ax.set_title("Coverage per depth band at each training checkpoint (validation starts)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_training.png", dpi=200)
    plt.close(fig)


def fig_timeline(traces):
    """Emitter depth and cumulative coverage against simulation time."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1]})
    # PPO is drawn last: the staged-exit run is identical to it until 24/24.
    for z, method in enumerate(("ppo_staged_exit", "coverage_feedback", "ppo")):
        row, trace, _ = traces[method]
        color = SERIES[method]
        top.plot(trace[:, 0], trace[:, 1], color=color, lw=2, label=LABELS[method], zorder=3 + z)
        bottom.plot(trace[:, 0], trace[:, 2], color=color, lw=2, label=LABELS[method],
                    drawstyle="steps-post", zorder=3 + z)
        full = trace[trace[:, 2] >= 24]
        if len(full):
            t_full = full[0, 0]
            bottom.plot(t_full, 24, "|", ms=12, mew=2, color=color)
        t_end = trace[-1, 0]
        n_dec = len(row["actions"].split())
        label_line_end(bottom, t_end, trace[-1, 2],
                       f"{t_end:.1f} s, {n_dec} decisions", color,
                       dy={"ppo": 0, "coverage_feedback": -11, "ppo_staged_exit": 11}[method])

    x_max = max(t[1][-1, 0] for t in traces.values()) * 1.22
    for band in BANDS:
        top.axhspan(band - 10, band + 10, color=GRID, alpha=0.7, lw=0, zorder=0)
        top.text(x_max - 0.3, band, f"{band} mm band", va="center", ha="right",
                 fontsize=8, color=INK_2)
    top.axhline(0, color=INK_2, lw=1)
    top.text(x_max - 0.3, -12, "entrance", va="top", ha="right", fontsize=8, color=INK_2)
    top.set_ylabel("Emitter depth past\nthe entrance (mm)")
    top.set_ylim(-80, 330)
    top.set_title("(a) Insertion depth (staged exit follows PPO until 24/24)")
    handles, labels = top.get_legend_handles_labels()
    order = [labels.index(LABELS[m]) for m in ("ppo", "coverage_feedback", "ppo_staged_exit")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False,
               ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=9)

    bottom.set_ylabel("Targets covered (of 24)")
    bottom.set_xlabel("Simulation time (s)")
    bottom.set_ylim(0, 26.5)
    bottom.set_yticks([0, 8, 16, 20, 24])
    bottom.axhline(20, color=INK_2, lw=1, ls="--")
    bottom.text(0.3, 20.4, "success needs at least 20", fontsize=8, color=INK_2)
    bottom.set_title("(b) Coverage; tick marks show when 24/24 was reached")
    bottom.set_xlim(0, x_max)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig3_timeline.png", dpi=200)
    plt.close(fig)


def fig_dropout(rows):
    """Total targets per scenario and method; incomplete episodes are outlined."""
    scenarios = list(dict.fromkeys(r["scenario"] for r in rows))
    methods = ["ppo", "ppo_replay", "coverage_feedback", "ppo_staged_exit"]
    total = np.zeros((len(methods), len(scenarios)))
    success = np.zeros_like(total)
    for i, m in enumerate(methods):
        for j, s in enumerate(scenarios):
            cell = [r for r in rows if r["method"] == m and r["scenario"] == s]
            total[i, j] = np.mean([int(r["total_targets"]) for r in cell])
            success[i, j] = sum(r["success"] == "True" for r in cell)

    ramp = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]  # 21, 22, 23, 24 targets
    cmap = matplotlib.colors.ListedColormap(ramp)
    norm = matplotlib.colors.BoundaryNorm([20.5, 21.5, 22.5, 23.5, 24.5], cmap.N)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(total, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(scenarios)), [s.replace("_", "\n") for s in scenarios])
    ax.set_yticks(range(len(methods)), [LABELS[m] for m in methods])
    ax.set_xticks(np.arange(-0.5, len(scenarios)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, lw=3)
    ax.tick_params(which="both", length=0)
    for i in range(len(methods)):
        for j in range(len(scenarios)):
            value = total[i, j]
            ok = success[i, j]
            text = f"{value:.0f}/24" + ("" if ok == 10 else f"\n{ok:.0f}/10 success")
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5,
                    color="white" if value >= 23.5 else INK,
                    fontweight="bold" if ok < 10 else "normal")
            if ok < 10:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False,
                                           ec="#d03b3b", lw=2))
    ax.set_title("Targets covered under scan dropout (same result on all 10 starts in each cell)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_dropout.png", dpi=200)
    plt.close(fig)


def fig_coverage_maps(maps):
    """Which of the 24 targets each method covered in the early 3 s dropout."""
    fig, axes = plt.subplots(1, len(maps), figsize=(11, 2.6))
    for ax, (method, (row, covered)) in zip(axes, maps.items()):
        for r, band in enumerate(BANDS):
            for s in range(1, 9):
                hit = (band, s) in covered
                ax.add_patch(plt.Rectangle((s - 1, r), 0.92, 0.9,
                                           color="#2a78d6" if hit else MISSED))
                if not hit:
                    ax.text(s - 0.54, r + 0.45, "✕", ha="center", va="center",
                            color="#d03b3b", fontsize=11, fontweight="bold")
        for r, band in enumerate(BANDS):
            n = sum(b == band for b, _ in covered)
            ax.text(8.15, r + 0.45, f"{n}/8", va="center", fontsize=9,
                    color="#d03b3b" if n < 6 else INK_2, fontweight="bold" if n < 6 else "normal")
        ax.set_xlim(-0.1, 8.9)
        ax.set_ylim(3, -0.1)
        ax.set_xticks(np.arange(8) + 0.46, range(1, 9))
        ax.set_yticks(np.arange(3) + 0.45, [f"{b} mm" for b in BANDS])
        ax.grid(False)
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_xlabel("Sector")
        ax.set_title(f"{LABELS[method]}\n{len(covered)}/24, {row['outcome']}", fontsize=10)
    fig.suptitle("Early 3 s dropout (4–7 s): covered targets in blue, missed targets marked ✕",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_coverage_maps.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    policy = PPO.load("models/hc20dt_ppo_4096.zip", device="cpu")
    env = PipeInspectionGym()
    nominal = read_csv("evaluation/starts.csv")[0]
    scenarios = {s["scenario"]: s for s in read_csv("evaluation/dropout_scenarios.csv")}

    fig_scene(env, policy, nominal)
    print("fig1_scene.png", flush=True)
    fig_training()
    print("fig2_training.png", flush=True)

    traces = {m: traced_episode(policy, env, nominal, scenarios["no_dropout"], m)
              for m in ("ppo", "coverage_feedback", "ppo_staged_exit")}
    fig_timeline(traces)
    print("fig3_timeline.png", flush=True)

    fig_dropout(read_csv("evaluation/dropout_results.csv"))
    print("fig4_dropout.png", flush=True)

    maps = {}
    for method in ("ppo", "ppo_replay", "coverage_feedback"):
        row, _, covered = traced_episode(policy, env, nominal, scenarios["early_3s"], method)
        maps[method] = (row, covered)
    fig_coverage_maps(maps)
    print("fig5_coverage_maps.png", flush=True)


if __name__ == "__main__":
    main()
