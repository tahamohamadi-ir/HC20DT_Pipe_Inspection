from pathlib import Path
from time import monotonic

import cv2
import mujoco
from stable_baselines3 import PPO

from demo.record_demo import draw_frame
from simulation.gym_env import PipeInspectionGym


def main():
    env = PipeInspectionGym()
    obs, _ = env.reset(options={"offsets_deg": (0.0, 0.0)})
    root = Path.cwd()
    policy = PPO.load(root / "models/hc20dt_ppo_4096.zip", device="cpu")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.96, 0.0, 1.12)
    camera.distance, camera.azimuth, camera.elevation = 2.1, 73, -16
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    options.sitegroup[:] = 0
    name = "HC20DT | Live inspection"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, 960, 720)
    try:
        with mujoco.Renderer(env.inspection.model, height=480, width=640) as renderer:
            last = float("-inf")
            start = monotonic()

            def show(label="Inspecting", force=False):
                nonlocal last
                t = env.inspection.data.time
                if not force and t - last < 0.04 - 1e-9:
                    return
                last = t
                image = cv2.cvtColor(draw_frame(renderer, env, camera, options), cv2.COLOR_RGB2BGR)
                counts = [sum(b == d for b, _ in env.inspection.covered) for d in (100, 200, 300)]
                cv2.rectangle(image, (0, 0), (639, 55), (32, 28, 22), -1)
                cv2.putText(image, f"{label}  |  {t:.1f} s", (12, 21),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                text = "     ".join(f"{d} mm: {n}/8" for d, n in zip((100, 200, 300), counts))
                cv2.putText(image, text, (12, 45), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (240, 235, 220), 1, cv2.LINE_AA)
                cv2.imshow(name, image)
                delay_ms = max(1, round(1000 * (start + t - monotonic())))
                if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
                if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                    raise KeyboardInterrupt

            show("Start", True)
            env.inspection.on_physics_step = show
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            print(info["outcome"], len(env.inspection.covered), flush=True)
            show(info["outcome"] + " | press any key to close", True)
            while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1:
                if cv2.waitKey(50) != -1:
                    break
    except KeyboardInterrupt:
        print("Live view closed.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
