from pathlib import Path
import shutil
import subprocess

import cv2
import mujoco
import numpy as np
from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym
from simulation.coverage import RAY_GROUPS


def draw_frame(renderer, env, camera, options):
    sim = env.inspection
    model, data = sim.model, sim.data
    renderer.update_scene(data, camera=camera, scene_option=options)
    scene = renderer.scene
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        name = model.geom(geom.objid).name
        if name.startswith(("pipe_", "rim_")):
            if model.geom_pos[geom.objid, 1] < -1e-8:
                geom.rgba[3] = 0
                geom.transparent = 1
            else:
                geom.rgba[:] = (0.32, 0.62, 0.74, 1) if name.startswith("pipe_") else (0.85, 0.68, 0.32, 1)
        elif name == "scanner_collision":
            geom.rgba[:] = (0.95, 0.39, 0.13, 1)

    site = data.site("scan_origin")
    direction = site.xmat.reshape(3, 3)[:, 2].copy()
    hit = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(model, data, site.xpos, direction, RAY_GROUPS,
                             True, model.body("scanner").id, hit, None)
    if distance >= 0 and scene.ngeom < scene.maxgeom:
        ray = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(ray, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).ravel(), np.array([0.3, 1, 0.3, 1], np.float32))
        mujoco.mjv_connector(ray, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003,
                             site.xpos, site.xpos + distance * direction)
        scene.ngeom += 1

    return renderer.render()


def main():
    encoder = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    if encoder is None:
        raise SystemExit("Install ffmpeg to encode the demo.")
    env = PipeInspectionGym()
    obs, _ = env.reset(options={"offsets_deg": (0.0, 0.0)})
    policy = PPO.load("models/hc20dt_ppo_4096.zip", device="cpu")
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.96, 0, 1.12)
    camera.distance, camera.azimuth, camera.elevation = 2.1, 73, -16
    options = mujoco.MjvOption()
    options.geomgroup[3] = 1
    options.sitegroup[:] = 0
    Path("demo").mkdir(exist_ok=True)
    command = [encoder, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "640x480", "-r", "25", "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", "demo/inspection.mp4"]
    with mujoco.Renderer(env.inspection.model, height=480, width=640) as renderer:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as video:
            last = float("-inf")
            def frame(label="Inspecting", force=False):
                nonlocal last
                t = env.inspection.data.time
                if not force and t - last < 0.08 - 1e-9:
                    return
                last = t
                image = draw_frame(renderer, env, camera, options)
                counts = [sum(b == d for b, _ in env.inspection.covered) for d in (100, 200, 300)]
                cv2.rectangle(image, (0, 0), (639, 55), (22, 28, 36), -1)
                cv2.putText(image, f"HC20DT  |  {label}  |  {t:.1f} s  |  2x",
                            (12, 21), 0, 0.54, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.putText(image, f"100 mm: {counts[0]}/8     200 mm: {counts[1]}/8     300 mm: {counts[2]}/8",
                            (12, 45), 0, 0.53, (220, 235, 240), 1, cv2.LINE_AA)
                video.stdin.write(image.tobytes())

            env.inspection.on_physics_step = frame
            frame("Start", True)
            while True:
                action, _ = policy.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            for _ in range(25):
                frame(info["outcome"], True)
            video.stdin.close()
            if video.wait():
                raise RuntimeError("Video encoding failed")
    print("demo/inspection.mp4", info["outcome"], len(env.inspection.covered))


if __name__ == "__main__":
    main()
