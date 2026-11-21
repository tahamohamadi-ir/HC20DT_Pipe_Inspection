from pathlib import Path
from time import monotonic, sleep

import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from stable_baselines3 import PPO

from simulation.gym_env import PipeInspectionGym


def main():
    root = Path(__file__).resolve().parents[1]
    policy = PPO.load(root / "models/hc20dt_ppo_4096.zip", device="cpu")
    env = PipeInspectionGym()
    obs, _ = env.reset(options={"offsets_deg": (0.0, 0.0)})
    data = env.inspection.data
    names = [f"joint_{i}" for i in range(1, 7)]

    rclpy.init()
    node = rclpy.create_node("pipe_inspection")
    joints_pub = node.create_publisher(JointState, "/joint_states", 10)
    pose_pub = node.create_publisher(PoseStamped, "/scanner_pose", 10)
    start = monotonic()
    last = float("-inf")

    def publish(force=False):
        nonlocal last
        if not force and data.time - last < 0.04 - 1e-9:
            return
        sleep(max(0.0, start + data.time - monotonic()))
        last = data.time
        stamp = node.get_clock().now().to_msg()

        joints = JointState()
        joints.header.stamp = stamp
        joints.name = names
        joints.position = [float(data.joint(n).qpos[0]) for n in names]
        joints.velocity = [float(data.joint(n).qvel[0]) for n in names]

        scanner = data.body("scanner")
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "base_link"
        p = pose.pose.position
        p.x, p.y, p.z = map(float, scanner.xpos)
        q = pose.pose.orientation
        q.w, q.x, q.y, q.z = map(float, scanner.xquat)

        joints_pub.publish(joints)
        pose_pub.publish(pose)

    try:
        env.inspection.on_physics_step = publish
        publish(force=True)
        while rclpy.ok():
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                publish(force=True)
                node.get_logger().info(
                    f"{info['outcome']}, targets={len(env.inspection.covered)}/24"
                )
                break
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()