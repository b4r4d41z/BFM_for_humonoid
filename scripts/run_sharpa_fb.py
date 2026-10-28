"""ROS2 endpoint for the authoritative collection simulator, not old IsaacLab."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", choices=["object_to_goal", "right_reach_object"], default="object_to_goal")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-state-age", type=float, required=True, help="Local receive age in seconds; never header age")
    args = p.parse_args()
    if not 0 < args.max_state_age < float('inf'):
        raise ValueError("max-state-age must be finite and positive")
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from dynamic_biped.msg import BfmAction, BfmState, RobotHandPosition
    from bfm.sharpa_runtime import SharpaRunner, fill_messages
    runner = SharpaRunner(args.checkpoint, args.task, args.device)
    rclpy.init()
    class Runtime(Node):
        def __init__(self):
            super().__init__("sharpa_fb_runtime")
            self.latest, self.received = None, 0.
            self.arm = self.create_publisher(JointState, "/robot_arm_joint_command", 1)
            self.hand = self.create_publisher(RobotHandPosition, "/bfm/command/hand_target", 1)
            self.action = self.create_publisher(BfmAction, "/bfm/action", 1)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(BfmState, "/bfm/state", self.receive, qos)
            self.create_timer(.1, self.tick)

        def receive(self, msg):
            if msg.header.frame_id != "world":
                raise ValueError("Expected world-frame measured BfmState")
            self.latest, self.received = msg, time.monotonic()

        def tick(self):
            if self.latest is None:
                return
            if time.monotonic()-self.received > args.max_state_age:
                raise RuntimeError("Measured BfmState stream stale; stopping publication")
            for topic in ("/robot_arm_joint_command", "/bfm/command/hand_target", "/bfm/action"):
                if any(info.node_name != self.get_name() or info.node_namespace != self.get_namespace() for info in self.get_publishers_info_by_topic(topic)):
                    raise RuntimeError(f"Competing publisher on {topic}; stop gateway/teleoperation before model control")
            arm, hand, action = fill_messages(runner.act(self.latest), JointState(), RobotHandPosition(), BfmAction())
            stamp = self.get_clock().now().to_msg()
            arm.header.stamp = hand.header.stamp = action.header.stamp = stamp
            self.arm.publish(arm); self.hand.publish(hand); self.action.publish(action)
    node = Runtime()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
