"""BFM-side ROS2 endpoint; simulator and scene remain external."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from bfm.fb.runtime import VisualRuntime, validate_ros_config
    from bfm.sharpa_runtime import VisualObservationBuffer, fill_messages
    from bfm.contracts.sharpa import ActionNormalizer
    from bfm.data.mcap import decode_image
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/runtime/isaac_ros2.yaml'))
    p.add_argument('--checkpoint');p.add_argument('--task-z');p.add_argument('--device',default='cpu')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());validate_ros_config(cfg)
    checkpoint=a.checkpoint or cfg['checkpoint'];task=a.task_z or cfg['task_z']
    if not checkpoint or not task:p.error('Supply checkpoint and task-z through CLI or config')
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image, JointState
        from dynamic_biped.msg import BfmAction, RobotHandPosition
    except ImportError as exc:
        raise RuntimeError('Source ROS2 and dynamic_biped message packages before running this endpoint') from exc
    runtime=VisualRuntime(checkpoint,a.device);runtime.load_task(task)
    rclpy.init()
    class Runtime(Node):
        def __init__(self):
            super().__init__('bfm_visual_runtime',parameter_overrides=[Parameter('use_sim_time',value=cfg['use_sim_time'])])
            self.buffer=VisualObservationBuffer(cfg['max_age_s'],cfg['cameras'],cfg['max_camera_skew_s'])
            self.publishers_={k:self.create_publisher(t,cfg['output_topics'][k],1)
                              for k,t in [('arm',JointState),('hand',RobotHandPosition),('action',BfmAction)]}
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(JointState,cfg['joint_topic'],lambda m:self.buffer.receive('joints',m,time.monotonic()),qos)
            for c in cfg['cameras']:
                self.create_subscription(Image,cfg['camera_topics'][c],lambda m,c=c:self.buffer.receive(c,decode_image(m),time.monotonic()),qos)
            self.create_timer(1/cfg['policy_hz'],self.tick)

        def tick(self):
            obs=self.buffer.snapshot(time.monotonic())
            if obs is None:return
            for topic in cfg['output_topics'].values():
                if any(i.node_name!=self.get_name() or i.node_namespace!=self.get_namespace() for i in self.get_publishers_info_by_topic(topic)):
                    raise RuntimeError(f'Competing publisher on {topic}; relinquish gateway/teleop control')
            physical=ActionNormalizer().physical(runtime.act(obs)).numpy()
            self.buffer.snapshot(time.monotonic())  # Include inference latency in age budget.
            messages=fill_messages(physical,JointState(),RobotHandPosition(),BfmAction())
            for key,message in zip(('arm','hand','action'),messages):
                message.header.stamp=self.get_clock().now().to_msg()
                self.publishers_[key].publish(message)
    node=Runtime()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
