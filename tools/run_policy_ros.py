"""BFM-side ROS2 endpoint; simulator and scene remain external."""
import argparse
from pathlib import Path
import sys
import time
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import yaml
    from bfm.fb.runtime import VisualRuntime, validate_ros_config
    from bfm.sharpa_runtime import VisualObservationBuffer, fill_messages
    from bfm.data.mcap import decode_image
    from bfm.fb.evaluation import Evaluation, RolloutLogger, validate_evaluation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/runtime/isaac_ros2.yaml'))
    p.add_argument('--checkpoint');p.add_argument('--task-z');p.add_argument('--device',default='cpu')
    p.add_argument('--mode', choices=['shadow', 'closed-loop'])
    p.add_argument('--warmup-ticks', type=int)
    p.add_argument('--max-policy-steps', type=int)
    p.add_argument('--max-arm-delta-rad', type=float)
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());validate_ros_config(cfg)
    mode = a.mode or cfg.get('mode', 'shadow')
    warmup = a.warmup_ticks if a.warmup_ticks is not None else cfg.get('warmup_ticks', 10)
    maximum = a.max_policy_steps if a.max_policy_steps is not None else cfg.get('max_policy_steps', 300)
    delta = a.max_arm_delta_rad if a.max_arm_delta_rad is not None else cfg['safety']['max_arm_delta_rad']
    validate_evaluation(mode, warmup, maximum, delta)
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
    logger = RolloutLogger(Path(cfg['logging']['output_root'])/uuid.uuid4().hex,
                           mode=mode, checkpoint=checkpoint, task_z=task)
    print(f'Rollout logs: {logger.directory}', flush=True)
    try:
        runtime=VisualRuntime(checkpoint,a.device);runtime.load_task(task, require_bound=True)
    except Exception as exc:
        logger.close(f'initialization failed: {exc}')
        raise
    rclpy.init()
    class Runtime(Node):
        def __init__(self):
            super().__init__('bfm_visual_runtime',parameter_overrides=[Parameter('use_sim_time',value=cfg['use_sim_time'])])
            self.buffer=VisualObservationBuffer(cfg['max_age_s'],cfg['cameras'],cfg['max_camera_skew_s'])
            self.publishers_={k:self.create_publisher(t,cfg['output_topics'][k],1)
                              for k,t in [('arm',JointState),('hand',RobotHandPosition),('action',BfmAction)]} if mode == 'closed-loop' else {}
            self.evaluation = Evaluation(runtime, self.buffer, logger, mode=mode, warmup_ticks=warmup,
                                         max_policy_steps=maximum, max_arm_delta_rad=delta)
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(JointState,cfg['joint_topic'],lambda m:self.receive('joints',m),qos)
            for c in cfg['cameras']:
                self.create_subscription(Image,cfg['camera_topics'][c],lambda m,c=c:self.receive(c,m),qos)
            self.create_timer(1/cfg['policy_hz'],self.tick)

        def receive(self, source, message):
            try:
                value = decode_image(message) if source != 'joints' else message
                self.buffer.receive(source, value, time.monotonic())
            except Exception as exc:
                self.evaluation.stop(f'observation callback failed: {exc}', invalid_observation=True)
                raise

        def check_publishers(self):
            for topic in cfg['output_topics'].values():
                infos = self.get_publishers_info_by_topic(topic)
                if len(infos) > 1 or any(i.node_name!=self.get_name() or i.node_namespace!=self.get_namespace() for i in infos):
                    raise RuntimeError(f'Competing publisher on {topic}; relinquish gateway/teleop control')

        def publish_action(self, physical):
            messages=fill_messages(physical,JointState(),RobotHandPosition(),BfmAction())
            stamp = self.get_clock().now().to_msg()
            for key,message in zip(('arm','hand','action'),messages):
                message.header.stamp=stamp
                self.publishers_[key].publish(message)

        def tick(self):
            self.evaluation.tick(self.publish_action, self.check_publishers)

    node = None
    try:
        node=Runtime()
        while rclpy.ok() and not node.evaluation.stopped:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        logger.close('keyboard_interrupt')
    except Exception as exc:
        logger.close(f'runtime failure: {type(exc).__name__}: {exc}')
        raise
    finally:
        logger.close('shutdown')
        if node is not None: node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
