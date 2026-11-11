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
    from bfm.fb.execution import ExecutionMonitor, fill_action, output_types
    from bfm.data.policy_time import SimPolicyClock
    from bfm.data.mcap import decode_image
    from bfm.fb.evaluation import Evaluation, RolloutLogger, validate_evaluation
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/runtime/isaac_ros2.yaml'))
    p.add_argument('--checkpoint');p.add_argument('--task-z');p.add_argument('--device',default='cpu')
    p.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'])
    p.add_argument('--profile-inference', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--mode', choices=['shadow', 'closed-loop'])
    p.add_argument('--warmup-ticks', type=int)
    p.add_argument('--max-policy-steps', type=int)
    p.add_argument('--max-arm-delta-rad', type=float)
    p.add_argument('--disable-arm-delta-limit', action='store_true',
                   help='Simulation-only: bypass per-tick arm target delta limiting (physical bounds still checked)')
    a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text())
    if a.precision is not None: cfg['inference_precision'] = a.precision
    if a.profile_inference is not None: cfg['profile_inference'] = a.profile_inference
    validate_ros_config(cfg)
    v4=cfg.get('control_contract_version',3)==4
    if v4 and (a.max_arm_delta_rad is not None or a.disable_arm_delta_limit):p.error('v4 delegates trajectory limits to the executor')
    mode = a.mode or cfg.get('mode', 'shadow')
    warmup = a.warmup_ticks if a.warmup_ticks is not None else cfg.get('warmup_ticks', 10)
    maximum = a.max_policy_steps if a.max_policy_steps is not None else cfg.get('max_policy_steps', 300)
    if a.disable_arm_delta_limit and a.max_arm_delta_rad is not None:
        p.error('--disable-arm-delta-limit and --max-arm-delta-rad cannot be used together')
    delta = (None if a.disable_arm_delta_limit else
             a.max_arm_delta_rad if a.max_arm_delta_rad is not None else cfg['safety']['max_arm_delta_rad'])
    if v4:delta=None
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
        if v4:
            from dynamic_biped.msg import BfmExecutionStatus
            from rosgraph_msgs.msg import Clock
            from rclpy.clock import Clock as ROSClock
            from rclpy.clock_type import ClockType
            if not all(hasattr(BfmAction(),k) for k in ("producer_session_id","action_id")):
                raise RuntimeError("Rebuild dynamic_biped v4 message interfaces")
    except ImportError as exc:
        raise RuntimeError('Source ROS2 and dynamic_biped message packages before running this endpoint') from exc
    logger = RolloutLogger(Path(cfg['logging']['output_root'])/uuid.uuid4().hex,
                           mode=mode, checkpoint=checkpoint, task_z=task)
    logger.summary.update(arm_delta_limit_enabled=delta is not None,
                          max_arm_delta_rad=delta)
    print(f'Rollout logs: {logger.directory}', flush=True)
    try:
        runtime=VisualRuntime(checkpoint,a.device, precision=cfg.get('inference_precision', 'fp32'),
                              profile=cfg.get('profile_inference', False))
        if v4:runtime.require_v4()
        runtime.load_task(task, require_bound=True)
    except Exception as exc:
        logger.close(f'initialization failed: {exc}')
        raise
    rclpy.init()
    class Runtime(Node):
        def __init__(self):
            super().__init__('bfm_visual_runtime',parameter_overrides=[Parameter('use_sim_time',value=cfg['use_sim_time'])])
            self.buffer=VisualObservationBuffer(cfg['max_age_s'],cfg['cameras'],cfg['max_camera_skew_s'])
            self.publishers_={k:self.create_publisher(t,cfg['output_topics'][k],1)
                              for k,t in output_types(4 if v4 else 3,BfmAction,JointState,RobotHandPosition).items()} if mode == 'closed-loop' else {}
            self.evaluation = Evaluation(runtime, self.buffer, logger, mode=mode, warmup_ticks=warmup,
                                         max_policy_steps=maximum, max_arm_delta_rad=delta, control_contract_version=4 if v4 else 3)
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(JointState,cfg['joint_topic'],lambda m:self.receive('joints',m),qos)
            for c in cfg['cameras']:
                self.create_subscription(Image,cfg['camera_topics'][c],lambda m,c=c:self.receive(c,m),qos)
            if v4:
                self.scheduler=SimPolicyClock(15)
                self.monitor=ExecutionMonitor(runtime.payload['execution'],cfg['ack_timeout_wall_s'])
                self.execution_events=(logger.directory/'execution.jsonl').open('w')
                self.create_subscription(BfmExecutionStatus,cfg['execution_status_topic'],self.acknowledgement,100)
                self.create_subscription(Clock,'/clock',self.simulation_clock,QoSProfile(depth=1000,reliability=ReliabilityPolicy.BEST_EFFORT))
                self.create_timer(.05,self.watchdog,clock=ROSClock(clock_type=ClockType.STEADY_TIME))
            else:self.create_timer(1/cfg['policy_hz'],self.tick)

        def flush_execution(self):
            import json
            while self.monitor.events:self.execution_events.write(json.dumps(self.monitor.events.pop(0))+'\n')
            self.execution_events.flush()

        def watchdog(self):
            try:self.monitor.check()
            except Exception as exc:
                self.flush_execution();self.evaluation.stop(str(exc));raise

        def acknowledgement(self, message):
            try:self.monitor.accept(message)
            except Exception as exc:
                self.evaluation.stop(str(exc));raise
            finally:self.flush_execution()

        def simulation_clock(self, message):
            ns=message.clock.sec*10**9+message.clock.nanosec
            if self.scheduler.last_ns is not None and ns<self.scheduler.last_ns:
                self.evaluation.stop('simulation_reset');return
            self.sim_ns=ns
            if self.scheduler.advance(ns):
                self.watchdog();self.tick()

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
            if v4:
                message=self.monitor.publish(fill_action(physical,BfmAction()),self.sim_ns)
                self.publishers_['action'].publish(message)
                self.flush_execution();return dict(producer_session_id=message.producer_session_id,action_id=message.action_id,acceptance_status='pending')
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
        while rclpy.ok() and (not node.evaluation.stopped or (v4 and bool(node.monitor.pending))):
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        logger.close('keyboard_interrupt')
    except Exception as exc:
        logger.close(f'runtime failure: {type(exc).__name__}: {exc}')
        raise
    finally:
        logger.close('shutdown')
        if node is not None:
            if v4:
                import json
                node.flush_execution();node.execution_events.close()
                logger.summary.update(producer_session_id=node.monitor.session,accepted_actions=len(node.monitor.completed),unacknowledged_actions=list(node.monitor.pending))
                (logger.directory/'summary.json').write_text(json.dumps(logger.summary,indent=2))
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':main()
