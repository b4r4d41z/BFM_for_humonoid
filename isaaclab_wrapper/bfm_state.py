"""Named measured-signal extraction; imports no Isaac modules at module load."""
import torch

from bfm.contracts.kuavo import (
    ARM_JOINTS, RIGHT_ARM, LEFT_ARM, RIGHT_FINGERS, OPEN_HAND, CLOSED_HAND,
    KuavoContract, ordered_indices,
)


def selected_limits(env, names):
    ids = ordered_indices(list(env.robot.joint_names), names)
    limits = getattr(env.robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        raise ValueError("Measured articulation soft_joint_pos_limits are required")
    if limits.ndim == 3:
        if not torch.allclose(limits, limits[0:1].expand_as(limits)):
            raise ValueError("Per-environment joint limits differ")
        limits = limits[0]
    selected = limits[ids].detach().cpu()
    if not torch.isfinite(selected).all():
        raise ValueError("Finite joint limits are required")
    return selected[:, 0].tolist(), selected[:, 1].tolist()


def contract_from_env(env, *, eef_body, eef_offset, policy_hz=10., left_hold=None):
    right_low, right_high = selected_limits(env, RIGHT_ARM)
    left_low, left_high = selected_limits(env, LEFT_ARM)
    if left_hold is None:
        ids = ordered_indices(list(env.robot.joint_names), LEFT_ARM)
        left_hold = env.robot.data.default_joint_pos[0, ids].detach().cpu().tolist()
    target_offset = [0., 0., float(env.cfg.container_base_cfg.spawn.size[2]) / 2 + float(env.cfg.ball_radius)]
    return KuavoContract(right_low, right_high, left_hold, left_low, left_high, eef_body, eef_offset,
                         control_dt=float(env.cfg.sim.dt) * env.cfg.decimation,
                         policy_dt=1 / policy_hz, container_target_offset=target_offset)


class MeasuredStateExtractor:
    def __init__(self, env, contract):
        self.env, self.contract = env, contract
        names = list(env.robot.joint_names)
        self.right = ordered_indices(names, RIGHT_ARM)
        self.fingers = ordered_indices(names, RIGHT_FINGERS)
        self.body = ordered_indices(list(env.robot.body_names), [contract.eef_body])[0]
        actual = contract_from_env(env, eef_body=contract.eef_body, eef_offset=contract.eef_offset,
                                   policy_hz=1 / contract.policy_dt, left_hold=contract.left_hold)
        if actual.to_dict() != contract.to_dict():
            # Only left claw state is a runtime choice, not an articulation property.
            actual.left_gripper_closed = contract.left_gripper_closed
            if actual.to_dict() != contract.to_dict():
                raise ValueError("Checkpoint contract differs from Isaac joint limits/timing/scene geometry")
        if list(env._ctrl_joint_names) != list(ARM_JOINTS):
            raise ValueError("Environment control ordering differs from canonical KUAVO ordering")

    @torch.no_grad()
    def __call__(self):
        e, c = self.env, self.contract
        data, origins = e.robot.data, e.scene.env_origins
        quat = data.body_quat_w[:, self.body]
        offset = torch.tensor(c.eef_offset, device=quat.device).expand(len(quat), -1)
        # Isaac quaternions are wxyz. Rotate a body-local TCP offset into world.
        uv = torch.cross(quat[:, 1:], offset, dim=-1)
        rotated = offset + 2 * (quat[:, :1] * uv + torch.cross(quat[:, 1:], uv, dim=-1))
        state = {
            "right_arm_q": data.joint_pos[:, self.right], "right_arm_dq": data.joint_vel[:, self.right],
            "right_gripper_q": data.joint_pos[:, self.fingers],
            "right_eef_position": data.body_pos_w[:, self.body] + rotated - origins,
            "red_ball_position": e.red_ball.data.root_pos_w - origins,
            "red_ball_velocity": e.red_ball.data.root_lin_vel_w,
            "container_position": e.container_base.data.root_pos_w - origins + torch.tensor(c.container_target_offset, device=origins.device),
            # Richer measured storage, excluded from the first model's view.
            "all_joint_q": data.joint_pos, "all_joint_dq": data.joint_vel,
        }
        return {k: v.detach().clone() for k, v in state.items()}


class IsaacTransitionBridge:
    """Captures true post-physics/pre-reset states using the environment's opt-in hook."""
    def __init__(self, env, contract):
        self.wrapped_env, self.env = env, env.unwrapped
        self.extract = MeasuredStateExtractor(self.env, contract)
        self.contract = contract
        if getattr(self.env, "_bfm_pre_reset_observer", None) is not None:
            raise RuntimeError("Another pre-reset observer is already installed")
        self.env._bfm_pre_reset_observer = self._capture
        self.captured = None

    def applied_action(self):
        # Read targets actually submitted to actuators after joint-limit clipping.
        e = self.env
        full = torch.empty((e.num_envs, 26), device=e.device)
        full[:, :14] = e._joint_pos_target
        opened = torch.tensor(OPEN_HAND, device=e.device)
        closed = torch.tensor(CLOSED_HAND, device=e.device)
        full[:, 14:20] = torch.where(e._left_claw_closed[:, None], closed, opened)
        full[:, 20:26] = torch.where(e._right_claw_closed[:, None], closed, opened)
        physical = torch.cat((full[:, :7], torch.where(e._right_claw_closed[:, None], 1., -1.)), -1)
        return physical.clone(), full.clone()

    def step(self, action):
        # Capture executed action inside the pre-reset callback too: reset mutates
        # the environment's claw state before env.step returns.
        self.captured = None
        self.executed = None
        result = self.wrapped_env.step(action)
        if self.captured is None or self.executed is None:
            raise RuntimeError("Missing pre-reset capture hook; refusing reset observations as transitions")
        state, terminated, truncated = self.captured
        return state, terminated, truncated, self.executed, result

    def _capture(self, terminated, truncated):
        self.captured = (self.extract(), terminated.detach().clone(), truncated.detach().clone())
        self.executed = self.applied_action()

    def close(self):
        self.env._bfm_pre_reset_observer = None
