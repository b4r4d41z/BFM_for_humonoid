"""Frozen task56 inference and explicit physical ROS controller messages."""
import numpy as np
import torch
from bfm.contracts.sharpa import GROUPS, JOINTS, SLICES, ActionNormalizer, flatten_state
from bfm.fb.checkpoint import load_model


class SharpaRunner:
    def __init__(self, checkpoint, task="object_to_goal", device="cpu", observation_profile="task56", *, expected_vision=None):
        self.model, self.payload = load_model(checkpoint, device, expected_profile=observation_profile,
                                             expected_vision=expected_vision)
        if self.payload["format"] != "vanilla_fb_sharpa_v3":
            raise ValueError("Sharpa runner refuses legacy checkpoints")
        reference = self.payload["reference_states"].to(device)
        a, b = {"object_to_goal": ("object_position", "goal_center"),
                "right_reach_object": ("right_eef_position", "object_position")}[task]
        rewards = -(reference[:, SLICES[a]] - reference[:, SLICES[b]]).norm(dim=-1)
        with torch.no_grad():
            if observation_profile == 'vision_proprio_v1':
                backward = self.payload['reference_backward'].to(device)
                if backward.shape != (len(reference), self.model.cfg.latent_dim) or not torch.isfinite(backward).all():
                    raise ValueError('Invalid visual reward reference embeddings')
                self.z = self.model.project_z((backward * rewards[:, None]).mean(0, keepdim=True))
            else:
                self.z = self.model.reward_inference(reference, rewards)
        self.actions = ActionNormalizer()

    @torch.no_grad()
    def act(self, measured_message):
        if self.model.cfg.observation_profile == 'vision_proprio_v1':
            state = measured_message  # Explicit named RGB/proprio dictionary, no BfmState.
        else:
            state = torch.tensor(flatten_state(measured_message)[None], dtype=torch.float32, device=self.model.device)
        return self.actions.physical(self.model.act(state, self.z, deterministic=True))[0].cpu().numpy()


class VisualObservationBuffer:
    """Latest received measurements in one monotonic clock; no header-clock mixing."""
    def __init__(self, max_age_s, cameras, max_camera_skew_s=None):
        if not np.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('Positive finite runtime freshness required')
        if max_camera_skew_s is not None and (not np.isfinite(max_camera_skew_s) or max_camera_skew_s <= 0):
            raise ValueError('Positive finite camera skew limit required')
        self.max_age, self.cameras, self.max_skew = max_age_s, tuple(cameras), max_camera_skew_s
        self.samples = {}

    def receive(self, source, value, receive_time):
        if source not in ('joints', *self.cameras) or not np.isfinite(receive_time):
            raise ValueError('Unknown visual source or invalid receive time')
        if source in self.samples and receive_time < self.samples[source][0]:
            raise ValueError('Receive clock moved backwards')
        self.samples[source] = (receive_time, value)

    def snapshot(self, tick_time):
        from bfm.contracts.vision import proprioception
        sources = ('joints', *self.cameras)
        if any(source not in self.samples for source in sources):
            return None
        ages = [tick_time-self.samples[source][0] for source in sources]
        if not np.isfinite(tick_time) or min(ages) < 0 or max(ages) > self.max_age:
            raise RuntimeError('Future/stale visual observation; stopping publication')
        if self.max_skew is not None and np.ptp([self.samples[c][0] for c in self.cameras]) > self.max_skew:
            raise RuntimeError('Runtime inter-camera skew exceeds configured limit')
        joints = self.samples['joints'][1]
        result = {'proprio': proprioception(joints.name, joints.position, joints.velocity)[None]}
        result.update({c: self.samples[c][1][None] for c in self.cameras})
        return result


def fill_messages(physical, arm, hand, action):
    # Message instances are injected to keep the contract testable without ROS.
    a = np.asarray(physical)
    ActionNormalizer().normalize(a)
    arm.name = list(JOINTS["left"] + JOINTS["right"])
    arm.position = np.r_[a[GROUPS["left_arm"]], a[GROUPS["right_arm"]]].tolist()
    hand.left_hand_position = np.rint(a[GROUPS["left_hand"]]).astype(int).tolist()
    hand.right_hand_position = np.rint(a[GROUPS["right_hand"]]).astype(int).tolist()
    for side in JOINTS:
        setattr(action, f"{side}_arm_joint_names", list(JOINTS[side]))
        setattr(action, f"{side}_arm_target", a[GROUPS[f"{side}_arm"]].tolist())
        # RobotHandPosition uses integer logical commands. Record what is sent.
        setattr(action, f"{side}_hand_target", [float(x) for x in getattr(hand, f"{side}_hand_position")])
    return arm, hand, action
