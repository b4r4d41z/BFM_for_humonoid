"""Frozen task56 inference and explicit physical ROS controller messages."""
import numpy as np
import torch
from bfm.contracts.sharpa import GROUPS, JOINTS, SLICES, ActionNormalizer, flatten_state
from bfm.fb.checkpoint import load_model


class SharpaRunner:
    def __init__(self, checkpoint, task="object_to_goal", device="cpu"):
        self.model, self.payload = load_model(checkpoint, device)
        if self.payload["format"] != "vanilla_fb_sharpa_v3":
            raise ValueError("Sharpa runner refuses legacy checkpoints")
        reference = self.payload["reference_states"].to(device)
        a, b = {"object_to_goal": ("object_position", "goal_center"),
                "right_reach_object": ("right_eef_position", "object_position")}[task]
        rewards = -(reference[:, SLICES[a]] - reference[:, SLICES[b]]).norm(dim=-1)
        with torch.no_grad():
            self.z = self.model.reward_inference(reference, rewards)
        self.actions = ActionNormalizer()

    @torch.no_grad()
    def act(self, measured_message):
        state = torch.tensor(flatten_state(measured_message)[None], dtype=torch.float32, device=self.model.device)
        return self.actions.physical(self.model.act(state, self.z, deterministic=True))[0].cpu().numpy()


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
