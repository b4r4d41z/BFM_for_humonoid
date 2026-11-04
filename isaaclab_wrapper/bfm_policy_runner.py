import torch

from bfm.contracts.kuavo import ARM_JOINTS, BFMStateAdapter, BFMActionAdapter, KuavoContract
from bfm.fb.checkpoint import load_model
from bfm.fb.context import infer_task
from .policy_scheduler import PolicyScheduler


class BFMPolicyRunner:
    def __init__(self, checkpoint_path, device="cpu", env_device=None, env_joint_names=ARM_JOINTS):
        self.model, self.checkpoint = load_model(checkpoint_path, device)
        if self.checkpoint["format"] != "vanilla_fb_v1":
            raise ValueError("Old IsaacLab/claw runner cannot execute Sharpa checkpoints; use scripts/run_sharpa_fb.py")
        self.contract = KuavoContract(**self.checkpoint["contract"])
        self.states = BFMStateAdapter()
        self.actions = BFMActionAdapter(self.contract, env_joint_names, env_device or device)
        self.scheduler = PolicyScheduler(policy_hz=1 / self.contract.policy_dt, control_hz=1 / self.contract.control_dt)
        self.reference = self.checkpoint["reference_states"].to(device)
        self.z = None
        self.last_normalized_action = None

    def set_goal(self, goal_spec):
        self.z = infer_task(self.model, goal_spec, self.reference).detach()
        self.scheduler.reset()

    def reset(self):
        # Retain the task prompt; discard action cache at every episode boundary.
        self.scheduler.reset()
        self.last_normalized_action = None

    @torch.no_grad()
    def act(self, environment_observation):
        if self.z is None:
            raise RuntimeError("Call set_goal before acting")
        state = self.states(environment_observation)
        if state.ndim != 2:
            raise ValueError("Runtime measured state must have an environment batch dimension")
        z = self.z.expand(len(state), -1) if len(self.z) == 1 else self.z
        def infer():
            self.last_normalized_action = self.model.act(state, z, deterministic=True)
            return self.actions(self.last_normalized_action)
        return self.scheduler.tick(infer)[0]
