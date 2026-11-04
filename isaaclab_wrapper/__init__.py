"""Lazy exports: importing the FB runtime does not import the BC policy stack."""
from importlib import import_module

_MODULES = {
    "ActionAdapter": "action_adapter", "IsaacCameraBridge": "camera_bridge",
    "create_isaaclab_env": "env_factory", "ObservationAdapter": "observation_adapter",
    "BCPolicyRunner": "policy_runner", "PolicyScheduler": "policy_scheduler",
    "RolloutRecorder": "rollout_recorder", "BFMPolicyRunner": "bfm_policy_runner",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)

__all__ = [
    "ActionAdapter",
    "BCPolicyRunner",
    "IsaacCameraBridge",
    "ObservationAdapter",
    "PolicyScheduler",
    "RolloutRecorder",
    "create_isaaclab_env",
    "BFMPolicyRunner",
]
