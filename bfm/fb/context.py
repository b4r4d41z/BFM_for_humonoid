"""Post-training task prompts only. This module is not imported by pretraining."""
import torch
from bfm.contracts.kuavo import STATE_FIELDS


def field_slices():
    out, offset = {}, 0
    for key, (width, _) in STATE_FIELDS.items():
        out[key] = slice(offset, offset + width)
        offset += width
    return out


def task_rewards(states, task):
    fields = field_slices()
    if task == "reach_red_ball":
        a, b = "right_eef_position", "red_ball_position"
    elif task == "move_red_ball_to_container":
        a, b = "red_ball_position", "container_position"
    else:
        raise ValueError(f"Unknown reward task: {task}")
    return -(states[:, fields[a]] - states[:, fields[b]]).norm(dim=-1)


@torch.no_grad()
def infer_task(model, spec, reference_states):
    if spec.get("type") == "goal":
        if "state" not in spec:
            raise ValueError("Goal requires a complete measured state, not partial coordinates")
        from bfm.contracts.kuavo import BFMStateAdapter
        state = BFMStateAdapter()(spec["state"])
        if state.ndim == 1:
            state = state[None]
        return model.goal_inference(state)
    if spec.get("type") == "reward":
        return model.reward_inference(reference_states, task_rewards(reference_states, spec["task"]))
    raise ValueError("Goal specification type must be goal or reward")
