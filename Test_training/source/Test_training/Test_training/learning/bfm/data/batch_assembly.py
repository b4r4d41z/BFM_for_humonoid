from __future__ import annotations

from typing import Any

import torch


def assemble_bfm_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("samples must not be empty")

    has_images = len(samples[0]["obs"]["images"]) > 0

    batch = {
        "obs": {
            "state": {
                "arm_joints": torch.stack(
                    [s["obs"]["state"]["arm_joints"] for s in samples], dim=0
                ),
                "hand_state": torch.stack(
                    [s["obs"]["state"]["hand_state"] for s in samples], dim=0
                ),
                "full": torch.stack(
                    [s["obs"]["state"]["full"] for s in samples], dim=0
                ),
            },
            "images": {},
            "text": [s["obs"]["text"] for s in samples],
            "timestamp": torch.stack(
                [s["obs"]["timestamp"] for s in samples], dim=0
            ),
        },
        "action": {
            "joint_target": torch.stack(
                [s["action"]["joint_target"] for s in samples], dim=0
            ),
            "hand_target": torch.stack(
                [s["action"]["hand_target"] for s in samples], dim=0
            ),
            "full": torch.stack(
                [s["action"]["full"] for s in samples], dim=0
            ),
        },
        "next_obs": {
            "state": {
                "arm_joints": torch.stack(
                    [s["next_obs"]["state"]["arm_joints"] for s in samples], dim=0
                ),
                "hand_state": torch.stack(
                    [s["next_obs"]["state"]["hand_state"] for s in samples], dim=0
                ),
                "full": torch.stack(
                    [s["next_obs"]["state"]["full"] for s in samples], dim=0
                ),
            }
        },
        "reward": torch.stack([s["reward"] for s in samples], dim=0),
        "done": torch.stack([s["done"] for s in samples], dim=0),
        "meta": [s["meta"] for s in samples],
    }

    if has_images:
        for key in samples[0]["obs"]["images"].keys():
            batch["obs"]["images"][key] = torch.stack(
                [s["obs"]["images"][key] for s in samples], dim=0
            )

    return batch