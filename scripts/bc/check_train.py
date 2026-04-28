from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bc.fb_cpr.agent import AgentConfig, FBCPRAgent, TrainConfig
from bc.fb_cpr.model import ModelConfig


def build_fake_batch(batch_size: int, h: int, w: int, device: str = "cpu") -> dict:
    return {
        "obs": {
            "state": {
                "full": torch.randn(batch_size, 26, device=device),
            },
            "images": {
                "head": torch.randint(0, 256, (batch_size, h, w, 3), device=device, dtype=torch.uint8),
                "left_wrist": torch.randint(0, 256, (batch_size, h, w, 3), device=device, dtype=torch.uint8),
                "right_wrist": torch.randint(0, 256, (batch_size, h, w, 3), device=device, dtype=torch.uint8),
            },
        },
        "action": {
            "full": torch.randn(batch_size, 26, device=device),
        },
    }


def run_forward_smoke(agent: FBCPRAgent, batch: dict) -> None:
    with torch.no_grad():
        pred = agent.model(batch)

    expected_shape = (batch["obs"]["state"]["full"].shape[0], 26)
    assert tuple(pred.shape) == expected_shape, (
        f"Forward output shape mismatch: expected {expected_shape}, got {tuple(pred.shape)}"
    )
    print(f"[OK] forward smoke test: output shape={tuple(pred.shape)}")


def run_one_step_train_smoke(agent: FBCPRAgent, batch: dict) -> None:
    metrics = agent.update(batch)
    required = ("loss", "action_mae", "grad_norm")

    for key in required:
        assert key in metrics, f"Missing metric '{key}' in update output"

    assert metrics["loss"] >= 0.0, f"Loss must be non-negative, got {metrics['loss']}"
    print(
        "[OK] one-step train smoke: "
        f"loss={metrics['loss']:.6f}, action_mae={metrics['action_mae']:.6f}, grad_norm={metrics['grad_norm']:.6f}"
    )


def run_shape_guard_checks(agent: FBCPRAgent, good_batch: dict) -> None:
    # 1) Missing key check
    bad_missing_key = {
        "obs": {
            "state": {"full": good_batch["obs"]["state"]["full"]},
            "images": {
                # "head" key intentionally removed
                "left_wrist": good_batch["obs"]["images"]["left_wrist"],
                "right_wrist": good_batch["obs"]["images"]["right_wrist"],
            },
        },
        "action": {"full": good_batch["action"]["full"]},
    }

    try:
        _ = agent.model(bad_missing_key)
        raise AssertionError("Expected KeyError for missing obs.images.head, but no error was raised")
    except KeyError as e:
        print(f"[OK] missing-key guard: {e}")

    # 2) Wrong state dim check
    bad_state_dim = build_fake_batch(
        batch_size=good_batch["obs"]["state"]["full"].shape[0],
        h=good_batch["obs"]["images"]["head"].shape[1],
        w=good_batch["obs"]["images"]["head"].shape[2],
        device=str(agent.device),
    )
    bad_state_dim["obs"]["state"]["full"] = torch.randn(
        bad_state_dim["obs"]["state"]["full"].shape[0], 25, device=agent.device
    )

    try:
        _ = agent.model(bad_state_dim)
        raise AssertionError("Expected ValueError for wrong obs.state.full dim, but no error was raised")
    except ValueError as e:
        print(f"[OK] state-dim guard: {e}")

    # 3) Wrong action dim check at train update
    bad_action_dim = build_fake_batch(
        batch_size=good_batch["obs"]["state"]["full"].shape[0],
        h=good_batch["obs"]["images"]["head"].shape[1],
        w=good_batch["obs"]["images"]["head"].shape[2],
        device=str(agent.device),
    )
    bad_action_dim["action"]["full"] = torch.randn(
        bad_action_dim["action"]["full"].shape[0], 25, device=agent.device
    )

    try:
        _ = agent.update(bad_action_dim)
        raise AssertionError("Expected ValueError for wrong action.full dim, but no error was raised")
    except ValueError as e:
        print(f"[OK] action-dim guard: {e}")


def main() -> None:
    parser = argparse.ArgumentParser("BFM model/agent smoke checks")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=84)
    parser.add_argument("--img_w", type=int, default=84)
    args = parser.parse_args()

    model_cfg = ModelConfig(state_dim=26, action_dim=26)
    agent_cfg = AgentConfig(
        device=args.device,
        train=TrainConfig(batch_size=args.batch_size, lr=1e-4, action_loss="mse"),
        compile=False,
    )
    agent = FBCPRAgent(model_cfg=model_cfg, agent_cfg=agent_cfg)

    fake_batch = build_fake_batch(args.batch_size, args.img_h, args.img_w, device=args.device)

    run_forward_smoke(agent, fake_batch)
    run_one_step_train_smoke(agent, fake_batch)
    run_shape_guard_checks(agent, fake_batch)

    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
