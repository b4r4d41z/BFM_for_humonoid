from __future__ import annotations

import argparse
import sys
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
_PY_PKG_ROOT = _REPO_ROOT / "source" / "Test_training" / "Test_training"

if str(_PY_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_PKG_ROOT))

from learning.bfm.buffers.buffers import OfflineTrajectoryBuffer
from learning.bfm.fb_cpr.agent import AgentConfig, FBCPRAgent, TrainConfig
from learning.bfm.fb_cpr.model import ModelConfig


def collect_h5_files(path: Path, max_files: int) -> list[Path]:
    if path.is_file():
        if path.suffix != ".h5":
            raise ValueError(f"Expected .h5 file, got: {path}")
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = sorted(path.glob("*.h5"))
    if len(files) == 0:
        raise FileNotFoundError(f"No .h5 files found in: {path}")

    return files[:max_files]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("BFM offline BC trainer")

    parser.add_argument("--data", type=str, required=True, help="Path to one .h5 file or directory with .h5")
    parser.add_argument("--max_files", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--updates", type=int, default=10_000, help="Number of optimizer update steps")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1, help="If >1, sample sequence batches [B,T,...]")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--action_loss", type=str, default="mse", choices=["mse", "smooth_l1"])

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="bfm_offline_bc.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data_path = Path(args.data).expanduser().resolve()
    h5_files = collect_h5_files(data_path, max_files=args.max_files)

    print("[BFM offline train] selected H5 files:")
    for i, p in enumerate(h5_files):
        print(f"  [{i}] {p}")

    buffer = OfflineTrajectoryBuffer.from_hdf5_files(
        hdf5_paths=h5_files,
        use_images=True,
        use_text=True,
        device=args.device,
        seed=42,
    )

    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        batch_size=args.batch_size,
        action_loss=args.action_loss,
    )
    agent_cfg = AgentConfig(device=args.device, train=train_cfg)

    model_cfg = ModelConfig(
        state_dim=26,
        action_dim=26,
    )
    agent = FBCPRAgent(model_cfg=model_cfg, agent_cfg=agent_cfg)

    print("[BFM offline train] start updates")
    for step in range(1, int(args.updates) + 1):
        if args.seq_len > 1:
            batch = buffer.sample_sequences(
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                device=args.device,
            )
        else:
            batch = buffer.sample_transitions(
                batch_size=args.batch_size,
                device=args.device,
            )

        metrics = agent.update(batch)

        if step % int(args.print_every) == 0 or step == 1:
            print(
                f"[BFM offline train] step={step} "
                f"loss={metrics['loss']:.6f} "
                f"action_mae={metrics['action_mae']:.6f} "
                f"grad_norm={metrics['grad_norm']:.6f}"
            )

    save_path = Path(args.save_path).expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path))
    print(f"[BFM offline train] saved checkpoint: {save_path}")


if __name__ == "__main__":
    main()
