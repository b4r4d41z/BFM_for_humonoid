from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


from bc.buffers.buffers import OfflineTrajectoryBuffer
from bc.fb_cpr.agent import AgentConfig, FBCPRAgent, TrainConfig
from bc.fb_cpr.model import ModelConfig


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

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Legacy single-device flag. If set, overrides both --buffer_device and --model_device "
            "(examples: cpu, cuda, cuda:0)."
        ),
    )
    parser.add_argument(
        "--buffer_device",
        type=str,
        default="cpu",
        help="Device for OfflineTrajectoryBuffer storage/sampling (recommended: cpu).",
    )
    parser.add_argument(
        "--model_device",
        type=str,
        default="cpu",
        help="Device for model/agent updates (examples: cpu, cuda, cuda:0).",
    )

    parser.add_argument("--updates", type=int, default=10_000, help="Number of optimizer update steps")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=1, help="If >1, sample sequence batches [B,T,...]")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=0.0)
    parser.add_argument("--action_loss", type=str, default="mse", choices=["mse", "smooth_l1"])

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--save_path", type=str, default="bc_offline.pt")

    # Full offline training system pieces
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Episode-level validation ratio")
    parser.add_argument("--split_seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument(
        "--split_by",
        type=str,
        default="episode",
        choices=["episode", "file", "bag_name"],
        help="How to group episodes before deterministic split.",
    )

    parser.add_argument("--eval_every", type=int, default=100, help="Run offline validation every N updates")
    parser.add_argument("--val_num_batches", type=int, default=16, help="Number of val batches per eval")
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_mae_full",
        choices=["val_mae_full", "val_mse_full", "val_mae_arm", "val_mae_hand"],
        help="Metric used for best-checkpoint selection (lower is better).",
    )

    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging (requires tensorboard package).",
    )
    parser.add_argument(
        "--tb_logdir",
        type=str,
        default="runs/bc",
        help="TensorBoard log root directory.",
    )
    parser.add_argument(
        "--tb_run_name",
        type=str,
        default=None,
        help="TensorBoard run subdirectory name (default: auto timestamp).",
    )
    parser.add_argument("--tb_flush_secs", type=int, default=10, help="TensorBoard SummaryWriter flush interval")

    return parser


def resolve_devices(args: argparse.Namespace) -> tuple[str, str]:
    if args.device is not None:
        buffer_device = args.device
        model_device = args.device
    else:
        buffer_device = args.buffer_device
        model_device = args.model_device

    if str(model_device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested model_device='{model_device}', but torch.cuda.is_available() is False."
        )
    return buffer_device, model_device


def build_buffer_from_files(
    h5_files: list[Path],
    *,
    buffer_device: str,
    seed: int,
    use_images: bool = True,
    use_text: bool = True,
) -> OfflineTrajectoryBuffer:
    return OfflineTrajectoryBuffer.from_hdf5_files(
        hdf5_paths=h5_files,
        use_images=use_images,
        use_text=use_text,
        device=buffer_device,
        seed=seed,
    )


def _episode_group_key(
    episode: dict[str, Any],
    episode_id: int,
    split_by: str,
    h5_files: list[Path],
) -> str:
    if split_by == "episode":
        return f"episode:{episode_id}"

    if split_by == "file":
        return f"file:{h5_files[episode_id].name}" if episode_id < len(h5_files) else f"file:{episode_id}"

    if split_by == "bag_name":
        meta = episode.get("meta", {}) if isinstance(episode, dict) else {}
        if isinstance(meta, dict) and "bag_name" in meta:
            return f"bag:{meta['bag_name']}"
        return f"bag_missing:{episode_id}"

    raise ValueError(f"Unsupported split_by: {split_by}")


def split_episode_ids(
    full_buffer: OfflineTrajectoryBuffer,
    h5_files: list[Path],
    *,
    val_ratio: float,
    split_seed: int,
    split_by: str,
) -> tuple[list[int], list[int], dict[str, Any]]:
    if full_buffer.num_episodes() < 2:
        raise ValueError(
            "Need at least 2 episodes to create train/val split. "
            f"Got num_episodes={full_buffer.num_episodes()}"
        )

    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    episode_ids = list(range(full_buffer.num_episodes()))
    grouped: dict[str, list[int]] = {}
    for ep_id in episode_ids:
        episode = full_buffer.get_episode(ep_id)
        key = _episode_group_key(episode, ep_id, split_by=split_by, h5_files=h5_files)
        grouped.setdefault(key, []).append(ep_id)

    group_keys = sorted(grouped.keys())
    rng = random.Random(split_seed)
    rng.shuffle(group_keys)

    target_val_episodes = max(1, int(math.ceil(len(episode_ids) * val_ratio)))

    val_ids: list[int] = []
    for gk in group_keys:
        val_ids.extend(grouped[gk])
        if len(val_ids) >= target_val_episodes:
            break

    val_set = set(val_ids)
    train_ids = [ep for ep in episode_ids if ep not in val_set]

    if len(train_ids) == 0:
        raise RuntimeError("Split produced empty train set. Reduce val_ratio or change split strategy.")

    intersection = sorted(set(train_ids).intersection(val_set))
    if len(intersection) > 0:
        raise RuntimeError(f"Train/val leakage detected. Intersection episode ids: {intersection}")

    split_info: dict[str, Any] = {
        "split_seed": split_seed,
        "split_by": split_by,
        "val_ratio": val_ratio,
        "num_episodes_total": len(episode_ids),
        "num_episodes_train": len(train_ids),
        "num_episodes_val": len(val_ids),
        "intersection_count": 0,
    }

    return train_ids, val_ids, split_info


def build_sub_buffer(
    source: OfflineTrajectoryBuffer,
    episode_ids: list[int],
    *,
    buffer_device: str,
    seed: int,
) -> OfflineTrajectoryBuffer:
    sub = OfflineTrajectoryBuffer(device=buffer_device, seed=seed)
    for ep_id in episode_ids:
        sub.add_episode(source.get_episode(ep_id))
    return sub


def _prepare_eval_batch(agent: FBCPRAgent, batch: dict[str, Any]) -> dict[str, Any]:
    # Reuse the exact same shaping/device logic as in training.
    return agent._prepare_batch_for_model(batch)  # noqa: SLF001 (intentional for consistency)


def evaluate(
    agent: FBCPRAgent,
    val_buffer: OfflineTrajectoryBuffer,
    *,
    batch_size: int,
    seq_len: int,
    val_num_batches: int,
    buffer_device: str,
) -> dict[str, float]:
    agent.eval()

    total = {
        "val_mse_full": 0.0,
        "val_mae_full": 0.0,
        "val_mse_arm": 0.0,
        "val_mae_arm": 0.0,
        "val_mse_hand": 0.0,
        "val_mae_hand": 0.0,
    }

    num = max(1, int(val_num_batches))
    with torch.no_grad():
        for _ in range(num):
            if seq_len > 1:
                batch = val_buffer.sample_sequences(batch_size=batch_size, seq_len=seq_len, device=buffer_device)
            else:
                batch = val_buffer.sample_transitions(batch_size=batch_size, device=buffer_device)

            model_batch = _prepare_eval_batch(agent, batch)
            pred = agent.model(model_batch)
            tgt = model_batch["action"]["full"]

            pred_arm, pred_hand = pred[:, :14], pred[:, 14:]
            tgt_arm, tgt_hand = tgt[:, :14], tgt[:, 14:]

            total["val_mse_full"] += float(torch.mean((pred - tgt) ** 2).item())
            total["val_mae_full"] += float(torch.mean(torch.abs(pred - tgt)).item())

            total["val_mse_arm"] += float(torch.mean((pred_arm - tgt_arm) ** 2).item())
            total["val_mae_arm"] += float(torch.mean(torch.abs(pred_arm - tgt_arm)).item())

            total["val_mse_hand"] += float(torch.mean((pred_hand - tgt_hand) ** 2).item())
            total["val_mae_hand"] += float(torch.mean(torch.abs(pred_hand - tgt_hand)).item())

    return {k: v / num for k, v in total.items()}


def checkpoint_paths(save_path: str) -> tuple[Path, Path, Path]:
    p = Path(save_path).expanduser().resolve()
    save_dir = p.parent if p.suffix else p
    save_dir.mkdir(parents=True, exist_ok=True)

    last_path = save_dir / "last.pt"
    best_path = save_dir / "best.pt"
    alias_path = p if p.suffix else (save_dir / "bc_offline.pt")
    return last_path, best_path, alias_path


def create_tensorboard_writer(args: argparse.Namespace, split_info: dict[str, Any]):
    if not args.tensorboard:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception as exc:  # pragma: no cover - dependency/runtime environment specific
        print(
            "[BFM offline train][warn] TensorBoard requested but unavailable: "
            f"{exc}. Install with: pip install tensorboard"
        )
        return None

    run_name = args.tb_run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.tb_logdir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir), flush_secs=int(args.tb_flush_secs))

    hparams = {
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip_norm": float(args.grad_clip_norm),
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "action_loss": str(args.action_loss),
        "eval_every": int(args.eval_every),
        "val_num_batches": int(args.val_num_batches),
        "split_seed": int(split_info["split_seed"]),
        "split_by": str(split_info["split_by"]),
        "val_ratio": float(split_info["val_ratio"]),
        "train_episodes": int(split_info["num_episodes_train"]),
        "val_episodes": int(split_info["num_episodes_val"]),
    }
    for key, value in hparams.items():
        writer.add_text(f"hparams/{key}", str(value), global_step=0)

    print(f"[BFM offline train] TensorBoard logging enabled: {run_dir}")
    return writer


def main() -> None:
    args = build_parser().parse_args()
    buffer_device, model_device = resolve_devices(args)

    data_path = Path(args.data).expanduser().resolve()
    h5_files = collect_h5_files(data_path, max_files=args.max_files)

    print("[BFM offline train] selected H5 files:")
    for i, p in enumerate(h5_files):
        print(f"  [{i}] {p}")

    full_buffer = build_buffer_from_files(
        h5_files,
        buffer_device=buffer_device,
        seed=args.split_seed,
        use_images=True,
        use_text=True,
    )

    train_ids, val_ids, split_info = split_episode_ids(
        full_buffer,
        h5_files,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        split_by=args.split_by,
    )

    train_buffer = build_sub_buffer(full_buffer, train_ids, buffer_device=buffer_device, seed=args.split_seed)
    val_buffer = build_sub_buffer(full_buffer, val_ids, buffer_device=buffer_device, seed=args.split_seed + 1)

    print(
        "[BFM offline train] split "
        f"seed={split_info['split_seed']} split_by={split_info['split_by']} val_ratio={split_info['val_ratio']:.3f} "
        f"train_eps={split_info['num_episodes_train']} val_eps={split_info['num_episodes_val']} "
        f"intersection={split_info['intersection_count']}"
    )
    print(
        "[BFM offline train] metric guide: "
        "train_loss=opt objective, train_mae=|pred-target| train batch, "
        "val_mae_full/arm/hand=validation |pred-target| (lower is better), "
        "val_mse_full=validation squared error, grad_norm=clipped gradient norm"
    )

    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        batch_size=args.batch_size,
        action_loss=args.action_loss,
    )
    agent_cfg = AgentConfig(device=model_device, train=train_cfg)

    model_cfg = ModelConfig(state_dim=26, action_dim=26)
    agent = FBCPRAgent(model_cfg=model_cfg, agent_cfg=agent_cfg)

    start_step = 1
    best_metric = float("inf")
    best_step = 0

    if args.resume_from:
        resume_path = Path(args.resume_from).expanduser().resolve()
        resume_extra = agent.load(str(resume_path))
        if isinstance(resume_extra, dict):
            start_step = int(resume_extra.get("step", 0)) + 1
            best_metric = float(resume_extra.get("best_metric", best_metric))
            best_step = int(resume_extra.get("best_step", best_step))
        print(
            f"[BFM offline train] resumed from {resume_path} "
            f"start_step={start_step} best_metric={best_metric:.6f} best_step={best_step}"
        )

    last_path, best_path, alias_path = checkpoint_paths(args.save_path)
    tb_writer = create_tensorboard_writer(args, split_info)

    print("[BFM offline train] start updates")
    print(f"[BFM offline train] buffer_device={buffer_device} model_device={model_device}")

    for step in range(start_step, int(args.updates) + 1):
        if args.seq_len > 1:
            batch = train_buffer.sample_sequences(batch_size=args.batch_size, seq_len=args.seq_len, device=buffer_device)
        else:
            batch = train_buffer.sample_transitions(batch_size=args.batch_size, device=buffer_device)

        train_metrics = agent.update(batch)
        if tb_writer is not None:
            tb_writer.add_scalar("train/loss", float(train_metrics["loss"]), step)
            tb_writer.add_scalar("train/action_mae", float(train_metrics["action_mae"]), step)
            tb_writer.add_scalar("train/grad_norm", float(train_metrics["grad_norm"]), step)
            tb_writer.add_scalar("train/lr", float(agent.learning_rate), step)

        run_eval = (step == start_step) or (step % int(args.eval_every) == 0) or (step == int(args.updates))

        if run_eval:
            val_metrics = evaluate(
                agent,
                val_buffer,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                val_num_batches=args.val_num_batches,
                buffer_device=buffer_device,
            )

            score = float(val_metrics[args.best_metric])
            if tb_writer is not None:
                for metric_name, metric_value in val_metrics.items():
                    tb_writer.add_scalar(f"val/{metric_name}", float(metric_value), step)
                tb_writer.add_scalar("val/best_metric_current_eval", score, step)
            improved = score < best_metric
            if improved:
                best_metric = score
                best_step = step
                if tb_writer is not None:
                    tb_writer.add_scalar("val/best_metric_so_far", best_metric, step)

            extra = {
                "step": step,
                "best_metric": best_metric,
                "best_step": best_step,
                "best_metric_name": args.best_metric,
                "split_info": split_info,
            }
            agent.save(str(last_path), extra=extra)
            # Keep backward-compatible alias at requested save_path.
            agent.save(str(alias_path), extra=extra)

            if improved:
                agent.save(str(best_path), extra=extra)
                print(
                    f"[BFM offline train] best updated at step={step} "
                    f"{args.best_metric}={score:.6f} path={best_path}"
                )

            if step % int(args.print_every) == 0 or step == start_step:
                best_metric_fragment = (
                    f"{args.best_metric}={score:.6f} "
                    if args.best_metric != "val_mae_full"
                    else ""
                )
                print(
                    f"[BFM offline train] step={step} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_mae={train_metrics['action_mae']:.6f} "
                    f"{best_metric_fragment}"
                    f"val_mae_full={val_metrics['val_mae_full']:.6f} "
                    f"val_mae_arm={val_metrics['val_mae_arm']:.6f} "
                    f"val_mae_hand={val_metrics['val_mae_hand']:.6f}"
                )
        else:
            if step % int(args.print_every) == 0:
                print(
                    f"[BFM offline train] step={step} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_mae={train_metrics['action_mae']:.6f}"
                )

    print(f"[BFM offline train] saved last checkpoint: {last_path}")
    print(f"[BFM offline train] saved best checkpoint: {best_path}")
    print(
        f"[BFM offline train] best summary: step={best_step} "
        f"{args.best_metric}={best_metric:.6f}"
    )
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()