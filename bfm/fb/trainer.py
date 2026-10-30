"""Reward-free offline FB training. No task rewards are computed here."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from bfm.data.dataset import BFMDataset
from bfm.data.replay_buffer import BFMReplayBuffer
from .agent import FBAgent, TrainConfig
from .model import ModelConfig
from .checkpoint import save_checkpoint, load_payload, resume_agent


def train(config, *, resume=None):
    allowed = {
        "dataset_schema",
        "dataset_manifest",
        "output_dir",
        "seed",
        "device",
        "batch_size",
        "training_steps",
        "checkpoint_interval",
        "logging_interval",
        "reference_count",
        "reference_seed",
        "model",
        "train",
        "observation_profile",
        "vision",
        "outcome",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown training keys: {set(config)-allowed}")

    cfg = {
        "seed": 0,
        "device": "cpu",
        "batch_size": 256,
        "training_steps": 100000,
        "checkpoint_interval": 10000,
        "logging_interval": 100,
        "reference_count": 4096,
        "reference_seed": 17,
        "model": {},
        "train": {},
        **config,
    }

    if cfg["batch_size"] < 2 or any(
        cfg[k] < 1
        for k in (
            "training_steps",
            "checkpoint_interval",
            "logging_interval",
            "reference_count",
        )
    ):
        raise ValueError("Positive training counts and batch_size>=2 required")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    if not resume and (output / "metrics.jsonl").exists():
        raise FileExistsError(
            "Output already contains a run; use --resume or a new output directory"
        )

    schema = cfg.get(
        "dataset_schema", "measured_v2"
    )  # Explicit old programmatic compatibility.
    profile = cfg.get("observation_profile", "task56")
    visual = profile == "vision_proprio_v1"

    if profile not in ("task56", "vision_proprio_v1") or (
        visual and schema != "sharpa_v3"
    ):
        raise ValueError("Observation profile incompatible with dataset schema")

    if schema == "sharpa_v3":
        from bfm.data.sharpa import SharpaDataset

        data = SharpaDataset(
            cfg["dataset_manifest"],
            split="train",
            observation_profile=profile,
            outcome=cfg.get("outcome", "all"),
        )
    elif schema == "measured_v2":
        data = BFMDataset(cfg["dataset_manifest"], split="train")
    else:
        raise ValueError(
            "dataset_schema must be sharpa_v3 or legacy measured_v2"
        )

    if visual:
        from bfm.data.visual_replay import VisualReplayBuffer
        from .observation import VisionConfig

        vision = asdict(VisionConfig(**cfg.get("vision", {})))
        replay = VisualReplayBuffer(
            data,
            cfg["seed"],
            cameras=vision["cameras"],
        )
        state_dim = vision["embedding_dim"]
    else:
        if cfg.get("vision"):
            raise ValueError("task56 cannot have vision configuration")

        replay = BFMReplayBuffer(data.episodes, cfg["seed"])
        vision, state_dim = {}, replay.data["observation"].shape[-1]

    requested = {
        "state_dim": state_dim,
        "action_dim": replay.data["action"].shape[-1],
        **cfg["model"],
        "observation_profile": profile,
        "vision": vision,
    }

    if (
        requested["state_dim"] != state_dim
        or requested["action_dim"] != replay.data["action"].shape[-1]
    ):
        raise ValueError("Model dimensions disagree with measured dataset")

    payload = load_payload(resume) if resume else None

    if payload and visual:
        # Reuse checkpoint-contained weights without redownloading the backbone.
        saved = payload["model_config"]["vision"]

        for k, value in cfg.get("vision", {}).items():
            if (
                k == "revision"
                and value
                == payload["run_config"]
                .get("vision", {})
                .get("revision")
            ):
                continue  # Original requested alias already resolved in model_config.

            if saved.get(k) != value:
                raise ValueError(
                    f"Resume visual configuration mismatch: {k}"
                )

        requested["vision"] = saved

    model_cfg = ModelConfig(**requested)

    agent = FBAgent(
        model_cfg,
        TrainConfig(**cfg["train"]),
        cfg["device"],
        initialize_pretrained=not bool(resume),
    )

    refs, ref_ids = replay.reference_states(
        cfg["reference_count"],
        cfg["reference_seed"],
    )

    if resume:
        for k in (
            "batch_size",
            "seed",
            "reference_count",
            "reference_seed",
        ):
            if payload["run_config"][k] != cfg[k]:
                raise ValueError(
                    f"Resume changes deterministic setting {k}"
                )

        resume_agent(
            payload,
            agent,
            replay,
            data,
        )

        refs, ref_ids = (
            payload["reference_states"],
            payload["reference_indices"],
        )

        if cfg["training_steps"] <= agent.step:
            raise ValueError(
                "training_steps must exceed the resumed checkpoint step"
            )
    else:
        agent.model.normalizer.fit(
            torch.cat(
                (
                    replay.data["observation"],
                    replay.data["next_observation"],
                )
            )
        )

    cfg["model"] = asdict(agent.model.cfg)
    cfg["train"] = asdict(agent.cfg)
    cfg["observation_profile"] = profile

    (output / "config.json").write_text(
        json.dumps(cfg, indent=2)
    )

    writer = SummaryWriter(
        log_dir=str(output / "tensorboard")
    )

    try:
        with (output / "metrics.jsonl").open("a") as log:
            while agent.step < cfg["training_steps"]:
                metrics = agent.update(
                    replay.sample_transitions(
                        cfg["batch_size"]
                    ),
                    policy_dt=data.contract.policy_dt,
                )

                if (
                    agent.step % cfg["logging_interval"] == 0
                    or agent.step == 1
                    or agent.step == cfg["training_steps"]
                ):
                    record = {
                        "step": agent.step,
                        **metrics,
                    }

                    line = json.dumps(
                        record,
                        allow_nan=False,
                    )

                    print(
                        line,
                        flush=True,
                    )

                    log.write(line + "\n")
                    log.flush()

                    for key, value in metrics.items():
                        if isinstance(
                            value,
                            (int, float, np.integer, np.floating),
                        ):
                            writer.add_scalar(
                                key,
                                float(value),
                                agent.step,
                            )
                        elif torch.is_tensor(value) and value.numel() == 1:
                            writer.add_scalar(
                                key,
                                value.detach().item(),
                                agent.step,
                            )

                    writer.flush()

                if (
                    agent.step % cfg["checkpoint_interval"] == 0
                    or agent.step == cfg["training_steps"]
                ):
                    save_checkpoint(
                        output / "last.pt",
                        agent,
                        replay,
                        data,
                        refs,
                        ref_ids,
                        cfg,
                    )
    finally:
        writer.flush()
        writer.close()

    return output / "last.pt"