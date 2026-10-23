"""Reward-free offline FB training. No task rewards are computed here."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from bfm.data.dataset import BFMDataset
from bfm.data.replay_buffer import BFMReplayBuffer
from .agent import FBAgent, TrainConfig
from .model import ModelConfig
from .checkpoint import save_checkpoint, load_payload, resume_agent


def train(config, *, resume=None):
    allowed = {"dataset_manifest", "output_dir", "seed", "device", "batch_size", "training_steps",
               "checkpoint_interval", "logging_interval", "reference_count", "reference_seed", "model", "train"}
    if set(config) - allowed:
        raise ValueError(f"Unknown training keys: {set(config)-allowed}")
    cfg = {"seed": 0, "device": "cpu", "batch_size": 256, "training_steps": 100000,
           "checkpoint_interval": 10000, "logging_interval": 100, "reference_count": 4096, "reference_seed": 17,
           "model": {}, "train": {}, **config}
    if cfg["batch_size"] < 2 or any(cfg[k] < 1 for k in ("training_steps", "checkpoint_interval", "logging_interval", "reference_count")):
        raise ValueError("Positive training counts and batch_size>=2 required")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "metrics.jsonl").exists():
        raise FileExistsError("Output already contains a run; use --resume or a new output directory")
    data = BFMDataset(cfg["dataset_manifest"], split="train")
    replay = BFMReplayBuffer(data.episodes, cfg["seed"])
    model_cfg = ModelConfig(**cfg["model"])
    if model_cfg.state_dim != replay.data["observation"].shape[-1] or model_cfg.action_dim != replay.data["action"].shape[-1]:
        raise ValueError("Model dimensions disagree with measured dataset")
    agent = FBAgent(model_cfg, TrainConfig(**cfg["train"]), cfg["device"])
    refs, ref_ids = replay.reference_states(cfg["reference_count"], cfg["reference_seed"])
    if resume:
        payload = load_payload(resume)
        for k in ("batch_size", "seed", "reference_count", "reference_seed"):
            if payload["run_config"][k] != cfg[k]:
                raise ValueError(f"Resume changes deterministic setting {k}")
        resume_agent(payload, agent, replay, data)
        refs, ref_ids = payload["reference_states"], payload["reference_indices"]
        if cfg["training_steps"] <= agent.step:
            raise ValueError("training_steps must exceed the resumed checkpoint step")
    else:
        agent.model.normalizer.fit(torch.cat((replay.data["observation"], replay.data["next_observation"])))
    cfg["model"], cfg["train"] = asdict(agent.model.cfg), asdict(agent.cfg)
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    with (output / "metrics.jsonl").open("a") as log:
        while agent.step < cfg["training_steps"]:
            metrics = agent.update(replay.sample_transitions(cfg["batch_size"]), policy_dt=data.contract.policy_dt)
            if agent.step % cfg["logging_interval"] == 0 or agent.step == 1 or agent.step == cfg["training_steps"]:
                record = {"step": agent.step, **metrics}
                line = json.dumps(record, allow_nan=False)
                print(line, flush=True)
                log.write(line + "\n"); log.flush()
            if agent.step % cfg["checkpoint_interval"] == 0 or agent.step == cfg["training_steps"]:
                save_checkpoint(output / "last.pt", agent, replay, data, refs, ref_ids, cfg)
    return output / "last.pt"
