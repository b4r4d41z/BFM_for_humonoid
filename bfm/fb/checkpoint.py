from dataclasses import asdict
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from bfm.contracts.kuavo import KuavoContract, STATE_DIM, STATE_FIELDS
from bfm.data.schema import SCHEMA_VERSION
from .model import FBModel, ModelConfig


def rng_state():
    n = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4])}


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


def save_checkpoint(path, agent, replay, dataset, reference_states, reference_indices, run_config):
    payload = {
        "format": "vanilla_fb_sharpa_v3" if getattr(dataset, "schema_version", None) == "3.0.0" else "vanilla_fb_v1",
        "schema_version": getattr(dataset, "schema_version", SCHEMA_VERSION),
        "model_config": asdict(agent.model.cfg), "train_config": asdict(agent.cfg),
        "model": agent.model.state_dict(), "target_F": agent.target_F.state_dict(), "target_B": agent.target_B.state_dict(),
        "fb_optimizer": agent.fb_optimizer.state_dict(), "actor_optimizer": agent.actor_optimizer.state_dict(),
        "step": agent.step, "rng": rng_state(), "replay_rng": replay.state_dict(),
        "contract": dataset.contract.to_dict(), "state_fields": getattr(dataset, "state_fields", list(STATE_FIELDS)),
        "dataset_fingerprint": dataset.fingerprint, "training_episode_ids": dataset.ids,
        "dataset_manifest": dataset.manifest, "run_config": run_config,
        "reference_states": reference_states.cpu(), "reference_indices": reference_indices,
    }
    payload['observation_profile'] = agent.model.cfg.observation_profile
    if agent.model.observation_encoder is not None:
        payload['observation_identity'] = agent.model.observation_encoder.identity()
        payload['target_fusion'] = agent.target_fusion.state_dict()
        # Compact backward reference embeddings are enough for reward inference;
        # avoid embedding thousands of raw RGB frames in every checkpoint.
        with torch.no_grad():
            payload['reference_backward'] = torch.cat([
                agent.model.backward_map(replay.observations(reference_indices[i:i+32], 'next_obs')).cpu()
                for i in range(0, len(reference_indices), 32)])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}-{uuid.uuid4().hex}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") == "vanilla_fb_sharpa_v3":
        from bfm.contracts.sharpa import SharpaContract, VERSION, FIELDS
        if payload.get("schema_version") != VERSION or payload.get("contract") != SharpaContract().to_dict() or payload.get("state_fields") != list(FIELDS):
            raise ValueError("Sharpa checkpoint contract/order mismatch")
        cfg = ModelConfig(**payload["model_config"])
        if payload.get('observation_profile') != cfg.observation_profile:
            raise ValueError('Checkpoint observation profile mismatch; old collection-v2 checkpoints are unsupported')
        if cfg.observation_profile == 'vision_proprio_v1':
            from .observation import VisionConfig
            vision = asdict(VisionConfig(**cfg.vision))
            if payload.get('observation_identity') != {'observation_profile': cfg.observation_profile, **vision}:
                raise ValueError('Checkpoint visual configuration/identity mismatch')
            if cfg.state_dim != vision['embedding_dim'] or cfg.action_dim != 26:
                raise ValueError('Visual embedding/action dimension mismatch')
            if payload.get('reference_backward') is None or 'target_fusion' not in payload:
                raise ValueError('Visual checkpoint missing reward references/target fusion')
        elif (cfg.state_dim, cfg.action_dim) != (56, 26):
            raise ValueError("Sharpa task56 checkpoint requires 56D states and 26D actions")
        return payload
    if payload.get("format") != "vanilla_fb_v1" or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported FB checkpoint/schema")
    if payload.get("state_fields") != list(STATE_FIELDS):
        raise ValueError("Checkpoint state ordering mismatch")
    cfg = ModelConfig(**payload["model_config"])
    if (cfg.state_dim, cfg.action_dim) != (STATE_DIM, 8):
        raise ValueError("Checkpoint does not match KUAVO state/action view")
    KuavoContract(**payload["contract"])
    return payload


def load_model(path, device="cpu", *, expected_profile=None, expected_vision=None):
    payload = load_payload(path)
    cfg = ModelConfig(**payload['model_config'])
    if expected_profile is not None and expected_profile != cfg.observation_profile:
        raise ValueError('Checkpoint observation profile differs from requested runtime')
    if expected_vision is not None and expected_vision != cfg.vision:
        raise ValueError('Checkpoint visual configuration differs from requested runtime')
    model = FBModel(cfg, initialize_pretrained=False).to(device)
    model.load_state_dict(payload["model"])
    if model.observation_encoder is not None:
        from .observation import weights_digest
        if weights_digest(model.observation_encoder.backbone) != cfg.vision['weights_sha256']:
            raise ValueError('Frozen backbone weights checksum mismatch')
    model.eval().requires_grad_(False)
    return model, payload


def resume_agent(payload, agent, replay, dataset):
    if payload["dataset_fingerprint"] != dataset.fingerprint or payload["contract"] != dataset.contract.to_dict():
        raise ValueError("Resume dataset/contract differs from checkpoint")
    # Default the new profile fields for pre-existing legacy KUAVO checkpoints.
    if asdict(ModelConfig(**payload["model_config"])) != asdict(agent.model.cfg) or payload["train_config"] != asdict(agent.cfg):
        raise ValueError("Resume architecture or optimizer configuration mismatch")
    agent.model.load_state_dict(payload["model"])
    if agent.model.observation_encoder is not None:
        from .observation import weights_digest
        if weights_digest(agent.model.observation_encoder.backbone) != agent.model.cfg.vision['weights_sha256']:
            raise ValueError('Resume frozen backbone weights checksum mismatch')
    agent.target_F.load_state_dict(payload["target_F"])
    agent.target_B.load_state_dict(payload["target_B"])
    if agent.target_fusion is not None:
        agent.target_fusion.load_state_dict(payload['target_fusion'])
    agent.fb_optimizer.load_state_dict(payload["fb_optimizer"])
    agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
    agent.step = payload["step"]
    replay.load_state_dict(payload["replay_rng"])
    restore_rng(payload["rng"])
