import copy
from dataclasses import dataclass
import math

import torch
from torch import nn

from .model import FBModel, ModelConfig
from .losses import actor_loss, fb_loss


@dataclass
class TrainConfig:
    discount: float = 0.99
    tau: float = 0.01
    learning_rate_fb: float = 1e-4
    learning_rate_actor: float = 1e-4
    orthonormalization_coefficient: float = 1.0
    gradient_clip_norm: float = 10.0
    fb_pessimism: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("Training configuration must be finite")
        if not 0 <= self.discount < 1 or not 0 <= self.tau <= 1:
            raise ValueError("discount in [0,1), tau in [0,1] required")
        if self.learning_rate_fb <= 0 or self.learning_rate_actor <= 0 or self.orthonormalization_coefficient < 0 or self.gradient_clip_norm < 0:
            raise ValueError("Invalid optimizer/orthogonality configuration")


def checked_step(loss, optimizer, parameters, clip):
    params = list(parameters)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite FB/actor loss: {loss.detach()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(params, clip if clip > 0 else float("inf"), error_if_nonfinite=True)
    optimizer.step()
    return norm


class FBAgent:
    def __init__(self, model_cfg: ModelConfig, train_cfg: TrainConfig, device="cpu"):
        self.model = FBModel(model_cfg).to(device)
        self.cfg = train_cfg
        self.target_F = copy.deepcopy(self.model.forwards).requires_grad_(False).eval()
        self.target_B = copy.deepcopy(self.model.backward).requires_grad_(False).eval()
        self.fb_parameters = list(self.model.forwards.parameters()) + list(self.model.backward.parameters())
        self.fb_optimizer = torch.optim.Adam(self.fb_parameters, lr=train_cfg.learning_rate_fb)
        self.actor_optimizer = torch.optim.Adam(self.model.actor.parameters(), lr=train_cfg.learning_rate_actor)
        self.step = 0

    @torch.no_grad()
    def soft_update(self):
        for online, target in ((self.model.forwards, self.target_F), (self.model.backward, self.target_B)):
            for p, t in zip(online.parameters(), target.parameters()):
                t.lerp_(p, self.cfg.tau)

    def update_actor(self, states, z):
        self.fb_optimizer.zero_grad(set_to_none=True)
        for p in self.fb_parameters:
            p.requires_grad_(False)
        try:
            actions = self.model.act(states, z.detach(), deterministic=False)
            loss = actor_loss(self.model.forward_map(states, actions, z.detach()), z.detach())
            norm = checked_step(loss, self.actor_optimizer, self.model.actor.parameters(), self.cfg.gradient_clip_norm)
        finally:
            for p in self.fb_parameters:
                p.requires_grad_(True)
        return {"actor_loss": loss.detach(), "actor_gradient_norm": norm,
                "actor_action_mean": actions.detach().mean(), "actor_action_std": actions.detach().std(correction=0),
                "actor_boundary_fraction": (actions.detach().abs() > 0.99).float().mean()}

    def update(self, batch, *, policy_dt=0.1):
        m = self.model
        s = batch["observation"].to(m.device, dtype=torch.float32)
        ns = batch["next_observation"].to(m.device, dtype=torch.float32)
        a = batch["action"].to(m.device, dtype=torch.float32)
        terminated = batch["terminated"].to(m.device)
        if terminated.dtype != torch.bool or terminated.shape != (len(s),):
            raise ValueError("terminated must be boolean [N]")
        if not torch.isfinite(a).all() or (a.abs() > 1.00001).any():
            raise ValueError("Replay actions must be finite in [-1,1]")
        durations = batch["duration"].to(m.device, dtype=torch.float32)
        if not math.isfinite(policy_dt) or policy_dt <= 0 or durations.shape != (len(s),) or not torch.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError("Transition durations must be positive finite [N]")
        discount = (self.cfg.discount ** (durations / policy_dt) * (~terminated)).unsqueeze(-1)
        # Uniform spherical latents: no rewards or demonstrated actions used to infer z.
        z = m.sample_z(len(s))
        with torch.no_grad():
            next_action = m.act(ns, z, deterministic=False)
            x = torch.cat((m._states(ns), next_action, z), -1)
            target_f = torch.stack([f(x) for f in self.target_F])
            target_b = self.target_B(m._states(ns))
        f, b = m.forward_map(s, a, z), m.backward_map(ns)
        loss, metrics = fb_loss(f, b, target_f, target_b, discount,
                               ortho_coef=self.cfg.orthonormalization_coefficient, pessimism=self.cfg.fb_pessimism)
        norm = checked_step(loss, self.fb_optimizer, self.fb_parameters, self.cfg.gradient_clip_norm)
        metrics.update({"fb_loss": loss, "gradient_norm": norm,
                        "forward_prediction_norm": f.norm(dim=-1).mean(),
                        "backward_embedding_norm": b.norm(dim=-1).mean(), "latent_norm": z.norm(dim=-1).mean()})
        metrics.update(self.update_actor(s, z))
        self.soft_update()
        self.step += 1
        result = {k: float(v.detach()) for k, v in metrics.items()}
        if not all(math.isfinite(v) for v in result.values()):
            raise FloatingPointError(f"Non-finite metric at step {self.step}: {result}")
        return result
