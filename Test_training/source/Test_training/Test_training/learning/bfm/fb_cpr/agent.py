from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import autograd

from ..fb.agent import FBAgent
from ..fb.agent import TrainConfig as FBTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import Config as FBcprModelConfig
from .model import FBcprModel, config_from_dict


@dataclasses.dataclass
class TrainConfig(FBTrainConfig):
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5

    reg_coeff: float = 1.0
    scale_reg: bool = True

    # for your current project:
    # - use_cpr=False: offline HDF5 training without expert_slicer/train split
    # - use_cpr=True: later, when you have separate expert/train buffers and sequence slicing
    use_cpr: bool = False

    # if z is already present in batch and you want to use it
    use_batch_z_if_available: bool = True

    # how to create z when batch has no z:
    # "backward_next_obs" -> z = project_z(B(next_obs))
    # "random"           -> random latent
    # "mixed"            -> mixture of random z and backward(next_obs)
    z_strategy: str = "backward_next_obs"
    random_z_ratio: float = 0.3

    # optional relabeling of z inside batch
    relabel_ratio: float | None = 1.0

    # discriminator regularization, only used when use_cpr=True
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0


@dataclasses.dataclass
class Config:
    model: FBcprModelConfig = dataclasses.field(default_factory=FBcprModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    cudagraphs: bool = False
    compile: bool = False


class FBcprAgent(FBAgent):
    def __init__(self, **kwargs):
        model_cfg = kwargs.get("model", {})
        train_cfg = kwargs.get("train", {})

        seq_length = int(model_cfg.get("seq_length", 1))
        batch_size = int(train_cfg.get("batch_size", 256))
        use_cpr = bool(train_cfg.get("use_cpr", False))

        # only force divisibility by seq_length when expert sequence encoding is enabled
        if use_cpr and seq_length > 1:
            batch_size = int(torch.ceil(torch.tensor([batch_size / seq_length])) * seq_length)
            kwargs.setdefault("train", {})
            kwargs["train"]["batch_size"] = batch_size

        self.cfg = config_from_dict(kwargs, Config)
        self._model = FBcprModel(**dataclasses.asdict(self.cfg.model))
        self._model.to(self.cfg.model.device)

        self.setup_training()
        self.setup_compile()

    def setup_training(self) -> None:
        super().setup_training()

        self._critic_map_paramlist = ()
        self._target_critic_map_paramlist = ()
        self.critic_optimizer = None
        self.discriminator_optimizer = None

        if hasattr(self._model, "_critic") and hasattr(self._model, "_target_critic"):
            self._critic_map_paramlist = tuple(self._model._critic.parameters())
            self._target_critic_map_paramlist = tuple(self._model._target_critic.parameters())

            self.critic_optimizer = torch.optim.Adam(
                self._model._critic.parameters(),
                lr=self.cfg.train.lr_critic,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay,
            )

        if hasattr(self._model, "_discriminator"):
            self.discriminator_optimizer = torch.optim.Adam(
                self._model._discriminator.parameters(),
                lr=self.cfg.train.lr_discriminator,
                capturable=self.cfg.cudagraphs and not self.cfg.compile,
                weight_decay=self.cfg.train.weight_decay_discriminator,
            )

    def setup_compile(self):
        super().setup_compile()

        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None

            self.update_actor = torch.compile(self.update_actor, mode=mode)

            if self.cfg.train.use_cpr:
                self.update_critic = torch.compile(self.update_critic, mode=mode)
                self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
                self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_actor = CudaGraphModule(self.update_actor, warmup=5)

            if self.cfg.train.use_cpr:
                self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
                self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
                self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    def _sample_train_batch(self, replay_buffer):
        batch_size = self.cfg.train.batch_size

        # case 1: replay_buffer itself has .sample(...)
        if hasattr(replay_buffer, "sample"):
            return replay_buffer.sample(batch_size)

        # case 2: replay_buffer is a dict-like object with "train"
        if isinstance(replay_buffer, Mapping):
            if "train" in replay_buffer and hasattr(replay_buffer["train"], "sample"):
                return replay_buffer["train"].sample(batch_size)

        raise ValueError(
            "Could not sample train batch. Expected either "
            "replay_buffer.sample(batch_size) or replay_buffer['train'].sample(batch_size)."
        )

    def _sample_expert_batch(self, replay_buffer):
        batch_size = self.cfg.train.batch_size

        if isinstance(replay_buffer, Mapping):
            if "expert_slicer" in replay_buffer and hasattr(replay_buffer["expert_slicer"], "sample"):
                return replay_buffer["expert_slicer"].sample(batch_size)
            if "expert" in replay_buffer and hasattr(replay_buffer["expert"], "sample"):
                return replay_buffer["expert"].sample(batch_size)

        return None

    def _extract_batch_tensors(
        self,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs = batch["observation"].to(self.device)
        action = batch["action"].to(self.device)
        next_obs = batch["next"]["observation"].to(self.device)

        if "done" in batch:
            done = batch["done"].to(self.device)
        elif "next" in batch and "terminated" in batch["next"]:
            done = batch["next"]["terminated"].to(self.device)
        else:
            done = torch.zeros((obs.shape[0], 1), device=self.device, dtype=torch.bool)

        if done.ndim == 1:
            done = done.unsqueeze(-1)

        done = done.bool()
        discount = self.cfg.train.discount * (~done).float()

        return obs, action, next_obs, discount

    @torch.no_grad()
    def _normalize_obs(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._model._obs_normalizer(obs)
        self._model._obs_normalizer(next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            obs = self._model._obs_normalizer(obs)
            next_obs = self._model._obs_normalizer(next_obs)

        return obs, next_obs

    @torch.no_grad()
    def _normalize_single_obs(self, obs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            return self._model._obs_normalizer(obs)

    @torch.no_grad()
    def _sample_random_z(self, batch_size: int) -> torch.Tensor:
        return self._model.sample_z(batch_size, device=self.device)

    @torch.no_grad()
    def _sample_goal_z(self, next_obs: torch.Tensor) -> torch.Tensor:
        z = self._model._backward_map(next_obs)
        z = self._model.project_z(z)
        return z

    @torch.no_grad()
    def _sample_train_z_from_next_obs(self, next_obs: torch.Tensor) -> torch.Tensor:
        strategy = self.cfg.train.z_strategy
        batch_size = next_obs.shape[0]

        if strategy == "random":
            return self._sample_random_z(batch_size)

        if strategy == "backward_next_obs":
            return self._sample_goal_z(next_obs)

        if strategy == "mixed":
            goal_z = self._sample_goal_z(next_obs)
            random_z = self._sample_random_z(batch_size)
            mask = (torch.rand((batch_size, 1), device=self.device) < self.cfg.train.random_z_ratio)
            return torch.where(mask, random_z, goal_z)

        raise ValueError(
            f"Unknown z_strategy='{strategy}'. "
            "Supported values: 'backward_next_obs', 'random', 'mixed'."
        )

    @torch.no_grad()
    def _get_train_z(self, batch: dict[str, Any], next_obs: torch.Tensor) -> torch.Tensor:
        if self.cfg.train.use_batch_z_if_available and "z" in batch:
            z = batch["z"].to(self.device)
            return self._model.project_z(z)

        return self._sample_train_z_from_next_obs(next_obs)

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor):
        if self.cfg.model.seq_length <= 1:
            z_expert = self._model._backward_map(next_obs).detach()
            z_expert = self._model.project_z(z_expert)
            return z_expert

        batch_size = next_obs.shape[0]
        seq_length = self.cfg.model.seq_length

        if batch_size % seq_length != 0:
            raise ValueError(
                f"Expert batch size ({batch_size}) must be divisible by seq_length ({seq_length})."
            )

        B_expert = self._model._backward_map(next_obs).detach()  # batch x d
        B_expert = B_expert.view(batch_size // seq_length, seq_length, B_expert.shape[-1])  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, seq_length, dim=0)  # batch x d
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        train_batch = self._sample_train_batch(replay_buffer)
        train_obs, train_action, train_next_obs, discount = self._extract_batch_tensors(train_batch)

        train_obs, train_next_obs = self._normalize_obs(train_obs, train_next_obs)
        train_z = self._get_train_z(train_batch, train_next_obs)

        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

        metrics: Dict[str, torch.Tensor] = {}

        # optional CPR branch for a later stage of your project
        if self.cfg.train.use_cpr:
            expert_batch = self._sample_expert_batch(replay_buffer)
            if expert_batch is None:
                raise ValueError(
                    "use_cpr=True, but no expert batch source was found. "
                    "Expected replay_buffer['expert_slicer'] or replay_buffer['expert']."
                )

            expert_obs, _, expert_next_obs, _ = self._extract_batch_tensors(expert_batch)
            expert_obs, expert_next_obs = self._normalize_obs(expert_obs, expert_next_obs)
            expert_z = self.encode_expert(next_obs=expert_next_obs)

            grad_penalty = (
                self.cfg.train.grad_penalty_discriminator
                if self.cfg.train.grad_penalty_discriminator > 0
                else None
            )

            metrics.update(
                self.update_discriminator(
                    expert_obs=expert_obs,
                    expert_z=expert_z,
                    train_obs=train_obs,
                    train_z=train_z,
                    grad_penalty=grad_penalty,
                )
            )

            if hasattr(self, "z_buffer"):
                z_new = self._sample_train_z_from_next_obs(train_next_obs).clone()
                self.z_buffer.add(z_new)

                if self.cfg.train.relabel_ratio is not None:
                    mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
                    train_z = torch.where(mask, z_new, train_z)

        else:
            if self.cfg.train.relabel_ratio is not None:
                z_new = self._sample_train_z_from_next_obs(train_next_obs)
                mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
                train_z = torch.where(mask, z_new, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )

        if self.cfg.train.use_cpr:
            metrics.update(
                self.update_critic(
                    obs=train_obs,
                    action=train_action,
                    discount=discount,
                    next_obs=train_next_obs,
                    z=train_z,
                )
            )

        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )

            if self.cfg.train.use_cpr and len(self._critic_map_paramlist) > 0:
                _soft_update_params(
                    self._critic_map_paramlist,
                    self._target_critic_map_paramlist,
                    self.cfg.train.critic_target_tau,
                )

        return metrics

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor,
        real_z: torch.Tensor,
        fake_obs: torch.Tensor,
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=real_obs.device)

        interpolates = torch.cat(
            [
                (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
                (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True),
            ],
            dim=1,
        )

        d_interpolates = self._model._discriminator.compute_logits(
            interpolates[:, 0:real_obs.shape[1]],
            interpolates[:, real_obs.shape[1]:],
        )

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor,
        expert_z: torch.Tensor,
        train_obs: torch.Tensor,
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        if self.discriminator_optimizer is None:
            raise RuntimeError("Discriminator optimizer is not initialized.")

        expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
        unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)

        expert_loss = -F.logsigmoid(expert_logits)
        unlabeled_loss = F.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        wgan_gp = None
        if grad_penalty is not None:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss = loss + grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean(),
                "disc_train_loss": unlabeled_loss.detach().mean(),
            }
            if wgan_gp is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()

        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.critic_optimizer is None:
            raise RuntimeError("Critic optimizer is not initialized.")

        num_parallel = self.cfg.model.archi.critic.num_parallel

        with torch.no_grad():
            reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)

            next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(
                next_Qs,
                self.cfg.train.critic_pessimism_penalty,
            )

            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        Qs = self._model._critic(obs, z, action)  # num_parallel x batch x 1
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }

        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        sampled_action = dist.sample(clip=self.cfg.train.stddev_clip)

        # FB part: always active
        Fs = self._model._forward_map(obs, z, sampled_action)  # num_parallel x batch x z_dim
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)

        Q_discriminator = None
        if self.cfg.train.use_cpr:
            Qs_discriminator = self._model._critic(obs, z, sampled_action)  # num_parallel x batch x 1
            _, _, Q_discriminator = self.get_targets_uncertainty(
                Qs_discriminator,
                self.cfg.train.actor_pessimism_penalty,
            )

            weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
            actor_loss = -Q_fb.mean() - self.cfg.train.reg_coeff * weight * Q_discriminator.mean()
        else:
            actor_loss = -Q_fb.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)

        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
            if Q_discriminator is not None:
                output_metrics["Q_discriminator"] = Q_discriminator.mean().detach()

        return output_metrics