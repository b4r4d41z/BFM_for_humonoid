from __future__ import annotations

import math
from contextlib import ContextDecorator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal


##########################
# Initialization utils
##########################


def weight_init(module: nn.Module) -> None:
    """
    Orthogonal init for Linear layers, zero bias.

    This is a simplified and stable subset of the initialization style
    used in metamotivo/nn_models.py.
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def soft_update_params(net: nn.Module, target_net: nn.Module, tau: float) -> None:
    """
    Soft-update target network parameters:
        target = (1 - tau) * target + tau * net
    """
    tau = float(min(max(tau, 0.0), 1.0))
    with torch.no_grad():
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(param.data, alpha=tau)


class eval_mode(ContextDecorator):
    """
    Temporarily switch one or more modules to eval mode.
    """

    def __init__(self, *models: nn.Module) -> None:
        self.models = models
        self.prev_states: list[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: Any) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Small helper modules
##########################


class Norm(nn.Module):
    """
    Normalize the last dimension to sqrt(dim), same idea as metamotivo.
    Useful later for latent z if needed.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class TruncatedNormal(pyd.Normal):
    """
    Normal distribution with output clamped to [low, high].
    Useful later if you decide to move from deterministic action prediction
    to stochastic policy outputs.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        low: float = -1.0,
        high: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: torch.Tensor) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(
        self,
        clip: float | None = None,
        sample_shape: torch.Size = torch.Size(),
    ) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


##########################
# Configs
##########################


@dataclass
class MLPConfig:
    hidden_dim: int = 256
    hidden_layers: int = 2
    use_layernorm: bool = False
    activation: str = "relu"   # {"relu", "tanh", "mish", "gelu"}
    dropout: float = 0.0


@dataclass
class VisionEncoderConfig:
    """
    Reserved for later vision integration.
    The default encoder is intentionally lightweight.
    """
    out_dim: int = 128
    channels: tuple[int, ...] = (32, 64, 128)
    kernel_size: int = 3
    stride: int = 2


##########################
# Activation factory
##########################


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "mish":
        return nn.Mish()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


##########################
# Basic building blocks
##########################


class MLPBlock(nn.Module):
    """
    A single linear -> optional norm -> activation -> optional dropout block.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str = "relu",
        use_layernorm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]

        if use_layernorm:
            layers.append(nn.LayerNorm(out_dim))

        layers.append(get_activation(activation))

        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Simple residual MLP block.
    Useful later if you decide to move from a plain MLP to a deeper network.
    """

    def __init__(
        self,
        dim: int,
        *,
        activation: str = "mish",
        use_layernorm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(dim, dim))
        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MLP(nn.Module):
    """
    Generic MLP:
        input -> hidden blocks -> output linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
        *,
        output_activation: str | None = None,
    ) -> None:
        super().__init__()

        if cfg.hidden_layers < 1:
            raise ValueError(f"hidden_layers must be >= 1, got {cfg.hidden_layers}")

        layers: list[nn.Module] = []

        in_dim = input_dim
        for _ in range(cfg.hidden_layers):
            layers.append(
                MLPBlock(
                    in_dim,
                    cfg.hidden_dim,
                    activation=cfg.activation,
                    use_layernorm=cfg.use_layernorm,
                    dropout=cfg.dropout,
                )
            )
            in_dim = cfg.hidden_dim

        layers.append(nn.Linear(in_dim, output_dim))

        if output_activation is not None:
            layers.append(get_activation(output_activation))

        self.net = nn.Sequential(*layers)
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


##########################
# Encoders
##########################


class StateEncoder(nn.Module):
    """
    Encoder for proprio/state vectors, e.g. obs.state.full.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.net = MLP(input_dim=input_dim, output_dim=output_dim, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleVisionEncoder(nn.Module):
    """
    Lightweight CNN encoder for images shaped [B, H, W, C] or [B, C, H, W].

    This is not required for the very first proprio-only baseline,
    but it is convenient to already have it in model_blocks.py.
    """

    def __init__(self, in_channels: int = 3, cfg: VisionEncoderConfig = VisionEncoderConfig()) -> None:
        super().__init__()

        c1, c2, c3 = cfg.channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=cfg.kernel_size, stride=cfg.stride, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(c3, cfg.out_dim)

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        - [B, H, W, C]
        - [B, C, H, W]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor with ndim=4, got shape {tuple(x.shape)}")

        if x.shape[-1] in (1, 3, 4):
            x = x.permute(0, 3, 1, 2).contiguous()

        x = x.float()
        if x.max() > 1.0:
            x = x / 255.0

        feat = self.conv(x).flatten(start_dim=1)
        return self.proj(feat)


class MultiViewVisionEncoder(nn.Module):
    """
    Encode multi-camera input:
        head, left_wrist, right_wrist

    Returns one fused embedding by concatenation + projection.
    """

    def __init__(
        self,
        view_encoder_out_dim: int = 128,
        fused_out_dim: int = 256,
        view_cfg: VisionEncoderConfig = VisionEncoderConfig(),
        fusion_cfg: MLPConfig = MLPConfig(hidden_dim=256, hidden_layers=2),
    ) -> None:
        super().__init__()

        internal_view_cfg = VisionEncoderConfig(
            out_dim=view_encoder_out_dim,
            channels=view_cfg.channels,
            kernel_size=view_cfg.kernel_size,
            stride=view_cfg.stride,
        )

        self.head_encoder = SimpleVisionEncoder(in_channels=3, cfg=internal_view_cfg)
        self.left_encoder = SimpleVisionEncoder(in_channels=3, cfg=internal_view_cfg)
        self.right_encoder = SimpleVisionEncoder(in_channels=3, cfg=internal_view_cfg)

        self.fusion = MLP(
            input_dim=3 * view_encoder_out_dim,
            output_dim=fused_out_dim,
            cfg=fusion_cfg,
        )

    def forward(
        self,
        head: torch.Tensor,
        left_wrist: torch.Tensor,
        right_wrist: torch.Tensor,
    ) -> torch.Tensor:
        head_feat = self.head_encoder(head)
        left_feat = self.left_encoder(left_wrist)
        right_feat = self.right_encoder(right_wrist)

        fused = torch.cat([head_feat, left_feat, right_feat], dim=-1)
        return self.fusion(fused)


##########################
# Fusion blocks
##########################


class ConcatFusion(nn.Module):
    """
    Simple feature fusion by concatenation followed by MLP projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.net = MLP(input_dim=input_dim, output_dim=output_dim, cfg=cfg)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if len(xs) == 0:
            raise ValueError("ConcatFusion received no tensors")
        return self.net(torch.cat(xs, dim=-1))


##########################
# Policy / prediction heads
##########################


class DeterministicPolicyHead(nn.Module):
    """
    Deterministic action head.
    Good default for the first offline behavior cloning baseline.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        cfg: MLPConfig,
        *,
        squash_output: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = MLP(input_dim=input_dim, output_dim=action_dim, cfg=cfg)
        self.squash_output = squash_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action = self.backbone(x)
        if self.squash_output:
            action = torch.tanh(action)
        return action


class GaussianPolicyHead(nn.Module):
    """
    Optional stochastic policy head.
    Not necessary for the first baseline, but useful later.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        cfg: MLPConfig,
        *,
        init_std: float = 0.2,
        squash_mean: bool = True,
    ) -> None:
        super().__init__()
        self.mu_net = MLP(input_dim=input_dim, output_dim=action_dim, cfg=cfg)
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))
        self.squash_mean = squash_mean

    def forward(self, x: torch.Tensor) -> TruncatedNormal:
        mu = self.mu_net(x)
        if self.squash_mean:
            mu = torch.tanh(mu)
        std = self.log_std.exp().expand_as(mu)
        return TruncatedNormal(mu, std)


class ValueHead(nn.Module):
    """
    Simple scalar value / critic head.
    Useful later for fb_cpr-style extensions.
    """

    def __init__(
        self,
        input_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.net = MLP(input_dim=input_dim, output_dim=1, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BinaryDiscriminatorHead(nn.Module):
    """
    Binary discriminator head.
    Useful later for expert / latent-conditioned discrimination.
    """

    def __init__(
        self,
        input_dim: int,
        cfg: MLPConfig,
    ) -> None:
        super().__init__()
        self.net = MLP(input_dim=input_dim, output_dim=1, cfg=cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))

    def compute_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def compute_reward(self, x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(x)
        s = torch.clamp(s, eps, 1.0 - eps)
        return s.log() - (1.0 - s).log()


##########################
# Factory helpers
##########################


def build_state_encoder(
    input_dim: int,
    output_dim: int,
    cfg: MLPConfig,
) -> StateEncoder:
    return StateEncoder(input_dim=input_dim, output_dim=output_dim, cfg=cfg)


def build_deterministic_policy_head(
    input_dim: int,
    action_dim: int,
    cfg: MLPConfig,
    *,
    squash_output: bool = False,
) -> DeterministicPolicyHead:
    return DeterministicPolicyHead(
        input_dim=input_dim,
        action_dim=action_dim,
        cfg=cfg,
        squash_output=squash_output,
    )


def build_gaussian_policy_head(
    input_dim: int,
    action_dim: int,
    cfg: MLPConfig,
    *,
    init_std: float = 0.2,
    squash_mean: bool = True,
) -> GaussianPolicyHead:
    return GaussianPolicyHead(
        input_dim=input_dim,
        action_dim=action_dim,
        cfg=cfg,
        init_std=init_std,
        squash_mean=squash_mean,
    )


def build_value_head(
    input_dim: int,
    cfg: MLPConfig,
) -> ValueHead:
    return ValueHead(input_dim=input_dim, cfg=cfg)


def build_binary_discriminator_head(
    input_dim: int,
    cfg: MLPConfig,
) -> BinaryDiscriminatorHead:
    return BinaryDiscriminatorHead(input_dim=input_dim, cfg=cfg)