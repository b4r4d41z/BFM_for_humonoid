"""Small MLP heads for Vanilla FB; no imitation objective."""
import torch
from torch import nn


def mlp(input_dim, output_dim, hidden_dim, hidden_layers):
    layers = []
    for i in range(hidden_layers):
        layers += [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class LatentActor(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, hidden_dim, hidden_layers):
        super().__init__()
        self.net = mlp(state_dim + latent_dim, action_dim, hidden_dim, hidden_layers)

    def forward(self, states, z, *, std=0., noise_clip=0.3):
        mean = self.net(torch.cat((states, z), -1)).tanh()
        if std == 0:
            return mean
        noise = (torch.randn_like(mean) * std).clamp(-noise_clip, noise_clip)
        raw = mean + noise
        # Reference TD3-style bounded Gaussian: clipped samples, straight-through
        # boundary gradient. This is not a learned-variance SAC policy.
        return raw + (raw.clamp(-1 + 1e-6, 1 - 1e-6) - raw).detach()
