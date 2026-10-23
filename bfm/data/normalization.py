import torch
from torch import nn


class StateNormalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, training_states):
        x = torch.as_tensor(training_states, dtype=torch.float64, device=self.mean.device)
        if x.ndim != 2 or len(x) < 2 or x.shape[-1] != len(self.mean) or not torch.isfinite(x).all():
            raise ValueError("Normalizer needs finite training states [N>=2,D]")
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0, correction=0).clamp_min(1e-3))

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean
