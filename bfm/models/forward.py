import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class BackwardEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = self.net(s)
        return Fnn.normalize(x, dim=-1)  # L2 normalizaton


class ForwardEncoder(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, goal_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        input_dim = state_dim + action_dim + goal_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a, z], dim=-1)
        return self.net(x)


# Тестирование (локальное)
if __name__ == "__main__":
    batch_size = 4
    state_dim = 16
    action_dim = 6
    goal_dim = 8
    hidden_dim = 64
    latent_dim = 32

    s = torch.randn(batch_size, state_dim)
    a = torch.randn(batch_size, action_dim)
    z = torch.randn(batch_size, goal_dim)

    B = BackwardEncoder(state_dim, hidden_dim, latent_dim)
    F = ForwardEncoder(state_dim, action_dim, goal_dim, hidden_dim, latent_dim)

    b_out = B(s)
    f_out = F(s, a, z)

    print("B(s):", b_out.shape, "(нормирован)")
    print("F(s,a,z):", f_out.shape)
