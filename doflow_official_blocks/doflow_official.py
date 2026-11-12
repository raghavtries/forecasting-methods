"""Copied from https://github.com/StatFusion/DoFlow_Causal_Time_Series (main_treatment.py)."""
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 30, num_layers: int = 3, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )

    def forward(self, x: torch.Tensor, hidden: tuple = None):
        out, (h_n, c_n) = self.lstm(x, hidden)
        h_top = h_n[-1]  # (B, H)
        return out, h_top, (h_n, c_n)


class CNF(nn.Module):
    def __init__(self, data_dim: int, cond_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.data_dim = data_dim
        self.cond_dim = cond_dim

        layers = []
        in_dim = data_dim + cond_dim + 1 # +1 for flow time s \in [0,1]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        x = states[:, :self.data_dim]
        cond  = states[:, self.data_dim:]
        inp = torch.cat([x, cond, s], dim=1)
        dx  = self.net(inp)
        return dx
    
    @torch.no_grad()
    def sample(self, cond_states, num_steps=20, method='rk4', forward=False, density=False, z0=None):
        # Minimal sampler compatible with training API: Euler fallback for simplicity.
        # Uses fixed-step integration over s in [0,1].
        B = cond_states.size(0)
        D = self.data_dim
        device = cond_states.device
        if z0 is None:
            y = torch.randn(B, D, device=device)
        else:
            y = z0

        # integrate ds (reverse-time for generative sampling)
        steps = num_steps
        dt = 1.0 / steps
        for k in range(steps):
            # reverse-time: s = 1 - t, so use sign = -1
            s = (1.0 - k * dt) if not forward else (k * dt)
            s_t = torch.full((B, 1), s, device=device)
            dy = self.forward(s_t, torch.cat([y, cond_states], dim=1))
            y = y - dy * dt if not forward else y + dy * dt
        return y, None

