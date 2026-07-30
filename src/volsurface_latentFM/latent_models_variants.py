"""
LatentFlowMatch — three interpolation variants in one class.

path='linear'    Constant velocity ODE  (Optimal Transport)
                 z_t = (1-t)x z_0 + t x z_1
                 v   = z_1 - z_0                  constant, t-independent

path='quadratic' Quadratic ODE  (accelerating path)
                 z_t = (1-t^2)xz_0 + t^2 x z_1
                 v   = 2t x (z_1 - z_0)             increases with t, slow start

path='vp'        Variance-Preserving probability flow ODE
                 z_t = tx z_1 + sqrt(1-t^2)xz_0        ||z_t|| approximately preserved
                 v   = z_1 - t/sqrt(1-t^2)xz_0          from Theorem 3 of Lipman et al.

All three share identical fit() / sample() / save() / load() interfaces.
The ODE is integrated with Euler steps in all cases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

"""
UNet-style MLP for latent-space generative models
===================================================
Replaces LatentMLP for FM / DSBM / DDPM in latent space.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Time embedding  
class SinusoidalEmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (math.log(10000) / (half - 1))
        )
        x = t[:, None] * freqs[None, :]
        return torch.cat([x.sin(), x.cos()], dim=-1)   # (B, dim)


# Adaptive Layer Normalisation block
class AdaLN(nn.Module):
    """
    Adaptive Layer Normalisation.
    Projects t_emb to (scale, shift) that modulate the normalised activation.

    y = scale(t) * LayerNorm(x) + shift(t)

    This is strictly more powerful than concatenating t at the input because
    the time embedding controls the gain AND bias of every feature, not just
    adds extra input channels once.
    """
    def __init__(self, feature_dim: int, t_emb_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim, elementwise_affine=False)
        # Project t_emb to scale + shift for this layer
        self.proj = nn.Sequential(
            nn.Linear(t_emb_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, x, t_emb):
        """
        x     : (B, feature_dim)
        t_emb : (B, t_emb_dim)
        """
        params       = self.proj(t_emb)                          # (B, 2*d)
        scale, shift = params.chunk(2, dim=-1)                   # (B, d) each
        return (1 + scale) * self.norm(x) + shift


# UNet block: Linear to SiLU to AdaLN
class UNetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, t_emb_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act    = nn.SiLU()
        self.ada_ln = AdaLN(out_dim, t_emb_dim)

    def forward(self, x, t_emb):
        return self.ada_ln(self.act(self.linear(x)), t_emb)



# UNet MLP  — drop-in replacement for LatentMLP
class LatentUNet(nn.Module):
    """
    UNet-style MLP velocity network for latent FM / DSBM / DDPM.

    Parameters
    ----------
    z_dim     : latent dimension (input and output size)
    hidden    : tuple of hidden sizes for each encoder level.
                Decoder mirrors them in reverse.
                e.g. (256, 512, 256) → 3 encoder levels, bottleneck at 512
    t_emb_dim : sinusoidal time embedding dimension
    """

    def __init__(self, z_dim: int, hidden=(64, 128, 64), t_emb_dim: int = 64):
        super().__init__()
        assert len(hidden) >= 2, "Need at least 2 hidden levels for skip connections"

        self.z_dim     = z_dim
        self.hidden    = hidden
        self.t_emb_dim = t_emb_dim
        self.t_embed   = SinusoidalEmbed(t_emb_dim)

        #  Encoder
        enc_dims = [z_dim] + list(hidden)
        self.encoder = nn.ModuleList([
            UNetBlock(enc_dims[i], enc_dims[i+1], t_emb_dim)
            for i in range(len(hidden))
        ])

        #  Decoder (mirrors encoder, skip connections double the input) 
        # Skip connections: all encoder levels except the bottleneck
        # dec level i receives: output of previous dec level
        #                     + skip from encoder level (n-i-2)
        dec_in  = []
        dec_out = []
        for i in range(len(hidden) - 1):
            skip_dim = enc_dims[-(i+2)]      # encoder level to skip from
            in_d     = hidden[-(i+1)] + skip_dim if i > 0 else hidden[-1] + skip_dim
            # Fix: track properly
            dec_in.append(in_d)
            dec_out.append(hidden[-(i+2)] if i < len(hidden)-2 else hidden[0])

        # Rebuild cleanly
        self.decoder = nn.ModuleList()
        prev = hidden[-1]                    # start from bottleneck output
        for i in range(len(hidden) - 1):
            skip_dim = enc_dims[-(i+2)]      # symmetric skip from encoder
            self.decoder.append(
                UNetBlock(prev + skip_dim, hidden[-(i+2)], t_emb_dim)
            )
            prev = hidden[-(i+2)]

        #  Output projection
        self.out = nn.Linear(prev, z_dim)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"LatentUNet  |  z_dim={z_dim}  hidden={hidden}  "
              f"t_emb_dim={t_emb_dim}  params={n_p:,}")

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z : (B, z_dim)
        t : (B,)  time in [0, 1]
        Returns velocity (B, z_dim)
        """
        t_emb = self.t_embed(t)              # (B, t_emb_dim)

        #  Encoder forward, cache skip activations 
        skips = []
        x     = z
        for i, block in enumerate(self.encoder):
            x = block(x, t_emb)
            if i < len(self.encoder) - 1:   # all except bottleneck
                skips.append(x)

        # Decoder forward with skip connections 
        for i, block in enumerate(self.decoder):
            skip = skips[-(i+1)]            # pop skips in reverse order
            x    = torch.cat([x, skip], dim=-1)
            x    = block(x, t_emb)

        return self.out(x)


# Keep original LatentMLP for backwards compatibility
class LatentMLP(nn.Module):
    """Original flat MLP — kept for comparison and backwards compatibility."""
    def __init__(self, z_dim, hidden=(128, 128, 128), t_emb_dim=32):
        super().__init__()
        self.t_embed = SinusoidalEmbed(t_emb_dim)
        layers, prev = [], z_dim + t_emb_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev    = h
        layers.append(nn.Linear(prev, z_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        return self.net(torch.cat([z, self.t_embed(t)], dim=-1))
    
# Main class
class LatentFlowMatch(nn.Module):
    """
    Unified Flow Matching with three path choices.

    Parameters
    ----------
    z_dim     : latent dimension  (must match frozen AE)
    hidden    : MLP hidden layer sizes
    ode_steps : Euler integration steps at sampling
    path      : 'linear' | 'quadratic' | 'vp'
    eps       : small constant to avoid division by zero in VP path near t=1
    """

    PATHS = ('linear', 'quadratic', 'vp', 'trigo')

    def __init__(self, z_dim=8, hidden=(64, 128, 64),
                 t_emb_dim=32, ode_steps=100, path='linear', eps=1e-3):
        super().__init__()
        assert path in self.PATHS, f"path must be one of {self.PATHS}"
        self.z_dim     = z_dim
        self.ode_steps = ode_steps
        self.path      = path
        self.eps       = eps
        self.net       = LatentUNet(z_dim, hidden, t_emb_dim=t_emb_dim)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"LatentFlowMatch  |  path={path}  z_dim={z_dim}  "
              f"ode_steps={ode_steps}  params={n_p:,}")

    #  Interpolation and target velocity 
    def _interpolate(self, z0, z1, t):
        """
        Returns (z_t, v_target) for each path choice.

        z0, z1 : (B, z_dim)  noise and data samples
        t      : (B,)        time in [0, 1]

        Linear (OT-FM)
        ──────────────
          z_t = (1-t) x z_0 + tx z_1
          v   = z_1 - z_0                      constant along each trajectory

        Quadratic
        ─────────
          z_t = (1-t^2)xz_0 + t^2xz_1
          v   = dz_t/dt = 2tx(z_1 - z_0)       accelerates toward data
          The path lingers near z_0 early and rushes to z_1 late.
          Useful when the data manifold is far from the noise distribution.

        Variance Preserving (probability flow ODE)
        ──────────────────────────────────────────
          z_t = txz_1 + sqrt(1-t^2)xz_0
          v   = dz_t/dt = z_1 - t/sqrt(1-t^2)xz_0
        ||z_t|| is approximately preserved: E[||z_t||^2] = E[||z_0||^2] = d            
        """
        t_ = t[:, None]                         # (B, 1) for broadcasting

        if self.path == 'linear':
            z_t     = (1.0 - t_) * z0 + t_ * z1
            v_target = z1 - z0

        elif self.path == 'quadratic':
            t2       = t_ ** 2
            z_t      = (1.0 - t2) * z0 + t2 * z1
            v_target = 2.0 * t_ * (z1 - z0)

        elif self.path == 'vp':
            sigma_t  = torch.sqrt(torch.clamp(1.0 - t_ ** 2, min=self.eps))
            z_t      = t_ * z1 + sigma_t * z0
            v_target = z1 - (t_ / sigma_t) * z0

        elif self.path == 'trigo':
            alpha = torch.cos(0.5 * torch.pi * t_)
            sigma = torch.sin(0.5 * torch.pi * t_)

            z_t = alpha * z0 + sigma * z1
            v_target = (
                    -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t_) * z0
                    +0.5 * torch.pi * torch.cos(0.5 * torch.pi * t_) * z1
                )

        return z_t, v_target

    #  Training

    def fit(self, z_codes, n_epochs=4000, batch_size=16, lr=3e-4,
            print_every=500, device=None):
        """
        z_codes : (n, z_dim) numpy array of training latent codes.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        z1  = torch.tensor(z_codes, dtype=torch.float32).to(dev)
        ldr = DataLoader(TensorDataset(z1), batch_size=batch_size,
                         shuffle=True, drop_last=False)
        opt  = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []

        for epoch in range(1, n_epochs + 1):
            self.train()
            ep_loss = 0.

            for (z1b,) in ldr:
                B  = len(z1b)
                # Sample time uniformly; avoid t=1 for VP (singularity)
                if self.path == 'vp':
                    t = torch.rand(B, device=dev) * (1.0 - self.eps)
                else:
                    t = torch.rand(B, device=dev)

                z0       = torch.randn_like(z1b)
                zt, vtgt = self._interpolate(z0, z1b, t)
                vpred    = self.net(zt, t)
                loss     = F.mse_loss(vpred, vtgt)

                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item()

            hist.append(ep_loss / len(ldr))
            if epoch % print_every == 0 or epoch == 1:
                print(f"  [{self.path.upper():>9}]  "
                      f"Ep {epoch:>5}/{n_epochs}  loss={hist[-1]:.6f}")

        return hist

    #  Sampling

    @torch.no_grad()
    def sample(self, n, device=None):
        """
        Euler ODE integration from z₀ ~ N(0,I) at t=0 to z₁ at t=1.
        Returns (n, z_dim) numpy array.
        """
        dev = device or next(self.parameters()).device
        self.eval()

        z  = torch.randn(n, self.z_dim, device=dev)
        dt = 1.0 / self.ode_steps

        for i in range(self.ode_steps):
            t_val = i / self.ode_steps
            # Clamp near t=1 for VP to avoid singularity
            if self.path == 'vp':
                t_val = min(t_val, 1.0 - self.eps)
            t = torch.full((n,), t_val, device=dev)
            z = z + dt * self.net(z, t)

        return z.cpu().numpy()

    #  Persistence 

    def save(self, path):
        torch.save(dict(
            state_dict = self.state_dict(),
            z_dim      = self.z_dim,
            hidden     = self.net.hidden,
            t_emb_dim  = self.net.t_emb_dim,
            ode_steps  = self.ode_steps,
            path       = self.path,
            eps        = self.eps,
        ), path)
        print(f"  Saved FM ({self.path}) → {path}")

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        m  = cls(z_dim     = ck["z_dim"],
                 hidden    = ck.get("hidden", (64, 128, 64)),
                 t_emb_dim = ck.get("t_emb_dim", 32),
                 ode_steps = ck["ode_steps"],
                 path      = ck["path"],
                 eps       = ck.get("eps", 1e-3))
        m.load_state_dict(ck["state_dict"])
        return m