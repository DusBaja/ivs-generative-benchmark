"""
Stage 2: Latent-Space Generative Models
=========================================
Both models operate on latent codes z in R^d produced by a frozen IVS_AE.

  LatentDDPM       — denoising diffusion in latent space
  LatentFlowMatch  — flow matching  in latent space

Both share the same interface:
    model.fit(z_codes)
    z_new = model.sample(n)

Usage
-----
    ae = IVS_AE.load("checkpoints/ivs_ae.pt")
    z  = ae.encode(surf_train_n)              # (n_train, z_dim)

    # train whichever you want — or both
    ddpm = LatentDDPM(z_dim=ae.z_dim)
    ddpm.fit(z)

    fm = LatentFlowMatch(z_dim=ae.z_dim)
    fm.fit(z)

    # decode with the same frozen AE
    z_gen    = ddpm.sample(100)               # or fm.sample(100)
    surf_gen = ae.decode(z_gen)              # (100, 512) normalised
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



# Time embedding  (same as before)
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
    Projects t_emb → (scale, shift) that modulate the normalised activation.

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

    Compared to LatentMLP
    """

    def __init__(self, z_dim: int, hidden=(64, 128, 64), t_emb_dim: int = 32):
        super().__init__()
        assert len(hidden) >= 2, "Need at least 2 hidden levels for skip connections"

        self.z_dim     = z_dim
        self.hidden    = hidden
        self.t_emb_dim = t_emb_dim
        self.t_embed   = SinusoidalEmbed(t_emb_dim)

        # Encoder
        enc_dims = [z_dim] + list(hidden)
        self.encoder = nn.ModuleList([
            UNetBlock(enc_dims[i], enc_dims[i+1], t_emb_dim)
            for i in range(len(hidden))
        ])

        # ── Decoder (mirrors encoder, skip connections double the input) ──
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

        # Output projection 
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

        #  Decoder forward with skip connections 
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
    

# Latent DDPM
class LatentDDPM(nn.Module):
    """
    DDPM operating on latent codes z in R^z_dim.
    Uses a cosine noise schedule and x₀-prediction parameterisation
    (more stable than ε-prediction for small datasets).

    Parameters
    ----------
    z_dim   : latent dimension (must match the frozen AE)
    T       : diffusion steps
    hidden  : MLP hidden layer sizes
    """
    def __init__(self, z_dim=8, T=200, hidden=(64, 128, 64)):
        super().__init__()
        self.z_dim = z_dim
        self.T     = T
        self.net   = LatentUNet(z_dim, hidden, t_emb_dim=32)

        # Cosine schedule
        s       = 0.008
        steps   = torch.arange(T + 1)
        alpha_c = torch.cos((steps / T + s) / (1 + s) * np.pi / 2) ** 2
        alpha_c = alpha_c / alpha_c[0]
        betas   = (1 - alpha_c[1:] / alpha_c[:-1]).clamp(max=0.999)
        alphas  = 1 - betas
        alpha_bar = torch.cumprod(alphas, 0)

        self.register_buffer("betas",     betas)
        self.register_buffer("alphas",    alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"LatentDDPM  |  z_dim={z_dim}  T={T}  params={n_p:,}")

    def _noise(self, z0, t_idx):
        """Forward process: q(z_t | z_0)"""
        ab  = self.alpha_bar[t_idx].reshape(-1, 1)
        eps = torch.randn_like(z0)
        return ab.sqrt() * z0 + (1 - ab).sqrt() * eps, eps

    def fit(self, z_codes, n_epochs=2000, batch_size=16, lr=3e-4,
            print_every=200, device=None):
        """z_codes : (n, z_dim) numpy array"""
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        z   = torch.tensor(z_codes, dtype=torch.float32).to(dev)
        ldr = DataLoader(TensorDataset(z), batch_size=batch_size,
                         shuffle=True, drop_last=False)
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []

        for epoch in range(1, n_epochs + 1):
            self.train()
            ep_loss = 0.
            for (zb,) in ldr:
                t_idx = torch.randint(0, self.T, (len(zb),), device=dev)
                zt, _  = self._noise(zb, t_idx)
                t_norm = t_idx.float() / self.T                # ∈ [0,1]
                z0_hat = self.net(zt, t_norm)                  # predict z₀
                loss   = F.mse_loss(z0_hat, zb)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item()
            hist.append(ep_loss / len(ldr))
            if epoch % print_every == 0 or epoch == 1:
                print(f"  [DDPM] Ep {epoch:>5}/{n_epochs}  loss={hist[-1]:.5f}")
        return hist

    @torch.no_grad()
    def sample(self, n, device=None):
        """Ancestral sampling → (n, z_dim) numpy array"""
        dev = device or next(self.parameters()).device
        self.eval()
        z = torch.randn(n, self.z_dim, device=dev)
        for t_idx in reversed(range(self.T)):
            t_norm = torch.full((n,), t_idx / self.T, device=dev)
            z0_hat = self.net(z, t_norm)

            ab   = self.alpha_bar[t_idx]
            ab_p = self.alpha_bar[t_idx - 1] if t_idx > 0 else torch.ones(1, device=dev)
            beta = self.betas[t_idx]

            # DDPM posterior mean
            mu = (ab_p.sqrt() * beta / (1 - ab)) * z0_hat + \
                 ((1 - ab_p) * self.alphas[t_idx].sqrt() / (1 - ab)) * z

            if t_idx > 0:
                sigma = ((1 - ab_p) / (1 - ab) * beta).sqrt()
                z     = mu + sigma * torch.randn_like(mu)
            else:
                z = mu
        return z.cpu().numpy()

    def save(self, path):
        torch.save(dict(state_dict=self.state_dict(),
                        z_dim=self.z_dim, T=self.T), path)
        print(f"  Saved DDPM → {path}")

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        m  = cls(z_dim=ck["z_dim"], T=ck["T"])
        m.load_state_dict(ck["state_dict"]); return m


# Latent Flow Matching
class LatentFlowMatch(nn.Module):
    """
    Conditional Flow Matching (Lipman et al. 2022) in latent space.
    Learns a velocity field v(z_t, t) such that the ODE
        dz/dt = v(z_t, t),  z_0 following N(0,I)
    transports noise to the data distribution.

    Training: regress v on straight-line paths 
              target velocity = z_1 - z_0  (constant along path)

    Parameters
    ----------
    z_dim   : latent dimension (must match the frozen AE)
    hidden  : MLP hidden layer sizes
    ode_steps : number of Euler steps at sampling
    """
    def __init__(self, z_dim=8, hidden=(64, 128, 64), ode_steps=100):
        super().__init__()
        self.z_dim     = z_dim
        self.ode_steps = ode_steps
        self.net       = LatentUNet(z_dim, hidden, t_emb_dim=32)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"LatentFlowMatch  |  z_dim={z_dim}  ode_steps={ode_steps}  params={n_p:,}")

    def fit(self, z_codes, n_epochs=2000, batch_size=16, lr=3e-4,
            print_every=200, device=None):
        """z_codes : (n, z_dim) numpy array"""
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        z1  = torch.tensor(z_codes, dtype=torch.float32).to(dev)   # data
        ldr = DataLoader(TensorDataset(z1), batch_size=batch_size,
                         shuffle=True, drop_last=False)
        opt  = torch.optim.Adam(self.parameters(), lr=lr)
        hist = []

        for epoch in range(1, n_epochs + 1):
            self.train()
            ep_loss = 0.

            for (z1b,) in ldr:

                B  = len(z1b)
                t  = torch.rand(B, device=dev)
                z0 = torch.randn_like(z1b)

                # reshape time for broadcasting
                t_ = t[:, None]

                # Variance-preserving trig interpolation
                alpha = torch.cos(0.5 * torch.pi * t_)
                sigma = torch.sin(0.5 * torch.pi * t_)

                zt = alpha * z0 + sigma * z1b

                # Target velocity
                v_target = (
                    -0.5 * torch.pi * torch.sin(0.5 * torch.pi * t_) * z0
                    +0.5 * torch.pi * torch.cos(0.5 * torch.pi * t_) * z1b
                )

                # Network prediction
                v_pred = self.net(zt, t)

                loss = F.mse_loss(v_pred, v_target)

                opt.zero_grad()
                loss.backward()
                opt.step()

                ep_loss += loss.item()

            hist.append(ep_loss / len(ldr))

            if epoch % print_every == 0 or epoch == 1:
                print(f"  [FM]   Ep {epoch:>5}/{n_epochs}  loss={hist[-1]:.5f}")

        return hist

    @torch.no_grad()
    def sample(self, n, device=None):
        """Euler ODE solve z_0→z_1  →  (n, z_dim) numpy array"""
        dev = device or next(self.parameters()).device
        self.eval()
        sigma = 1 # To try and get tails (extreme events) but increase arbitrage cases
        z  = sigma * torch.randn(n, self.z_dim, device=dev)
        dt = 1.0 / self.ode_steps
        for i in range(self.ode_steps):
            t = torch.full((n,), i / self.ode_steps, device=dev)
            z = z + dt * self.net(z, t)
        return z.cpu().numpy()

    def save(self, path):
        torch.save(dict(state_dict=self.state_dict(),
                        z_dim=self.z_dim, ode_steps=self.ode_steps), path)
        print(f"  Saved FM → {path}")

    @classmethod
    def load(cls, path, device=None):
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        m  = cls(z_dim=ck["z_dim"], ode_steps=ck["ode_steps"])
        m.load_state_dict(ck["state_dict"])
        return m
