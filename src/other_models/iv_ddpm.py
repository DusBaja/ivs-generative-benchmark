"""
IV-DDPM: Unconditional DDPM for Implied Volatility Surfaces
============================================================
Faithfully adapted from Jin & Agarwal (2025) arXiv:2511.07571
  + Austinjinc/diffusion-paper-code: model.py / diffusion.py / losses.py

Adaptations from the original conditional 9x9 model:
  - Unconditional  (no conditioning surface / EWMAs / scalar risk indicators)
  - 16x32 grid     (original: 9x9)  — pool 2x2 instead of 3x3
  - [-1,1] normalisation from fit_normalizer (original: log-standardise)
  - Arbitrage penalty axes corrected for (B,1,16[tenor],32[money]) layout

Everything else is kept exactly as in the original:
  - DoubleConv with AdaGN conditioning 
  - V-prediction parametrisation
  - Cosine noise schedule
  - SNR-weighted arbitrage penalty  ← key paper contribution
  - EMA model (beta=0.9)
  - AdamW + ReduceLROnPlateau
  - Gradient clip at 0.15
  - DDPM and DDIM samplers
"""

import copy, math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


# Noise schedule  (get_cosine_schedule)
def _cosine_schedule(T, s=0.008):
    steps      = torch.linspace(0, T, T + 1)
    alpha_bar  = torch.cos(((steps / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar  = alpha_bar / alpha_bar[0]        # ← replace /= with = ... /
    betas      = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(0.0001, 0.9999)
    return betas


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        half_dim   = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim, num_groups=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 15),
            nn.SiLU(),
            nn.Linear(15, out_channels * 4),
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.act1  = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2  = nn.SiLU()

    def forward(self, x, comb_emb):
        s1, b1, s2, b2 = self.projection(comb_emb).chunk(4, dim=1)

        h = self.conv1(x)
        h = self.norm1(h)
        h = h * s1[:, :, None, None] + b1[:, :, None, None]
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = h * s2[:, :, None, None] + b2[:, :, None, None]
        return self.act2(h)


class IVUNet(nn.Module):
    """
    Unconditional UNet for (B, 1, 16, 32) surfaces.
    Mirrors ConditionalUnet with:
      - scalar MLP removed  (emb_dim = 2 x t_emb_dim)
      - MaxPool2d(2,2) and ConvTranspose2d(2,2) for 16x32 grid
        (originally used 3x3 for 9x9)
    """
    def __init__(self, enc_channels=32, bottle_channels=64, emb_dim=128):
        super().__init__()
        assert emb_dim % 2 == 0
        self.time_mlp = SinusoidalPositionEmbeddings(emb_dim // 2)
        # Without scalars we still need emb_dim total → double the time emb
        self.time_proj = nn.Sequential(
            nn.Linear(emb_dim // 2, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.enc1   = DoubleConv(1,               enc_channels,    emb_dim)
        self.pool   = nn.MaxPool2d(kernel_size=2, stride=2)        # 16x32 to 8x16
        self.bottle = DoubleConv(enc_channels,    bottle_channels, emb_dim)
        self.up     = nn.ConvTranspose2d(bottle_channels, enc_channels,
                                         kernel_size=2, stride=2)  # 8x16 to 16x32
        self.dec1   = DoubleConv(enc_channels * 2, enc_channels,   emb_dim)
        self.out_conv = nn.Conv2d(enc_channels, 1, kernel_size=1)

        n = sum(p.numel() for p in self.parameters())
        print(f"IVUNet  |  enc={enc_channels}  bot={bottle_channels}  "
              f"emb={emb_dim}  params={n:,}")

    def forward(self, x_t, time_t):
        """
        x_t    : (B, 1, 16, 32)  noised surface
        time_t : (B,)  integer timestep
        """
        t_emb    = self.time_mlp(time_t.float())   # (B, emb//2)
        comb_emb = self.time_proj(t_emb)            # (B, emb)

        skip = self.enc1(x_t, comb_emb)             # (B, enc, 16, 32)
        h    = self.pool(skip)                       # (B, enc,  8, 16)
        h    = self.bottle(h, comb_emb)              # (B, bot,  8, 16)
        h    = self.up(h)                            # (B, enc, 16, 32)
        h    = torch.cat([h, skip], dim=1)           # (B, enc*2, 16, 32)
        h    = self.dec1(h, comb_emb)
        return self.out_conv(h)                      # (B, 1, 16, 32)


class ArbitragePenalties(nn.Module):
    def __init__(self, device, tau_grid):
        super().__init__()
        # taus along tenor dim (dim 2)
        taus = torch.tensor(tau_grid, dtype=torch.float32, device=device)
        self.register_buffer('taus', taus.view(1, 1, -1, 1))  # (1,1,16,1)

        # Smile: convexity along moneyness (dim 3) → kernel (1,3)
        smile_k = torch.tensor([[[[1, -2, 1]]]], dtype=torch.float32, device=device)
        self.smile_conv = nn.Conv2d(1, 1, kernel_size=(1, 3), bias=False)
        self.smile_conv.weight.data = smile_k
        self.smile_conv.weight.requires_grad = False

        # Calendar: non-decreasing total variance along tenor (dim 2) → kernel (2,1)
        ttm_k = torch.tensor([[[[1], [-1]]]], dtype=torch.float32, device=device)
        self.ttm_conv = nn.Conv2d(1, 1, kernel_size=(2, 1), bias=False)
        self.ttm_conv.weight.data = ttm_k
        self.ttm_conv.weight.requires_grad = False

    def forward(self, surface_iv_pct):
        """
        surface_iv_pct : (B, 1, 16, 32)  IV in % (denormalised)
        Returns smile_loss, ttm_loss  each (B,)
        """
        # Smile: penalise concave smile (second diff < 0)
        smile_curv = self.smile_conv(surface_iv_pct)
        smile_loss = F.relu(-smile_curv).pow(2).mean(dim=(1, 2, 3))

        # Calendar: penalise decreasing total variance across tenors
        total_var  = surface_iv_pct.pow(2) * self.taus      # (B,1,16,32)
        ttm_diff   = self.ttm_conv(total_var)                # (B,1,15,32)
        ttm_loss   = F.relu(ttm_diff).pow(2).mean(dim=(1, 2, 3))

        return smile_loss, ttm_loss

class _EMA:
    def __init__(self, beta=0.9):
        self.beta = beta

    def update(self, ema_model, model):
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data = self.beta * ep.data + (1 - self.beta) * p.data

class IVDDPM:
    """
    Unconditional IV-DDPM.

    Parameters
    ----------
    tau_grid     : (16,) maturities in years — for calendar penalty
    lo / hi      : scalars from fit_normalizer (surface normaliser)
    T            : diffusion timesteps
    enc_channels : UNet encoder channels
    bot_channels : UNet bottleneck channels
    emb_dim      : time embedding dimension (must be even)
    lambda_smile : smile penalty weight
    lambda_ttm   : calendar penalty weight
    ema_beta     : EMA decay (0.9 originally)
    """

    def __init__(self, tau_grid, lo, hi, T=500,
                 enc_channels=32, bot_channels=64, emb_dim=128,
                 lambda_smile=0.01, lambda_ttm=0.01,
                 ema_beta=0.9, device=None):

        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.T            = T
        self.lo           = float(lo)
        self.hi           = float(hi)
        self.lambda_smile = lambda_smile
        self.lambda_ttm   = lambda_ttm
        self.tau_grid     = np.asarray(tau_grid)

        # Noise schedule  (from diffusion.py)
        betas         = _cosine_schedule(T).to(self.device)
        alphas        = 1. - betas
        alpha_bar     = torch.cumprod(alphas, 0)
        self.betas     = betas
        self.alphas    = alphas
        self.alpha_bar = alpha_bar                        # (T,)

        # SNR weights  (key contribution: arb penalty weighted by SNR)
        self.snr = alpha_bar / (1 - alpha_bar + 1e-8)    # (T,)

        # Model + EMA  (mirrors Trainer.__init__)
        self.model     = IVUNet(enc_channels, bot_channels, emb_dim).to(self.device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self._ema      = _EMA(beta=ema_beta)

        # Arbitrage penalty 
        self.arb = ArbitragePenalties(self.device, tau_grid)

        self._arch = dict(enc_channels=enc_channels,
                          bot_channels=bot_channels,
                          emb_dim=emb_dim)

    # Forward noising 

    def _noise(self, x0, t):
        """
        V-prediction parametrisation  (from diffusion.py noise_surfaces):
          x_t = √ᾱ·x₀ + √(1−ᾱ)·ε
          v   = √ᾱ·ε  − √(1−ᾱ)·x₀
        """
        ab  = self.alpha_bar[t][:, None, None, None]
        eps = torch.randn_like(x0)
        xt  = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        v   = ab.sqrt() * eps - (1 - ab).sqrt() * x0
        return xt, v

    # Loss 

    def _loss(self, x0_norm):
        B = x0_norm.shape[0]
        t = torch.randint(1, self.T, (B,), device=self.device)

        xt, v_target = self._noise(x0_norm, t)
        v_pred       = self.model(xt, t)
        mse          = F.mse_loss(v_pred, v_target)

        # Recover x0 estimate for arbitrage penalty  (from one_epoch)
        ab     = self.alpha_bar[t][:, None, None, None]
        x0_hat = ab.sqrt() * xt - (1 - ab).sqrt() * v_pred

        # Denormalise to % for physically interpretable penalty
        x0_pct = (x0_hat + 1) / 2 * (self.hi - self.lo) + self.lo  # (B,1,16,32)

        smile_loss, ttm_loss = self.arb(x0_pct)
        struct = self.lambda_smile * smile_loss + self.lambda_ttm * ttm_loss  # (B,)

        # SNR weighting  (exact from diffusion.py one_epoch)
        snr_w = self.snr[t]                                          # (B,)
        total = mse + (snr_w * struct).mean()

        return total, mse, smile_loss.mean(), ttm_loss.mean()

    # Training  
    def fit(self, surf_train_n, surf_val_n=None,
            n_epochs=1000, batch_size=32, lr=3e-4,
            print_every=100, patience=200):
        """
        surf_train_n : (n, 1, 16, 32) or (n, 16, 32) normalised to [-1, 1]
        surf_val_n   : optional validation set (same shape)
        """
        dev = self.device

        def prep(s):
            a = np.asarray(s, dtype=np.float32)
            if a.ndim == 3: a = a[:, None]
            return torch.tensor(a).to(dev)

        x_tr  = prep(surf_train_n)
        ldr   = DataLoader(TensorDataset(x_tr), batch_size=batch_size,
                           shuffle=True, drop_last=False)
        has_v = surf_val_n is not None
        if has_v:
            x_val = prep(surf_val_n)

        # Exact optimiser + scheduler from diffusion.py
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=0.8, patience=100, min_lr=1e-6
        )

        hist = []
        best_val, best_state, no_improve = float('inf'), None, 0

        for epoch in range(1, n_epochs + 1):
            self.model.train()
            ep_tot = ep_mse = ep_smile = ep_ttm = 0.

            for (xb,) in ldr:
                opt.zero_grad()
                tot, mse, smile, ttm = self._loss(xb)

                if torch.isnan(tot) or torch.isinf(tot):
                    print(f"  ⚠ NaN/Inf at epoch {epoch} — stopping"); break

                tot.backward()
                # Exact clip from diffusion.py
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.15)
                opt.step()
                self._ema.update(self.ema_model, self.model)

                ep_tot   += tot.item()
                ep_mse   += mse.item()
                ep_smile += smile.item()
                ep_ttm   += ttm.item()

            n = len(ldr)
            ep_tot /= n; ep_mse /= n; ep_smile /= n; ep_ttm /= n
            hist.append(ep_tot)

            # Validation + early stopping (patience only after warmup)
            if has_v:
                self.model.eval()
                with torch.no_grad():
                    vl, *_ = self._loss(x_val)
                vl = vl.item()
                sch.step(vl)
                if vl < best_val - 1e-7:
                    best_val   = vl
                    best_state = {k: v.cpu().clone()
                                  for k, v in self.ema_model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stop  epoch={epoch}  best_val={best_val:.5f}")
                    break

            if epoch % print_every == 0 or epoch == 1:
                msg = (f"  [IV-DDPM] Ep {epoch:>5}/{n_epochs}  "
                       f"loss={ep_tot:.5f}  mse={ep_mse:.5f}  "
                       f"smile={ep_smile:.5f}  cal={ep_ttm:.5f}")
                if has_v: msg += f"  val={best_val:.5f}"
                print(msg)

        if best_state and has_v:
            self.ema_model.load_state_dict(
                {k: v.to(dev) for k, v in best_state.items()}
            )
            print(f"  Restored best EMA  val={best_val:.5f}")

        return hist

    # Sampling 
    @torch.no_grad()
    def sample(self, n, sampler='ddim', ddim_steps=100, eta=0.0):
        """
        sampler    : 'ddpm' (T steps, slow) | 'ddim' (ddim_steps, fast)
        eta        : DDIM stochasticity  0.0=deterministic  1.0=DDPM
        Returns (n, 16, 32) in %
        """
        dev = self.device
        self.ema_model.eval()
        x = torch.randn(n, 1, 16, 32, device=dev)

        if sampler == 'ddpm':
            x = self._ddpm(x, n, dev)
        else:
            x = self._ddim(x, n, dev, ddim_steps, eta)

        # Denormalise [-1,1] → %
        surf = (x + 1) / 2 * (self.hi - self.lo) + self.lo
        return surf.squeeze(1).cpu().numpy()           # (n, 16, 32)

    def _ddpm(self, x, n, dev):
        rng = reversed(range(1, self.T))
        if TQDM:
            rng = tqdm(rng, total=self.T-1, desc="DDPM", leave=False)
        for i in rng:
            t          = torch.full((n,), i, device=dev, dtype=torch.long)
            ab         = self.alpha_bar[i]
            ab_p       = self.alpha_bar[i-1] if i > 0 else torch.tensor(1.)
            beta       = self.betas[i]
            alpha      = self.alphas[i]

            v_pred     = self.ema_model(x, t)
            x0_hat     = ab.sqrt() * x - (1 - ab).sqrt() * v_pred

            mean       = (ab_p.sqrt() * beta / (1 - ab)) * x0_hat + \
                         (alpha.sqrt() * (1 - ab_p) / (1 - ab)) * x
            post_var   = beta * (1 - ab_p) / (1 - ab)
            noise      = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x          = mean + post_var.sqrt() * noise
        return x

    def _ddim(self, x, n, dev, steps, eta):
        """Exact logic from diffusion.py _sample_ddim"""
        ts = torch.linspace(self.T - 1, 0, steps + 1).long()
        rng = range(steps)
        if TQDM:
            rng = tqdm(rng, total=steps, desc="DDIM", leave=False)
        for i in rng:
            t_cur  = ts[i].item()
            t_nxt  = ts[i+1].item()
            t      = torch.full((n,), t_cur, device=dev, dtype=torch.long)
            ab     = self.alpha_bar[t_cur]
            ab_n   = self.alpha_bar[t_nxt] if t_nxt >= 0 else torch.tensor(1.)

            v_pred = self.ema_model(x, t)
            x0_hat = ab.sqrt() * x - (1 - ab).sqrt() * v_pred
            eps    = ab.sqrt() * v_pred + (1 - ab).sqrt() * x

            sigma  = eta * ((1-ab_n)/(1-ab) * (1 - ab/ab_n)).clamp(min=0).sqrt()
            c1     = (1 - ab_n - sigma**2).clamp(min=0).sqrt()
            noise  = torch.randn_like(x) if i < steps-1 else torch.zeros_like(x)
            x      = ab_n.sqrt() * x0_hat + c1 * eps + sigma * noise
        return x

    # Persistence 
    def save(self, path):
        torch.save(dict(
            ema_state    = self.ema_model.state_dict(),
            model_state  = self.model.state_dict(),
            T            = self.T,
            lo           = self.lo,
            hi           = self.hi,
            tau_grid     = self.tau_grid,
            lambda_smile = self.lambda_smile,
            lambda_ttm   = self.lambda_ttm,
            arch         = self._arch,
        ), path)
        print(f"  Saved IV-DDPM → {path}")

    @classmethod
    def load(cls, path, device=None):
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck  = torch.load(path, map_location=dev, weights_only=False)
        arch = ck.get("arch", dict(enc_channels=32, bot_channels=64, emb_dim=128))
        m   = cls(
            tau_grid     = ck["tau_grid"],
            lo           = ck["lo"],
            hi           = ck["hi"],
            T            = ck["T"],
            lambda_smile = ck["lambda_smile"],
            lambda_ttm   = ck["lambda_ttm"],
            device       = dev,
            **arch,
        )
        m.ema_model.load_state_dict(ck["ema_state"])
        m.model.load_state_dict(ck["model_state"])
        return m
