"""
VolGAN Training — corrected arbitrage penalty
=============================================
Adapted from Vuletic et al. (2023)

Use directional ReLU penalties — only penalise VIOLATIONS:
    Calendar  : relu(\sigma(\tau_i) - \sigma(\tau_{i+1}))  — penalise decreasing tenor
    Call spread: relu(C(K_i) - C(K_{i-1}))                 — penalise increasing call price
    Butterfly  : relu(-d^2C/dK^2)                          — penalise concave smile

Two penalty modes available:
    'iv'  — penalties computed directly on IV surface (fast, no BS needed)
    'bs'  — penalties computed on BS call prices (exact VolGAN formulation)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.stats import norm as sp_norm
from src.other_models.VolGan import Generator, Discriminator
from src.other_models.diffusion_cond_ext import denormalize


def _smallBS_tensor(m, tau, sigma, r=0.0):
    """
    Relative call price C/S via Black-Scholes.
    All inputs: (B, NT, NM) tensors.
    m   : log-moneyness = log(K/F)
    tau : time to expiry (years)
    sigma : implied vol (fraction)
    """
    norm = torch.distributions.Normal(0., 1.)
    eps  = 1e-8
    sig  = sigma.clamp(min=eps)
    tau_ = tau.clamp(min=eps)
    d1   = (-m + tau_ * (r + 0.5 * sig**2)) / (sig * tau_.sqrt())
    d2   = d1 - sig * tau_.sqrt()
    C    = norm.cdf(d1) - torch.exp(m - r*tau_) * norm.cdf(d2)
    return C.clamp(min=1e-10)


def arb_penalty_iv(surf_pct, tau_grid=None):
    """
    Fast IV-space arbitrage penalty.
    surf_pct : (B, NT, NM) in fraction (e.g. 0.20 for 20%)

    Calendar  : sigma non-decreasing in tenor  
    Butterfly : smile convex in moneyness 
    Call spread: sigma non-increasing in m     
    """
    # Calendar spread
    diff_tau = surf_pct[:, 1:, :] - surf_pct[:, :-1, :]   # (B,NT-1,NM)
    p_cal    = F.relu(-diff_tau).pow(2).mean()

    # Butterfly (convexity)
    d2m      = (surf_pct[:, :, 2:] - 2*surf_pct[:, :, 1:-1]
                + surf_pct[:, :, :-2])                     # (B,NT,NM-2)
    p_but    = F.relu(-d2m).pow(2).mean()

    # Call spread (monotonicity in m)
    diff_m   = surf_pct[:, :, 1:] - surf_pct[:, :, :-1]   # (B,NT,NM-1)
    p_call   = F.relu(diff_m).pow(2).mean()

    return p_cal, p_but, p_call


def arb_penalty_bs(surf_pct, m_grid, tau_grid, device):
    """
    Exact VolGAN arbitrage penalty via BS call prices.
    surf_pct : (B, NT, NM) in fraction

    Uses penalty matrices from VolGAN (penalty_mutau_tensor).
    """
    B, NT, NM = surf_pct.shape
    m   = torch.tensor(m_grid,   dtype=torch.float32, device=device)
    tau = torch.tensor(tau_grid, dtype=torch.float32, device=device)

    # Expand to (B, NT, NM)
    m_t   = m.view(1, 1, NM).expand(B, NT, NM)
    tau_t = tau.view(1, NT, 1).expand(B, NT, NM)

    C = _smallBS_tensor(m_t, tau_t, surf_pct)    # (B, NT, NM)

    # Calendar: C(τ_{i+1}) ≥ C(τ_i) → penalise violations
    p_cal  = F.relu(C[:, :-1, :] - C[:, 1:, :]).pow(2).mean()

    # Call spread: C(m_{i+1}) ≤ C(m_i) → penalise violations
    p_call = F.relu(C[:, :, 1:] - C[:, :, :-1]).pow(2).mean()

    # Butterfly: C convex in m
    d2m    = C[:, :, 2:] - 2*C[:, :, 1:-1] + C[:, :, :-2]
    p_but  = F.relu(-d2m).pow(2).mean()

    return p_cal, p_but, p_call


# Training
def train_volgan(
    surf_train_n,           # (N, 1, NT, NM) normalised surfaces
    lo_v, hi_v,             # normaliser scalars
    m_new, tau_new,         # moneyness and tenor grids
    device,
    noise_dim   = 32,
    hidden_dim  = 256,
    n_epochs    = 1500,
    n_grad      = 100,      # gradient-matching pre-phase
    lr_g = 1e-4,
    lr_d = 1e-4,
    batch_size  = 32,
    lambda_cal  = 1.0,      # relative penalty weights (calibrated in phase 1)
    lambda_but  = 1.0,
    lambda_call = 1.0,
    penalty_mode = 'iv',    # 'iv' (fast) or 'bs' (exact VolGAN)
    print_every = 100,
    save_path   = "data/treated/volgan_checkpoint.pt",
):
    NT = surf_train_n.shape[2]
    NM = surf_train_n.shape[3]
    Ng = NT * NM

    # Flatten and build consecutive pairs
    flat = torch.tensor(
        surf_train_n.squeeze(1).reshape(-1, Ng), dtype=torch.float32
    ).to(device)
    cond_vg   = flat[:-1]              # σ_{t-1}  normalised
    target_vg = flat[1:] - flat[:-1]  # increment (normalised)
    N_tr      = cond_vg.shape[0]
    print(f"VolGAN: {N_tr} consecutive pairs  |  surface dim = {Ng}")

    # ── Helper: denorm + reshape ──────────────────────────────────────────────
    lo_t = torch.tensor(lo_v, device=device)
    hi_t = torch.tensor(hi_v, device=device)

    def to_pct(surf_flat):
        """(B, Ng) normalised → (B, NT, NM) fraction"""
        x01  = (surf_flat + 1.0) / 2.0
        pct  = (x01 * (hi_t - lo_t) + lo_t) / 100.0   # → fraction
        return pct.reshape(-1, NT, NM)

    # ── Penalty function ──────────────────────────────────────────────────────
    def penalty(fake_surf_flat, cond_flat):
        """fake_surf_flat: (B, Ng) normalised absolute surface"""
        surf_pct = to_pct(fake_surf_flat)
        if penalty_mode == 'bs':
            pc, pb, pm = arb_penalty_bs(surf_pct, m_new, tau_new, device)
        else:
            pc, pb, pm = arb_penalty_iv(surf_pct)
        return pc, pb, pm

    # ── Models ────────────────────────────────────────────────────────────────
    gen  = Generator(noise_dim=noise_dim, cond_dim=Ng,
                     hidden_dim=hidden_dim, output_dim=Ng).to(device)
    disc = Discriminator(in_dim=2*Ng, hidden_dim=hidden_dim).to(device)
    opt_g = torch.optim.RMSprop(gen.parameters(),  lr=lr_g)
    opt_d = torch.optim.RMSprop(disc.parameters(), lr=lr_d)
    bce   = nn.BCELoss()
    print(f"Generator  params: {sum(p.numel() for p in gen.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in disc.parameters()):,}")

    # Phase 1: Gradient magnitude matching
    # Calibrates lambda_cal, lambda_but, lambda_call so their gradient norms
    # match the adversarial gradient norm to balanced multi-objective training
    print(f"\nPhase 1: gradient matching ({n_grad} epochs) ...")
    bce_norms, cal_norms, but_norms, call_norms = [], [], [], []
    gen.train()

    for _ in tqdm(range(n_grad)):
        perm = torch.randperm(N_tr)
        for i in range(0, N_tr, batch_size):
            idx  = perm[i:i+batch_size]
            cond = cond_vg[idx]; B = cond.shape[0]
            noise = torch.randn(B, noise_dim, device=device)
            fake  = gen(noise, cond)
            fake_abs = fake + cond              # absolute surface

            def _grad_norm(loss_val):
                opt_g.zero_grad()
                loss_val.backward(retain_graph=True)
                return sum(
                    p.grad.data.norm(2).item()**2
                    for p in gen.parameters() if p.grad is not None
                ) ** 0.5

            # Adversarial gradient norm
            d_fake = disc(torch.cat([cond, fake], dim=-1))
            n_bce  = _grad_norm(bce(d_fake, torch.ones_like(d_fake)))
            bce_norms.append(n_bce)

            # Penalty gradient norms
            pc, pb, pm = penalty(fake_abs, cond)
            n_cal  = _grad_norm(pc)
            n_but  = _grad_norm(pb)
            n_call = _grad_norm(pm)
            cal_norms.append(n_cal)
            but_norms.append(n_but)
            call_norms.append(n_call)
            opt_g.zero_grad()

    # Calibrated weights: scale so each penalty's gradient for adversarial norm
    eps = 1e-12
    lam_cal  = float(np.mean(bce_norms) / (np.mean(cal_norms)  + eps))
    lam_but  = float(np.mean(bce_norms) / (np.mean(but_norms)  + eps))
    lam_call = float(np.mean(bce_norms) / (np.mean(call_norms) + eps))
    print(f"  λ_cal={lam_cal:.4f}  λ_but={lam_but:.4f}  λ_call={lam_call:.4f}")

    # Phase 2: Adversarial training
    print(f"\nPhase 2: adversarial training ({n_epochs} epochs) ...")
    ldr = DataLoader(
        TensorDataset(cond_vg, target_vg),
        batch_size=batch_size, shuffle=True, drop_last=False
    )
    gen_losses, disc_losses, arb_losses = [], [], []
    gen.train(); disc.train()

    for epoch in tqdm(range(1, n_epochs + 1)):
        gl_ep = dl_ep = al_ep = 0.; n_batches = 0

        for cond, real in ldr:
            B     = cond.shape[0]
            ones  = torch.ones(B, 1,  device=device)
            zeros = torch.zeros(B, 1, device=device)

            # Discriminator
            noise = torch.randn(B, noise_dim, device=device)
            fake  = gen(noise, cond).detach()
            opt_d.zero_grad()
            d_real = disc(torch.cat([cond, real], dim=-1))
            d_fake = disc(torch.cat([cond, fake], dim=-1))
            d_loss = 0.5 * (bce(d_real, ones) + bce(d_fake, zeros))
            d_loss.backward(); opt_d.step()

            # Generator 
            noise    = torch.randn(B, noise_dim, device=device)
            fake     = gen(noise, cond)
            fake_abs = fake + cond                  # absolute normalised surface
            d_fake   = disc(torch.cat([cond, fake], dim=-1))
            adv_loss = bce(d_fake, ones)

            # Directional arbitrage penalties (only penalise violations)
            pc, pb, pm = penalty(fake_abs, cond)
            arb_loss = lam_cal * pc + lam_but * pb + lam_call * pm

            g_loss = adv_loss + arb_loss
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            gl_ep  += g_loss.item()
            dl_ep  += d_loss.item()
            al_ep  += arb_loss.item()
            n_batches += 1

        gen_losses.append(gl_ep / n_batches)
        disc_losses.append(dl_ep / n_batches)
        arb_losses.append(al_ep / n_batches)

        if epoch % print_every == 0 or epoch == 1:
            print(f"  Ep {epoch:>5}/{n_epochs}  "
                  f"G={gen_losses[-1]:.4f}  "
                  f"D={disc_losses[-1]:.4f}  "
                  f"arb={arb_losses[-1]:.5f}")

    # Save 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'gen'       : gen.state_dict(),
        'disc'      : disc.state_dict(),
        'lam_cal'   : lam_cal,
        'lam_but'   : lam_but,
        'lam_call'  : lam_call,
        'noise_dim' : noise_dim,
        'hidden_dim': hidden_dim,
        'NT'        : NT,
        'NM'        : NM,
        'lo'        : lo_v,
        'hi'        : hi_v,
    }, save_path)
    print(f"\n  Saved VolGAN → {save_path}")

    return gen, disc, gen_losses, disc_losses, arb_losses



# Generation
def sample_volgan(gen, surf_cond_n, N_GEN, noise_dim, lo_v, hi_v, device):
    """
    Generate N_GEN surfaces conditioned on surf_cond_n.

    surf_cond_n : (1, 1, NT, NM) or (NT, NM) — the conditioning surface
                  (typically the last training surface)
    Returns (N_GEN, NT, NM) in %
    """
    gen.eval()
    NT = surf_cond_n.shape[-2]
    NM = surf_cond_n.shape[-1]
    Ng = NT * NM

    cond_flat = torch.tensor(
        surf_cond_n.reshape(1, Ng), dtype=torch.float32, device=device
    ).expand(N_GEN, -1)                               # (N_GEN, Ng)

    with torch.no_grad():
        noise = torch.randn(N_GEN, noise_dim, device=device)
        fake  = gen(noise, cond_flat)                 # (N_GEN, Ng) increment
        surf_norm = (fake + cond_flat).cpu()          # absolute normalised

    # Denormalise → %
    lo_t = torch.tensor(lo_v); hi_t = torch.tensor(hi_v)
    x01  = (surf_norm + 1.0) / 2.0
    surf_pct = (x01 * (hi_t - lo_t) + lo_t).numpy()  # (N_GEN, Ng)
    return surf_pct.reshape(N_GEN, NT, NM)


def sample_volgan_diverse(gen, surf_train_n, N_GEN, noise_dim,
                          lo_v, hi_v, device, n_seeds=10):
    """
    Generate N_GEN surfaces by conditioning on n_seeds different
    training surfaces (spread across history).
    Gives more diverse output than conditioning on a single surface.
    Returns (N_GEN, NT, NM) in %
    """
    NT  = surf_train_n.shape[2]
    NM  = surf_train_n.shape[3]
    n_per_seed = N_GEN // n_seeds

    # Evenly spaced seeds from training history
    seed_idx = np.linspace(0, len(surf_train_n)-1, n_seeds, dtype=int)
    parts = []
    for idx in seed_idx:
        s = surf_train_n[idx]   # (1, NT, NM) or (NT, NM)
        parts.append(
            sample_volgan(gen, s, n_per_seed, noise_dim, lo_v, hi_v, device)
        )
    surfs = np.concatenate(parts, axis=0)    # (N_GEN≈, NT, NM)

    # Top up to exactly N_GEN if needed
    if len(surfs) < N_GEN:
        extra = sample_volgan(gen, surf_train_n[-1], N_GEN - len(surfs),
                              noise_dim, lo_v, hi_v, device)
        surfs = np.concatenate([surfs, extra], axis=0)

    return surfs[:N_GEN]
