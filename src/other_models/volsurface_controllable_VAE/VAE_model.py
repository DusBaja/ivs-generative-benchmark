"""
Controllable VAE for IVS — Wang et al. (2025)
=============================================
Key fixes vs first version:
  1. Train on same normalized space as the diffusion model (% values, same lo/hi)
     → denormalization in comparison cell works with the same lo/hi, no mismatch
  2. KL warmup: β ramps from 0 → β_target over warmup_epochs
     → prevents posterior collapse (decoder ignoring z and y)
  3. Smaller default hidden layers [64, 32] suitable for ~70 training surfaces
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc1      = nn.Linear(in_dim,  out_dim)
        self.fc2      = nn.Linear(out_dim, out_dim)
        self.shortcut = nn.Linear(in_dim,  out_dim) if in_dim != out_dim else nn.Identity()
        self.act      = nn.ReLU()

    def forward(self, x):
        return self.act(self.fc2(self.act(self.fc1(x))) + self.shortcut(x))


class Encoder(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim, hidden=(64, 32)):
        super().__init__()
        layers, prev = [], x_dim + y_dim
        for h in hidden:
            layers.append(ResBlock(prev, h)); prev = h
        self.net    = nn.Sequential(*layers)
        self.mu     = nn.Linear(prev, latent_dim)
        self.logvar = nn.Linear(prev, latent_dim)

    def forward(self, x, y):
        h = self.net(torch.cat([x, y], -1))
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, x_dim, y_dim, latent_dim, hidden=(32, 64)):
        super().__init__()
        layers, prev = [], latent_dim + y_dim
        for h in hidden:
            layers.append(ResBlock(prev, h)); prev = h
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(prev, x_dim)

    def forward(self, z, y):
        return self.out(self.net(torch.cat([z, y], -1)))


# ─────────────────────────────────────────────────────────────────────────────
# Controllable VAE
# ─────────────────────────────────────────────────────────────────────────────

class IVS_VAE(nn.Module):
    FEATURE_NAMES = ["level", "slope", "curvature", "term_struct"]

    def __init__(
        self,
        n_features:  int   = 2,
        x_dim:       int   = 512,          # 16 × 32
        latent_dim:  int   = 5,
        hidden_enc:  Tuple = (64, 32),     # smaller → better for ~70 samples
        hidden_dec:  Tuple = (32, 64),
        beta:        float = 1e-4,         # target KL weight (after warmup)
    ):
        super().__init__()
        assert 1 <= n_features <= 4
        self.n_features = n_features
        self.x_dim      = x_dim
        self.latent_dim = latent_dim
        self.beta       = beta
        self.encoder    = Encoder(x_dim, n_features, latent_dim, hidden_enc)
        self.decoder    = Decoder(x_dim, n_features, latent_dim, hidden_dec)

        active = self.FEATURE_NAMES[:n_features]
        print(f"IVS_VAE  |  {n_features} feature(s): {active}")
        print(f"         |  x_dim={x_dim}  z_dim={latent_dim}  β_target={beta}")

    @staticmethod
    def reparametrize(mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z          = self.reparametrize(mu, logvar)
        return self.decoder(z, y), mu, logvar

    def loss(self, x, x_hat, mu, logvar, beta):
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kl, recon, kl

    # ── generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, y, z=None, device=None):
        dev = device or next(self.parameters()).device
        y   = y.to(dev)
        z   = torch.randn(len(y), self.latent_dim, device=dev) if z is None else z.to(dev)
        self.eval()
        return self.decoder(z, y)

    # ── training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        surf_train_n:  np.ndarray,    # (n, 1, 16, 32) or (n, 512) — NORMALISED surfaces
                                      # must be in the SAME space used by your other models
                                      # i.e. normalize(raw_iv_pct, lo, hi) from fit_normalizer
        feat_train_n:  np.ndarray,    # (n, 4)  min-max normalised features
        surf_val_n:    Optional[np.ndarray] = None,
        feat_val_n:    Optional[np.ndarray] = None,
        n_epochs:      int   = 3000,
        batch_size:    int   = 16,    # small dataset → small batch
        lr:            float = 1e-3,
        warmup_epochs: int   = 500,   # β ramps 0 → beta over this many epochs
        device:        Optional[torch.device] = None,
        print_every:   int   = 200,
        patience:      int   = 400,
    ):
        """
        surf_train_n  MUST be the same normalised space you use for all other models,
        i.e. the output of:
            normalize(resize_to_16x32(raw_iv_pct, device), lo, hi)
        where lo, hi come from fit_normalizer on the raw % training surfaces.
        """
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(dev)

        def prep(s):
            a = np.asarray(s, dtype=np.float32)
            if a.ndim == 4: a = a[:, 0]          # (n, 16, 32)
            return torch.tensor(a.reshape(len(a), -1)).to(dev)  # (n, 512)

        def prep_y(f):
            return torch.tensor(
                f[:, :self.n_features].astype(np.float32)
            ).to(dev)

        x_tr = prep(surf_train_n);  y_tr = prep_y(feat_train_n)
        loader = DataLoader(TensorDataset(x_tr, y_tr),
                            batch_size=batch_size, shuffle=True, drop_last=False)

        has_val = surf_val_n is not None and feat_val_n is not None
        if has_val:
            x_val = prep(surf_val_n);  y_val = prep_y(feat_val_n)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        history = {"train_loss": [], "recon": [], "kl": [], "val_loss": [], "beta": []}
        best_val, best_state, no_improve = float("inf"), None, 0

        for epoch in range(1, n_epochs + 1):
            # ── KL warmup ────────────────────────────────────────────────────
            beta_now = self.beta * min(1.0, epoch / max(warmup_epochs, 1))

            self.train()
            ep_loss = ep_recon = ep_kl = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                x_hat, mu, lv = self(xb, yb)
                loss, recon, kl = self.loss(xb, x_hat, mu, lv, beta_now)
                loss.backward(); opt.step()
                ep_loss += loss.item(); ep_recon += recon.item(); ep_kl += kl.item()

            n_bat = len(loader)
            history["train_loss"].append(ep_loss  / n_bat)
            history["recon"].append(ep_recon / n_bat)
            history["kl"].append(ep_kl    / n_bat)
            history["beta"].append(beta_now)

            if has_val:
                self.eval()
                with torch.no_grad():
                    xh, mu, lv = self(x_val, y_val)
                    vl, _, _   = self.loss(x_val, xh, mu, lv, beta_now)
                history["val_loss"].append(vl.item())
                if vl.item() < best_val - 1e-7:
                    best_val   = vl.item()
                    best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= patience:
                    print(f"\n  Early stop at epoch {epoch}  (best val={best_val:.6f})")
                    break

            if epoch % print_every == 0 or epoch == 1:
                msg = (f"  Ep {epoch:>5}/{n_epochs}  "
                       f"recon={history['recon'][-1]:.5f}  "
                       f"kl={history['kl'][-1]:.4f}  "
                       f"β={beta_now:.2e}")
                if has_val: msg += f"  val={history['val_loss'][-1]:.5f}"
                print(msg)

        if best_state:
            self.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
            print(f"  Restored best  val={best_val:.6f}")

        return history

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(dict(state_dict=self.state_dict(), n_features=self.n_features,
                        x_dim=self.x_dim, latent_dim=self.latent_dim, beta=self.beta), path)
        print(f"  Saved → {path}")

    @classmethod
    def load(cls, path: str, device=None):
        ck  = torch.load(path, map_location=device or "cpu", weights_only=False)
        vae = cls(n_features=ck["n_features"], x_dim=ck["x_dim"],
                  latent_dim=ck["latent_dim"],  beta=ck["beta"])
        vae.load_state_dict(ck["state_dict"])
        return vae


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def unflatten(x_flat, n_tenor=16, n_money=32):
    """(n, 512) → (n, 1, 16, 32)"""
    return x_flat.reshape(-1, 1, n_tenor, n_money)

"""
IVS Generation & 3D Visualization
===================================
Generates surfaces from the trained VAE and plots:
  - Individual 3D generated surfaces (grid)
  - The mean generated surface vs the empirical mean
"""

import matplotlib.pyplot as plt
from matplotlib import cm


# ─────────────────────────────────────────────────────────────────────────────
# Un-normalise helper  (inverse of your fit_normalizer / normalize)
# ─────────────────────────────────────────────────────────────────────────────

def denormalize(x_norm, lo, hi):
    """
    Inverse of normalize(x, lo, hi).
    Works for torch tensors or numpy arrays.
    lo, hi : scalars or broadcastable arrays
    """
    if torch.is_tensor(x_norm):
        x_norm = x_norm.cpu().numpy()
    return x_norm * (hi - lo) + lo


# ─────────────────────────────────────────────────────────────────────────────
# Generate a batch of surfaces
# ─────────────────────────────────────────────────────────────────────────────

def generate_surfaces(
    vae:        IVS_VAE,
    y_norm:     torch.Tensor,       # (n, n_features)  normalised target features
    lo,  hi,                        # normalisation bounds from fit_normalizer
    device=None,
    as_pct:     bool = True,        # return in %  (multiply fraction × 100)
):
    """
    Returns
    -------
    surfs : (n, 16, 32)  generated surfaces  in fraction or %
    """
    dev   = device or next(vae.parameters()).device
    x_gen = vae.generate(y=y_norm, device=dev)          # (n, 512)  normalised
    surfs = unflatten(x_gen).squeeze(1).cpu().numpy()   # (n, 16, 32)
    surfs = denormalize(surfs, lo, hi)                  # back to IV fraction
    if as_pct:
        surfs = surfs * 100.0
    return surfs


# ─────────────────────────────────────────────────────────────────────────────
# 3D surface plot  (single surface)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_one_surface(ax, surf, tau_new, m_new, title="", cmap="viridis", alpha=0.85):
    M, T   = np.meshgrid(m_new, tau_new)          # (16, 32) each
    surf2d = surf                                  # (16, 32)

    ax.plot_surface(M, T, surf2d, cmap=cmap, alpha=alpha,
                    linewidth=0, antialiased=True)
    ax.set_xlabel("Moneyness K/S", fontsize=8, labelpad=4)
    ax.set_ylabel("Tenor (yr)",    fontsize=8, labelpad=4)
    ax.set_zlabel("IV (%)",        fontsize=8, labelpad=4)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# Grid of individual generated surfaces
# ─────────────────────────────────────────────────────────────────────────────

def plot_generated_grid(
    surfs:   np.ndarray,     # (n, 16, 32)  in %
    tau_new: np.ndarray,
    m_new:   np.ndarray,
    n_cols:  int  = 3,
    feat_labels: list = None,   # optional text per surface
    suptitle: str = "Generated IVS samples",
    save_path: str = None,
):
    n      = len(surfs)
    n_rows = int(np.ceil(n / n_cols))

    fig = plt.figure(figsize=(5 * n_cols, 4.2 * n_rows))
    for i, surf in enumerate(surfs):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        title = feat_labels[i] if feat_labels else f"Sample {i+1}"
        _plot_one_surface(ax, surf, tau_new, m_new, title=title)

    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Mean surface  (generated vs empirical)
# ─────────────────────────────────────────────────────────────────────────────

def plot_mean_surface(
    surfs_gen:  np.ndarray,          # (n, 16, 32)  generated, in %
    tau_new:    np.ndarray,
    m_new:      np.ndarray,
    surfs_real: np.ndarray = None,   # (n_real, 16, 32)  real data, in %  (optional)
    save_path:  str = None,
):
    mean_gen  = surfs_gen.mean(axis=0)    # (16, 32)
    n_panels  = 2 if surfs_real is not None else 1

    fig = plt.figure(figsize=(7 * n_panels, 5.5))

    ax1 = fig.add_subplot(1, n_panels, 1, projection="3d")
    _plot_one_surface(ax1, mean_gen, tau_new, m_new,
                      title=f"Mean generated  (n={len(surfs_gen)})", cmap="plasma")

    if surfs_real is not None:
        mean_real = surfs_real.mean(axis=0)
        ax2 = fig.add_subplot(1, n_panels, 2, projection="3d")
        _plot_one_surface(ax2, mean_real, tau_new, m_new,
                          title=f"Empirical mean  (n={len(surfs_real)})", cmap="viridis")

        # align z-axes
        zmin = min(mean_gen.min(), mean_real.min())
        zmax = max(mean_gen.max(), mean_real.max())
        for ax in [ax1, ax2]:
            ax.set_zlim(zmin, zmax)

    plt.suptitle("Mean IVS — Generated vs Empirical", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()