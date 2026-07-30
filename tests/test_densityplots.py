from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from scipy.stats import gaussian_kde

from src.other_models.diffusion_cond_ext import (
    VolaDiff,
    denormalize,
    fit_normalizer,
    resize_to_16x32,
)
from src.other_models.volsurface_controllable_VAE.VAE_model import IVS_VAE, unflatten
from src.other_models.volsurface_controllable_VAE.ivs_features import (FeatureNormalizer,extract_features)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _interactive_backend() -> None:
    """Enable an interactive matplotlib backend for local plot display."""
    plt.ion()
    backend = matplotlib.get_backend()
    if backend.lower() == "agg":
        try:
            matplotlib.use("Qt5Agg", force=True)
            plt.ion()
        except Exception:
            pass


def _show_current_plot() -> None:
    plt.draw()
    plt.show(block=False)
    plt.pause(0.1)


def _kde_plot(ax: plt.Axes, data: np.ndarray, label: str, color: str) -> None:
    data = np.asarray(data, dtype=float)
    if data.size < 2:
        return
    kde = gaussian_kde(data)
    xs = np.linspace(max(-0.05, data.min()), min(0.6, data.max()), 300)
    ax.plot(xs, kde(xs), label=label, color=color, linewidth=2)


def _build_data() -> dict:
    treated = REPO_ROOT / "data" / "treated"
    if not treated.exists():
        pytest.skip("Treated data folder is missing.")

    cube_path = treated / "vol_surface_cube_test.npz"
    if not cube_path.exists():
        pytest.skip(f"Required test cube data not found: {cube_path}")

    checkpoint_best = treated / "best_model_checkpoint_Best.pt"
    checkpoint_pen = treated / "best_model_checkpoint_arbitrage_pen.pt"
    volgan_ckpt = treated / "volgan_checkpoint.pt"
    vae_ckpt = treated / "vae_checkpoint.pt"
    ivs_norm = treated / "ivs_vae_feat_norm.npz"

    for path in [checkpoint_best, checkpoint_pen, volgan_ckpt, vae_ckpt, ivs_norm]:
        if not path.exists():
            pytest.skip(f"Required checkpoint file not found: {path}")

    np.random.seed(0)
    torch.manual_seed(0)

    data = np.load(cube_path)
    cube_raw = data["cube"]
    m_grid = data["m_grid"]
    tenor_days = data["tenor_days"]
    tau_grid = tenor_days / 365.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    diff = VolaDiff()
    diff.device = device
    diff.model = diff.model.to(device)
    diff.best_model = diff.best_model.to(device)
    diff.load(str(checkpoint_best))

    diffpen = VolaDiff()
    diffpen.device = device
    diffpen.model = diffpen.model.to(device)
    diffpen.best_model = diffpen.best_model.to(device)
    diffpen.load(str(checkpoint_pen))

    ckpt = torch.load(volgan_ckpt, map_location=device, weights_only=False)
    try:
        from src.other_models.VolGan import Generator
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"VolGAN dependencies are missing: {exc}")

    gen_sim = Generator(
        noise_dim=ckpt["noise_dim"],
        cond_dim=ckpt["cond_dim"],
        hidden_dim=ckpt["hidden_dim"],
        output_dim=ckpt["target_dim"],
        mean_in=ckpt["m_in"],
        std_in=ckpt["sigma_in"],
        mean_out=ckpt["m_out"],
        std_out=ckpt["sigma_out"],
    ).to(device)
    gen_sim.load_state_dict(ckpt["gen_state_dict"])
    gen_sim.eval()

    ck_vae = torch.load(vae_ckpt, map_location=device, weights_only=False)
    vae = IVS_VAE(
        n_features=ck_vae["n_features"],
        x_dim=ck_vae["x_dim"],
        latent_dim=ck_vae["latent_dim"],
        beta=ck_vae["beta"],
    ).to(device)
    vae.load_state_dict(ck_vae["state_dict"])
    vae.eval()

    vae_lo = torch.tensor(ck_vae["surf_lo"], dtype=torch.float32, device=device)
    vae_hi = torch.tensor(ck_vae["surf_hi"], dtype=torch.float32, device=device)

    fn_data = np.load(ivs_norm)
    feat_norm = FeatureNormalizer()
    feat_norm.lo, feat_norm.hi = fn_data["lo"], fn_data["hi"]

    return {
        "cube_raw": cube_raw,
        "m_grid": m_grid,
        "tau_grid": tau_grid,
        "device": device,
        "diff": diff,
        "diffpen": diffpen,
        "gen_sim": gen_sim,
        "vae": vae,
        "vae_lo": vae_lo,
        "vae_hi": vae_hi,
        "feat_norm": feat_norm,
        "ckpt": ckpt,
    }


def _entangle(s2d: np.ndarray, lk: int, lt: int) -> np.ndarray:
    out = np.zeros(lk * lt, dtype=float)
    for i in range(lt):
        out[i * lk : (i + 1) * lk] = s2d[:, i]
    return out


@pytest.mark.integration
def test_density_plots_display() -> None:
    _interactive_backend()
    data = _build_data()

    cube_raw = data["cube_raw"]
    m_grid = data["m_grid"]
    tau_grid = data["tau_grid"]
    device = data["device"]
    diff = data["diff"]
    diffpen = data["diffpen"]
    gen_sim = data["gen_sim"]
    vae = data["vae"]
    vae_lo = data["vae_lo"]
    vae_hi = data["vae_hi"]
    feat_norm = data["feat_norm"]
    ckpt = data["ckpt"]

    N_GEN = 100
    m_idx = np.linspace(0, len(m_grid) - 1, 32, dtype=int)
    real_iv_raw = cube_raw[m_idx, :, :].transpose(2, 1, 0)
    real_iv = resize_to_16x32(real_iv_raw, device)
    real_iv_frac = real_iv / 100.0
    x_train = resize_to_16x32(real_iv_raw[: int(0.8 * real_iv_raw.shape[0])], device)
    lo, hi = fit_normalizer(x_train)
    m_new = np.linspace(m_grid[m_idx].min(), m_grid[m_idx].max(), 32)
    tau_new = np.linspace(tau_grid.min(), tau_grid.max(), 16)

    with torch.no_grad():
        gen_raw = diff.sample_uncond(batch_size=N_GEN, return_history=False)
        gen_raw2 = diffpen.sample_uncond(batch_size=N_GEN, return_history=False)

    diff_iv_raw = denormalize(gen_raw, lo, hi).clamp(min=0.5)[:, 0].cpu().numpy()
    diff_iv_raw2 = denormalize(gen_raw2, lo, hi).clamp(min=0.5)[:, 0].cpu().numpy()

    _m_idx = np.array([np.argmin(np.abs(m_grid - mi)) for mi in ckpt["m"]])
    _t_idx = np.array([np.argmin(np.abs(tau_grid - ti)) for ti in ckpt["tau"]])
    cube_sub = cube_raw[np.ix_(_m_idx, _t_idx)] / 100.0
    surf_flat = np.stack([_entangle(cube_sub[:, :, t], ckpt["lk"], ckpt["lt"]) for t in range(cube_sub.shape[2])], axis=0)
    log_iv_all = np.log(np.clip(surf_flat, 1e-6, None))
    cond_all = np.concatenate([np.zeros((len(surf_flat), 3)), log_iv_all], axis=1)

    idx_s = np.random.choice(len(cond_all) - 1, N_GEN, replace=True)
    cond_t = torch.tensor(cond_all[idx_s], dtype=torch.float32, device=device)
    with torch.no_grad():
        noise = torch.randn(N_GEN, ckpt["noise_dim"], device=device)
        out_gen = gen_sim(noise, cond_t)

    log_inc = out_gen[:, 1:].cpu().numpy()
    past = cond_t[:, 3:].cpu().numpy()
    new_log = past + log_inc
    gan_iv_mat = np.exp(new_log.reshape(N_GEN, ckpt["lt"], ckpt["lk"]))
    gan_iv_raw = resize_to_16x32(gan_iv_mat, device).squeeze(1).cpu().numpy()

    features_real = extract_features(real_iv_frac, tau_new, m_new, verbose=False)
    feat_real_n = feat_norm.transform(features_real)
    feat_real_n_ctrl = feat_real_n[:, : vae.n_features]
    idx = np.random.choice(len(feat_real_n_ctrl), size=N_GEN, replace=True)
    y_vae = torch.tensor(feat_real_n_ctrl[idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        x_gen = vae.generate(y=y_vae, device=device)
    vae_iv_raw = denormalize(unflatten(x_gen).squeeze(1).cpu(), vae_lo, vae_hi)

    flat_real = real_iv_frac.reshape(-1)
    flat_diff = diff_iv_raw.reshape(-1) / 100.0
    flat_gan = gan_iv_raw.reshape(-1)
    flat_diffpen = diff_iv_raw2.reshape(-1) / 100.0
    flat_vae = vae_iv_raw.reshape(-1) / 100.0

    colors = {
        "Real": "black",
        "Diffusion": "steelblue",
        "VolGAN": "forestgreen",
        "Diffusion+Pen": "darkorange",
        "VAE": "purple",
    }

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    for label, data_arr in [
        ("Real", flat_real),
        ("Diffusion", flat_diff),
        ("VolGAN", flat_gan),
        ("Diffusion+Pen", flat_diffpen),
        ("VAE", flat_vae),
    ]:
        _kde_plot(ax1, data_arr, label=label, color=colors[label])
    ax1.set_xlim(-0.05, 0.6)
    ax1.set_ylim(0, 10)
    ax1.set_title("Distribution of Implied Volatility — Multi-model")
    ax1.set_xlabel("Implied Volatility")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.tight_layout()
    _show_current_plot()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(flat_real, bins=50, alpha=0.4, label="Real", density=True)
    ax2.hist(flat_diff, bins=50, alpha=0.4, label="Diffusion", density=True)
    ax2.hist(flat_gan, bins=50, alpha=0.4, label="VolGAN", density=True)
    ax2.hist(flat_vae, bins=50, alpha=0.4, label="VAE", density=True)
    ax2.set_title("Histogram of Implied Volatility")
    ax2.set_xlabel("Implied Volatility")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    _show_current_plot()

    real_np = real_iv_frac[:, 0].cpu().numpy()
    diff_np = diff_iv_raw / 100.0
    diffp_np = diff_iv_raw2 / 100.0
    gan_np = gan_iv_raw
    vae_np = vae_iv_raw / 100.0
    models = {
        "Real": real_np,
        "Diffusion": diff_np,
        "Diffusion+Pen": diffp_np,
        "VolGAN": gan_np,
        "VAE": vae_np,
    }

    fig3, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    t_indices = [2, 8, 13]
    t_labels = [f"Short  τ={tau_new[i]:.2f}yr" for i in t_indices]
    for ax, t_idx, t_lbl in zip(axes, t_indices, t_labels):
        for name, arr in models.items():
            ax.plot(m_new, arr[:, t_idx, :].mean(0), label=name, color=colors[name], linewidth=2)
        ax.set_title(t_lbl)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Implied Volatility")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig3.suptitle("Average Volatility Smile", fontsize=14, y=1.02)
    plt.tight_layout()
    _show_current_plot()

    fig4, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, t_idx, t_lbl in zip(axes, t_indices, t_labels):
        for name, arr in models.items():
            skews = np.gradient(arr[:, t_idx, :], m_new, axis=1)
            ax.plot(m_new, skews.mean(0), label=name, color=colors[name], linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(t_lbl)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("dIV / dK")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig4.suptitle("Average Volatility Skew  (dIV/dK)", fontsize=14, y=1.02)
    plt.tight_layout()
    _show_current_plot()

    assert fig1 is not None
    assert fig2 is not None
    assert fig3 is not None
    assert fig4 is not None

    # Optionally block and keep plot windows open for interactive inspection.
    # Set environment variable BLOCK_PLOTS=1 to enable.
    if os.environ.get("BLOCK_PLOTS", "").lower() in ("1", "true", "yes"):
        try:
            print("BLOCK_PLOTS set — showing plots until closed.")
            plt.show(block=True)
        except Exception:
            # If the display/backend doesn't support blocking show, fall back to a long pause.
            plt.pause(60.0)


if __name__ == "__main__":
    test_density_plots_display()
