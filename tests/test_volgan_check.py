from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from src.other_models.VolGan import Discriminator, Generator
from src.other_models.diffusion_cond_ext import resize_to_16x32


pytestmark = pytest.mark.integration


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


FIGS_DIR = repo_root() / "figs"


def entangle_surface(s2d: np.ndarray, lk: int, lt: int) -> np.ndarray:
    out = np.zeros(lk * lt, dtype=float)
    for i in range(lt):
        out[i * lk : (i + 1) * lk] = s2d[:, i]
    return out


def load_volgan_artifacts() -> dict:
    treated = repo_root() / "data" / "treated"
    cube_path = treated / "vol_surface_cube_test.npz"
    ckpt_path = treated / "volgan_checkpoint.pt"

    if not cube_path.exists():
        pytest.skip(f"Required cube not found: {cube_path}")
    if not ckpt_path.exists():
        pytest.skip(f"Required VolGAN checkpoint not found: {ckpt_path}")

    raw = np.load(cube_path, allow_pickle=True)
    cube = np.asarray(raw["cube"], dtype=float)
    m_grid = np.asarray(raw["m_grid"], dtype=float)
    tenor_days = np.asarray(raw["tenor_days"], dtype=float)
    asof_dates = np.asarray(raw["asof_dates"])
    tau_grid = tenor_days / 365.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    generator = Generator(
        noise_dim=ckpt["noise_dim"],
        cond_dim=ckpt["cond_dim"],
        hidden_dim=ckpt["hidden_dim"],
        output_dim=ckpt["target_dim"],
        mean_in=ckpt["m_in"],
        std_in=ckpt["sigma_in"],
        mean_out=ckpt["m_out"],
        std_out=ckpt["sigma_out"],
    ).to(device)
    generator.load_state_dict(ckpt["gen_state_dict"])
    generator.eval()

    discriminator = Discriminator(
        in_dim=ckpt["cond_dim"] + ckpt["target_dim"],
        hidden_dim=ckpt["hidden_dim"],
        mean=torch.cat((ckpt["m_in"], ckpt["m_out"]), dim=-1),
        std=torch.cat((ckpt["sigma_in"], ckpt["sigma_out"]), dim=-1),
    ).to(device)
    discriminator.load_state_dict(ckpt["disc_state_dict"])
    discriminator.eval()

    m_idx = np.array([np.argmin(np.abs(m_grid - mi)) for mi in ckpt["m"]])
    t_idx = np.array([np.argmin(np.abs(tau_grid - ti)) for ti in ckpt["tau"]])
    cube_sub = cube[np.ix_(m_idx, t_idx)] / 100.0

    surf_flat = np.stack(
        [entangle_surface(cube_sub[:, :, t], ckpt["lk"], ckpt["lt"]) for t in range(cube_sub.shape[2])],
        axis=0,
    )
    log_iv_all = np.log(np.clip(surf_flat, 1e-6, None))
    cond_all = np.concatenate([np.zeros((len(surf_flat), 3)), log_iv_all], axis=1)

    return {
        "cube": cube,
        "m_grid": m_grid,
        "tenor_days": tenor_days,
        "tau_grid": tau_grid,
        "asof_dates": asof_dates,
        "ckpt": ckpt,
        "generator": generator,
        "discriminator": discriminator,
        "device": device,
        "cond_all": cond_all,
    }


def _interactive_backend() -> None:
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


def _figure_path(name: str) -> Path:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    return FIGS_DIR / name


def emit(capsys: Any, message: str) -> None:
    with capsys.disabled():
        print(message, flush=True)


def sample_volgan_surfaces(data: dict, n_gen: int = 8) -> np.ndarray:
    ckpt = data["ckpt"]
    generator = data["generator"]
    device = data["device"]
    cond_all = data["cond_all"]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(cond_all) - 1, n_gen, replace=True)
    cond_t = torch.tensor(cond_all[idx], dtype=torch.float32, device=device)
    noise = torch.randn(n_gen, ckpt["noise_dim"], device=device)

    with torch.no_grad():
        out = generator(noise, cond_t)

    log_inc = out[:, 1:].detach().cpu().numpy()
    past = cond_t[:, 3:].detach().cpu().numpy()
    new_log = past + log_inc
    gan_iv_mat = np.exp(new_log.reshape(n_gen, ckpt["lt"], ckpt["lk"]))
    gan_iv_raw = resize_to_16x32(gan_iv_mat, device).squeeze(1).cpu().numpy() * 100.0
    return gan_iv_raw


def print_surface_diagnostics(name: str, surface: np.ndarray, m_grid: np.ndarray, tau_grid: np.ndarray) -> None:
    finite = surface[np.isfinite(surface)]
    print(f"\n{name}")
    print(f"  shape: {surface.shape}")
    print(f"  m-grid: [{float(m_grid.min()):.4f}, {float(m_grid.max()):.4f}]")
    print(f"  tau:    [{float(tau_grid.min()):.4f}, {float(tau_grid.max()):.4f}]")
    print(
        "  stats: "
        f"min={float(finite.min()):.4f}  "
        f"mean={float(finite.mean()):.4f}  "
        f"max={float(finite.max()):.4f}"
    )
    print(f"  above 100%: {float(np.mean(finite > 100) * 100):.3f}%")
    print("  top-left 3x3:\n" + np.array2string(surface[:3, :3], precision=2, suppress_small=True))


def test_volgan_checkpoint_loads_and_roundtrips(capsys):
    data = load_volgan_artifacts()
    ckpt = data["ckpt"]
    generator = data["generator"]
    discriminator = data["discriminator"]

    emit(capsys, f"\nVolGAN checkpoint keys: {sorted(ckpt.keys())}")
    emit(
        capsys,
        "generator dims: "
        f"{{'noise_dim': {ckpt['noise_dim']}, 'cond_dim': {ckpt['cond_dim']}, "
        f"'hidden_dim': {ckpt['hidden_dim']}, 'target_dim': {ckpt['target_dim']}}}",
    )
    emit(capsys, f"grid dims: {{'lk': {ckpt['lk']}, 'lt': {ckpt['lt']}}}")
    emit(capsys, f"condition matrix shape: {data['cond_all'].shape}")

    cond = torch.zeros(2, ckpt["cond_dim"], device=data["device"])
    noise = torch.randn(2, ckpt["noise_dim"], device=data["device"])
    with torch.no_grad():
        out = generator(noise, cond)
        disc_out = discriminator(torch.cat([cond, out], dim=-1))

    emit(capsys, f"generator output shape: {tuple(out.shape)}")
    emit(capsys, f"generator output range: {float(out.min())} {float(out.max())}")
    emit(capsys, f"discriminator output shape: {tuple(disc_out.shape)}")
    emit(capsys, f"discriminator output range: {float(disc_out.min())} {float(disc_out.max())}")

    assert out.shape == (2, ckpt["target_dim"])
    assert disc_out.shape == (2, 1)
    assert torch.isfinite(out).all()
    assert torch.isfinite(disc_out).all()
    assert float(disc_out.min()) >= 0.0
    assert float(disc_out.max()) <= 1.0


def test_volgan_generated_surfaces_are_positive_and_reasonable(capsys):
    _interactive_backend()
    data = load_volgan_artifacts()
    ckpt = data["ckpt"]
    surfaces = sample_volgan_surfaces(data, n_gen=8)

    emit(capsys, "\nGenerated VolGAN surfaces:")
    emit(capsys, f"  shape: {surfaces.shape}")
    emit(capsys, f"  finite: {bool(np.isfinite(surfaces).all())}")
    emit(capsys, f"  min: {float(np.min(surfaces)):.4f}")
    emit(capsys, f"  mean: {float(np.mean(surfaces)):.4f}")
    emit(capsys, f"  max: {float(np.max(surfaces)):.4f}")
    emit(capsys, f"  pct > 100%: {float(np.mean(surfaces > 100) * 100):.3f}%")

    m_plot = np.linspace(data["m_grid"].min(), data["m_grid"].max(), surfaces.shape[2])
    tau_plot = np.linspace(data["tau_grid"].min(), data["tau_grid"].max(), surfaces.shape[1])
    with capsys.disabled():
        print_surface_diagnostics("VolGAN sample #0", surfaces[0], m_plot, tau_plot)
        print_surface_diagnostics("VolGAN sample #last", surfaces[-1], m_plot, tau_plot)

    fig = plt.figure(figsize=(16, 5))
    M, T = np.meshgrid(m_plot, tau_plot)
    plot_surfaces = [(surfaces.mean(axis=0), "Mean generated surface"), (surfaces[0], "Sample #0"), (surfaces[-1], "Sample #last")]
    for axis_index, (surface, title) in enumerate(plot_surfaces, start=1):
        ax = fig.add_subplot(1, 3, axis_index, projection="3d")
        ax.plot_surface(M, T, surface, cmap="plasma", edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Tenor (years)")
        ax.set_zlabel("IV %")

    fig.suptitle("VolGAN generated volatility surfaces", y=1.02)
    plt.tight_layout()
    fig_path = _figure_path("volgan_check.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    emit(capsys, f"\nSaved plot: {fig_path}")
    _show_current_plot()

    assert surfaces.shape == (8, 16, 32)
    assert np.isfinite(surfaces).all()
    assert float(np.min(surfaces)) > 0.0
    assert float(np.mean(surfaces)) > 0.0
    assert float(np.max(surfaces)) < 100.0
    assert float(np.mean(surfaces > 100)) == 0.0

    assert fig_path.exists()

    if os.getenv("BLOCK_PLOTS", "0").lower() in ("1", "true", "yes"):
        emit(capsys, "BLOCK_PLOTS set — leaving VolGAN figure open until closed.")
        plt.show(block=True)
