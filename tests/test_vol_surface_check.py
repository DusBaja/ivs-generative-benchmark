from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest


pytestmark = pytest.mark.integration


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


FIGS_DIR = repo_root() / "figs"


def load_treated_cube() -> dict:
    cube_path = repo_root() / "data" / "treated" / "vol_surface_cube_test.npz"

    if not cube_path.exists():
        pytest.skip(f"No treated surface cube found at {cube_path}")

    data = np.load(cube_path)
    cube = np.asarray(data["cube"], dtype=float)
    m_grid = np.asarray(data["m_grid"], dtype=float)
    tenor_days = np.asarray(data["tenor_days"], dtype=float)

    return {
        "cube_path": cube_path,
        "cube": cube,
        "m_grid": m_grid,
        "tenor_days": tenor_days,
        "tau": tenor_days / 365.0,
        "keys": list(data.files),
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


def print_surface_diagnostics(name: str, surface: np.ndarray, m_grid: np.ndarray, tau: np.ndarray) -> None:
    finite = surface[np.isfinite(surface)]
    print(f"\n{name}")
    print(f"  shape: {surface.shape}")
    print(f"  m-grid: [{float(m_grid.min()):.4f}, {float(m_grid.max()):.4f}]")
    print(f"  tau:    [{float(tau.min()):.4f}, {float(tau.max()):.4f}]")
    print(
        "  stats: "
        f"min={float(finite.min()):.4f}  "
        f"mean={float(finite.mean()):.4f}  "
        f"max={float(finite.max()):.4f}"
    )
    print(f"  above 100%: {float(np.mean(finite > 100) * 100):.3f}%")
    print("  top-left 3x3:\n" + np.array2string(surface[:3, :3], precision=2, suppress_small=True))


def test_treated_surface_cube_has_expected_shape_and_keys(capsys):
    data = load_treated_cube()

    emit(capsys, f"\nKeys: {data['keys']}")
    emit(capsys, f"cube shape: {data['cube'].shape}")
    emit(capsys, f"m grid shape: {data['m_grid'].shape}")
    emit(capsys, f"tenor days shape: {data['tenor_days'].shape}")

    assert "cube" in data["keys"]
    assert "m_grid" in data["keys"]
    assert "tenor_days" in data["keys"]
    assert data["cube"].ndim == 3
    assert data["cube"].shape[0] == len(data["m_grid"])
    assert data["cube"].shape[1] == len(data["tenor_days"])


def test_treated_surface_cube_values_are_reasonable(capsys):
    data = load_treated_cube()
    cube = data["cube"]

    emit(capsys, "\nCube summary:")
    emit(capsys, f"  finite: {bool(np.isfinite(cube).all())}")
    emit(capsys, f"  min: {float(np.min(cube)):.4f}")
    emit(capsys, f"  mean: {float(np.mean(cube)):.4f}")
    emit(capsys, f"  max: {float(np.max(cube)):.4f}")
    emit(capsys, f"  pct > 100%: {float(np.mean(cube > 100) * 100):.3f}%")

    assert np.isfinite(cube).all()
    assert float(np.min(cube)) > 0.0
    assert float(np.mean(cube)) > 0.0
    assert float(np.max(cube)) < 200.0
    assert float(np.mean(cube > 100)) < 0.05


def test_treated_surface_samples_print_and_stay_positive(capsys):
    data = load_treated_cube()
    cube = data["cube"]
    m_grid = data["m_grid"]
    tau = data["tau"]

    for idx in [0, cube.shape[2] // 2, cube.shape[2] - 1]:
        with capsys.disabled():
            print_surface_diagnostics(f"Surface #{idx}", cube[:, :, idx], m_grid, tau)
        surface = cube[:, :, idx]
        assert np.isfinite(surface).all()
        assert float(np.min(surface)) > 0.0
        assert float(np.max(surface)) < 200.0


def test_treated_surface_mean_plot(capsys):
    _interactive_backend()
    data = load_treated_cube()
    cube = data["cube"]
    m_grid = data["m_grid"]
    tau = data["tau"]

    mean_surface = cube.mean(axis=2)
    sample_indices = [0, cube.shape[2] // 2, cube.shape[2] - 1]

    fig = plt.figure(figsize=(16, 5))
    M, T = np.meshgrid(m_grid, tau, indexing="ij")
    surfaces = [(mean_surface, "Mean surface")] + [(cube[:, :, idx], f"Surface #{idx}") for idx in sample_indices]

    for axis_index, (surface, title) in enumerate(surfaces, start=1):
        ax = fig.add_subplot(1, 4, axis_index, projection="3d")
        ax.plot_surface(M, T, surface, cmap="viridis", edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Moneyness")
        ax.set_ylabel("Tenor (years)")
        ax.set_zlabel("IV %")

    fig.suptitle("Treated volatility cube surfaces", y=1.02)
    plt.tight_layout()
    fig_path = _figure_path("vol_surface_check.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    emit(capsys, f"\nSaved plot: {fig_path}")
    _show_current_plot()

    assert fig_path.exists()

    if os.getenv("BLOCK_PLOTS", "0").lower() in ("1", "true", "yes"):
        emit(capsys, "BLOCK_PLOTS set — leaving vol surface figure open until closed.")
        plt.show(block=True)
