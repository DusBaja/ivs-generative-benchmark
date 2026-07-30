from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm as scipy_norm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_TREATED = REPO_ROOT / "data" / "treated"


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


def _relative_call(sigma: np.ndarray, m: np.ndarray, tau: np.ndarray, r: float = 0.0) -> np.ndarray:
    M, T = np.meshgrid(m, tau, indexing="ij")
    s = np.clip(sigma, 1e-8, None)
    T = np.clip(T, 1e-8, None)

    d1 = (-np.log(M) + T * (r + 0.5 * s**2)) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)

    return scipy_norm.cdf(d1) - M * np.exp(-r * T) * scipy_norm.cdf(d2)


def _arbitrage_details_single(surface: np.ndarray, m: np.ndarray, tau: np.ndarray, r: float = 0.0, eps: float = 1e-10) -> dict:
    sigma = np.asarray(surface, dtype=float)
    if np.nanmedian(sigma) > 2.0:
        sigma = sigma / 100.0

    C = _relative_call(sigma, m, tau, r=r)
    dm = np.diff(m)
    dt = np.diff(tau)

    calendar_raw = (C[:, :-1] - C[:, 1:]) / dt[None, :]
    calendar_violation = np.maximum(calendar_raw, 0.0)

    mono_raw = (C[1:, :] - C[:-1, :]) / dm[:, None]
    mono_violation = np.maximum(mono_raw, 0.0)

    left_slope = (C[1:-1, :] - C[:-2, :]) / dm[:-1, None]
    right_slope = (C[2:, :] - C[1:-1, :]) / dm[1:, None]
    butterfly_raw = left_slope - right_slope
    butterfly_violation = np.maximum(butterfly_raw, 0.0)

    p1 = float(calendar_violation.sum())
    p2 = float(mono_violation.sum())
    p3 = float(butterfly_violation.sum())

    return {
        "p_calendar": p1,
        "p_monotonicity": p2,
        "p_butterfly": p3,
        "phi": p1 + p2 + p3,
        "n_calendar_viol": int((calendar_violation > eps).sum()),
        "n_monotonicity_viol": int((mono_violation > eps).sum()),
        "n_butterfly_viol": int((butterfly_violation > eps).sum()),
        "calendar_matrix": calendar_violation,
        "monotonicity_matrix": mono_violation,
        "butterfly_matrix": butterfly_violation,
        "call_price": C,
    }


def _load_cube(cube_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not DATA_TREATED.exists():
        pytest.skip("Treated data folder is missing.")
    if not cube_path.exists():
        pytest.skip(f"Required cube file not found: {cube_path}")

    data = np.load(cube_path, allow_pickle=True)
    cube = data["cube"]
    m_grid = data["m_grid"] if "m_grid" in data.files else np.linspace(0.70, 1.30, cube.shape[0])
    tenor_days = data["tenor_days"] if "tenor_days" in data.files else np.array([7, 14, 21, 30, 60, 90, 180, 270, 365], dtype=float)
    dates = data["asof_dates"] if "asof_dates" in data.files else np.arange(cube.shape[2])
    tau = tenor_days / 365.0
    return cube, m_grid, tau, dates


def _dataset_label(cube_name: str) -> str:
    if cube_name == "vol_surface_cube.npz":
        return "SX5E (vol_surface_cube.npz)"
    if cube_name == "vol_surface_spx_long.npz":
        return "SPX (vol_surface_spx_long.npz)"
    return cube_name


@pytest.mark.integration
@pytest.mark.parametrize("cube_name", ["vol_surface_cube.npz", "vol_surface_spx_long.npz"])
def test_arbitrage_penalties_visualisation(cube_name: str) -> None:
    _interactive_backend()
    cube_path = DATA_TREATED / cube_name
    cube, m_grid, tau, dates = _load_cube(cube_path)
    dataset_label = _dataset_label(cube_name)

    rows = []
    details = []
    for k in range(cube.shape[2]):
        surface = cube[:, :, k]
        if not np.isfinite(surface).any():
            continue

        surface = (
            pd.DataFrame(surface)
            .interpolate(axis=0)
            .interpolate(axis=1)
            .ffill()
            .bfill()
            .values
        )

        d = _arbitrage_details_single(surface, m_grid, tau, r=0.0)
        details.append(d)
        rows.append(
            {
                "idx": k,
                "date": dates[k],
                "p_calendar": d["p_calendar"],
                "p_monotonicity": d["p_monotonicity"],
                "p_butterfly": d["p_butterfly"],
                "phi": d["phi"],
                "n_calendar_viol": d["n_calendar_viol"],
                "n_monotonicity_viol": d["n_monotonicity_viol"],
                "n_butterfly_viol": d["n_butterfly_viol"],
            }
        )

    assert rows, f"No valid surfaces found in cube {cube_name}."

    arb_df = pd.DataFrame(rows).sort_values("phi", ascending=False)
    assert np.isfinite(arb_df["phi"]).all()
    assert arb_df["phi"].max() >= 0.0

    plt.figure(figsize=(10, 4))
    plt.plot(arb_df.sort_values("idx")["idx"], arb_df.sort_values("idx")["phi"])
    plt.title(f"[{dataset_label}] Total arbitrage penalty phi across traded cube dates")
    plt.xlabel("Surface index")
    plt.ylabel("phi")
    plt.grid(True)
    _show_current_plot()

    plt.figure(figsize=(7, 4))
    plt.hist(arb_df["phi"], bins=30)
    plt.title(f"[{dataset_label}] Distribution of arbitrage penalty phi")
    plt.xlabel("phi")
    plt.ylabel("count")
    plt.grid(True)
    _show_current_plot()

    plt.figure(figsize=(8, 4))
    plt.boxplot([
        arb_df["p_calendar"],
        arb_df["p_monotonicity"],
        arb_df["p_butterfly"],
    ], labels=["calendar", "monotonicity", "butterfly"])
    plt.title(f"[{dataset_label}] Arbitrage penalty decomposition")
    plt.ylabel("penalty")
    plt.grid(True)
    _show_current_plot()

    worst_idx = int(arb_df.iloc[0]["idx"])
    worst = details[rows.index(next(r for r in rows if r["idx"] == worst_idx))]

    plt.figure(figsize=(8, 4))
    plt.imshow(worst["calendar_matrix"].T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(f"[{dataset_label}] Calendar violations: worst surface")
    plt.xlabel("moneyness index")
    plt.ylabel("tenor interval index")
    _show_current_plot()

    plt.figure(figsize=(8, 4))
    plt.imshow(worst["monotonicity_matrix"].T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(f"[{dataset_label}] Monotonicity violations: worst surface")
    plt.xlabel("moneyness interval index")
    plt.ylabel("tenor index")
    _show_current_plot()

    plt.figure(figsize=(8, 4))
    plt.imshow(worst["butterfly_matrix"].T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.title(f"[{dataset_label}] Butterfly violations: worst surface")
    plt.xlabel("moneyness interior index")
    plt.ylabel("tenor index")
    _show_current_plot()

    if os.getenv("BLOCK_PLOTS", "0") == "1":
        print(f"BLOCK_PLOTS=1 set; waiting for plot windows for {dataset_label} to be closed.")
        plt.show(block=True)
