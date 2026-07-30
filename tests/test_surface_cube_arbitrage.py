from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm as scipy_norm


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_call(
    sigma: np.ndarray,
    m: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """
    Relative Black-Scholes call price:
        c(m,tau) = N(d1) - m exp(-r tau) N(d2)

    sigma: (n_m, n_T), IV in fraction, e.g. 0.20 not 20
    m:     (n_m,)
    tau:   (n_T,)
    """
    M, T = np.meshgrid(m, tau, indexing="ij")
    s = np.clip(np.asarray(sigma, dtype=float), 1e-8, None)
    T = np.clip(T, 1e-8, None)

    d1 = (-np.log(M) + T * (r + 0.5 * s**2)) / (s * np.sqrt(T))
    d2 = d1 - s * np.sqrt(T)

    return scipy_norm.cdf(d1) - M * np.exp(-r * T) * scipy_norm.cdf(d2)


def arbitrage_details_single(
    surface: np.ndarray,
    m: np.ndarray,
    tau: np.ndarray,
    r: float = 0.0,
    eps: float = 1e-10,
) -> dict:
    sigma = np.asarray(surface, dtype=float)

    # If vols are stored in percent, convert to fraction.
    if np.nanmedian(sigma) > 2.0:
        sigma = sigma / 100.0

    C = relative_call(sigma, m, tau, r=r)

    dm = np.diff(m)
    dt = np.diff(tau)

    # Calendar: call price should increase with maturity.
    calendar_raw = (C[:, :-1] - C[:, 1:]) / dt[None, :]
    calendar_violation = np.maximum(calendar_raw, 0.0)

    # Call monotonicity: call price should decrease as m = K/S increases.
    mono_raw = (C[1:, :] - C[:-1, :]) / dm[:, None]
    mono_violation = np.maximum(mono_raw, 0.0)

    # Butterfly: call price should be convex in m.
    left_slope = (C[1:-1, :] - C[:-2, :]) / dm[:-1, None]
    right_slope = (C[2:, :] - C[1:-1]) / dm[1:, None]
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
    }


def _fill_nan_2d(surface: np.ndarray) -> np.ndarray:
    return (
        pd.DataFrame(surface)
        .interpolate(axis=0, limit_direction="both")
        .interpolate(axis=1, limit_direction="both")
        .ffill()
        .bfill()
        .values
    )


@pytest.fixture(scope="session")
def existing_cube_data() -> dict:
    """

    This fixture does NOT rebuild the cube and does NOT save anything.
    """
    cube_path = repo_root() / "data" / "treated" / "vol_surface_cube.npz"

    if not cube_path.exists():
        pytest.skip(f"No treated cube found at {cube_path}")

    data = np.load(cube_path, allow_pickle=True)

    cube = np.asarray(data["cube"])
    m_grid = (
        np.asarray(data["m_grid"], dtype=float)
        if "m_grid" in data.files
        else np.linspace(0.70, 1.30, cube.shape[0])
    )
    tenor_days = (
        np.asarray(data["tenor_days"], dtype=float)
        if "tenor_days" in data.files
        else np.array([7, 14, 21, 30, 60, 90, 180, 270, 365], dtype=float)
    )
    dates = (
        np.asarray(data["asof_dates"])
        if "asof_dates" in data.files
        else np.arange(cube.shape[2])
    )

    return {
        "cube_path": cube_path,
        "keys": list(data.files),
        "cube": cube,
        "m_grid": m_grid,
        "tenor_days": tenor_days,
        "dates": dates,
        "tau": tenor_days / 365.0,
    }


@pytest.fixture(scope="session")
def arbitrage_df(existing_cube_data: dict) -> pd.DataFrame:
    cube = existing_cube_data["cube"]
    m_grid = existing_cube_data["m_grid"]
    tau = existing_cube_data["tau"]
    dates = existing_cube_data["dates"]

    rows = []

    for k in range(cube.shape[2]):
        surface = cube[:, :, k]

        if not np.isfinite(surface).any():
            continue

        surface = _fill_nan_2d(surface)
        d = arbitrage_details_single(surface, m_grid, tau, r=0.0)

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

    assert rows, "No valid surfaces found in cube."

    return pd.DataFrame(rows).sort_values("phi", ascending=False).reset_index(drop=True)


@pytest.mark.integration
def test_existing_cube_has_required_arrays(existing_cube_data: dict):
    print("\nKeys:", existing_cube_data["keys"])

    assert "cube" in existing_cube_data["keys"]
    assert existing_cube_data["cube"].ndim == 3


@pytest.mark.integration
def test_existing_cube_shape_matches_grids(existing_cube_data: dict):
    cube = existing_cube_data["cube"]
    m_grid = existing_cube_data["m_grid"]
    tenor_days = existing_cube_data["tenor_days"]
    dates = existing_cube_data["dates"]

    print("cube shape:", cube.shape)
    print("m grid:", m_grid.shape, float(np.nanmin(m_grid)), float(np.nanmax(m_grid)))
    print("tenor days:", tenor_days)
    print("n dates:", len(dates))

    assert cube.shape[0] == len(m_grid)
    assert cube.shape[1] == len(tenor_days)
    assert cube.shape[2] == len(dates)


@pytest.mark.integration
def test_existing_cube_grids_are_valid(existing_cube_data: dict):
    m_grid = existing_cube_data["m_grid"]
    tenor_days = existing_cube_data["tenor_days"]
    tau = existing_cube_data["tau"]

    assert np.all(np.isfinite(m_grid))
    assert np.all(np.isfinite(tenor_days))
    assert np.all(np.diff(m_grid) > 0), "m_grid must be strictly increasing."
    assert np.all(np.diff(tenor_days) > 0), "tenor_days must be strictly increasing."
    assert np.all(tau > 0), "All maturities must be positive."


@pytest.mark.integration
def test_existing_cube_values_are_valid(existing_cube_data: dict):
    cube = existing_cube_data["cube"]

    assert np.isfinite(cube).any(), "Cube has no finite values."
    assert np.nanmin(cube) > 0.0, f"Min IV non-positive: {np.nanmin(cube)}"

    i_atm = int(np.argmin(np.abs(existing_cube_data["m_grid"] - 1.0)))
    atm_slice = cube[i_atm, :, :]

    assert np.isfinite(atm_slice).any(), "ATM slice is entirely NaN/inf."


@pytest.mark.integration
def test_existing_cube_calendar_arbitrage_is_negligible(arbitrage_df: pd.DataFrame):
    """
    Calendar was already basically clean in the notebook.
    This test is a real gate because p_calendar should be numerical noise only.
    """
    max_calendar = float(arbitrage_df["p_calendar"].max())
    print("\nMax calendar penalty:", max_calendar)

    assert np.isfinite(max_calendar)
    assert max_calendar < 1e-8


@pytest.mark.integration
def test_existing_cube_arbitrage_summary_is_finite(arbitrage_df: pd.DataFrame):
    """
    Diagnostic test: prints the same summary as the notebook.
    Does not fail because monotonicity/butterfly arbitrage exists.
    """
    print("\nTop 20 arbitrage surfaces:")
    print(arbitrage_df.head(20).to_string(index=False))

    print("\nArbitrage summary:")
    print("Mean phi:", float(arbitrage_df["phi"].mean()))
    print("Median phi:", float(arbitrage_df["phi"].median()))
    print("Max phi:", float(arbitrage_df["phi"].max()))
    print("Zero-arb %:", float(100 * (arbitrage_df["phi"] <= 1e-10).mean()))

    assert not arbitrage_df.empty
    assert np.all(np.isfinite(arbitrage_df["phi"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(arbitrage_df["p_calendar"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(arbitrage_df["p_monotonicity"].to_numpy(dtype=float)))
    assert np.all(np.isfinite(arbitrage_df["p_butterfly"].to_numpy(dtype=float)))


@pytest.mark.integration
def test_existing_cube_worst_surface_is_reported(arbitrage_df: pd.DataFrame):
    worst = arbitrage_df.iloc[0]

    print("\nWorst surface:")
    print(worst.to_string())

    assert int(worst["idx"]) >= 0
    assert float(worst["phi"]) >= 0.0
    assert float(worst["p_monotonicity"]) >= 0.0
    assert float(worst["p_butterfly"]) >= 0.0


@pytest.mark.integration
def test_existing_cube_zero_arbitrage_ratio_is_computable(arbitrage_df: pd.DataFrame):
    zero_arb_pct = float(100 * (arbitrage_df["phi"] <= 1e-10).mean())

    print("\nZero-arb %:", zero_arb_pct)

    assert np.isfinite(zero_arb_pct)
    assert 0.0 <= zero_arb_pct <= 100.0