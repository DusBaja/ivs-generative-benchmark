"""
    - LatentSB/SBJTS/LightSB outputs are true simulated paths when their NPZ contains `paths`.
    - Most FM models are static surface generators. To include them in path criteria,
      this bridge groups independently generated surfaces into pseudo-paths. Their dynamic
      scores are therefore IID-pseudo-path diagnostics, not a true temporal path claim.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pickle
import numpy as np
import pandas as pd

from scipy.interpolate import RegularGridInterpolator
from scipy.stats import wasserstein_distance, ks_2samp, norm

def to_fraction(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return x / 100.0 if np.isfinite(med) and med > 2.0 else x


def to_percent(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return x * 100.0 if np.isfinite(med) and med < 2.0 else x


def ensure_surfaces_t_m(x: np.ndarray) -> np.ndarray:
    """Convert common arrays to (N,Ntau,Nm)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 4 and x.shape[1] == 1:     # (N,1,T,M)
        return x[:, 0]
    if x.ndim == 3:                         # already (N,T,M)
        return x
    if x.ndim == 2:                         # one surface (T,M)
        return x[None]
    raise ValueError(f"Cannot convert array with shape {x.shape} to (N,Ntau,Nm)")


def ensure_paths_t_m(x: np.ndarray) -> np.ndarray:
    """Convert common path arrays to (N_paths,path_length,Ntau,Nm)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 4:
        raise ValueError(f"Expected 4D paths, got {x.shape}")
    # Ambiguity between (P,L,T,M) and (P,L,M,T). We leave as-is; caller must resample/transpose if needed.
    return x


def resample_surfaces_t_m(surfaces_t_m: np.ndarray, tau_src: np.ndarray, m_src: np.ndarray,
                          tau_tgt: np.ndarray, m_tgt: np.ndarray) -> np.ndarray:
    """Bilinear interpolation from source (tau,m) grid to target (tau,m) grid."""
    surfaces = ensure_surfaces_t_m(surfaces_t_m)
    tau_src = np.asarray(tau_src, dtype=float)
    m_src = np.asarray(m_src, dtype=float)
    tau_tgt = np.asarray(tau_tgt, dtype=float)
    m_tgt = np.asarray(m_tgt, dtype=float)

    t_order = np.argsort(tau_src)
    m_order = np.argsort(m_src)
    tau_src = tau_src[t_order]
    m_src = m_src[m_order]
    surfaces = surfaces[:, t_order, :][:, :, m_order]

    TT, MM = np.meshgrid(tau_tgt, m_tgt, indexing="ij")
    pts = np.column_stack([TT.ravel(), MM.ravel()])
    out = np.empty((surfaces.shape[0], len(tau_tgt), len(m_tgt)), dtype=float)
    for i, s in enumerate(surfaces):
        f = RegularGridInterpolator((tau_src, m_src), s, bounds_error=False, fill_value=None)
        out[i] = f(pts).reshape(len(tau_tgt), len(m_tgt))
    return out

def resample_paths_t_m(paths_t_m: np.ndarray, tau_src: np.ndarray, m_src: np.ndarray,
                       tau_tgt: np.ndarray, m_tgt: np.ndarray) -> np.ndarray:
    p = np.asarray(paths_t_m, dtype=float)
    if p.ndim != 4:
        raise ValueError(f"Expected paths (P,L,T,M), got {p.shape}")
    P, L = p.shape[:2]
    flat = p.reshape(P * L, p.shape[2], p.shape[3])
    flat_rs = resample_surfaces_t_m(flat, tau_src, m_src, tau_tgt, m_tgt)
    return flat_rs.reshape(P, L, len(tau_tgt), len(m_tgt))

def load_latentsb_generated_paths_npz(path: str | Path) -> Dict[str, Any]:
    """Load our generated NPZ into canonical true path format (P,L,T,M)."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    m_grid = np.asarray(data["m_grid"], dtype=float)
    tau_grid = np.asarray(data["tenor_days"], dtype=float) / 365.0 if "tenor_days" in data.files else np.asarray(data["tau_grid"], dtype=float)
    if "paths" in data.files:
        paths = np.asarray(data["paths"], dtype=float)
        # Our convention: (P,L,Nm,Ntau) -> canonical (P,L,Ntau,Nm)
        if paths.ndim != 4:
            raise ValueError(f"paths in {path} should be 4D, got {paths.shape}")
        paths_t_m = np.transpose(paths, (0, 1, 3, 2))
    elif "cube" in data.files:
        cube = np.asarray(data["cube"], dtype=float)
        # cube (Nm,Ntau,N) -> surfaces (N,Ntau,Nm), then one pseudo path
        if cube.shape[0] == len(m_grid) and cube.shape[1] == len(tau_grid):
            surf_m_t = np.moveaxis(cube, 2, 0)
            surf_t_m = np.transpose(surf_m_t, (0, 2, 1))
            paths_t_m = surf_t_m[None]
        else:
            raise ValueError(f"Cannot infer cube orientation for {path}: {cube.shape}")
    else:
        raise KeyError(f"{path} has neither paths nor cube")
    return {
        "paths_t_m_frac": to_fraction(paths_t_m),
        "surfaces_t_m_frac": to_fraction(paths_t_m.reshape(-1, paths_t_m.shape[-2], paths_t_m.shape[-1])),
        "m_grid": m_grid,
        "tau_grid": tau_grid,
        "path_kind": "true_path",
        "source_path": str(path),
    }


def static_surfaces_to_pseudo_paths(surfaces_t_m_frac: np.ndarray, path_length: int,
                                    n_paths: Optional[int] = None, seed: int = 42,
                                    prepend_initial_surface: Optional[np.ndarray] = None) -> np.ndarray:
    """Group independent static samples into pseudo paths so path diagnostics can be computed.

    This does NOT turn a static generator into a real path model. It only creates an IID path
    diagnostic under the same metrics.
    """
    s = to_fraction(ensure_surfaces_t_m(surfaces_t_m_frac))
    rng = np.random.default_rng(seed)
    if prepend_initial_surface is not None:
        # first surface in each pseudo path is the last observed train surface; generated surfaces fill remaining steps
        L_gen = max(int(path_length) - 1, 1)
        max_paths = len(s) // L_gen
        if n_paths is None:
            n_paths = max_paths
        n_paths = int(min(n_paths, max_paths))
        idx = rng.permutation(len(s))[: n_paths * L_gen]
        body = s[idx].reshape(n_paths, L_gen, s.shape[1], s.shape[2])
        init = np.asarray(prepend_initial_surface, dtype=float)[None, None]
        init = np.repeat(init, n_paths, axis=0)
        return np.concatenate([init, body], axis=1)
    else:
        max_paths = len(s) // int(path_length)
        if n_paths is None:
            n_paths = max_paths
        n_paths = int(min(n_paths, max_paths))
        if n_paths < 1:
            raise ValueError(f"Not enough surfaces ({len(s)}) for path_length={path_length}")
        idx = rng.permutation(len(s))[: n_paths * int(path_length)]
        return s[idx].reshape(n_paths, int(path_length), s.shape[1], s.shape[2])


# Metrics: surface + path + arbitrage
def _finite_flat(x):
    x = np.asarray(x, dtype=float).ravel()
    return x[np.isfinite(x)]


def _robust_scale(y):
    y = _finite_flat(y)
    if len(y) < 2:
        return 1.0
    iqr = np.nanquantile(y, 0.75) - np.nanquantile(y, 0.25)
    if np.isfinite(iqr) and iqr > 1e-12:
        return float(iqr)
    sd = np.nanstd(y)
    if np.isfinite(sd) and sd > 1e-12:
        return float(sd)
    return max(abs(np.nanmedian(y)), 1e-6)


def _w1_scaled(x, y):
    x = _finite_flat(x); y = _finite_flat(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(wasserstein_distance(x, y) / _robust_scale(y))


def _ks(x, y):
    x = _finite_flat(x); y = _finite_flat(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return float(ks_2samp(x, y).statistic)


def dlog_iv_path(paths_t_m_frac: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(paths_t_m_frac, dtype=float), 1e-8, 10.0)
    if p.shape[1] < 2:
        return np.empty((p.shape[0], 0) + p.shape[2:])
    return np.diff(np.log(p), axis=1)


def path_step_norms(paths_t_m_frac: np.ndarray) -> np.ndarray:
    d = dlog_iv_path(paths_t_m_frac)
    if d.size == 0:
        return np.array([])
    return np.sqrt(np.nanmean(d.reshape(d.shape[0], d.shape[1], -1) ** 2, axis=2)).ravel()


def path_autocorr_factor(paths_t_m_frac: np.ndarray) -> float:
    d = dlog_iv_path(paths_t_m_frac)
    if d.shape[1] < 3:
        return np.nan
    x0 = d[:, :-1].reshape(-1)
    x1 = d[:, 1:].reshape(-1)
    if np.nanstd(x0) < 1e-12 or np.nanstd(x1) < 1e-12:
        return np.nan
    return float(np.corrcoef(x0, x1)[0, 1])


def surface_features(surfaces_t_m_frac: np.ndarray, tau_grid: np.ndarray, m_grid: np.ndarray) -> pd.DataFrame:
    s = to_fraction(ensure_surfaces_t_m(surfaces_t_m_frac))
    tau_grid = np.asarray(tau_grid, dtype=float)
    m_grid = np.asarray(m_grid, dtype=float)
    atm_idx = int(np.nanargmin(np.abs(m_grid - 1.0)))
    short_idx = 0
    long_idx = len(tau_grid) - 1
    rows = []
    for surf in s:
        smile = surf[short_idx]
        grad = np.gradient(smile, m_grid)
        curv = np.gradient(grad, m_grid)
        rows.append({
            "atm_level": float(surf[short_idx, atm_idx]),
            "atm_slope": float(grad[atm_idx]),
            "atm_curvature": float(curv[atm_idx]),
            "term_slope": float((surf[long_idx, atm_idx] - surf[short_idx, atm_idx]) / max(tau_grid[long_idx] - tau_grid[short_idx], 1e-8)),
            "global_mean": float(np.nanmean(surf)),
            "global_std": float(np.nanstd(surf)),
            "short_left": float(surf[short_idx, 0]),
            "short_right": float(surf[short_idx, -1]),
        })
    return pd.DataFrame(rows)


def relative_call_from_iv(sigma_t_m_frac: np.ndarray, m_grid: np.ndarray, tau_grid: np.ndarray, r: float = 0.0) -> np.ndarray:
    sigma = np.clip(to_fraction(sigma_t_m_frac), 1e-8, 10.0)
    if sigma.ndim == 2:
        sigma = sigma[None]
    tau = np.asarray(tau_grid, dtype=float)
    logm = np.log(np.clip(np.asarray(m_grid, dtype=float), 1e-8, None))
    T, M = np.meshgrid(tau, logm, indexing="ij")
    T = np.clip(T, 1e-12, None)[None]
    M = M[None]
    d1 = (-M + (float(r) + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.clip(norm.cdf(d1) - np.exp(M - float(r) * T) * norm.cdf(d2), 1e-12, None)


def arbitrage_diagnostics(surfaces_t_m_frac: np.ndarray, tau_grid: np.ndarray, m_grid: np.ndarray) -> Dict[str, float]:
    C = relative_call_from_iv(surfaces_t_m_frac, m_grid, tau_grid)
    tau = np.asarray(tau_grid, dtype=float)
    logm = np.log(np.clip(np.asarray(m_grid, dtype=float), 1e-8, None))
    dt = np.diff(tau)
    dm = np.diff(logm)
    cal = np.maximum(0.0, -(C[:, 1:, :] - C[:, :-1, :]) / dt[None, :, None])
    mono = np.maximum(0.0, (C[:, :, 1:] - C[:, :, :-1]) / dm[None, None, :])
    d2 = C[:, :, 2:] - 2.0 * C[:, :, 1:-1] + C[:, :, :-2]
    bfly = np.maximum(0.0, -d2)
    p1 = cal.sum(axis=(1,2)); p2 = mono.sum(axis=(1,2)); p3 = bfly.sum(axis=(1,2)); phi = p1 + p2 + p3
    return {
        "mean_phi": float(np.nanmean(phi)),
        "q95_phi": float(np.nanquantile(phi, 0.95)),
        "zero_phi_rate": float(np.nanmean(phi <= 1e-12)),
        "n_calendar_viol": int(np.sum(cal > 1e-10)),
        "n_monotonicity_viol": int(np.sum(mono > 1e-10)),
        "n_butterfly_viol": int(np.sum(bfly > 1e-10)),
        "calendar_mean": float(np.nanmean(p1)),
        "monotonicity_mean": float(np.nanmean(p2)),
        "butterfly_mean": float(np.nanmean(p3)),
    }


def rolling_reference_paths(train_t_m_frac: np.ndarray, valid_t_m_frac: np.ndarray, path_length: int) -> Tuple[np.ndarray, np.ndarray]:
    train = to_fraction(ensure_surfaces_t_m(train_t_m_frac))
    valid = to_fraction(ensure_surfaces_t_m(valid_t_m_frac))
    L = int(min(path_length, len(valid)))
    valid_path = valid[:L][None]
    windows = []
    if len(train) >= L:
        for i in range(len(train) - L + 1):
            windows.append(train[i:i+L])
    train_paths = np.asarray(windows, dtype=float) if windows else train[-L:][None]
    return train_paths, valid_path


def common_path_quality_metrics(paths_t_m_frac: np.ndarray,
                                train_t_m_frac: np.ndarray,
                                valid_t_m_frac: np.ndarray,
                                tau_grid: np.ndarray,
                                m_grid: np.ndarray,
                                label: str = "",
                                path_kind: str = "true_path") -> Dict[str, float]:
    p = to_fraction(np.asarray(paths_t_m_frac, dtype=float))
    if p.ndim != 4:
        raise ValueError(f"Expected paths (P,L,T,M), got {p.shape}")
    train = to_fraction(ensure_surfaces_t_m(train_t_m_frac))
    valid = to_fraction(ensure_surfaces_t_m(valid_t_m_frac))
    P, L, T, M = p.shape
    gen_surf = p.reshape(P * L, T, M)
    ref_train_paths, ref_valid_path = rolling_reference_paths(train, valid, min(L, len(valid)))

    # Surface distribution block.
    feat_gen = surface_features(gen_surf, tau_grid, m_grid)
    feat_valid = surface_features(valid, tau_grid, m_grid)
    feat_train = surface_features(train, tau_grid, m_grid)
    fs = []
    for col in feat_gen.columns:
        fs.append(0.70 * _w1_scaled(feat_gen[col], feat_valid[col]) + 0.30 * _w1_scaled(feat_gen[col], feat_train[col]))
    level_score = float(np.nanmean(fs))

    # Dynamic/path realism, computed within each path.
    gd = dlog_iv_path(p).reshape(-1)
    vd = dlog_iv_path(ref_valid_path).reshape(-1)
    td = dlog_iv_path(ref_train_paths).reshape(-1)
    delta_score = float(0.70 * _w1_scaled(gd, vd) + 0.30 * _w1_scaled(gd, td) + 0.10 * _ks(gd, vd))
    step_score = float(0.70 * _w1_scaled(path_step_norms(p), path_step_norms(ref_valid_path)) + 0.30 * _w1_scaled(path_step_norms(p), path_step_norms(ref_train_paths)))

    # Initial consistency from last training surface to first generated surface.
    last = train[-1]
    init_d = np.sqrt(np.nanmean((np.log(np.clip(p[:, 0], 1e-8, 10.0)) - np.log(np.clip(last, 1e-8, 10.0)))**2, axis=(1,2)))
    ref_init = np.sqrt(np.nanmean((np.log(np.clip(valid[0], 1e-8, 10.0)) - np.log(np.clip(last, 1e-8, 10.0)))**2))
    initial_score = float(abs(np.log((np.nanmedian(init_d) + 1e-12) / (ref_init + 1e-12))))

    # Autocorrelation / temporal structure.
    gen_ac = path_autocorr_factor(p)
    val_ac = path_autocorr_factor(ref_valid_path)
    tr_ac = path_autocorr_factor(ref_train_paths)
    ref_ac = np.nanmean([x for x in [val_ac, tr_ac] if np.isfinite(x)]) if np.isfinite(val_ac) or np.isfinite(tr_ac) else 0.0
    autocorr_score = float(abs((gen_ac if np.isfinite(gen_ac) else 0.0) - ref_ac))

    arb = arbitrage_diagnostics(gen_surf, tau_grid, m_grid)
    arb_score = float(np.log1p(arb["mean_phi"]) + 0.50*np.log1p(arb["q95_phi"]) + 0.20*(1.0-arb["zero_phi_rate"]))
    calendar_count_penalty = float(np.log1p(arb["n_calendar_viol"] / 1000.0))
    smile_shape_count_penalty = float(np.log1p((arb["n_monotonicity_viol"] + arb["n_butterfly_viol"]) / 1000.0))

    surface_distribution_block = level_score
    surface_noarb_block = arb_score + 0.20*calendar_count_penalty + 0.20*smile_shape_count_penalty
    dynamic_realism_block = 0.55*delta_score + 0.30*step_score + 0.15*autocorr_score
    initial_consistency_block = initial_score

    # For pseudo-IID paths, report same metrics but add a small transparency penalty in the final ranking.
    pseudo_path_penalty = 0.15 if path_kind != "true_path" else 0.0
    path_quality_common_score = (
        0.40 * surface_distribution_block
        + 0.28 * surface_noarb_block
        + 0.27 * dynamic_realism_block
        + 0.05 * initial_consistency_block
        + pseudo_path_penalty
    )

    return {
        "label": label,
        "path_kind": path_kind,
        "n_paths": int(P),
        "path_length": int(L),
        "path_quality_common_score": float(path_quality_common_score),
        "surface_distribution_block": float(surface_distribution_block),
        "surface_noarb_block": float(surface_noarb_block),
        "dynamic_realism_block": float(dynamic_realism_block),
        "initial_consistency_block": float(initial_consistency_block),
        "pseudo_path_penalty": float(pseudo_path_penalty),
        "level_score": level_score,
        "delta_score": delta_score,
        "step_score": step_score,
        "autocorr_score": autocorr_score,
        "initial_score": initial_score,
        "arb_score": arb_score,
        "calendar_count_penalty": calendar_count_penalty,
        "smile_shape_count_penalty": smile_shape_count_penalty,
        **arb,
    }


def compare_path_model_dict(paths_dict: Dict[str, np.ndarray],
                            train_t_m_frac: np.ndarray,
                            valid_t_m_frac: np.ndarray,
                            tau_grid: np.ndarray,
                            m_grid: np.ndarray,
                            path_kinds: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    rows = []
    path_kinds = path_kinds or {}
    for name, paths in paths_dict.items():
        try:
            metrics = common_path_quality_metrics(paths, train_t_m_frac, valid_t_m_frac, tau_grid, m_grid,
                                                  label=name, path_kind=path_kinds.get(name, "true_path"))
            metrics["model"] = name
            metrics["status"] = "ok"
            rows.append(metrics)
        except Exception as exc:
            rows.append({"model": name, "label": name, "status": "failed", "error": repr(exc)})
    df = pd.DataFrame(rows)
    if "path_quality_common_score" in df.columns:
        df = df.sort_values(["status", "path_quality_common_score"], ascending=[True, True])
    return df

