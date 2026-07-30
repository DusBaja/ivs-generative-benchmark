"""
IVS Shape Feature Extraction — Wang et al. (2025), Section 3
=============================================================
Adapted for real market data (SX5E, absolute moneyness K/S).

STRATEGY:
  Level       : direct linear interpolation at lm=0  (no regression bias)
  Slope       : local quadratic fit using only the N_LOCAL points nearest ATM
  Curvature   : same local quadratic, second derivative at lm=0
  Term-struct : linear fit in τ at the ATM column

m_grid must be absolute moneyness K/S (e.g. 0.7…1.3).
ATM is at lm = log(K/S) = 0.
"""

import numpy as np
from sklearn.linear_model import Ridge


# ─────────────────────────────────────────────────────────────────────────────
# Single-surface extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_single(surface_frac, tau_grid, m_grid,
                             n_local=12, ts_deg=1, alpha=1e-4):
    """
    Parameters
    ----------
    surface_frac : (16, 32)  IV fraction, axis-0=tenor, axis-1=moneyness
    tau_grid     : (16,)  years
    m_grid       : (32,)  absolute moneyness K/S
    n_local      : number of moneyness points centred on ATM for smile fit
                   (use fewer points → less wing contamination)
    ts_deg       : polynomial degree for term-structure fit

    Returns
    -------
    (4,) : [level, slope, curvature, term_struct]
    """
    lm_grid = np.log(m_grid)                      # ATM at lm = 0
    atm_col = np.argmin(np.abs(lm_grid))

    smile   = surface_frac[0, :]                  # (32,)  at shortest tenor

    # ── 1. Level  : direct interpolation at lm = 0 ───────────────────────
    level = float(np.interp(0.0, lm_grid, smile))

    # ── 2. Slope & curvature : LOCAL quadratic around ATM ────────────────
    half  = n_local // 2
    lo    = max(atm_col - half, 0)
    hi    = min(atm_col + half + 1, len(lm_grid))
    lm_loc  = lm_grid[lo:hi]
    sml_loc = smile[lo:hi]

    lm_scale = max(np.abs(lm_loc).max(), 1e-6)
    lm_n     = lm_loc / lm_scale

    # Quadratic design matrix  [1, lm_n, lm_n²]
    A = np.column_stack([lm_n**0, lm_n**1, lm_n**2])
    reg = Ridge(alpha=alpha, fit_intercept=False)
    reg.fit(A, sml_loc)
    c = reg.coef_                   # [a0, a1, a2] in normalised space

    # Slope  ∂σ/∂lm at lm=0  →  a1 / lm_scale
    slope     = c[1] / lm_scale
    # Curvature ∂²σ/∂lm² at lm=0  →  2·a2 / lm_scale²
    curvature = 2.0 * c[2] / lm_scale**2

    # ── 3. Term-structure : linear fit in τ at ATM column ────────────────
    ts        = surface_frac[:, atm_col]          # (16,)
    tau_ref   = tau_grid[0]
    tau_scale = max(tau_grid[-1] - tau_grid[0], 1e-6)
    tau_n     = (tau_grid - tau_ref) / tau_scale

    A_ts = np.column_stack([tau_n**k for k in range(ts_deg + 1)])
    reg2 = Ridge(alpha=alpha, fit_intercept=False)
    reg2.fit(A_ts, ts)
    d = reg2.coef_

    term_struct = d[1] / tau_scale                # ∂σ/∂τ in years⁻¹

    return np.array([level, slope, curvature, term_struct], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Batch extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(real_iv_frac, tau_new, m_new,
                     n_local=12, ts_deg=1, alpha=1e-4, verbose=True):
    """
    Parameters
    ----------
    real_iv_frac : (n, 1, 16, 32) or (n, 16, 32)  IV fraction
    tau_new      : (16,)  years
    m_new        : (32,)  absolute moneyness K/S

    Returns
    -------
    features : (n, 4)   [level, slope, curvature, term_struct]
    """
    arr = real_iv_frac
    if hasattr(arr, "cpu"): arr = arr.cpu().numpy()
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 4: arr = arr[:, 0, :, :]

    n = arr.shape[0]
    features = np.zeros((n, 4))
    for i in range(n):
        features[i] = extract_features_single(
            arr[i], tau_new, m_new,
            n_local=n_local, ts_deg=ts_deg, alpha=alpha
        )
        if verbose and (i + 1) % 10 == 0:
            print(f"  [{i+1:>4}/{n}]", end="\r")
    if verbose:
        print(f"  [{n:>4}/{n}]  done ✓")
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

def sanity_check(real_iv_frac, tau_new, m_new, features):
    arr = real_iv_frac
    if hasattr(arr, "cpu"): arr = arr.cpu().numpy()
    if arr.ndim == 4: arr = arr[:, 0, :, :]

    lm_grid = np.log(m_new)
    atm_col = np.argmin(np.abs(lm_grid))

    direct = np.array([np.interp(0.0, lm_grid, arr[i, 0, :]) for i in range(len(arr))])
    reg    = features[:, 0]
    err    = np.abs(reg - direct) / (np.abs(direct) + 1e-8)

    print("\nSANITY CHECK  (level: regression vs direct interpolation at lm=0, τ_min)")
    print(f"  ATM col       : {atm_col}  K/S={m_new[atm_col]:.4f}  "
          f"log(K/S)={lm_grid[atm_col]:.4f}")
    print(f"  Direct interp : mean={direct.mean():.4f}  std={direct.std():.4f}")
    print(f"  Regression    : mean={reg.mean():.4f}  std={reg.std():.4f}")
    print(f"  Relative error: mean={err.mean()*100:.2f}%  max={err.max()*100:.2f}%")
    print(f"  {'✓  Good' if err.mean() < 0.05 else '⚠  Still off — try reducing n_local'}")
    print(f"\n  tau_new range : [{tau_new[0]:.4f}, {tau_new[-1]:.4f}] yr")
    print(f"  lm range      : [{lm_grid[0]:.4f}, {lm_grid[-1]:.4f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Normaliser
# ─────────────────────────────────────────────────────────────────────────────

class FeatureNormalizer:
    """Min-max normaliser, fit on train set only."""
    def __init__(self): self.lo = self.hi = None

    def fit(self, f):
        self.lo, self.hi = f.min(0), f.max(0); return self

    def transform(self, f):
        r = np.where(self.hi - self.lo > 1e-8, self.hi - self.lo, 1.0)
        return (f - self.lo) / r

    def inverse_transform(self, fn):
        r = np.where(self.hi - self.lo > 1e-8, self.hi - self.lo, 1.0)
        return fn * r + self.lo

    def fit_transform(self, f): return self.fit(f).transform(f)


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = ["Level  Σ₀,₀", "Slope  Σ₀,₁", "Curvature  Σ₀,₂", "Term-struct  Σ₁,₀"]

def print_feature_summary(features, label=""):
    tag = f" ({label})" if label else ""
    print(f"\nFeature statistics{tag}:")
    print(f"  {'Feature':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 52)
    for k, name in enumerate(FEATURE_NAMES):
        v = features[:, k]
        print(f"  {name:<20} {v.mean():8.4f} {v.std():8.4f} {v.min():8.4f} {v.max():8.4f}")

def plot_features(features, label="", save_path=None):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, name, k in zip(axes.ravel(), FEATURE_NAMES, range(4)):
        v = features[:, k]
        ax.hist(v, bins=20, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.axvline(v.mean(), color="tomato", lw=1.5, label=f"μ={v.mean():.4f}")
        ax.set_title(name, fontsize=11); ax.set_xlabel("Value"); ax.set_ylabel("Count")
        ax.legend(fontsize=9)
    plt.suptitle(f"IVS Shape Features" + (f" — {label}" if label else ""), fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()