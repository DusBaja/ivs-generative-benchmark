"""
Multi-Objective Optuna Search for IVS_AE
==========================================
Objectives (both minimised):
  1. Arbitrage score  — mean φ over generated surfaces via batch_phi
  2. SWD              — Sliced Wasserstein Distance (real vs generated)

Search space: lambda_cal, lambda_but, beta, z_dim

No external dependencies beyond the VAE and your batch_phi function.
"""

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from scipy.stats import norm

def _denorm_to_fraction(x_hat_np, lo_v, hi_v):
    """
    Normalised [-1, 1] output from decoder → fraction IV (e.g. 0.20 for 20%).
    x_hat_np : (N, 512) numpy
    """
    return ((x_hat_np + 1.0) / 2.0 * (hi_v - lo_v) + lo_v) / 100.0

def relative_call(sigma, m, tau, r=0.0):
    """
    sigma : (Ntau, Nm) in FRACTION  (e.g. 0.20 for 20%)
    m     : (Nm,)  log-moneyness = log(K/F)
    tau   : (Ntau,)
    """
    sigma = np.asarray(sigma)
    if sigma.shape != (len(tau), len(m)):
        raise ValueError(f"sigma shape {sigma.shape} != ({len(tau)}, {len(m)})")

    T, M = np.meshgrid(tau, m, indexing='ij')
    vol  = np.clip(sigma, 1e-8, None)
    T    = np.clip(T, 1e-12, None)

    d1 = (-M + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    C  = norm.cdf(d1) - np.exp(M - r*T) * norm.cdf(d2)
    return np.clip(C, 1e-12, None)


def compute_phi(sigma, m, tau, r=0.0):
    C  = relative_call(sigma, m, tau, r=r)
    dm = np.diff(m)
    dt = np.diff(tau)

    dC_dm  = (C[:, 1:] - C[:, :-1]) / dm[None, :]
    p_call = np.maximum(dC_dm, 0.0).sum()

    d2C    = C[:, 2:] - 2 * C[:, 1:-1] + C[:, :-2]
    p_bfly = np.maximum(-d2C, 0.0).sum()

    p_cal  = 0   # disabled per user's evaluation code

    phi    = p_call + p_bfly + p_cal
    return p_call, p_bfly, p_cal, phi


def batch_phi(surfaces, m, tau):
    """
    surfaces : (N, NT, NM) in FRACTION
    m        : (NM,) log-moneyness  i.e. np.log(m_new)
    tau      : (NT,) maturities in years
    """
    N   = len(surfaces)
    res = {'call': np.zeros(N), 'butterfly': np.zeros(N),
           'calendar': np.zeros(N), 'phi': np.zeros(N)}
    for k in range(N):
        c, b, cal, phi    = compute_phi(surfaces[k], m, tau)
        res['call'][k]      = c
        res['butterfly'][k] = b
        res['calendar'][k]  = cal
        res['phi'][k]       = phi
    return res


# Sliced Wasserstein Distance
def sliced_wasserstein(X, Y, n_proj=200, seed=0):
    """
    X, Y : (N, D) numpy — surfaces flattened to D = NT*NM
    """
    rng  = np.random.default_rng(seed)
    D    = X.shape[1]
    dirs = rng.standard_normal((n_proj, D))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    px = np.sort(X @ dirs.T, axis=0)
    py = np.sort(Y @ dirs.T, axis=0)

    if len(px) != len(py):
        t  = np.linspace(0, 1, min(len(px), len(py)))
        tx = np.linspace(0, 1, len(px))
        ty = np.linspace(0, 1, len(py))
        px = np.array([np.interp(t, tx, px[:, j]) for j in range(n_proj)]).T
        py = np.array([np.interp(t, ty, py[:, j]) for j in range(n_proj)]).T

    return float(np.mean(np.abs(px - py)))


# Objective
def make_objective(surf_train_n, surf_val_n,
                    m_grid, tau_grid, lo_v, hi_v,
                    n_epochs_trial, device):
    """
    m_grid   : (NM,) actual moneyness K/S — log is taken internally
    tau_grid : (NT,) maturities in years
    """
    from src.volsurface_latentFM.ivs_autoencoder import IVS_AE

    NT, NM   = len(tau_grid), len(m_grid)
    m_log    = np.log(m_grid)          # log-moneyness for batch_phi

    # Real surfaces for SWD — fraction, flattened
    def _prep_real(s):
        a = np.asarray(s, dtype=np.float32)
        if a.ndim == 4: a = a[:, 0]
        return _denorm_to_fraction(a.reshape(-1, NT*NM), lo_v, hi_v)

    real_frac = _prep_real(surf_val_n)   # (N_val, NT*NM) fraction

    def objective(trial):
        lambda_cal = trial.suggest_float("lambda_cal", 1e-3, 5e-1, log=True)
        lambda_but = trial.suggest_float("lambda_but", 1e-3, 5e-1, log=True)
        beta       = trial.suggest_float("beta",       1e-4, 2e-2, log=True)
        z_dim      = trial.suggest_int  ("z_dim",      4,    12)

        ae = IVS_AE(
            z_dim      = z_dim,
            beta       = beta,
            lambda_cal = lambda_cal,
            lambda_but = lambda_but,
        ).to(device)

        try:
            ae.fit(
                surf_train_n, m_grid, tau_grid,
                surf_val_n    = surf_val_n,
                n_epochs      = n_epochs_trial,
                batch_size    = 32,
                lr            = 1e-3,
                warmup_epochs = min(300, n_epochs_trial // 3),
                patience      = max(150, n_epochs_trial // 5),
                print_every   = n_epochs_trial + 1,   # silent
                lo            = lo_v,
                hi            = hi_v,
                device        = device,
            )
        except Exception as e:
            raise optuna.exceptions.TrialPruned(f"Training failed: {e}")

        # Generate surfaces from prior
        N_gen = 300
        ae.eval()
        with torch.no_grad():
            z_gen  = torch.randn(N_gen, z_dim, device=device)
            x_gen  = ae.decoder(z_gen).cpu().numpy()        # (N, 512)

        gen_frac = _denorm_to_fraction(x_gen, lo_v, hi_v)  # (N, 512) fraction
        gen_surf = gen_frac.reshape(N_gen, NT, NM)          # (N, NT, NM)

        # Objective 1: mean arbitrage phi
        pen      = batch_phi(gen_surf, m_log, tau_grid)
        arb      = float(np.mean(pen['phi']))

        if not np.isfinite(arb):
            raise optuna.exceptions.TrialPruned("Non-finite arb score")

        #  Objective 2: SWD 
        swd = sliced_wasserstein(real_frac, gen_frac)

        if not np.isfinite(swd):
            raise optuna.exceptions.TrialPruned("Non-finite SWD")

        trial.set_user_attr("arb",  arb)
        trial.set_user_attr("swd",  swd)
        trial.set_user_attr("call", float(np.mean(pen['call'])))
        trial.set_user_attr("bfly", float(np.mean(pen['butterfly'])))

        print(f"  Trial {trial.number:>3}  "
              f"λ_cal={lambda_cal:.2e}  λ_but={lambda_but:.2e}  "
              f"β={beta:.2e}  z={z_dim}  "
              f"→  φ={arb:.4f}  swd={swd:.4f}")

        return arb, swd

    return objective


def run_optuna_search(surf_train_n, surf_val_n,
                       m_grid, tau_grid, lo_v, hi_v,
                       n_trials=40, n_epochs_trial=800,
                       device=None, study_name="ivs_ae_pareto"):
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        directions  = ["minimize", "minimize"],
        sampler     = TPESampler(seed=42),
        study_name  = study_name,
    )
    study.optimize(
        make_objective(surf_train_n, surf_val_n,
                        m_grid, tau_grid, lo_v, hi_v,
                        n_epochs_trial, dev),
        n_trials          = n_trials,
        show_progress_bar = True,
    )

    print(f"\n  Pareto-front: {len(study.best_trials)} trials")
    for t in study.best_trials:
        print(f"    Trial {t.number:>3}  "
              f"φ={t.values[0]:.4f}  swd={t.values[1]:.4f}  "
              f"params={t.params}")
    return study


def plot_pareto(study):
    import matplotlib.pyplot as plt

    all_arb = [t.values[0] for t in study.trials if t.values is not None]
    all_swd = [t.values[1] for t in study.trials if t.values is not None]
    par_arb = [t.values[0] for t in study.best_trials]
    par_swd = [t.values[1] for t in study.best_trials]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(all_arb, all_swd, c='lightgrey', s=40, label='All trials')
    ax.scatter(par_arb, par_swd, c='crimson',   s=80, zorder=5,
               label='Pareto front')
    for t in study.best_trials:
        ax.annotate(str(t.number), (t.values[0], t.values[1]),
                    fontsize=7, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel("Mean arbitrage φ  (lower = fewer violations)")
    ax.set_ylabel("Sliced Wasserstein Distance  (lower = better density)")
    ax.set_title("IVS_AE Pareto Front")
    ax.legend(); plt.tight_layout(); plt.show()
    return fig


def pick_best_trial(study, arb_weight=0.5):
    """
    Scalarise the Pareto front with arb_weight in [0,1].
    arb_weight=1 to minimise arbitrage only
    arb_weight=0 to minimise SWD only
    """
    pareto = study.best_trials
    arbs   = np.array([t.values[0] for t in pareto])
    swds   = np.array([t.values[1] for t in pareto])
    an     = (arbs - arbs.min()) / (arbs.ptp() + 1e-12)
    sn     = (swds - swds.min()) / (swds.ptp() + 1e-12)
    best   = pareto[int(np.argmin(arb_weight*an + (1-arb_weight)*sn))]
    print(f"\n  Best (arb_weight={arb_weight}): Trial {best.number}  "
          f"φ={best.values[0]:.4f}  swd={best.values[1]:.4f}")
    print(f"  → {best.params}")
    return best
