"""
Surface plotting for the IVS LDM pipeline.

    plot_single(surf, tau_new, m_new)          — one 3-D surface
    plot_mean(surfs, tau_new, m_new)           — mean of N surfaces
    plot_single_and_mean(surfs, tau_new, m_new)— both side-by-side
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def _draw(ax, surf, tau_new, m_new, title="", cmap="viridis"):
    """Draw one (16, 32) surface on a pre-created 3-D Axes."""
    M, T = np.meshgrid(m_new, tau_new)          # both (16, 32)
    ax.plot_surface(M, T, surf, cmap=cmap,
                    alpha=0.90, linewidth=0, antialiased=True)
    ax.set_xlabel("Moneyness K/S", fontsize=8, labelpad=3)
    ax.set_ylabel("Tenor (yr)",    fontsize=8, labelpad=3)
    ax.set_zlabel("IV (%)",        fontsize=8, labelpad=3)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9, pad=6)



# Public API
def plot_single(surf, tau_new, m_new,
                title="Generated IVS", cmap="viridis", save_path=None):
    """
    Plot one IVS in 3-D.

    Parameters
    ----------
    surf     : (16, 32)  IV in %
    tau_new  : (16,)     maturities in years
    m_new    : (32,)     absolute moneyness K/S
    """
    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    _draw(ax, surf, tau_new, m_new, title=title, cmap=cmap)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_mean(surfs, tau_new, m_new,
              label="", cmap="plasma", save_path=None):
    """
    Plot the mean of N generated surfaces in 3-D.

    Parameters
    ----------
    surfs    : (N, 16, 32)  IV in %
    """
    mean_surf = surfs.mean(axis=0)              # (16, 32)
    title     = f"Mean generated surface  (N={len(surfs)})"
    if label:
        title = f"{label} — {title}"

    fig = plt.figure(figsize=(7, 5))
    ax  = fig.add_subplot(111, projection="3d")
    _draw(ax, mean_surf, tau_new, m_new, title=title, cmap=cmap)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_single_and_mean(surfs, tau_new, m_new,
                         idx=0, label="",
                         cmap_single="viridis", cmap_mean="plasma",
                         save_path=None):
    """
    Side-by-side: one sampled surface (left) and the mean (right).

    Parameters
    ----------
    surfs : (N, 16, 32)  IV in %
    idx   : which sample to show on the left (default 0)
    """
    mean_surf = surfs.mean(axis=0)
    prefix    = f"{label} — " if label else ""

    fig = plt.figure(figsize=(13, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    _draw(ax1, surfs[idx], tau_new, m_new,
          title=f"{prefix}Sample #{idx}", cmap=cmap_single)

    ax2 = fig.add_subplot(122, projection="3d")
    _draw(ax2, mean_surf, tau_new, m_new,
          title=f"{prefix}Mean  (N={len(surfs)})", cmap=cmap_mean)

    # Shared z-axis range
    zmin = min(surfs[idx].min(), mean_surf.min())
    zmax = max(surfs[idx].max(), mean_surf.max())
    for ax in [ax1, ax2]:
        ax.set_zlim(zmin, zmax)

    plt.suptitle(f"{prefix}Generated IVS", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
