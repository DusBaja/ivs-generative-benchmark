# Latent Flow Matching for Arbitrage-Aware Implied Volatility Surface Generation


## Setup

Create and activate a virtual environment, then install the project from `pyproject.toml` in editable mode.

```bash
cd /path/to/this/repo-root
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## What this repository contains

- A clean Python project layout using `src/`
- A minimal API to:
  - prepare data (placeholders)
  - fit an unconditional model
  - generate the possible volatility surface
- Multiple CLI entrypoints: `other_models`,`volsurface_latentSB` and `volsurface_latentFM`.

## Project structure

```text

├── src/                                      # Main Python package and scripts
│   ├── constructor_surface_cube.py
│   ├── volsurface_path_bridge.py
│   ├── other_models/                         # Model implementations and training
│   │   ├── __init__.py
│   │   ├── HGB_RF.py
│   │   ├── VolGan.py
│   │   ├── cont_simulations.py
│   │   ├── diffusion_arbitrage_penality.py
│   │   ├── diffusion_cond_ext.py
│   │   ├── iv_ddpm.py
│   │   ├── models.py
│   │   ├── volgan_train.py
│   │   └── volsurface_controllable_VAE/      # Controllable VAE module
│   │       ├── VAE_model.py
│   │       └── ivs_features.py
│   ├── volsurface_latentFM/                  # Latent-factor models and utilities
│   │   ├── __init__.py
│   │   ├── ivs_autoencoder.py
│   │   ├── ivs_ldm_plot.py
│   │   ├── latent_models.py
│   │   ├── latent_models_variants.py
│   │   └── optuna_vae.py
│   └── volsurface_latentSB/                  # LightSB- and SBJTS-based models
│       ├── __init__.py
│       ├── lightsb_vol_surface_github_torchcompat.py
│       ├── sbjts_autoencoder_vol_surface.py
│       └── sbjts_pca_vol_surface.py
├── data/
│   ├── raw/
│   │   └── SPX_Option_Data/                  # Download from OptionsDX
│   └── treated/                              # Processed volatility-surface cubes, checkpoints and derived datasets
│       └── selected_retrain/
├── tests/                                    # Pytest test suite
├── figs/                                     # Generated figures
│   └── selected_retrain/
├── Full_models_comparaison.ipynb             # Training, optimisation and visualization of all our models
├── Paper_illustration.ipynb                  # Our main results
├── ablation_study.ipynb                      
└── pyproject.toml                            # Project configuration and dependencies
```

## Data

The experiments in this repository are based on historical SPX option-chain data obtained from **OptionsDX**.

The raw option-chain files are **not included** in this repository. Users should download the corresponding SPX historical option-chain data directly from OptionsDX in accordance with the provider's terms of use.

After downloading, place the files in:

```text
data/raw/SPX_Option_Data/
```

The processed datasets and trained model checkpoints included under `data/treated/` are derived from these raw option-chain files and are used to reproduce the experiments presented in this repository.

## Volatility surface cube building 

#### From OptionsDX prices (2020-2023) - SPX Index

```bash
# constructor script with the SPX input format
PYTHONPATH=src python -m constructor_surface_cube \
  --input-format spx-txt \
  --data-dir data/raw/SPX_Option_Data \
  --out data/treated/vol_surface_spx_long.npz \
  --verbose
```

Full single-line command with all parameters:

```bash
PYTHONPATH=src python -m constructor_surface_cube --input-format spx-txt --data-dir data/raw/SPX_Option_Data --out data/treated/vol_surface_spx_long.npz --verbose
```

## Latent Flow Matching models

The main Flow Matching models are implemented in:

```text
src/volsurface_latentFM/
├── ivs_autoencoder.py
├── latent_models.py
├── latent_models_variants.py
├── ivs_ldm_plot.py
└── optuna_vae.py
```

The Flow Matching pipeline is currently run from the notebooks:

```text
Full_models_comparaison.ipynb
Paper_illustration.ipynb
```

The notebooks first train or load the
arbitrage-aware VAE/autoencoder, encode the volatility surfaces into latent vectors, train the latent
Flow Matching model, then decode generated latent samples back into implied-volatility surfaces.

The main model is a latent Flow Matching model using a time-conditioned latent U-Net. The implemented variants include:

```text
L-FM linear
L-FM quadratic
L-FM VP
L-FM trigo
```

The variants are defined in `latent_models_variants.py`. In the main comparison notebook, these models
are trained and compared against the diffusion model, VolGAN, VolGAN raw, Cont-Vuletic WMC, SBJTS-PCA, SBJTS-AE/VAE and LightSB.


## VolGAN and VolGAN raw

**"VolGAN: a generative model for arbitrage-free implied volatility surfaces" - M. Vuletić and R. Cont (2024)**
The VolGAN baseline is implemented in:

```text
src/other_models/VolGan.py
src/other_models/volgan_train.py
```

VolGAN is used as an adversarial benchmark for implied-volatility surface generation. The generator takes random noise together with conditioning variables, while the discriminator compares real and generated volatility-surface increments. The implementation also includes smoothness and arbitrage-aware penalty terms in moneyness and maturity directions.

There are two implementations:

```text
VolGAN raw
VolGAN
```

`VolGAN raw` corresponds to the direct generated samples before the final arbitrage-aware filtering or
post-processing step. `VolGAN` corresponds to the post-processed version used in the final comparison
tables.



## Simulated volatility surface cube build (from Rama Cont and Milena Vuletic)

**"Simulation of Arbitrage-Free Implied Volatility Surfaces" - M. Vuletić and R. Cont (2022)**

Build a weighted Monte Carlo of volatility-surface paths (Cont & Vuletic):

```bash
# run the cont_simulations module (in other_models)
PYTHONPATH=src python -m other_models.cont_simulations \
  --input data/treated/vol_surface_cube.npz \
  --out data/treated/simulated_vol_surface.npz \
  --n-output-paths 1 \
  --path-length 103 \
  --n-candidate-paths 50000 \
  --candidate-batch-size 128 \
  --shock-scale 0.5 \
  --require-zero-path \
  --surface-arb-tol 1e-8 \
  --path-arb-tol 1e-8
```

The shock scale controls the magnitude of shocks applied during simulation.

Full single-line command with all parameters:

```bash
PYTHONPATH=src python -m other_models.cont_simulations --input data/treated/vol_surface_cube.npz --out data/treated/simulated_vol_surface.npz --n-output-paths 1 --path-length 103 --n-candidate-paths 50000 --candidate-batch-size 128 --shock-scale 0.5 --require-zero-path --surface-arb-tol 1e-8 --path-arb-tol 1e-8
```

## Schrödinger Bridge latent models

The repository contains three Schrödinger-Bridge-based models in:

```text
src/volsurface_latentSB/
├── sbjts_pca_vol_surface.py
├── sbjts_autoencoder_vol_surface.py
└── lightsb_vol_surface_github_torchcompat.py
```

These models are currently applied to volatility-surface cubes stored in `data/treated/`.

All commands should be run from the repository root with `PYTHONPATH=src`.


#### SBJTS-PCA

SBJTS-PCA applies Schrödinger Bridge with Jumps in a PCA latent space. It converts the IV surfaces into log-IV, projects them onto PCA factors, simulates latent jump-diffusion paths, and decodes them back into volatility surfaces.

```bash
PYTHONPATH=src python -m volsurface_latentSB.sbjts_pca_vol_surface \
  --input data/treated/vol_surface_spx_long.npz \
  --out data/treated/sbjts_pca_vol_surface.npz
```

#### SBJTS-AE/VAE

SBJTS-AE/VAE uses the same SBJTS latent jump-diffusion idea, but replaces the PCA representation with a beta-VAE latent representation. The VAE is trained on normalised log-IV surfaces, and posterior means are used as the latent time
series.

```bash
PYTHONPATH=src python -m volsurface_latentSB.sbjts_autoencoder_vol_surface \
  --input data/treated/vol_surface_spx_long.npz \
  --out data/treated/sbjts_autoencoder_vol_surface.npz \
  --autoencoder-checkpoint data/treated/sbjts_surface_autoencoder.pt
```

To reuse an existing VAE checkpoint, add:

```bash
--load-autoencoder-if-exists
```

#### LightSB

**Inspired from "Light Schrodinger Bride" - A. Korotin, N. Gushchin and E. Burnaev (2024).**
LightSB applies a lightweight Schrödinger Bridge solver in PCA latent space. It fits a diagonal Gaussian-mixture bridge transition, samples candidate latent paths, decodes them back into IV surfaces, and applies the same static-arbitrage selection step.

```bash
PYTHONPATH=src python -m volsurface_latentSB.lightsb_vol_surface_github_torchcompat \
  --input data/treated/vol_surface_spx_long.npz \
  --out data/treated/lightsb_vol_surface.npz
```


## Tests

Run the tests from the repository root:

```bash
python -m pytest tests/test_densityplots.py -q -s -r s
python -m pytest tests/test_surface_cube_arbitrage.py -q -s -r s
python -m pytest tests/test_vol_surface_check.py -q -s -r s
python -m pytest tests/test_volgan_check.py -q -s -r s
python -m pytest tests/test_visualisation_penalties.py -q -s -r s
```

If we want the plots to stay open until you close them manually:

```bash
BLOCK_PLOTS=1 python -m pytest tests/test_visualisation_penalties.py -q -s -r s
BLOCK_PLOTS=1 python -m pytest tests/test_vol_surface_check.py -q -s -r s
BLOCK_PLOTS=1 python -m pytest tests/test_volgan_check.py -q -s -r s
```

If the code is ran from a different environment, replace the `python -m pip install -e .` and `python -m pytest ...` commands with the equivalent commands for that environment.
