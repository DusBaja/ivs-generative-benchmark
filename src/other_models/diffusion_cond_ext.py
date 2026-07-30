from __future__ import annotations

from typing import Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def resize_to_16x32(x_np: np.ndarray, device):
    x = torch.from_numpy(x_np).float().unsqueeze(1).to(device)  # (N,1,H,W)
    return F.interpolate(x, size=(16, 32), mode="bilinear", align_corners=False)

def fit_normalizer(x_train: torch.Tensor):
    
    flat = x_train.flatten()
    lo = torch.quantile(flat, 0.05)
    hi = torch.quantile(flat, 0.95)# BEST AT 0.95 
    return lo.detach(), hi.detach()

def normalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    den = (hi - lo).clamp_min(1e-8)
    x01 = (x - lo) / den
    return (x01 * 2.0 - 1.0).contiguous()

def denormalize(x: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    x01 = (x + 1.0) / 2.0 #REMOVED Clamp for better queue!!!
    #x01 = (x.clamp(-1, 1) + 1.0) / 2.0
    return x01 * (hi - lo) + lo

def show_grid(tensor: torch.Tensor, title: str, vmin=None, vmax=None, n_show=8):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    t = tensor.detach().cpu()
    if t.ndim == 4:
        t = t[:, 0]
    if t.ndim == 2:
        t = t[None]
    n = min(n_show, t.shape[0])
    plt.figure(figsize=(12, 2.6))
    last = None
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        last = ax.imshow(t[i].numpy(), aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle(title)
    if last is not None:
        plt.colorbar(last, fraction=0.025, pad=0.02)
    plt.show()

def show_denoising(history, title="Denoising trajectory"):
    steps = [0, len(history)//3, 2*len(history)//3, len(history)-1]

    flat = torch.cat([h.flatten() for h in history])
    vmin = torch.quantile(flat, 0.05).item()
    vmax = torch.quantile(flat, 0.95).item()

    plt.figure(figsize=(10, 2.7))
    last = None
    for j, s in enumerate(steps):
        ax = plt.subplot(1, len(steps), j + 1)
        last = ax.imshow(
            history[s][0, 0].cpu().numpy(),
            aspect="auto",
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(f"step {s}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(title)
    plt.colorbar(last, fraction=0.03, pad=0.02)
    plt.show()


@dataclass
class SurfacePreprocessor_diffusion:
    cond_mean_: Optional[np.ndarray] = field(default=None, init=False)
    cond_std_: Optional[np.ndarray] = field(default=None, init=False)

    @staticmethod
    def expand_conditions(cond, size: Tuple[int, int]):
        H, W = size
        if isinstance(cond, torch.Tensor):
            return cond[:, :, None, None].expand(-1, -1, H, W)
        cond = np.asarray(cond)
        return (
            cond[:, :, None, None]
            .repeat(H, axis=2)
            .repeat(W, axis=3)
            .astype(np.float32)
        )

    def normalize_conditions(self, cond: np.ndarray, eps: float = 1e-8, fit: bool = False):
        cond = np.asarray(cond, dtype=np.float32)
        if fit:
            self.cond_mean_ = cond.mean(axis=0)
            self.cond_std_ = cond.std(axis=0)
            self.cond_std_[self.cond_std_ < eps] = 1.0
        if self.cond_mean_ is None or self.cond_std_ is None:
            raise RuntimeError("Condition normalizer not fitted.")
        return ((cond - self.cond_mean_) / self.cond_std_).astype(np.float32)


class UNet(nn.Module):
    """
    UNet adapted to surface generation.
    input: (B,1(+n_cond),16,32)
    output: (B,1,16,32)
    """

    def __init__(self, n_cond: int = 0):
        super().__init__()
        self.n_cond = int(n_cond)
        self.emb_dim_1 = 128
        self.emb_dim_2 = 256

        self.act = nn.SiLU()

        self.pre_conv_1 = nn.Conv2d(1 + self.n_cond, 64, 3, padding=1, padding_mode="reflect")
        self.pre_conv_2 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect")

        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.post_conv_1 = nn.Conv2d(64, 64, 3, padding=1, padding_mode="reflect")
        self.post_conv_2 = nn.Conv2d(64, 1, 3, padding=1, padding_mode="reflect")

        self.down_layers = nn.ModuleList([
            nn.Conv2d(64, self.emb_dim_1, kernel_size=5, padding=2, padding_mode="reflect"),
            nn.Conv2d(self.emb_dim_1, self.emb_dim_2, kernel_size=5, padding=2, padding_mode="reflect"),
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(self.emb_dim_2, self.emb_dim_1, kernel_size=5, padding=2, padding_mode="reflect"),
            nn.Conv2d(self.emb_dim_1, 64, kernel_size=5, padding=2, padding_mode="reflect"),
        ])

        self.t_emb_1 = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, self.emb_dim_2))
        self.t_emb_2 = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, self.emb_dim_1))

        self.t_res_emb_0 = nn.Sequential(nn.Linear(1, 64), nn.Sigmoid(), nn.Linear(64, 64))
        self.t_res_emb_1 = nn.Sequential(nn.Linear(1, self.emb_dim_1), nn.Sigmoid(), nn.Linear(self.emb_dim_1, self.emb_dim_1))
        self.t_res_emb_2 = nn.Sequential(nn.Linear(1, self.emb_dim_2), nn.Sigmoid(), nn.Linear(self.emb_dim_2, self.emb_dim_2))

    def forward(self, x: torch.Tensor, t: torch.Tensor, y=None, plot: bool = False, cond: Optional[torch.Tensor] = None):
        if cond is not None and self.n_cond > 0:
            cond_maps = SurfacePreprocessor_diffusion.expand_conditions(cond, size=(x.shape[2], x.shape[3]))
            if isinstance(cond_maps, np.ndarray):
                cond_maps = torch.from_numpy(cond_maps)
            cond_maps = cond_maps.to(x.device)
            x = torch.cat([x, cond_maps], dim=1)

        batch_size = x.shape[0]
        latents = []

        x = self.pre_conv_1(x)
        x = self.pre_conv_2(x)
        res = x

        t_res_emb_0 = self.t_res_emb_0(t).view(batch_size, 64, 1, 1)
        t_res_emb_1 = self.t_res_emb_1(t).view(batch_size, self.emb_dim_1, 1, 1)
        t_res_emb_2 = self.t_res_emb_2(t).view(batch_size, self.emb_dim_2, 1, 1)

        t_emb_1 = self.t_emb_1(t).view(batch_size, self.emb_dim_2, 1, 1)
        t_emb_2 = self.t_emb_2(t).view(batch_size, self.emb_dim_1, 1, 1)

        for layer in self.down_layers:
            x = self.act(layer(x))
            latents.append(x)
            x = self.down(x)

        for i, layer in enumerate(self.up_layers):
            x = self.up(x)
            if i == 0:
                x = x * t_emb_1 + latents.pop() * t_res_emb_2
            if i == 1:
                x = x * t_emb_2 + latents.pop() * t_res_emb_1
            x = self.act(layer(x))

        x = x + res * t_res_emb_0
        x = self.post_conv_1(x)
        x = self.act(x)
        x = self.post_conv_2(x)
        return x


class VolaDiff:
    """
    - training objective: predict x0 (clean surface) from x_t
    - sampler uses x_pred as x0 
    """

    def __init__(self, n_cond: int = 0, t_min: float = 1e-4, n_steps: int = 100):
        self.n_cond = int(n_cond)
        self.n_steps = int(n_steps)
        self.T_max = 1.0
        self.dt = self.T_max / self.n_steps
        self.t_min = float(t_min)
        self.t_max = 1.0 - 1e-4

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet(n_cond=self.n_cond).to(self.device)
        self.exp_model = copy.deepcopy(self.model)
        self.best_model = copy.deepcopy(self.model)

    def load(self,path):
        self.best_model.load_state_dict(torch.load(path,weights_only=True))

    def clamp_t(self, t: torch.Tensor):
        return torch.clamp(t, min=self.t_min, max=self.t_max)

    def var_inst(self, t):
        return -2 * torch.log(1 - torch.pow(t, 2))

    def alpha(self, t):
        return torch.exp(-0.5 * self.var_inst(t))

    def var(self, t):
        return 1 - torch.exp(-0.5 * self.var_inst(t))

    def corrupt(self, x0: torch.Tensor, t: torch.Tensor):
        t = self.clamp_t(t)
        noise = torch.randn_like(x0)
        xt = x0 * torch.sqrt(self.alpha(t)) + torch.sqrt(self.var(t)) * noise
        return xt, noise

    def mse(self, a, b):
        return torch.mean((a - b) ** 2)

    def _maybe_drop_cond(self, cond: Optional[torch.Tensor], p_drop: float):
        if cond is None or p_drop <= 0.0:
            return cond
        if p_drop >= 1.0:
            return None
        b = cond.shape[0]
        keep = (torch.rand(b, device=cond.device) > p_drop)
        if keep.all():
            return cond
        if (~keep).all():
            return None
        cond2 = cond.clone()
        cond2[~keep] = 0.0
        return cond2

    def train(self, data_loader, params, n_epochs=100, save_model=True, plotting=True, printing=True):
        device = self.device
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        batches_per_epoch = len(data_loader)
        total_iters = int(n_epochs) * int(batches_per_epoch)

        losses = []
        ema_losses = []
        best_ema = float("inf")

        self.exp_model.load_state_dict(self.model.state_dict())

        for epoch in range(int(n_epochs)):
            for i, batch in enumerate(data_loader):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    vol_target, cond = batch
                    vol_target = vol_target.to(device).view(-1, 1, 16, 32)
                    cond = cond.to(device)
                    cond = self._maybe_drop_cond(cond, float(params.get("cond_drop_prob", 0.0)))
                else:
                    vol_target = batch.to(device).view(-1, 1, 16, 32)
                    cond = None

                t = torch.rand(vol_target.shape[0], 1, 1, 1, device=device)
                noisy_x, noise = self.corrupt(vol_target, t)

                _pred = self.model(noisy_x, t, vol_target, cond=cond)

                loss = self.mse(_pred, vol_target)

                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())
                ema_losses.append(
                    0.99 * ema_losses[-1] + 0.01 * loss.item() if len(ema_losses) > 0 else loss.item()
                )

                effective_iter = epoch * batches_per_epoch + i
                if total_iters > 0 and effective_iter > 0.25 * total_iters:
                    decay = 0.995
                    for p, ep in zip(self.model.parameters(), self.exp_model.parameters()):
                        ep.data.mul_(decay).add_(p.data, alpha=1 - decay)

                if ema_losses[-1] < best_ema:
                    best_ema = ema_losses[-1]
                    self.best_model = copy.deepcopy(self.exp_model)

            if printing and (epoch % 10 == 0 or epoch == int(n_epochs) - 1):
                print(f"Epoch {epoch:03d} | EMA loss: {ema_losses[-1]:.6f}")

        if plotting:
            plt.plot(np.log(losses), label="loss")
            plt.plot(np.log(ema_losses), label="EMA loss")
            plt.legend()
            plt.xlabel("iteration")
            plt.ylabel("log loss")
            plt.show()

        if save_model:
            torch.save(self.best_model.state_dict(), "data/treated/best_model_checkpoint_cond.pt")

        if printing:
            print("Training complete. Best EMA loss:", best_ema)

    @torch.no_grad()
    def denoise_x(self, z: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor):
        """
        
        """
        t = self.clamp_t(t)
        var_t = torch.clamp(self.var(t), min=1e-6)
        var_tm1 = torch.clamp(self.var(t - self.dt), min=0.0)
        coef = var_tm1 / var_t
        noise = torch.randn_like(z)
        return coef * z + (1 - coef) * x0_pred + torch.sqrt(coef * (var_t - var_tm1)) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, cond: Optional[torch.Tensor] = None, cfg_scale: float = 0.0,
               return_history: bool = False, printing: bool = False):
        device = self.device
        z = torch.randn(batch_size, 1, 16, 32, device=device)
        history = []

        if cond is not None:
            cond = cond.to(device)
            if cond.ndim == 1:
                cond = cond[None, :]
            if cond.shape[0] != batch_size:
                raise ValueError("cond batch size must match batch_size")

        for k in range(self.n_steps):
            t = torch.full((batch_size, 1, 1, 1), 1 - k / self.n_steps, device=device)
            t = self.clamp_t(t)


            if self.n_cond > 0:
                cond_u = torch.zeros(batch_size, self.n_cond, device=device)
            else:
                cond_u = None

            x0_u = self.best_model(z, t, z, cond=cond_u)

            if cond is None or float(cfg_scale) == 0.0:
                x0 = x0_u
            else:
                x0_c = self.best_model(z, t, z, cond=cond)
                eta = float(cfg_scale)
                x0 = (1.0 + eta) * x0_c - eta * x0_u

            if return_history:
                history.append(z.clone())

            z = self.denoise_x(z, t, x0)

        if return_history:
            history.append(z.clone())
            return history
        return z

    @torch.no_grad()
    def sample_uncond(self, batch_size: int, return_history: bool = False):
        return self.sample(batch_size=batch_size, cond=None, cfg_scale=0.0, return_history=return_history)


class VolCondDataset(Dataset):
    def __init__(self, surfaces: torch.Tensor, cond: np.ndarray):
        if surfaces.ndim != 4 or surfaces.shape[1] != 1:
            raise ValueError(f"surfaces must have shape (n,1,H,W), got {tuple(surfaces.shape)}")
        cond_t = torch.from_numpy(np.asarray(cond)).float()
        if surfaces.shape[0] != cond_t.shape[0]:
            raise ValueError("surfaces and cond must have same length")
        self.surfaces = surfaces.float()
        self.cond = cond_t

    def __len__(self):
        return self.surfaces.shape[0]

    def __getitem__(self, idx):
        return self.surfaces[idx], self.cond[idx]


def run_training_from_npz(
    npz_path: Path,
    device,
    train_ratio=0.8,
    batch_size=16,
    n_epochs=20,
    n_gen=8,
    *,
    cond_key: Optional[str] = None,
    cond_drop_prob: float = 0.1,
    cfg_scale: float = 2.0,
    t_min: float = 1e-4,
    save_ckpt_to_treated: bool = True,
):
    raw = np.load(npz_path, allow_pickle=True)
    cube = raw["cube"]
    surf_np = cube.transpose(2, 1, 0)  # (date,T,m)

    n_dates = surf_np.shape[0]
    split = int(train_ratio * n_dates)
    surf_train = surf_np[:split]
    surf_test = surf_np[split:]

    x_train = resize_to_16x32(surf_train, device)
    x_test = resize_to_16x32(surf_test, device)

    lo, hi = fit_normalizer(x_train)
    vol_train = normalize(x_train, lo, hi)
    vol_test = normalize(x_test, lo, hi)

    if cond_key is None:
        loader = DataLoader(vol_train, batch_size=batch_size, shuffle=True, drop_last=True)
        diff = VolaDiff(n_cond=0, t_min=t_min, n_steps=100)
    else:
        cond_raw = np.asarray(raw[cond_key])  
        cond_train = cond_raw[:split]
        cond_test  = cond_raw[split:]

        cprep = SurfacePreprocessor_diffusion()
        cond_train_n = cprep.normalize_conditions(cond_train, fit=True)
        cond_test_n  = cprep.normalize_conditions(cond_test,  fit=False)

        ds = VolCondDataset(vol_train, cond_train_n)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

        diff = VolaDiff(
            n_cond=cond_train_n.shape[1],
            t_min=t_min,
            n_steps=100,
        )

    diff.device = device
    diff.model = diff.model.to(device)
    diff.exp_model = diff.exp_model.to(device)
    diff.best_model = diff.best_model.to(device)

    diff.train(
        data_loader=loader,
        params={"cond_drop_prob": float(cond_drop_prob)},
        n_epochs=int(n_epochs),
        save_model=True,
        plotting=True,
        printing=True,
    )

    if save_ckpt_to_treated:
        treated = Path.cwd() / "data" / "treated"
        treated.mkdir(parents=True, exist_ok=True)
        ckpt = treated / "best_model_checkpoint_cond.pt"
        torch.save(diff.best_model.state_dict(), ckpt)
        print("Saved checkpoint to:", ckpt)

    real = vol_test[: min(n_gen, vol_test.shape[0])]

    if cond_key is None:
        hist = diff.sample_uncond(batch_size=n_gen, return_history=True)
    else:
        cond_for_gen = torch.from_numpy(cond_test_n[:n_gen]).float().to(device)
        hist = diff.sample(
            batch_size=n_gen,
            cond=cond_for_gen,
            cfg_scale=float(cfg_scale),
            return_history=True,
        )
    gen = hist[-1]

    real_back = denormalize(real, lo, hi)
    gen_back = denormalize(gen, lo, hi)

    show_grid(real_back, "REAL", vmin=None, vmax=None)
    show_grid(gen_back,  "GEN",  vmin=None, vmax=None)


    #vmin = torch.quantile(real_back.flatten(), 0.05).item()
    #vmax = torch.quantile(real_back.flatten(), 0.95).item()
    #show_grid(real_back, "REAL test (de-normalized)", vmin=vmin, vmax=vmax)
    #show_grid(gen_back, "GENERATED (de-normalized)", vmin=vmin, vmax=vmax)

    #alln = torch.cat([real.flatten(), gen.flatten()])
    #vmin_n = torch.quantile(alln, 0.05).item()
    #vmax_n = torch.quantile(alln, 0.95).item()
    #show_grid(real, "REAL test (normalized)", vmin=vmin_n, vmax=vmax_n)
    #show_grid(gen, "GENERATED (normalized)", vmin=vmin_n, vmax=vmax_n)

    #plt.hist(real.flatten().detach().cpu().numpy(), bins=60, alpha=0.5, label="real")
    #plt.hist(gen.flatten().detach().cpu().numpy(), bins=60, alpha=0.5, label="gen")
    #plt.legend()
    #plt.show()

    show_denoising(hist, title="Denoising trajectory (z-space)")

    return {
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "real_norm": real,
        "gen_norm": gen,
        "real_denorm": real_back,
        "gen_denorm": gen_back,
    }
