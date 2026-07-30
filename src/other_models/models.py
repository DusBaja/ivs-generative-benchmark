
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any, List

import numpy as np
import joblib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
import optuna


@dataclass
class SurfacePreprocessor:
    """
    - cube input expected as (n_m, n_T, n_dates) = (moneyness, tenor, date), reorders to (n_dates, n_T, n_m)
    - resizes each surface to (H,W) = size (default 16x32)
    - normalizes to [-1,1] using train quantiles q_low 1% /q_high 99%
   + conditional variables normalization
    """
    size: Tuple[int, int] = (16, 32)
    q_low: float = 0.01
    q_high: float = 0.99

    lo_: Optional[float] = field(default=None, init=False)
    hi_: Optional[float] = field(default=None, init=False)

    cond_mean_: Optional[np.ndarray] = field(default=None, init=False)
    cond_std_: Optional[np.ndarray] = field(default=None, init=False)

    def transform_raw(self, cube: np.ndarray) -> np.ndarray:
        """
        cube: (n_m, n_T, n_dates) to  (n_dates, 1, H, W) 
        """
        surf = cube.transpose(2, 1, 0)  # (n_dates, n_T, n_m)
        t = torch.from_numpy(surf).float().unsqueeze(1)  # (n_dates,1,n_T,n_m)
        t_rs = F.interpolate(t, size=self.size, mode="bilinear", align_corners=False)
        return t_rs.cpu().numpy().astype(np.float32)

    @staticmethod
    def _normalize_np(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scale = max(float(hi - lo), 1e-12)
        x01 = (x - lo) / scale
        x11 = np.clip(x01 * 2.0 - 1.0, -1.0, 1.0)
        return x11.astype(np.float32)

    def inverse_transform(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Inverse of surface normalization (back to vol units).
        x_norm in [-1,1], shape (n,1,H,W) or (n,H,W)
        """
        if self.lo_ is None or self.hi_ is None:
            raise RuntimeError("Preprocessor not fitted; cannot inverse_transform.")
        x = x_norm
        if x.ndim == 3:
            x = x[:, None, :, :]
        x = np.clip(x, -1.0, 1.0)
        x01 = (x + 1.0) / 2.0
        return x01 * (self.hi_ - self.lo_) + self.lo_


    def normalize_conditions(self, cond: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        cond: (n_dates, n_cond)
        If fit=True, fits mean/std on this array (train window only, no leakage).
        Returns normalized array.
        """
        cond = np.asarray(cond, dtype=np.float32)
        if cond.ndim != 2:
            raise ValueError(f"cond must be 2D (n_dates,n_cond), got shape={cond.shape}")
        if fit or self.cond_mean_ is None or self.cond_std_ is None:
            mu = np.nanmean(cond, axis=0)
            sig = np.nanstd(cond, axis=0)
            sig = np.where(sig < 1e-12, 1.0, sig)
            self.cond_mean_ = mu.astype(np.float32)
            self.cond_std_ = sig.astype(np.float32)
        return ((cond - self.cond_mean_) / self.cond_std_).astype(np.float32)



    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error over all entries (scalar)."""
        yt = y_true.reshape(-1).astype(np.float64)
        yp = y_pred.reshape(-1).astype(np.float64)
        return float(np.mean((yt - yp) ** 2))

    @staticmethod
    def mse_per_surface(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Per-surface MSE (shape: (n_surfaces,))."""
        yt = y_true.reshape(y_true.shape[0], -1).astype(np.float64)
        yp = y_pred.reshape(y_pred.shape[0], -1).astype(np.float64)
        return np.mean((yt - yp) ** 2, axis=1).astype(np.float64)

@dataclass
class RandomForestSurfaceForecaster:
    n_estimators: int = 500
    max_depth: Optional[int] = 7
    max_features: Union[str, float] = "sqrt"
    min_samples_split: int = 5
    min_samples_leaf: int = 5
    random_state: int = 42
    n_jobs: int = -1
    bootstrap: bool = True
    oob_score: bool = False
    verbose: int = 0
    preprocessor: SurfacePreprocessor = field(default_factory=SurfacePreprocessor)
    model_: Optional[RandomForestRegressor] = field(default=None, init=False)
    info_: Dict[str, object] = field(default_factory=dict, init=False)

    def _make_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            verbose=self.verbose,
        )

    def tune_with_optuna(self,cube: np.ndarray,cond: Optional[np.ndarray] = None,train_ratio: float = 0.8,n_trials: int = 100,seed: int = 42,timeout: Optional[int] = None) -> Dict[str, object]:
        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=10),
                "max_depth": trial.suggest_int("max_depth", 5, 9, step=1),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7, 1.0]),
                "min_samples_split": trial.suggest_int("min_samples_split", 4, 12),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 10),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            }
            rf_trial = RandomForestSurfaceForecaster(**params, random_state=seed, n_jobs=-1, verbose=0)
            metrics = rf_trial.fit(cube, cond=cond, train_ratio=train_ratio)
            val_mse = metrics.get("val_mse", np.nan)
            #if not np.isfinite(val_mse):
            #    return 1e6
            return float(val_mse)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    def fit(self,cube: np.ndarray,cond: Optional[np.ndarray] = None,train_ratio: float = 0.8,use_optuna: bool = False,optuna_n_trials: int = 100,optuna_seed: int = 42,optuna_timeout: Optional[int] = None) -> Dict[str, float]:
        optuna_result = None
        if use_optuna:
            optuna_result = self.tune_with_optuna(
                cube=cube, cond=cond, train_ratio=train_ratio,
                n_trials=optuna_n_trials, seed=optuna_seed, timeout=optuna_timeout
            )
            best = optuna_result["best_params"]
            self.n_estimators = best.get("n_estimators", self.n_estimators)
            self.max_depth = best.get("max_depth", self.max_depth)
            self.max_features = best.get("max_features", self.max_features)
            self.min_samples_split = best.get("min_samples_split", self.min_samples_split)
            self.min_samples_leaf = best.get("min_samples_leaf", self.min_samples_leaf)
            self.bootstrap = best.get("bootstrap", self.bootstrap)

        x_raw = self.preprocessor.transform_raw(cube)
        n_dates = x_raw.shape[0]
        n_pairs = n_dates - 1
        n_train_pairs = max(1, int(train_ratio * n_pairs))

    
        x_fit = torch.from_numpy(x_raw[: n_train_pairs + 1]).float()
        lo = torch.quantile(x_fit, self.preprocessor.q_low).item()
        hi = torch.quantile(x_fit, self.preprocessor.q_high).item()
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        self.preprocessor.lo_ = lo
        self.preprocessor.hi_ = hi

        x = self.preprocessor._normalize_np(x_raw, lo, hi)  # (n_dates,1,H,W)

        Xsurf = x[:-1]
        Ysurf = x[1:]
        Xf = Xsurf.reshape(n_pairs, -1)
        Yf = Ysurf.reshape(n_pairs, -1)

        if cond is not None:
            cond = np.asarray(cond, dtype=np.float32)
            if cond.shape[0] != n_dates:
                raise ValueError(f"cond length mismatch: cond has {cond.shape[0]} dates, surfaces have {n_dates}")
            
            cond_train = cond[: n_train_pairs + 1]
            cond_norm = self.preprocessor.normalize_conditions(cond_train, fit=True)
            cond_all = self.preprocessor.normalize_conditions(cond, fit=False)
            Xcond = cond_all[:-1]  # align with X_t
            Xf = np.concatenate([Xf, Xcond], axis=1)

        Xtr, Ytr = Xf[:n_train_pairs], Yf[:n_train_pairs]
        Xva, Yva = Xf[n_train_pairs:], Yf[n_train_pairs:]

        self.model_ = self._make_model()
        self.model_.fit(Xtr, Ytr)

        Ytr_pred = self.model_.predict(Xtr)
        train_mse = SurfacePreprocessor.mse(Ytr, Ytr_pred)
        if Xva.shape[0] > 0:
            Yva_pred = self.model_.predict(Xva)
            val_mse = SurfacePreprocessor.mse(Yva, Yva_pred)
        else:
            val_mse = float("nan")

        self.info_ = {
            "model": "RandomForest",
            "n_surfaces": n_dates,
            "train_ratio": train_ratio,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "uses_cond": cond is not None,
            "n_cond": int(cond.shape[1]) if cond is not None else 0,
        }
        if optuna_result is not None:
            self.info_["optuna_best_val_mse"] = optuna_result["best_value"]
            self.info_["optuna_best_params"] = optuna_result["best_params"]

        return {"train_mse": float(train_mse), "val_mse": float(val_mse)}

    def predict_cube_next(self, cube: np.ndarray, cond: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")

        x_raw = self.preprocessor.transform_raw(cube)
        if self.preprocessor.lo_ is None or self.preprocessor.hi_ is None:
            raise RuntimeError("Preprocessor not fitted (lo_/hi_ missing).")

        x = self.preprocessor._normalize_np(x_raw, self.preprocessor.lo_, self.preprocessor.hi_)
        n_dates = x.shape[0]
        n_pairs = n_dates - 1
        Xf = x[:-1].reshape(n_pairs, -1)

        if cond is not None:
            cond = np.asarray(cond, dtype=np.float32)
            if cond.shape[0] != n_dates:
                raise ValueError(f"cond length mismatch: cond has {cond.shape[0]} dates, surfaces have {n_dates}")
            if self.preprocessor.cond_mean_ is None or self.preprocessor.cond_std_ is None:
                raise RuntimeError("Conditional normalization not fitted (cond_mean_/cond_std_ missing). Fit with cond first.")
            cond_all = self.preprocessor.normalize_conditions(cond, fit=False)
            Xcond = cond_all[:-1]
            Xf = np.concatenate([Xf, Xcond], axis=1)

        Ypred = self.model_.predict(Xf).reshape(
            n_pairs, 1, self.preprocessor.size[0], self.preprocessor.size[1]
        )
        return Ypred.astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        if self.model_ is None:
            raise RuntimeError("Nothing to save: model not fitted.")
        path = Path(path)
        payload = {
            "model": self.model_,
            "info": self.info_,
            "preprocessor": {
                "size": self.preprocessor.size,
                "q_low": self.preprocessor.q_low,
                "q_high": self.preprocessor.q_high,
                "lo": self.preprocessor.lo_,
                "hi": self.preprocessor.hi_,
                "cond_mean": None if self.preprocessor.cond_mean_ is None else self.preprocessor.cond_mean_.tolist(),
                "cond_std": None if self.preprocessor.cond_std_ is None else self.preprocessor.cond_std_.tolist(),
            }
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RandomForestSurfaceForecaster":
        payload = joblib.load(Path(path))
        obj = cls()
        obj.model_ = payload["model"]
        obj.info_ = payload.get("info", {})
        pp = payload.get("preprocessor", {})
        obj.preprocessor.size = tuple(pp.get("size", obj.preprocessor.size))
        obj.preprocessor.q_low = float(pp.get("q_low", obj.preprocessor.q_low))
        obj.preprocessor.q_high = float(pp.get("q_high", obj.preprocessor.q_high))
        obj.preprocessor.lo_ = pp.get("lo", None)
        obj.preprocessor.hi_ = pp.get("hi", None)
        cm = pp.get("cond_mean", None)
        cs = pp.get("cond_std", None)
        if cm is not None and cs is not None:
            obj.preprocessor.cond_mean_ = np.asarray(cm, dtype=np.float32)
            obj.preprocessor.cond_std_ = np.asarray(cs, dtype=np.float32)
        return obj


@dataclass
class HistGBPCASurfaceForecaster:
    n_components: Union[int, float] = 0.8
    max_depth: int = 6
    learning_rate: float = 0.05
    max_iter: int = 300
    l2_regularization: float = 1e-3
    max_bins: int = 255
    random_state: int = 42
    min_samples_leaf: int = 5

    preprocessor: SurfacePreprocessor = field(default_factory=SurfacePreprocessor)

    pca_: Optional[PCA] = field(default=None, init=False)
    model_: Optional[MultiOutputRegressor] = field(default=None, init=False)
    info_: Dict[str, object] = field(default_factory=dict, init=False)

    def _make_model(self) -> MultiOutputRegressor:
        base = HistGradientBoostingRegressor(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            max_bins=self.max_bins,
            random_state=self.random_state,
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    def tune_with_optuna(
        self,
        cube: np.ndarray,
        cond: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        n_trials: int = 40,
        seed: int = 42,
        timeout: Optional[int] = None
    ) -> Dict[str, object]:

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_components": trial.suggest_int("n_components", 5, 20),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_iter": trial.suggest_int("max_iter", 100, 600),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 200, step=20),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 1e-1, log=True),
            }
            model = HistGBPCASurfaceForecaster(**params, random_state=seed)
            metrics = model.fit(cube, cond=cond, train_ratio=train_ratio)
            val_mse = metrics.get("val_mse", np.nan)
            if not np.isfinite(val_mse):
                return 1e6
            return float(val_mse)

        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return {"best_value": float(study.best_value), "best_params": dict(study.best_params)}

    def fit(
        self,
        cube: np.ndarray,
        cond: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        use_optuna: bool = False,
        optuna_n_trials: int = 40,
        optuna_seed: int = 42,
        optuna_timeout: Optional[int] = None
    ) -> Dict[str, float]:

        optuna_result = None
        if use_optuna:
            optuna_result = self.tune_with_optuna(
                cube=cube, cond=cond, train_ratio=train_ratio,
                n_trials=optuna_n_trials, seed=optuna_seed, timeout=optuna_timeout
            )
            best = optuna_result["best_params"]
            self.n_components = best.get("n_components", self.n_components)
            self.max_depth = best.get("max_depth", self.max_depth)
            self.learning_rate = best.get("learning_rate", self.learning_rate)
            self.max_iter = best.get("max_iter", self.max_iter)
            self.min_samples_leaf = best.get("min_samples_leaf", self.min_samples_leaf)
            self.l2_regularization = best.get("l2_regularization", self.l2_regularization)

        x_raw = self.preprocessor.transform_raw(cube)
        n_dates = x_raw.shape[0]
        n_pairs = n_dates - 1
        n_train_pairs = max(1, int(train_ratio * n_pairs))

        x_fit = torch.from_numpy(x_raw[: n_train_pairs + 1]).float()
        lo = torch.quantile(x_fit, self.preprocessor.q_low).item()
        hi = torch.quantile(x_fit, self.preprocessor.q_high).item()
        if hi - lo < 1e-12:
            hi = lo + 1e-12
        self.preprocessor.lo_ = lo
        self.preprocessor.hi_ = hi

        x = self.preprocessor._normalize_np(x_raw, lo, hi)  # (n_dates,1,H,W)

        X = x[:-1].reshape(n_pairs, -1)
        Y = x[1:].reshape(n_pairs, -1)

        if cond is not None:
            cond = np.asarray(cond, dtype=np.float32)
            if cond.shape[0] != n_dates:
                raise ValueError(f"cond length mismatch: cond has {cond.shape[0]} dates, surfaces have {n_dates}")
            cond_train = cond[: n_train_pairs + 1]
            _ = self.preprocessor.normalize_conditions(cond_train, fit=True)
            cond_all = self.preprocessor.normalize_conditions(cond, fit=False)
            Xcond = cond_all[:-1]  # align with X_t
        else:
            Xcond = None

        Xtr, Ytr = X[:n_train_pairs], Y[:n_train_pairs]
        Xva, Yva = X[n_train_pairs:], Y[n_train_pairs:]

        self.pca_ = PCA(n_components=self.n_components, svd_solver="full", random_state=self.random_state)
        Zx_tr = self.pca_.fit_transform(Xtr)
        Zy_tr = self.pca_.transform(Ytr)

        if Xcond is not None:
            Zx_tr = np.concatenate([Zx_tr, Xcond[:n_train_pairs]], axis=1)

        self.model_ = self._make_model()
        self.model_.fit(Zx_tr, Zy_tr)

        Zy_tr_pred = self.model_.predict(Zx_tr)
        train_mse = SurfacePreprocessor.mse(Zy_tr, Zy_tr_pred)

        if Xva.shape[0] > 0:
            Zx_va = self.pca_.transform(Xva)
            Zy_va = self.pca_.transform(Yva)
            if Xcond is not None:
                Zx_va = np.concatenate([Zx_va, Xcond[n_train_pairs:]], axis=1)
            Zy_va_pred = self.model_.predict(Zx_va)
            val_mse = SurfacePreprocessor.mse(Zy_va, Zy_va_pred)
        else:
            val_mse = float("nan")

        self.info_ = {
            "model": "HistGB + PCA",
            "n_components": self.n_components,
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "uses_cond": cond is not None,
            "n_cond": int(cond.shape[1]) if cond is not None else 0,
        }
        if optuna_result is not None:
            self.info_["optuna_best_val_mse"] = optuna_result["best_value"]
            self.info_["optuna_best_params"] = optuna_result["best_params"]

        return {"train_mse": float(train_mse), "val_mse": float(val_mse)}

    def predict_cube_next(self, cube: np.ndarray, cond: Optional[np.ndarray] = None) -> np.ndarray:
        if self.model_ is None or self.pca_ is None:
            raise RuntimeError("Model not fitted.")

        x_raw = self.preprocessor.transform_raw(cube)
        if self.preprocessor.lo_ is None or self.preprocessor.hi_ is None:
            raise RuntimeError("Preprocessor not fitted (lo_/hi_ missing).")
        x = self.preprocessor._normalize_np(x_raw, self.preprocessor.lo_, self.preprocessor.hi_)

        n_dates = x.shape[0]
        n_pairs = n_dates - 1

        X = x[:-1].reshape(n_pairs, -1)
        Zx = self.pca_.transform(X)

        if cond is not None:
            cond = np.asarray(cond, dtype=np.float32)
            if cond.shape[0] != n_dates:
                raise ValueError(f"cond length mismatch: cond has {cond.shape[0]} dates, surfaces have {n_dates}")
            if self.preprocessor.cond_mean_ is None or self.preprocessor.cond_std_ is None:
                raise RuntimeError("Conditional normalization not fitted. Fit with cond first.")
            cond_all = self.preprocessor.normalize_conditions(cond, fit=False)
            Zx = np.concatenate([Zx, cond_all[:-1]], axis=1)

        Zy_pred = self.model_.predict(Zx)
        Y_pred = self.pca_.inverse_transform(Zy_pred)

        return Y_pred.reshape(n_pairs, 1, self.preprocessor.size[0], self.preprocessor.size[1]).astype(np.float32)

    def save(self, path: Union[str, Path]) -> None:
        payload = {
            "model": self.model_,
            "pca": self.pca_,
            "info": self.info_,
            "preprocessor": {
                "size": self.preprocessor.size,
                "q_low": self.preprocessor.q_low,
                "q_high": self.preprocessor.q_high,
                "lo": self.preprocessor.lo_,
                "hi": self.preprocessor.hi_,
                "cond_mean": None if self.preprocessor.cond_mean_ is None else self.preprocessor.cond_mean_.tolist(),
                "cond_std": None if self.preprocessor.cond_std_ is None else self.preprocessor.cond_std_.tolist(),
            },
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "HistGBPCASurfaceForecaster":
        payload = joblib.load(Path(path))
        obj = cls()
        obj.model_ = payload["model"]
        obj.pca_ = payload["pca"]
        obj.info_ = payload.get("info", {})
        pp = payload.get("preprocessor", {})
        obj.preprocessor.size = tuple(pp.get("size", obj.preprocessor.size))
        obj.preprocessor.q_low = float(pp.get("q_low", obj.preprocessor.q_low))
        obj.preprocessor.q_high = float(pp.get("q_high", obj.preprocessor.q_high))
        obj.preprocessor.lo_ = pp.get("lo", None)
        obj.preprocessor.hi_ = pp.get("hi", None)
        cm = pp.get("cond_mean", None)
        cs = pp.get("cond_std", None)
        if cm is not None and cs is not None:
            obj.preprocessor.cond_mean_ = np.asarray(cm, dtype=np.float32)
            obj.preprocessor.cond_std_ = np.asarray(cs, dtype=np.float32)
        return obj


def show_grid_np(arr: np.ndarray, title: str, n_show: int = 8,
                 vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
    x = np.asarray(arr)
    if x.ndim == 4:
        x = x[:, 0]
    if x.ndim == 2:
        x = x[None, :, :]
    n = min(n_show, x.shape[0])
    plt.figure(figsize=(12, 2.6))
    last = None
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        last = ax.imshow(x[i], aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle(title)
    plt.colorbar(last, fraction=0.025, pad=0.02)
    plt.show()


def plot_mse_curve(mse: np.ndarray, train_split: Optional[int] = None,
    title: str = "MSE on train vs test",) -> None:
    mse = np.asarray(mse).reshape(-1)
    x = np.arange(len(mse))

    plt.figure(figsize=(10, 3))
    plt.plot(x, mse, label="MSE")

    if train_split is not None:
        plt.axvline(train_split, color="red", linestyle="--", linewidth=1.5, label="train/test split")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_true_pred_error_grids(y_true: np.ndarray,y_pred: np.ndarray,n_show: int = 8,title_prefix: str = "",vmin: Optional[float] = None,vmax: Optional[float] = None) -> None:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if vmin is None or vmax is None:
        vmin = float(min(y_true.min(), y_pred.min()))
        vmax = float(max(y_true.max(), y_pred.max()))

    show_grid_np(y_true, f"{title_prefix}TRUE", n_show=n_show, vmin=vmin, vmax=vmax)
    show_grid_np(y_pred, f"{title_prefix}PRED", n_show=n_show, vmin=vmin, vmax=vmax)



def evaluate_and_plot_forecast(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_show: int = 8,
    title: str = "Forecast evaluation",
    train_split: Optional[int] = None,  
) -> Dict[str, Any]:

    mse = SurfacePreprocessor.mse_per_surface(y_true, y_pred)

    print(f"{title}: mean MSE = {float(np.mean(mse)):.6f} | median MSE = {float(np.median(mse)):.6f}")

    plot_mse_curve(mse, train_split=train_split, title=f"{title} — MSE per surface (split shown)")
    plot_true_pred_error_grids(y_true, y_pred, n_show=n_show, title_prefix=f"{title} — ")

    #return {"mse": nmse}
