from __future__ import annotations
import numpy as np
from typing import Self, Literal
from .rho import NormalizationResult
from dataclasses import dataclass, replace, asdict
from .. import Vector
from typing import Callable
import warnings
import matplotlib.pyplot as plt
from functools import cached_property
from scipy.optimize import least_squares
import jax
import optax
import jax.numpy as jnp
from tqdm.auto import tqdm
from scipy.stats import qmc
from jax_tqdm import scan_tqdm
from scipy.stats import norm
from pathlib import Path
from ..pipeline import Result, ResultMeta
import h5py
from abc import ABC, abstractmethod


class HDF5Serializable(ABC):
    """Mixin for classes that can save/load to HDF5 files.
    
    Provides common implementation of save/from_path that delegates
    to abstract to_hdf5_group/from_hdf5_group methods.
    """
    
    def save(self, path: str | Path, exist_ok: bool = False) -> None:
        """Save object to HDF5 file.
        
        Parameters
        ----------
        path : str or Path
            Path to save the HDF5 file
        exist_ok : bool, default: False
            If True, overwrite existing file
        """
        path = Path(path)
        
        if path.exists() and not exist_ok:
            raise FileExistsError(
                f"File {path} already exists. Use exist_ok=True to overwrite."
            )
        
        with h5py.File(path, 'w') as f:
            self.to_hdf5_group(f)
    
    @classmethod
    def from_path(cls, path: str | Path) -> Self:
        """Load object from HDF5 file.
        
        Parameters
        ----------
        path : str or Path
            Path to the HDF5 file
            
        Returns
        -------
        Self
            Loaded instance
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist")
        
        with h5py.File(path, 'r') as f:
            return cls.from_hdf5_group(f)
    
    @abstractmethod
    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write this object to an HDF5 group.
        
        Parameters
        ----------
        group : h5py.Group
            HDF5 group to write to
        """
        ...
    
    @classmethod
    @abstractmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Read this object from an HDF5 group.
        
        Parameters
        ----------
        group : h5py.Group
            HDF5 group to read from
            
        Returns
        -------
        Self
            Reconstructed instance
        """
        ...


@dataclass(frozen=True)
class ExtrapolationModel(HDF5Serializable):
    """Extrapolation model for the gsf at low or high energies.

    Args:
        fn: Function to evaluate the model, defined on log scale.
        params: Parameters of the model.
        support: Support of the model.
    """
    support: tuple[float, float]                     # (Eγ_min, Eγ_max)
    fn: Callable[[float, tuple[float, ...]], float]   # model(Eγ, params)
    params: tuple[float, ...]

    def __post_init__(self):
        if self.support[1] <= self.support[0]:
            raise ValueError("Support must be ordered: Eγ_min < Eγ_max")

    def __call__(self, Eγ: float) -> float:
        return self.fn(Eγ, self.params)

    @classmethod
    def on_range(cls, min: float, max: float, **kwargs) -> Self:
        return cls(support=(min, max), **kwargs)

    def plot(self, ax: plt.Axes = None, e: np.ndarray = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        x = e if e is not None else np.linspace(self.support[0], self.support[1], 100)
        y = np.exp(self.fn(x, self.params))
        ax.plot(x, y, **kwargs)
        return ax

    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write ExtrapolationModel to an HDF5 group."""
        group.attrs['class_name'] = 'ExtrapolationModel'
        group.attrs['support_min'] = self.support[0]
        group.attrs['support_max'] = self.support[1]
        group.attrs['fn_name'] = _get_function_name(self.fn)
        group.create_dataset('params', data=np.array(self.params))
    
    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Read ExtrapolationModel from an HDF5 group."""
        support = (float(group.attrs['support_min']), 
                  float(group.attrs['support_max']))
        fn_name = group.attrs['fn_name']
        fn = _get_function_by_name(fn_name)
        params = tuple(group['params'][:])
        return cls(support=support, fn=fn, params=params)


@dataclass(kw_only=True)
class FitResult(Result[Vector], HDF5Serializable):
    T: Vector
    model: ExtrapolationModel
    loss: Vector
    dof: int                               # n - 2
    sigma2: float                          # residual variance at the final params (log-domain)
    XtX_inv_centered: np.ndarray           # 2x2 inverse curvature from centered OLS (design [1, Ec])
    E_mu: float                            # center used for Ec = E - E_mu
    meta: ResultMeta = None                # Metadata for Result protocol

    def __post_init__(self):
        # Initialize meta if not provided
        if self.meta is None:
            self.meta = ResultMeta(method='fit')

    def __unwrap__(self) -> Vector:
        """Unwrap protocol for Result - returns the fitted Vector T."""
        return self.T

    def bands(
        self,
        e: np.ndarray = None,
        level: float = 0.95,
        kind: Literal["mean", "pred"] = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Curvature-based bands on T(E) (not log T): returns (lo, hi).
        - 'mean'  : confidence band for the fitted mean curve
        - 'pred'  : prediction band (mean band + residual variance term)
        """
        E = e if e is not None else np.linspace(*self.model.support, 100)
        Ec = E - self.E_mu
        # linear predictor (log): μ = a + b E
        a, b = self.model.params
        log_mu = a + b * E

        # Var(μ(E)) = x^T Σ x where x = [1, Ec] and Σ = sigma2 * XtX_inv (centered design)
        Sigma = self.sigma2 * self.XtX_inv_centered
        x1 = np.ones_like(Ec)
        x2 = Ec
        var_mu = (
            Sigma[0, 0] * x1 * x1
            + 2.0 * Sigma[0, 1] * x1 * x2
            + Sigma[1, 1] * x2 * x2
        )
        if kind == "pred":
            var_mu = var_mu + self.sigma2  # add noise for prediction band

        z = float(norm.ppf(0.5 * (1.0 + level)))
        half = z * np.sqrt(np.maximum(var_mu, 0.0))
        log_lo, log_hi = log_mu - half, log_mu + half
        return np.exp(log_lo), np.exp(log_hi)

    def plot(self, ax: plt.Axes = None, e: np.ndarray = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(nrows=2, sharex=True)
        
        # If the user provided just one axis, they probably dont care
        # about the residuals
        ax = np.atleast_1d(ax)
        if len(ax) == 1:
            ax0 = ax[0]
            ax1 = None
        else:
            ax0 = ax[0]
            ax1 = ax[1]


        e = e if e is not None else np.linspace(*self.model.support, 100)
        data = self.T.vloc[self.model.support[0]:self.model.support[1]]
        data.plot(ax=ax0)
        hat = self.model(e)
        ax0.plot(e, np.exp(hat), **kwargs)
        ax0.fill_between(e,*self.bands(e=e), color='gray', alpha=0.2, **kwargs)
        if ax1 is not None:
            residual = data - np.exp(self.model(data.X))
            residual.plot(ax=ax1)
            ax1.axhline(0, color='black', alpha=0.5)
            ax1.set_ylabel("Residuals")
        return ax

    @property
    def support(self) -> tuple[float, float]:
        return self.model.support

    def __call__(self, Eγ: float) -> float:
        return self.model(Eγ)

    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write FitResult to an HDF5 group."""
        # Save metadata
        group.attrs['class_name'] = 'FitResult'
        group.attrs['stage'] = str(self.meta.stage) if self.meta.stage is not None else ''
        group.attrs['method'] = self.meta.method if self.meta.method is not None else ''
        
        # Save scalar fields
        group.attrs['dof'] = self.dof
        group.attrs['sigma2'] = self.sigma2
        group.attrs['E_mu'] = self.E_mu
        
        # Save ExtrapolationModel using its own method
        model_grp = group.create_group('model')
        self.model.to_hdf5_group(model_grp)
        
        # Save XtX_inv_centered
        group.create_dataset('XtX_inv_centered', data=self.XtX_inv_centered)
        
        # Save T (Vector)
        t_grp = group.create_group('T')
        # Save Vector data manually
        t_grp.create_dataset('values', data=self.T.values)
        t_grp.create_dataset('X', data=self.T.X)
        # Save Vector metadata if needed
        if hasattr(self.T, 'metadata'):
            meta_dict = asdict(self.T.metadata)
            meta_grp = t_grp.create_group('metadata')
            for key, value in meta_dict.items():
                if value is not None and not isinstance(value, dict):
                    meta_grp.attrs[key] = str(value)
        
        # Save loss (Vector)
        loss_grp = group.create_group('loss')
        if isinstance(self.loss, Vector):
            loss_grp.create_dataset('values', data=self.loss.values)
            loss_grp.create_dataset('X', data=self.loss.X)
            if hasattr(self.loss, 'metadata'):
                loss_meta_dict = asdict(self.loss.metadata)
                loss_meta_grp = loss_grp.create_group('metadata')
                for key, value in loss_meta_dict.items():
                    if value is not None and not isinstance(value, dict):
                        loss_meta_grp.attrs[key] = str(value)
        elif isinstance(self.loss, list):
            # Handle empty list case
            loss_grp.attrs['is_empty'] = True

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Read FitResult from an HDF5 group."""
        # Load metadata
        class_name = group.attrs['class_name']
        if class_name != 'FitResult':
            raise ValueError(f"Group contains {class_name}, not FitResult")
        
        stage_str = group.attrs['stage']
        method = group.attrs['method'] if group.attrs['method'] != '' else None
        from ..pipeline import Stage
        stage = Stage.from_any(stage_str) if stage_str != '' else None
        meta = ResultMeta(stage=stage, method=method)
        
        # Load scalar fields
        dof = int(group.attrs['dof'])
        sigma2 = float(group.attrs['sigma2'])
        E_mu = float(group.attrs['E_mu'])
        
        # Load ExtrapolationModel using its own method
        model_grp = group['model']
        model = ExtrapolationModel.from_hdf5_group(model_grp)
        
        # Load XtX_inv_centered
        XtX_inv_centered = group['XtX_inv_centered'][:]
        
        # Load T (Vector)
        t_grp = group['T']
        T = Vector(X=t_grp['X'][:], values=t_grp['values'][:])
        
        # Load loss (Vector or empty list)
        loss_grp = group['loss']
        if 'is_empty' in loss_grp.attrs and loss_grp.attrs['is_empty']:
            loss = []
        else:
            loss = Vector(X=loss_grp['X'][:], values=loss_grp['values'][:])
        
        return cls(
            T=T,
            model=model,
            loss=loss,
            dof=dof,
            sigma2=sigma2,
            XtX_inv_centered=XtX_inv_centered,
            E_mu=E_mu,
            meta=meta
        )


def exp_decay_log(Eγ: float, params: tuple[float, float]) -> float:
    """
    """
    logA, inv_tau = params
    return logA - inv_tau * Eγ

def linear_log(E: np.ndarray | jnp.ndarray, params: tuple[float, float]) -> np.ndarray | jnp.ndarray:
    """log T(E) = a + b E  (b may be positive or negative)."""
    a, b = params
    return a + b * E


# Function registry for ExtrapolationModel serialization
_MODEL_FUNCTION_REGISTRY = {
    'exp_decay_log': exp_decay_log,
    'linear_log': linear_log,
}

def _get_function_name(fn: Callable) -> str:
    """Get the name of a function from the registry."""
    for name, func in _MODEL_FUNCTION_REGISTRY.items():
        if func is fn:
            return name
    raise ValueError(f"Function {fn} not found in registry")

def _get_function_by_name(name: str) -> Callable:
    """Get a function from the registry by name."""
    if name not in _MODEL_FUNCTION_REGISTRY:
        raise ValueError(f"Function {name} not found in registry")
    return _MODEL_FUNCTION_REGISTRY[name]


def _safe_log(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, float)
    floor = max(1e-300, float(np.median(y[y > 0])) * 1e-30 if np.any(y > 0) else 1e-300)
    return np.log(np.clip(y, floor, np.inf))

def _ols_centered(E: np.ndarray, y: np.ndarray):
    """
    OLS for y ≈ a0 + b Ec with Ec = E - mean(E).
    Returns:
      a0_hat, b_hat, sigma2, XtX_inv_centered, E_mu
    """
    E = np.asarray(E, float); y = np.asarray(y, float)
    E_mu = float(E.mean())
    Ec = E - E_mu
    X = np.column_stack([np.ones_like(Ec), Ec])
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ (X.T @ y)  # [a0, b]
    yhat = X @ beta
    resid = y - yhat
    dof = max(len(E) - 2, 1)
    sigma2 = float((resid @ resid) / dof)
    a0_hat, b_hat = map(float, beta)
    return a0_hat, b_hat, sigma2, XtX_inv, E_mu


# ------------------- Main fit -------------------
def fit(
    T: Vector,                                # Vector with T.X, T.values
    mask: np.ndarray,                      # boolean mask same length as T.values
    *,
    # refinement
    refine_steps: int = 0,                 # 0 = pure OLS; >0 = refine with Optax
    try_both_signs: bool = True,           # try both slope families and pick best
    loss_fn: Callable[[jnp.ndarray], jnp.ndarray] | None = None,  # loss on residuals; default Huber
    opt: optax.GradientTransformation | None = None,
    nu_min: float = 1e-8,                  # positive floor for slope magnitude during refine
    reg_lambda: float = 1e-4,              # tiny tether to OLS init in optimized coords
    clip_norm: float = 1.0,                # gradient clipping
) -> FitResult:
    """
    Default: OLS (centered) with curvature bands. Optional JAX refinement that:
      - fixes the slope sign chosen from OLS (or tries both),
      - optimizes a positive magnitude via softplus (cannot flip sign).
    Returns a log-linear model: log T(E) = a + b E (with chosen sign baked into b).
    """
    # ---- data slice ----
    E_all = np.asarray(T.X, float)
    V_all = np.asarray(T.values, float)
    m = np.asarray(mask, bool)
    if m.shape != V_all.shape:
        raise ValueError("mask must have the same shape as T.values")
    E = E_all[m]
    V = V_all[m]
    if E.size < 3:
        raise ValueError("mask selects too few points (need ≥ 3).")
    if np.any(V <= 0):
        raise ValueError("T(E) must be > 0 in masked region (log-domain).")
    y = _safe_log(V)

    # ---- OLS (centered) ----
    a0_hat, b_hat, sigma2_lr, XtX_inv_c, E_mu = _ols_centered(E, y)

    # slope families to consider
    signs = [np.sign(b_hat) if b_hat != 0.0 else 1.0]
    if try_both_signs:
        s2 = -signs[0]
        if s2 != signs[0]:
            signs.append(s2)

    # helper: evaluate loss for a given (a0, b) in centered space
    def eval_loss_centered(a0: float, b: float) -> float:
        Ec = E - E_mu
        yhat = a0 + b * Ec
        r = jnp.asarray(yhat - y)
        if loss_fn is None:
            val = jnp.mean(optax.huber_loss(r, delta=1.0))
        else:
            val = loss_fn(r)
        return float(val)

    # ---- refinement (optional) with fixed sign, positive magnitude ----
    def refine_for_sign(sgn: float):
        # parameterization: b = sgn * nu, with nu = nu_min + softplus(s_raw) > 0
        # OLS magnitude as init
        nu0 = max(abs(b_hat), nu_min)
        # invert softplus approximately
        s0 = float(jnp.log(jnp.expm1(nu0 - nu_min))) if nu0 > nu_min else -20.0
        theta0 = jnp.asarray([a0_hat, s0], dtype=jnp.float32)  # [a0, s_raw]
        theta_ref = theta0

        Ec = jnp.asarray(E - E_mu, dtype=jnp.float32)
        yj = jnp.asarray(y, dtype=jnp.float32)

        def objective(theta: jnp.ndarray) -> jnp.ndarray:
            a0, s_raw = theta[0], theta[1]
            nu = nu_min + jax.nn.softplus(s_raw)
            b = sgn * nu
            yhat = a0 + b * Ec
            r = yhat - yj
            data = (jnp.mean(optax.huber_loss(r, delta=1.0)) if loss_fn is None
                    else loss_fn(r))
            reg = reg_lambda * jnp.sum((theta - theta_ref) ** 2)
            return data + reg

        if refine_steps <= 0:
            # no refine: just evaluate OLS-initialized family
            b0 = float(sgn * nu0)
            return a0_hat, b0, eval_loss_centered(a0_hat, b0)

        # refine
        gopt = opt or optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(1e-1))
        state = gopt.init(theta0)

        @jax.jit
        def step(theta, state):
            val, grads = jax.value_and_grad(objective)(theta)
            updates, state = gopt.update(grads, state, theta)
            theta = optax.apply_updates(theta, updates)
            return theta, state, val

        theta = theta0
        best_theta, best_val = theta, jnp.inf
        for _ in range(int(refine_steps)):
            theta, state, val = step(theta, state)
            better = val < best_val
            best_theta = jax.lax.select(better, theta, best_theta)
            best_val = jax.lax.select(better, val, best_val)

        a0_fin = float(best_theta[0])
        nu_fin = float(nu_min + jax.nn.softplus(best_theta[1]))
        b_fin = float(sgn * nu_fin)
        return a0_fin, b_fin, float(best_val)

    # run families and pick best
    candidates = []
    for sgn in signs:
        a0_f, b_f, val = refine_for_sign(float(sgn))
        candidates.append((val, a0_f, b_f))
    val_best, a0_best, b_best = min(candidates, key=lambda t: t[0])

    # map to uncentered (a, b) for the final model
    # log T(E) = a0 + b (E - E_mu) = (a0 - b E_mu) + b E
    a_final = a0_best - b_best * E_mu
    b_final = b_best

    # residual variance at final params (log-domain)
    yhat_final = a_final + b_final * E
    resid = y - yhat_final
    dof = max(E.size - 2, 1)
    sigma2_final = float((resid @ resid) / dof)

    lo = E.min()
    hi = E.max()

    model = ExtrapolationModel(fn=linear_log, params=(a_final, b_final), support=(lo, hi))
    return FitResult(
        T=T,
        model=model,
        dof=dof,
        sigma2=sigma2_final,
        XtX_inv_centered=XtX_inv_c,
        E_mu=E_mu,
        loss=[]
    )


@dataclass(frozen=True)
class TSpectrum(HDF5Serializable):
    T: Vector
    low: ExtrapolationModel | FitResult = None
    high: ExtrapolationModel | FitResult = None
    scaling: float = 1.0 # Normalization factor

    def __post_init__(self):
        # Check that the support of the low and high models are ordered
        if self.low is not None and self.high is not None:
            if self.low.support[1] > self.high.support[0]:
                raise ValueError("Low model support must be before high model support")

        # the support should span [0, Sn]
        if self.leftmost() > 0: # or sn:
            warnings.warn(UserWarning("Leftmost energy is greater than 0"))
        

    def leftmost(self) -> float:
        if self.low is not None:
            return self.low.support[0]
        return self.T.X_index.leftmost

    def rightmost(self) -> float:
        if self.high is not None:
            return self.high.support[1]
        return self.T.X_index.rightmost

    def __call__(self, Eγ):
        """
        Return T(Eγ) across full domain.
        Accepts float or np.ndarray. Works by:
        - low region:  use low model (T)
        - high region: use high model (T)
        - mid region:  log-linear interpolation of measured T on the grid
        """
        # ensure array
        x = np.asarray(Eγ, dtype=float)
        scalar = x.ndim == 0
        if scalar:
            x = x[None]

        # boundaries (handle missing low/high gracefully)
        low_hi  = self.low.support[1]  if self.low  is not None else -np.inf
        high_lo = self.high.support[0] if self.high is not None else  np.inf

        # masks
        m_low  = x <  low_hi
        m_high = x >  high_lo
        m_mid  = ~(m_low | m_high)

        out = np.empty_like(x, dtype=float)

        # --- low region via model (model should return T, not log T) ---
        if np.any(m_low):
            if self.low is None:
                raise ValueError("Low model is not set, undefined behavior for {x} < {low_hi}")
            out[m_low] = np.exp(self.low(x[m_low]))

        # --- high region via model ---
        if np.any(m_high):
            if self.high is None:
                raise ValueError("High model is not set, undefined behavior for {x} > {high_lo}")
            out[m_high] = np.exp(self.high(x[m_high]))

        # --- mid region: interpolate in log-space on the measured grid ---
        if np.any(m_mid):
            # precompute energy grid and log T samples
            Egrid = np.asarray(self.T.X, dtype=float)
            Tvals = np.asarray(self.T.values, dtype=float)
            if Egrid.ndim != 1:
                raise ValueError("T.X must be 1D.")
            if np.any(Tvals <= 0):
                raise ValueError("T.values must be > 0 for log interpolation.")
            logT = np.log(Tvals)

            # np.interp does linear interp on y; we want linear on logT
            # x outside [Egrid.min, Egrid.max] shouldn’t occur for the mid region if supports are set correctly;
            # but if it does, we clamp.
            xm = np.clip(x[m_mid], Egrid[0], Egrid[-1])
            logT_mid = np.interp(xm, Egrid, logT)
            out[m_mid] = np.exp(logT_mid)

        # apply global scaling
        out *= float(self.scaling)

        return float(out[0]) if scalar else out


    @cached_property
    def Tlog(self) -> np.ndarray:
        return np.log(self.T)

    def plot(self, ax: plt.Axes = None, Sn: float = None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        self.T.plot(ax=ax, **kwargs)
        if self.low is not None:
            e = None
            if Sn is not None:
                e = np.linspace(0, self.low.support[1], 100)
            self.low.plot(ax=ax, e=e, **kwargs)
        if self.high is not None:
            e = None
            if Sn is not None:
                e = np.linspace(self.high.support[0], Sn, 100)
            self.high.plot(ax=ax, e=e, **kwargs)
        ax.set_yscale('log')
        return ax


    def integrate(
        self,
        Eγ_min: float,
        Eγ_max: float,
        *,
        n: int = 1024,
        weight: Callable[[np.ndarray], np.ndarray] | None = None,
        return_std: bool = False,
    ) -> float | tuple[float, float]:
        """
        Piecewise integral over [Eγ_min, Eγ_max] with optional 1σ uncertainty.
        - Mean is always computed by numeric quadrature of self(E).
        - Uncertainty (delta-method) is included only for segments backed by a FitResult.
        - weight(E) can supply a kernel; defaults to 1.
        - scaling affects both mean and std (std scales by |scaling|).
        """
        if weight is None:
            weight = lambda x: np.ones_like(x, dtype=float)

        E_left = float(Eγ_min)
        E_right = float(Eγ_max)
        if E_right <= E_left:
            raise ValueError("Eγ_max must be greater than Eγ_min. Got {E_right} <= {E_left}")

        # Mean integral by quadrature over the full interval
        x = np.linspace(E_left, E_right, int(n))
        w = weight(x)
        fx = np.array([self(e) for e in x], dtype=float)
        mean_int = float(np.trapezoid(w * fx, x)) 

        if not return_std:
            return mean_int

        # Accumulate variance only from fitted tails that are FitResult
        var_total = 0.0

        def add_fit_variance(fr, seg_lo, seg_hi):
            nonlocal var_total
            # gradient components: dI/da = ∫ w*T, dI/db = ∫ w*E*T over the segment
            xs = np.linspace(seg_lo, seg_hi, max(64, int(n * (seg_hi - seg_lo) / (E_right - E_left))))
            ww = weight(xs)
            # model params (a,b) live in fr.model.params; model.fn returns log T
            a, b = fr.model.params
            logT = fr.model.fn(xs, (a, b))
            Tx = np.exp(logT)
            g_a = float(np.trapezoid(ww * Tx, xs))
            g_b = float(np.trapezoid(ww * xs * Tx, xs))
            g = np.array([g_a, g_b], float)

            # covariance in (a,b):
            Sigma_ab = _cov_ab_from_fit(fr)
            var = float(g @ Sigma_ab @ g.T)
            var_total += var

        # Low segment variance
        if self.low is not None and isinstance(self.low, FitResult):
            # low model used for E < low.support[1]
            #seg = _seg_overlap(E_left, E_right, self.leftmost(), self.low.support[1])
            seg_lo = E_left
            seg_hi = min(E_right, self.low.support[1])
            print(f"{E_left=}, {E_right=}")
            print(f"{seg_lo=}, {seg_hi=}")
            if seg_hi > seg_lo:
                add_fit_variance(self.low, seg_lo, seg_hi)

        # High segment variance
        if self.high is not None and isinstance(self.high, FitResult):
            # high model used for E > high.support[0]
            #seg = _seg_overlap(E_left, E_right, self.high.support[0], self.rightmost())
            seg_lo = max(E_left, self.high.support[0])
            seg_hi = E_right
            print(f"{seg_lo=}, {seg_hi=}")
            if seg_hi > seg_lo:
                add_fit_variance(self.high, seg_lo, seg_hi)

        # Treat low/high fits as independent; scale variance by scaling^2
        std_int = np.sqrt(max(var_total, 0.0))
        return mean_int, std_int

    def integrate_parts(self, Sn: float, **kwargs) -> dict[str, float | tuple[float, float]]:
        low = self.integrate(0, self.low.support[1], **kwargs)
        mid = self.integrate(self.low.support[1], self.high.support[0], **kwargs)
        high = self.integrate(self.high.support[0], Sn, **kwargs)
        return {
            "low": low,
            "mid": mid,
            "high": high,
        }

    def to_hdf5_group(self, group: h5py.Group) -> None:
        """Write TSpectrum to an HDF5 group."""
        group.attrs['class_name'] = 'TSpectrum'
        group.attrs['scaling'] = self.scaling
        
        # Save T (Vector)
        t_grp = group.create_group('T')
        t_grp.create_dataset('values', data=self.T.values)
        t_grp.create_dataset('X', data=self.T.X)
        if hasattr(self.T, 'metadata'):
            meta_dict = asdict(self.T.metadata)
            meta_grp = t_grp.create_group('metadata')
            for key, value in meta_dict.items():
                if value is not None and not isinstance(value, dict):
                    meta_grp.attrs[key] = str(value)
        
        # Save low (ExtrapolationModel | FitResult | None)
        if self.low is not None:
            low_grp = group.create_group('low')
            if isinstance(self.low, FitResult):
                low_grp.attrs['type'] = 'FitResult'
                self.low.to_hdf5_group(low_grp)
            elif isinstance(self.low, ExtrapolationModel):
                low_grp.attrs['type'] = 'ExtrapolationModel'
                self.low.to_hdf5_group(low_grp)
        else:
            group.attrs['low_is_none'] = True
        
        # Save high (ExtrapolationModel | FitResult | None)
        if self.high is not None:
            high_grp = group.create_group('high')
            if isinstance(self.high, FitResult):
                high_grp.attrs['type'] = 'FitResult'
                self.high.to_hdf5_group(high_grp)
            elif isinstance(self.high, ExtrapolationModel):
                high_grp.attrs['type'] = 'ExtrapolationModel'
                self.high.to_hdf5_group(high_grp)
        else:
            group.attrs['high_is_none'] = True
    
    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Read TSpectrum from an HDF5 group."""
        # Load metadata
        class_name = group.attrs['class_name']
        if class_name != 'TSpectrum':
            raise ValueError(f"Group contains {class_name}, not TSpectrum")
        
        scaling = float(group.attrs['scaling'])
        
        # Load T (Vector)
        t_grp = group['T']
        T = Vector(X=t_grp['X'][:], values=t_grp['values'][:])
        
        # Load low
        if 'low_is_none' in group.attrs and group.attrs['low_is_none']:
            low = None
        else:
            low_grp = group['low']
            low_type = low_grp.attrs['type']
            if low_type == 'FitResult':
                low = FitResult.from_hdf5_group(low_grp)
            elif low_type == 'ExtrapolationModel':
                low = ExtrapolationModel.from_hdf5_group(low_grp)
            else:
                raise ValueError(f"Unknown low type: {low_type}")
        
        # Load high
        if 'high_is_none' in group.attrs and group.attrs['high_is_none']:
            high = None
        else:
            high_grp = group['high']
            high_type = high_grp.attrs['type']
            if high_type == 'FitResult':
                high = FitResult.from_hdf5_group(high_grp)
            elif high_type == 'ExtrapolationModel':
                high = ExtrapolationModel.from_hdf5_group(high_grp)
            else:
                raise ValueError(f"Unknown high type: {high_type}")
        
        return cls(T=T, low=low, high=high, scaling=scaling)



def _seg_overlap(a0: float, a1: float, b0: float, b1: float) -> tuple[float, float] | None:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return (lo, hi) if hi > lo else None

def _cov_ab_from_fit(fr) -> np.ndarray:
    # Map centered covariance to (a,b) using J = [[1, -E_mu],[0,1]]
    Sigma_c = fr.sigma2 * np.asarray(fr.XtX_inv_centered, float)
    J = np.array([[1.0, -float(fr.E_mu)],
                  [0.0,  1.0            ]], float)
    return J @ Sigma_c @ J.T  # 2x2

    
# --- helpers: map FitResult -> Cov(a,b), integrate exp(a+bE) numerically ---
def _cov_ab_from_fit(fr) -> np.ndarray:
    Sigma_c = fr.sigma2 * np.asarray(fr.XtX_inv_centered, float)
    J = np.array([[1.0, -float(fr.E_mu)],
                  [0.0,  1.0            ]], float)
    return J @ Sigma_c @ J.T  # 2x2

def _int_exp_linear(a: float, b: float, lo: float, hi: float,
                    weight, n_quad: int = 256) -> float:
    if hi <= lo:
        return 0.0
    x = np.linspace(lo, hi, n_quad)
    w = weight(x)
    fx = np.exp(a + b * x)
    return float(np.trapezoid(w * fx, x))

def _mid_integral_fixed(T, scaling: float, lo: float, hi: float,
                        weight, n_quad: int = 512) -> float:
    if hi <= lo:
        return 0.0
    x = np.linspace(lo, hi, n_quad)
    # log-interpolate the measured T on its grid
    Egrid = np.asarray(T.X, float)
    Tvals = np.asarray(T.values, float)
    logT = np.log(np.clip(Tvals, 1e-300, np.inf))
    logT_mid = np.interp(np.clip(x, Egrid[0], Egrid[-1]), Egrid, logT)
    fx = np.exp(logT_mid) * float(scaling)
    return float(np.trapezoid(weight(x) * fx, x))

def mc_integral_check(
    spec, E_min: float, E_max: float,
    *, n_samples: int = 2000, n_quad: int = 256,
    weight = None, seed: int = 0
):
    """
    Monte-Carlo sanity check for integral uncertainty due to tail fits.
    Returns dict with delta vs MC stats and the sample array.
    """
    if weight is None:
        weight = lambda x: np.ones_like(x, float)

    rng = np.random.default_rng(seed)

    # 1) Deterministic mid integral (no variance unless you add it)
    low_hi  = spec.low.support[1]  if spec.low  is not None else E_min
    high_lo = spec.high.support[0] if spec.high is not None else E_max
    I_mid = _mid_integral_fixed(spec.T, spec.scaling,
                                max(E_min, low_hi), min(E_max, high_lo),
                                weight, n_quad=max(256, n_quad))

    # 2) Prepare tails: means/covariances in (a,b)
    # Low
    low_seg = None
    if spec.low is not None:
        aL, bL = spec.low.model.params if hasattr(spec.low, "model") else spec.low.params
        loL, hiL = spec.low.support
        muL = np.array([aL, bL], float)
        SigmaL = _cov_ab_from_fit(spec.low) if hasattr(spec.low, "XtX_inv_centered") else None
        low_seg = (muL, SigmaL, loL, hiL)

    # High
    high_seg = None
    if spec.high is not None:
        aH, bH = spec.high.model.params if hasattr(spec.high, "model") else spec.high.params
        loH, hiH = spec.high.support
        muH = np.array([aH, bH], float)
        SigmaH = _cov_ab_from_fit(spec.high) if hasattr(spec.high, "XtX_inv_centered") else None
        high_seg = (muH, SigmaH, loH, hiH)

    # 3) Draw samples and integrate tails
    samples = np.empty(n_samples, float)
    for i in range(n_samples):
        I = I_mid

        if low_seg is not None:
            mu, Sig, lo, hi = low_seg
            # if no covariance (pure model), keep mean only
            if Sig is None:
                I += spec.scaling * _int_exp_linear(mu[0], mu[1],
                                                    max(E_min, lo), min(E_max, hi),
                                                    weight, n_quad)
            else:
                a,b = rng.multivariate_normal(mu, Sig)
                I += spec.scaling * _int_exp_linear(a, b,
                                                    max(E_min, lo), min(E_max, hi),
                                                    weight, n_quad)

        if high_seg is not None:
            mu, Sig, lo, hi = high_seg
            if Sig is None:
                I += spec.scaling * _int_exp_linear(mu[0], mu[1],
                                                    max(E_min, lo), min(E_max, hi),
                                                    weight, n_quad)
            else:
                a,b = rng.multivariate_normal(mu, Sig)
                I += spec.scaling * _int_exp_linear(a, b,
                                                    max(E_min, lo), min(E_max, hi),
                                                    weight, n_quad)

        samples[i] = I

    # 4) Delta-method reference from your own integrate(..., return_std=True)
    mean_delta, std_delta = spec.integrate(E_min, E_max, return_std=True)

    return {
        "delta_mean": mean_delta,
        "delta_std":  std_delta,
        "mc_mean":    float(samples.mean()),
        "mc_std":     float(samples.std(ddof=1)),
        "ratio_mc_over_delta": float(samples.std(ddof=1) / (std_delta if std_delta>0 else np.nan)),
        "samples":    samples,
        "I_mid":      I_mid,
    }


def mc_integral_check_exact(spec, E_min, E_max, *, n_samples=2000, n_quad=256, seed=0):
    """
    Draw (a,b) for each FitResult tail, but evaluate *exactly* the same integrand as TSpectrum.__call__.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(E_min, E_max, n_quad)

    # Snapshot the original tails
    low_orig  = spec.low
    high_orig = spec.high

    def draw_params(fr):
        # mean + cov in (a,b); if no covariance, return mean only
        if not hasattr(fr, "XtX_inv_centered"):
            return None, None, np.array(fr.params if hasattr(fr, "params") else fr.model.params, float)
        # FitResult case
        mu = np.array(fr.model.params, float)
        Sigma_c = fr.sigma2 * np.asarray(fr.XtX_inv_centered, float)
        J = np.array([[1.0, -float(fr.E_mu)], [0.0, 1.0]], float)
        Sigma = J @ Sigma_c @ J.T
        return mu, Sigma, mu  # (mean, cov, fallback)

    # Prepare low/high sampling info
    low_mu,  low_Sig,  low_mean  = draw_params(spec.low)  if spec.low  is not None else (None, None, None)
    high_mu, high_Sig, high_mean = draw_params(spec.high) if spec.high is not None else (None, None, None)

    samples = np.empty(n_samples, float)

    for i in range(n_samples):
        # For each draw, replace spec.low/high with a *temporary* model that has the sampled params,
        # but keep the same supports and call path.
        if spec.low is not None:
            if low_Sig is None:
                ab = low_mean
            else:
                ab = rng.multivariate_normal(low_mu, low_Sig)
            # rebuild a minimal drop-in with same interface (params, support, fn returns *log T* or *T* exactly like yours)
            spec.low = type(low_orig)(support=low_orig.support, fn=low_orig.fn, params=tuple(ab)) if hasattr(low_orig, "fn") else low_orig

        if spec.high is not None:
            if high_Sig is None:
                ab = high_mean
            else:
                ab = rng.multivariate_normal(high_mu, high_Sig)
            spec.high = type(high_orig)(support=high_orig.support, fn=high_orig.fn, params=tuple(ab)) if hasattr(high_orig, "fn") else high_orig

        # Use the *exact* integrand
        fx = np.array([spec(e) for e in x], float)
        samples[i] = float(np.trapezoid(fx, x))

    # Restore originals
    spec.low, spec.high = low_orig, high_orig

    # Reference delta from your method
    delta_mean, delta_std = spec.integrate(E_min, E_max, return_std=True)

    return {
        "delta_mean": delta_mean,
        "delta_std":  float(delta_std),
        "mc_mean":    float(samples.mean()),
        "mc_std":     float(samples.std(ddof=1)),
        "ratio_mc_over_delta": float(samples.std(ddof=1) / (float(delta_std) if delta_std > 0 else np.nan)),
        "samples":    samples,
    }