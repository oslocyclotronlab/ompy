from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, value_and_grad
from tqdm.auto import tqdm

from ..array import Matrix, Vector
from ..array.index import Index
from ..array.ops import exeg_to_efeg
from .jax_kl import kl_div
from ..pipeline import lift, Result, Settings, template_strategy
from ..helpers import make_ax
from ..stubs import Axes
from typing import override

type Optimizer = optax.GradientTransformationExtraArgs

"""
- Should FG be row-normalized or should Fg[cut] be row-normalized?
"""

@dataclass(kw_only=True)
class DecompositionSettings(Settings):
    """Settings for the direct decomposition optimizer.
    
    Attributes:
        iterations: Number of optimization iterations
        optimizer: Optax optimizer (default: Adam with lr=1e-3)
        pre_normalize: If True, row-normalize input FG matrix
        disable_tqdm: If True, disable progress bar
        mask: Optional mask for valid data points in FG matrix
        lam_rho2: L2 regularization weight for rho smoothness
        lam_T2: L2 regularization weight for T smoothness
        G_eg: Optional response matrix for gamma-ray axis
        G_ex: Optional response matrix for excitation axis
        leave_tqdm: If True, keep progress bar after completion
        Ef_index: Template index for Ef axis (for ensemble consistency).
            When set, exeg_to_efeg is called without cutting, then P is sliced
            to match this template. This ensures all ensemble members have 
            compatible output grids.
        Eg_index: Template index for Eg axis (for ensemble consistency).
            When set, ensures consistent Eg axis across ensemble members.
    
    Example:
        Process ensemble with consistent grids:
        
        >>> # Get template from first member
        >>> result_first = decompose(ensemble[0])
        >>> template = extract_template(result_first)
        >>> 
        >>> # Create settings with template
        >>> settings = DecompositionSettings(**template)
        >>> results = decompose(ensemble, settings=settings)
        >>> 
        >>> # Or update existing settings
        >>> settings = settings.update(**template)
    """
    iterations: int = 500
    optimizer: Optimizer = optax.adam(1e-3)
    pre_normalize: bool = True
    disable_tqdm: bool = False
    mask: np.ndarray | None = None
    lam_rho2: float = 1e-2
    lam_T2: float = 1e-3
    G_eg: Matrix | None = None
    G_ex: Matrix | None = None
    leave_tqdm: bool = True
    
    # Template indices for ensemble processing
    Ef_index: Index | None = None
    Eg_index: Index | None = None
    

@dataclass
class DecompositionResultTuple:
    rho: Vector
    T: Vector

@dataclass(kw_only=True)
class DecompositionResult(Result[DecompositionResultTuple]):
    rho: Vector
    T: Vector
    FG: Matrix
    P: Matrix
    loss: Vector
    R: Matrix | None = None
    mask: np.ndarray | None = None
    FG_efeg: Matrix | None = None
    settings: DecompositionSettings

    @override
    def __unwrap__(self) -> DecompositionResultTuple:
        return DecompositionResultTuple(self.rho, self.T)

    def to_template_kwargs(self) -> dict:
        """Extract template parameters as kwargs for ensemble processing.
        
        Used by template_strategy to automatically extract template from
        first result and inject into subsequent calls.
        
        Returns:
            Dictionary with 'Ef_index' and 'Eg_index' keys
            
        Example:
            >>> result = decompose(ensemble[0])
            >>> template_kwargs = result.to_template_kwargs()
            >>> # These get merged into kwargs automatically by template_strategy
        """
        return {
            'Ef_index': self.rho.X_index,
            'Eg_index': self.T.X_index,
        }

    def plot(self, ax: Axes | None = None, **kwargs):
        """Plot rho and T side by side.
        
        Args:
            ax: Optional matplotlib axes (creates 1x2 subplot if None)
            **kwargs: Additional keyword arguments passed to Vector.plot()
            
        Returns:
            Array of matplotlib axes
        """
        ax: np.ndarray = make_ax(ax, ncols=2, constrained_layout=True)  # type: ignore
        self.rho.plot(ax=ax[0], **kwargs)
        self.T.plot(ax=ax[1], **kwargs)
        return ax

    def plot_compare(
        self,
        ax: Axes | None = None,
        abs_kwargs: dict | None = None,
        rel_kwargs: dict | None = None,
        kl_kwargs: dict | None = None,
        perplexity_kwargs: dict | None = None,
        pearson_kwargs: dict | None = None,
        chisq_kwargs: dict | None = None,
        ignore_mask: bool = False,
        **kwargs,
    ):
        """Create 4x2 comparison plot showing FG, P, and various error metrics.
        
        Args:
            ax: Optional matplotlib axes (creates 4x2 subplot if None)
            abs_kwargs: Keyword arguments for absolute error plot
            rel_kwargs: Keyword arguments for relative error plot
            kl_kwargs: Keyword arguments for KL divergence plot
            perplexity_kwargs: Keyword arguments for perplexity plot
            pearson_kwargs: Keyword arguments for Pearson residuals plot
            chisq_kwargs: Keyword arguments for chi-squared residuals plot
            ignore_mask: If False, set masked regions to NaN in diagnostic plots
            **kwargs: Keyword arguments passed to FG and P plots
            
        Returns:
            Array of matplotlib axes
        """
        ax: np.ndarray = make_ax(
            ax,
            nrows=4,
            ncols=2,
            constrained_layout=True,
            sharex=True,
            sharey=True,
            figsize=(10, 16),
        )  # type: ignore
        ax = np.ravel(ax)

        # Row 1: Data and Model
        vmin = min(self.FG.min(), self.P.min())
        vmax = max(self.FG.max(), self.P.max())
        kwargs = {} if kwargs is None else kwargs
        kwargs = {"vmin": vmin, "vmax": vmax} | kwargs
        self.FG.plot(ax=ax[0], **kwargs)
        self.P.plot(ax=ax[1], **kwargs)

        # Row 2: Basic errors
        err = self.FG - self.P
        rel_err = err / self.FG

        # Apply mask if requested
        if not ignore_mask and self.mask is not None:
            err.values = np.where(self.mask, err.values, np.nan)
            rel_err.values = np.where(self.mask, rel_err.values, np.nan)

        abs_kwargs = {} if abs_kwargs is None else abs_kwargs
        rel_kwargs = {} if rel_kwargs is None else rel_kwargs

        err.plot(ax=ax[2], **abs_kwargs)
        rel_err.plot(ax=ax[3], **rel_kwargs)

        ax[2].set_title("Absolute Error")
        ax[3].set_title("Relative Error")

        # Row 3: KL divergence and Perplexity
        # KL(P||Q) = P*log(P/Q) for discrete distributions
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        FG_safe = np.where(self.FG.values > eps, self.FG.values, eps)
        P_safe = np.where(self.P.values > eps, self.P.values, eps)
        
        kl_div_values = FG_safe * np.log(FG_safe / P_safe)
        perplexity_values = np.exp(kl_div_values)
        
        # Apply mask if requested
        if not ignore_mask and self.mask is not None:
            kl_div_values = np.where(self.mask, kl_div_values, np.nan)
            perplexity_values = np.where(self.mask, perplexity_values, np.nan)
        
        kl_div = Matrix(Ex=self.FG.Ex, Eg=self.FG.Eg, values=kl_div_values)
        kl_div.title = "KL Divergence"
        
        perplexity = Matrix(Ex=self.FG.Ex, Eg=self.FG.Eg, values=perplexity_values)
        perplexity.title = "Perplexity"

        kl_kwargs = {} if kl_kwargs is None else kl_kwargs
        perplexity_kwargs = {} if perplexity_kwargs is None else perplexity_kwargs

        kl_div.plot(ax=ax[4], **kl_kwargs)
        perplexity.plot(ax=ax[5], **perplexity_kwargs)

        # Row 4: Pearson and Chi-squared residuals
        # Pearson residuals: (observed - expected) / sqrt(expected)
        pearson_values = (self.FG.values - self.P.values) / np.sqrt(P_safe)
        
        # Chi-squared residuals: (observed - expected)^2 / expected
        chisq_values = (self.FG.values - self.P.values)**2 / P_safe
        
        # Apply mask if requested
        if not ignore_mask and self.mask is not None:
            pearson_values = np.where(self.mask, pearson_values, np.nan)
            chisq_values = np.where(self.mask, chisq_values, np.nan)
        
        pearson = Matrix(Ex=self.FG.Ex, Eg=self.FG.Eg, values=pearson_values)
        pearson.title = "Pearson Residuals"
        
        chisq = Matrix(Ex=self.FG.Ex, Eg=self.FG.Eg, values=chisq_values)
        chisq.title = "Chi-squared Residuals"

        pearson_kwargs = {} if pearson_kwargs is None else pearson_kwargs
        chisq_kwargs = {} if chisq_kwargs is None else chisq_kwargs

        pearson.plot(ax=ax[6], **pearson_kwargs)
        chisq.plot(ax=ax[7], **chisq_kwargs)

        xlabel = ax[0].get_xlabel()
        ylabel = ax[0].get_ylabel()
        for i in range(8):
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

        ax[0].figure.supxlabel(xlabel)
        ax[0].figure.supylabel(ylabel)
        return ax

    def plot_compare_2(
        self,
        ax: Axes | None = None,
        abs_kwargs: dict | None = None,
        rel_kwargs: dict | None = None,
        **kwargs,
    ):
        """Create 2x2 comparison plot in Ef×Eg space.
        
        Similar to plot_compare but transforms FG and P to Ef×Eg coordinates.
        
        Args:
            ax: Optional matplotlib axes (creates 2x2 subplot if None)
            abs_kwargs: Keyword arguments for absolute error plot
            rel_kwargs: Keyword arguments for relative error plot
            **kwargs: Keyword arguments passed to FG and P plots
            
        Returns:
            Array of matplotlib axes
        """
        ax: np.ndarray = make_ax(
            ax,
            nrows=2,
            ncols=2,
            constrained_layout=True,
            sharex=True,
            sharey=True,
            figsize=(10, 10),
        )  # type: ignore
        ax = np.ravel(ax)
        
        FG = exeg_to_efeg(self.FG, cut=True)
        P = exeg_to_efeg(self.P, cut=True)
        ef0 = P.Ef[0]
        FG = FG.loc[f">{ef0}" :, :]
        
        vmin = min(FG.min(), P.min())
        vmax = max(FG.max(), P.max())
        kwargs = {} if kwargs is None else kwargs
        kwargs = {"vmin": vmin, "vmax": vmax} | kwargs

        FG.plot(ax=ax[0], **kwargs)
        P.plot(ax=ax[1], **kwargs)
        
        err = FG - P
        rel_err = err / FG

        abs_kwargs = {} if abs_kwargs is None else abs_kwargs
        rel_kwargs = {} if rel_kwargs is None else rel_kwargs

        err.plot(ax=ax[2], **abs_kwargs)
        rel_err.plot(ax=ax[3], **rel_kwargs)

        ax[2].set_title("Absolute Error")
        ax[3].set_title("Relative Error")

        xlabel = ax[0].get_xlabel()
        ylabel = ax[0].get_ylabel()
        for i in range(4):
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

        ax[0].figure.supxlabel(xlabel)
        ax[0].figure.supylabel(ylabel)
        return ax


# 1) Build a static interpolation plan for ef = Ex - Eg onto Ef edges (linear interp)
def make_interp_plan_exeg(FG: Matrix, Ef_edges: np.ndarray):
    Ex = jnp.array(FG.Ex, dtype=jnp.float32)  # (nx,)
    Eg = jnp.array(FG.Eg, dtype=jnp.float32)  # (ng,)
    Ef_edges = jnp.array(Ef_edges, dtype=jnp.float32)  # (nf+1,)

    ex = Ex[:, None]  # (nx,1)
    eg = Eg[None, :]  # (1,ng)
    ef = ex - eg  # (nx,ng)

    # upper bin index, clip to valid interior
    i1 = jnp.clip(
        jnp.searchsorted(Ef_edges, ef, side="right") - 1, 0, Ef_edges.size - 2
    )
    i0 = jnp.maximum(i1 - 1, 0)

    e0 = Ef_edges[i0]
    e1 = Ef_edges[i1 + 1]
    denom = jnp.maximum(e1 - e0, 1e-12)
    w1 = jnp.clip((ef - e0) / denom, 0.0, 1.0)
    w0 = 1.0 - w1

    plan = {
        "i0": i0,
        "i1": i1,
        "w0": w0,
        "w1": w1,
        "Ex": Ex,
        "Eg": Eg,
        "Ef_edges": Ef_edges,
    }
    return plan


# 2) Log-space model on Ex×Eg:
#    logP(ex,eg) = log rho(ef) + log T(eg) - logZ_row(ex),
#    with rho linearly interpolated on Ef bins.
def build_P_exeg_logspace(theta_rho, theta_T, plan):
    i0, i1, w0, w1 = plan["i0"], plan["i1"], plan["w0"], plan["w1"]

    # Unconstrained params -> log-rho, log-T
    log_rho_grid = theta_rho  # (nf,)
    log_T_grid = theta_T  # (ng,)

    # Interpolate rho at each (ex,eg) in *value* space, stably via log-sum-exp
    log_r0 = log_rho_grid[i0]  # (nx,ng)
    log_r1 = log_rho_grid[i1]  # (nx,ng)
    log_r_interp = jnp.log(w0 + 1e-20) + log_r0
    log_r_interp = jnp.logaddexp(log_r_interp, jnp.log(w1 + 1e-20) + log_r1)

    log_T = log_T_grid[None, :]  # (1,ng)
    logP = log_r_interp + log_T  # (nx,ng)

    # Row-normalize gauge (softmax along Eg). This is in log-space -> subtract logsumexp.
    logP = logP - jax.scipy.special.logsumexp(logP, axis=1, keepdims=True)
    P = jnp.exp(logP)  # (nx,ng), each row sums to 1
    return P


# 3) Smoothness penalties (optional)
def smooth_penalty_1d(theta, lam1=0.0, lam2=1e-2):
    d1 = jnp.diff(theta)
    d2 = jnp.diff(theta, n=2)
    return lam1 * jnp.sum(d1**2) + lam2 * jnp.sum(d2**2)


def setup(
    FG: Matrix, 
    Ef_index: Index | None = None,
    Eg_index: Index | None = None
) -> tuple[Matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Setup the decomposition by transforming FG to Ef×Eg space.
    
    Args:
        FG: First-generation matrix in Ex×Eg space
        Ef_index: Optional template index for Ef axis (for ensemble consistency)
        Eg_index: Optional template index for Eg axis (for ensemble consistency)
    
    Returns:
        Tuple of (P matrix, Ef array, Eg array, Ex array)
    """
    assert FG.Ex_index.is_uniform(), "Ex must be uniform"
    assert FG.Eg_index.is_uniform(), "Eg must be uniform"
    
    # Determine if we need template-based cutting
    use_template = Ef_index is not None or Eg_index is not None
    
    if use_template:
        # Get full transformation without cutting
        P = exeg_to_efeg(FG, cut=False)
        
        # Create a template matrix with desired indices for cutting
        # Use existing P indices as defaults if templates not provided
        template_Ef_index = Ef_index if Ef_index is not None else P.X_index
        template_Eg_index = Eg_index if Eg_index is not None else P.Y_index
        
        # Create dummy template matrix with desired axes
        template = Matrix(
            X=template_Ef_index,
            Y=template_Eg_index,
            values=np.zeros((len(template_Ef_index), len(template_Eg_index)))
        )
        
        # Cut P to match template
        P = P.cut_like(template, inplace=False)
    else:
        # Use original heuristic cut
        P = exeg_to_efeg(FG, cut=True)
    
    # Extract axes from the resulting matrix (don't reuse Index objects)
    return P, P.Ef, P.Eg, FG.Ex


# 4) Optimizer that mirrors your 'optimize' API but works in Ex×Eg directly
def optimize(
    FG: Matrix,
    settings: DecompositionSettings = DecompositionSettings(),
    **kwargs,
):
    """
    Same return type as your optimize(), but internally builds P in Ex×Eg
    with a log-space model and row-softmax normalization.
    """
    settings, _ = settings.consume(kwargs, error_on_leftover=True)

    if jnp.any(FG.values < 0):
        raise ValueError("FG values must be non-negative.")

    if (mask := settings.mask) is None:
        mask = np.isfinite(np.asarray(FG.values)) & (np.asarray(FG.values) > 1e-3)
    mask = jnp.array(mask)

    # Row-normalize data in Ex×Eg if requested (same as your row_normalize)  [gauge fix]
    if settings.pre_normalize:
        FG = FG / FG.values.sum(axis=1, keepdims=True)  # your helper does this too
    P_data = jnp.array(FG.values, dtype=jnp.float32)  # (nx,ng)

    # Reuse your ef grid by calling exeg_to_efeg once (book-keeping stays aligned)
    FG_efeg, Ef, Eg, Ex = setup(
        FG, 
        Ef_index=settings.Ef_index,
        Eg_index=settings.Eg_index
    )
    Ef = np.asarray(Ef, dtype=np.float32)
    Eg = np.asarray(Eg, dtype=np.float32)

    # Build interpolation plan onto Ef edges
    #Ef_edges = np.linspace(
    #    Ef[0] - (Ef[1] - Ef[0]) / 2, Ef[-1] + (Ef[1] - Ef[0]) / 2, len(Ef) + 1
    #).astype(np.float32)
    plan = make_interp_plan_exeg(FG, Ef)

    # Init params in log-space: start close to empirical marginals
    # Avoid zeros; add eps and renormalize for stability.
    row_sum = np.asarray(FG_efeg.sum(axis=1)) + 1e-12
    col_sum = np.asarray(P_data.sum(axis=0)) + 1e-12
    rho0 = row_sum / row_sum.sum()
    T0 = col_sum / col_sum.sum()
    theta_rho = jnp.log(jnp.asarray(rho0, dtype=jnp.float32))
    theta_T = jnp.log(jnp.asarray(T0, dtype=jnp.float32))

    params = {"theta_rho": theta_rho, "theta_T": theta_T}
    optimizer = settings.optimizer
    opt_state = optimizer.init(params)


    if (G_eg := settings.G_eg) is not None:
        G_eg = jnp.array(G_eg.values)
    if (G_ex := settings.G_ex) is not None:
        G_ex = jnp.array(G_ex.values)

    @jit
    def loss_and_grad_fn(params):
        P_hat = build_P_exeg_logspace(params["theta_rho"], params["theta_T"], plan)
        # KL(P_data || P_hat) over observed cells only (your kl_div has P, Q order)
        data = jnp.where(mask, P_data, 0.0)
        model = jnp.where(mask, P_hat, 1e-12)
        if G_eg is not None:
            model = model @ G_eg
        if G_ex is not None:
            model = G_ex @ model
        nll = kl_div(
            data, model
        )[mask].sum()  # reuse your KL helper :contentReference[oaicite:4]{index=4}
        reg = smooth_penalty_1d(params["theta_rho"], lam2=settings.lam_rho2) + smooth_penalty_1d(
            params["theta_T"], lam2=settings.lam_T2
        )
        return nll + reg

    loss_and_grad = jit(value_and_grad(loss_and_grad_fn))

    losses = np.zeros(settings.iterations, dtype=np.float64)
    bar = range(settings.iterations) if settings.disable_tqdm else tqdm(range(settings.iterations), leave=settings.leave_tqdm)
    for i in bar:
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses[i] = float(loss)
        if not settings.disable_tqdm:
            bar.set_postfix_str(f"loss: {loss:.2e}")

    # Extract rho/T on your Vector grid (Ef/Eg) and build P Matrix on Ex×Eg
    theta_rho, theta_T = params["theta_rho"], params["theta_T"]
    rho = jnp.exp(theta_rho)
    rho = rho / (rho.sum() + 1e-12)
    T = jnp.exp(theta_T)
    T = T / (T.sum() + 1e-12)

    rho_v = Vector(
        E=np.asarray(Ef),
        values=np.array(rho),
        xlabel=r'$E_\mathrm{in}$',
        ylabel="amplitude [arbitrary]",
        title="rho",
    )  # :contentReference[oaicite:5]{index=5}
    T_v = Vector(
        E=np.asarray(Eg),
        values=np.array(T),
        xlabel=r"$E_\gamma$",
        ylabel="amplitude [arbitrary]",
        title="T",
    )  # :contentReference[oaicite:6]{index=6}

    # Model P on Ex×Eg from final params
    P_hat = build_P_exeg_logspace(theta_rho, theta_T, plan)
    if G_eg is not None:
        P_hat = P_hat @ G_eg
        T_v.values = T_v @ G_eg
    if G_ex is not None:
        P_hat = G_ex @ P_hat
    P_m = Matrix(Ex=np.asarray(FG.Ex), Eg=np.asarray(FG.Eg), values=np.array(P_hat))
    P_m.title = "fitted P (row-normalized)"

    losses_v = Vector(i=np.arange(settings.iterations), values=losses, name="Loss", xlabel="iteration", vlabel="loss")

    # Also keep your ef×eg view for debugging if you want:
    # (use your existing nld_T_product for exact consistency)
    # P_efeg = exeg_to_efeg(P_m, cut=True)   # reuse your mapping helper :contentReference[oaicite:7]{index=7}

    return DecompositionResult(
        rho=rho_v,
        T=T_v,
        FG=FG,
        P=P_m,
        loss=losses_v,
        R=None,
        mask=np.asarray(mask),
        settings=settings,
    )  # :contentReference[oaicite:8]{index=8}


def extract_template(result: DecompositionResult) -> dict:
    """Extract template indices from a decomposition result for ensemble processing.
    
    Use this to ensure consistent grids when processing an ensemble of matrices.
    
    Args:
        result: A DecompositionResult from decomposing a single matrix
        
    Returns:
        Dictionary with 'Ef_index' and 'Eg_index' suitable for passing to
        DecompositionSettings.update()
        
    Example:
        >>> # Process first member to get template
        >>> first_result = decompose(ensemble[0])
        >>> template = extract_template(first_result)
        >>> 
        >>> # Process full ensemble with consistent grids
        >>> settings = DecompositionSettings(**template)
        >>> results = decompose(ensemble, settings=settings)
        >>> 
        >>> # Or update existing settings
        >>> settings = settings.update(**template)
    """
    return {
        'Ef_index': result.rho.X_index,
        'Eg_index': result.T.X_index,
        'mask': None,
    }


@lift(expects='first generation', produces='decomposed', ensemble_strategy=template_strategy)
def decompose(FG: Matrix, *args, **kwargs) -> DecompositionResult:
    return optimize(FG, *args, **kwargs)