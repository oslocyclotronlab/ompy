from jax_tqdm import scan_tqdm, PBar
from dataclasses import dataclass
from functools import partial
import optax
from typing import Callable
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from dataclasses import field



from ..array import Vector
from ..pipeline.result import Settings, Result

def ct(E, eshift, T):
    return jnp.exp((E - eshift) / T) / T


def ct_log(E, eshift, T):
    return (E - eshift) / T - jnp.log(T)


@dataclass(kw_only=True)
class NormalizationSettings(Settings):
    iterations: int = 100
    leave_tqdm: bool = True
    discrete_mask: np.ndarray | None = None
    model_mask: np.ndarray | None = None
    model: Callable = ct_log
    center_E: bool = True
    disable_tqdm: bool = False
    initial: dict[str, float] = field(default_factory=dict)
    sample_rho_at_sn: bool = False


@dataclass
class NormalizationResult(Result[Vector]):
    rho: Vector
    theta: dict[str, float]
    loss: Vector
    discrete: Vector
    settings: NormalizationSettings
    rho_at_sn: tuple[float, float, float]

    def plot(self, ax=None):
        if ax is None:
            import matplotlib.gridspec as gridspec

            fig = plt.figure(figsize=(6, 10), constrained_layout=False)
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.3)
            ax = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
            ax[0].sharex(ax[1])
        self.rho.plot(ax=ax[0], label=r"Normalized $\rho$")
        self.discrete.plot(ax=ax[0], label="Levels")

        # We plot the interpolation of the model from a bit before the first model point to a bit after the last model point
        start = 0.8*self.rho.X[np.argmax(self.settings.model_mask)]
        sn, rho_at_sn, rho_at_sn_err = self.rho_at_sn
        stop = sn
        erange = np.linspace(start, stop, 100)
        model_params = {k: v for k, v in self.theta.items() if k in ['T', 'eshift']}
        model = self.settings.model(erange, **model_params)
        model = jnp.exp(model)
        ax[0].plot(erange, model, label="Model")
        ax[0].errorbar(sn, rho_at_sn, rho_at_sn_err, color="k", label=r"$\rho(S_n)$", marker="o", markersize=3)

        ymax = max(self.discrete.max(), self.rho.max())
        ymax = max(ymax, rho_at_sn + 2 * rho_at_sn_err)
        ax[0].set_yscale("log")
        ax[0].set_ylim(5e-1, 1.1 * ymax)
        ax[0].grid(True, which="major")
        ax[0].grid(True, which="minor", lw=0.5, color="0.65")

        residual = np.log(self.rho) - np.log(self.discrete)
        residual[~np.isfinite(residual.values) | (self.rho < 1e-1)] = np.nan
        residual.plot(ax=ax[1])
        ymin, ymax = ax[1].get_ylim()
        loss = optax.losses.huber_loss(np.log(self.rho.values), np.log(self.discrete.values), delta=1.0)
        loss = residual.clone(values=loss)
        loss.plot(ax=ax[1])
        ax[1].set_ylabel("Log residual")
        ax[1].set_title("Residual for discrete fit")
        ax[1].set_ylim(ymin, ymax)


        residual = np.log(self.rho) - self.settings.model(self.rho.X, **model_params)
        residual = residual.as_numpy(copy=True)
        residual[~np.isfinite(residual) | (self.rho < 1e-1)] = np.nan
        residual.plot(ax=ax[2])
        model = self.settings.model(self.rho.X, **model_params)
        loss = optax.losses.huber_loss(np.log(self.rho.values), np.log(model), delta=1.0)
        loss = residual.clone(values=loss)
        ax2 = ax[2].twinx()
        loss.plot(ax=ax2, color='C1')
        ax2.set_ylabel("Huber loss")
        ax[2].set_ylabel("Log residual")
        ax[2].set_title("Residuals for model fit")
        i = np.argmax(self.settings.model_mask[::-1])
        i = i + 1
        stop = 1.1*self.rho.X[len(self.rho.X) - i]
        ax[2].set_xlim(start, stop)

        vals = residual[self.settings.model_mask]
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)
        m = max(abs(ymin), abs(ymax))
        ax[2].set_ylim(-1.2*m, 1.2*m)

        for s, e in zip(*make_mask_span(self.settings.discrete_mask)):
            ax[1].axvspan(
                self.rho.X[s],
                self.rho.X[e],
                color="grey",
                alpha=0.1,
            )
            ax[0].axvspan(
                self.rho.X[s],
                self.rho.X[e],
                color="grey",
                alpha=0.1,
                label="Discrete fit range",
            )

        for s, e in zip(*make_mask_span(self.settings.model_mask)):
            ax[2].axvspan(
                self.rho.X[s],
                self.rho.X[e],
                color='C3',
                alpha=0.1,
            )
            ax[0].axvspan(
                self.rho.X[s],
                self.rho.X[e],
                color='C3',
                alpha=0.1,
                label="Model fit range",
            )

        ax[0].xaxis.set_major_locator(MultipleLocator(1))

        ax[1].axhline(0, color="k")
        ax[2].axhline(0, color="k")
        ax[0].set_xlabel("")
        ax[1].set_xlabel("")
        #fig.legend(loc='center left', bbox_to_anchor=(0.5, 0.5), frameon=False, bbox_transform=fig.transFigure)
        ax[0].legend()
        return ax

    def _prepare_loss_computation(self):
        """Prepare data needed for loss computation.
        
        Returns:
            dict: Dictionary containing all data needed for loss calculations
        """
        # Extract energy and values
        E, rho_values = self.rho.unpack()
        E_, discrete_values = self.discrete.unpack()
        
        # Convert to log space
        eps = 1e-5
        rho_log = np.log(rho_values + eps)
        discrete_log = np.log(discrete_values + eps)
        
        # Extract theta values
        A = self.theta['A']
        alpha = self.theta['alpha']
        T = self.theta['T']
        eshift = self.theta['eshift']
        
        # Reverse-transform rho to get original (before normalization)
        rho_log_original = rho_log - A - alpha * E
        
        # Extract rho_at_sn
        sn, rho_at_sn, rho_at_sn_err = self.rho_at_sn
        rho_at_sn_log = np.log(rho_at_sn)
        rho_at_sn_err_log = np.log(rho_at_sn_err)
        
        return {
            'E': E,
            'rho': rho_log_original,
            'discrete': discrete_log,
            'A': A,
            'alpha': alpha,
            'T': T,
            'eshift': eshift,
            'sn': sn,
            'rho_at_sn': rho_at_sn_log,
            'rho_at_sn_err': rho_at_sn_err_log,
        }
    
    def loss_discrete(self, weighted=True):
        """Compute discrete region loss.
        
        Args:
            weighted: If True, return weighted loss. If False, return unweighted sum.
            
        Returns:
            float: The discrete region loss
        """
        data = self._prepare_loss_computation()
        
        # Apply normalization to rho
        rho = data['A'] + data['rho'] + data['alpha'] * data['E']
        
        # Compute loss
        loss = optax.losses.huber_loss(rho, data['discrete'], delta=1.0)
        discrete_loss = loss[self.settings.discrete_mask]
        
        if weighted:
            w_disc = self.settings.discrete_mask.sum()
            return float(discrete_loss.sum() / w_disc)
        else:
            return float(discrete_loss.sum())
    
    def loss_model(self, weighted=True):
        """Compute model region loss.
        
        Args:
            weighted: If True, return weighted loss. If False, return unweighted sum.
            
        Returns:
            float: The model region loss
        """
        data = self._prepare_loss_computation()
        
        # Apply normalization to rho
        rho = data['A'] + data['rho'] + data['alpha'] * data['E']
        
        # Compute model at model mask points
        model = ct_log(data['E'][self.settings.model_mask], data['eshift'], data['T'])
        model_loss = optax.losses.huber_loss(model, rho[self.settings.model_mask], delta=1.0)
        
        if weighted:
            w_model = self.settings.model_mask.sum()
            return float(model_loss.sum() / w_model)
        else:
            return float(model_loss.sum())
    
    def loss_sn(self, weighted=True):
        """Compute S_n loss.
        
        Args:
            weighted: If True, return weighted loss. If False, return unweighted loss.
            
        Returns:
            float: The S_n loss
        """
        data = self._prepare_loss_computation()
        
        # Compute model at S_n
        model_at_sn = ct_log(data['sn'], data['eshift'], data['T'])
        loss = (model_at_sn - data['rho_at_sn']) ** 2
        
        if weighted:
            w_sn = 1.0 / data['rho_at_sn_err']**2
            return float(loss * w_sn)
        else:
            return float(loss)
    
    def summary(self):
        """Print a summary of the normalization results."""
        theta = self.theta.copy()
        A = theta.pop('A')
        alpha = theta.pop('alpha')
        
        print("=" * 60)
        print("Normalization Results Summary")
        print("=" * 60)
        
        print("\nOptimization Parameters:")
        print(f"  A       = {A:12.6f}")
        print(f"  alpha   = {alpha:12.6f}")
        
        print("\nModel Parameters:")
        for key, value in theta.items():
            print(f"  {key:8s} = {value:12.6f}")
        
        print(f"\nFinal Total Loss: {self.loss.values[-1]:.6e}")
        
        print("\nLoss by Region:")
        print(f"  {'Region':<15} {'Unweighted':>15} {'%':>8} {'Weighted':>15} {'%':>8}")
        print(f"  {'-'*15} {'-'*15} {'-'*8} {'-'*15} {'-'*8}")
        
        loss_disc_uw = self.loss_discrete(weighted=False)
        loss_disc_w = self.loss_discrete(weighted=True)
        
        loss_model_uw = self.loss_model(weighted=False)
        loss_model_w = self.loss_model(weighted=True)
        
        loss_sn_uw = self.loss_sn(weighted=False)
        loss_sn_w = self.loss_sn(weighted=True)
        
        total_unweighted = loss_disc_uw + loss_model_uw + loss_sn_uw
        total_weighted = loss_disc_w + loss_model_w + loss_sn_w
        
        pct_disc_uw = 100 * loss_disc_uw / total_unweighted
        pct_disc_w = 100 * loss_disc_w / total_weighted
        print(f"  {'Discrete':<15} {loss_disc_uw:>15.6e} {pct_disc_uw:>7.2f}% {loss_disc_w:>15.6e} {pct_disc_w:>7.2f}%")
        
        pct_model_uw = 100 * loss_model_uw / total_unweighted
        pct_model_w = 100 * loss_model_w / total_weighted
        print(f"  {'Model':<15} {loss_model_uw:>15.6e} {pct_model_uw:>7.2f}% {loss_model_w:>15.6e} {pct_model_w:>7.2f}%")
        
        pct_sn_uw = 100 * loss_sn_uw / total_unweighted
        pct_sn_w = 100 * loss_sn_w / total_weighted
        print(f"  {'S_n':<15} {loss_sn_uw:>15.6e} {pct_sn_uw:>7.2f}% {loss_sn_w:>15.6e} {pct_sn_w:>7.2f}%")
        
        print(f"  {'-'*15} {'-'*15} {'-'*8} {'-'*15} {'-'*8}")
        print(f"  {'Total':<15} {total_unweighted:>15.6e} {'100.00%':>8} {total_weighted:>15.6e} {'100.00%':>8}")
        
        print("=" * 60)

    def __unwrap__(self):
        return self.rho


def make_mask_span(mask):
    # Plot span where the  mask is true
    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    # handle if mask starts or ends True
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)-1]
    return starts, ends


def loss_fn(
    theta, E, rho, discrete, discrete_mask, model_mask, sn, rho_at_sn, rho_at_sn_err,
    center_E=True,
):
    # We assume we are in log space
    # A, alpha = theta['A'], theta['alpha']

    (A, alpha), (T, eshift) = theta
    E_centered = E - E.mean() if center_E else E
    rho = A + rho + alpha * E_centered  # (E - E[-1])# (E - 7)
    #loss = (rho - discrete) ** 2
    loss = optax.losses.huber_loss(rho, discrete, delta=1.0)
    discrete_loss = loss[discrete_mask]
    model = ct_log(E[model_mask], eshift, T)
    model_at_sn = ct_log(sn, eshift, T)
    model_loss_at_sn = (model_at_sn - rho_at_sn) ** 2
    #model_loss_at_rho = (model - rho[model_mask]) ** 2
    model_loss_at_rho = optax.losses.huber_loss(model, rho[model_mask], delta=1.0)

    w_disc = discrete_mask.sum()
    w_model = model_mask.sum()
    w_sn = 1 / rho_at_sn_err**2

    return (
        1 / w_disc * discrete_loss.sum()
        + 1 / w_sn * model_loss_at_sn
        + 1 / w_model * model_loss_at_rho.sum()
    )


def make_normalizer(
    discrete: Vector,
    rho_at_sn: tuple[float, float, float],
    settings: NormalizationSettings = NormalizationSettings(),
    opt=optax.adam(1e-4),
    **kwargs
):
    """Create a normalizer function that can be vmapped.
    
    This factory function handles all non-JAX setup (Vector unpacking, unit conversions,
    mask processing) and returns a pure JAX function that can be vmapped over multiple
    rho vectors.
    
    Args:
        discrete: Discrete level density (Vector)
        rho_at_sn: Tuple of (sn, rho_value, rho_err) at neutron separation energy
        settings: Normalization settings
        opt: Optax optimizer
        **kwargs: Additional settings to update
        
    Returns:
        normalize_fn: Pure JAX function(rho_values, init, pbar_id) -> (theta, loss_history, normalized_values)
            - rho_values: Raw rho values array
            - init: Initial guess ((A, alpha), (T, eshift))
            - pbar_id: Progress bar ID for vmap (None for single call)
            
    Example:
        >>> normalizer = make_normalizer(discrete, rho_at_sn, settings)
        >>> # Single normalization
        >>> theta, loss, norm_rho = normalizer(rho_values, init=None, pbar_id=None)
        >>> # Batched normalization with vmap
        >>> vmapped = jax.vmap(normalizer, in_axes=(0, None, 0))
        >>> batch_theta, batch_loss, batch_norm = vmapped(batch_rhos, None, pbar_ids)
    """
    # Process settings
    if kwargs:
        settings = settings.update(**kwargs)
    
    # Extract and convert discrete to log space
    discrete_converted = discrete.to_mid().to_unit("MeV")
    E, discrete_values = discrete_converted.unpack()
    discrete_values = jnp.log(discrete_values + 1e-5)
    
    # Process rho_at_sn
    sn, rho_at_sn_val, rho_at_sn_err = rho_at_sn
    
    # Process masks
    if settings.discrete_mask is None:
        discrete_mask = jnp.ones(discrete_converted.shape, bool)
    else:
        discrete_mask = settings.discrete_mask
        
    if settings.model_mask is None:
        model_mask = jnp.ones(discrete_converted.shape, bool)
    else:
        model_mask = settings.model_mask

    discrete_mask = np.asarray(discrete_mask, dtype=bool)
    model_mask = np.asarray(model_mask, dtype=bool)

    A, alpha = settings.initial.get('A', None), settings.initial.get('alpha', None)
    if A is None or alpha is None:
        def initial_gauges(rho):
            A_, alpha_ = estimate_A_alpha_leastsq(rho, E=E, mask_disc=discrete_mask, Sn=sn, rho_Sn=rho_at_sn_val, rho_disc=discrete_values)
            A_ = A if A is not None else jnp.log(A_)
            alpha_ = alpha if alpha is not None else alpha_
            return (A_, alpha_)
    else:
        A = A if A is not None else 1.0
        alpha = alpha if alpha is not None else 0.0
        initial_gauges = lambda rho: (A, alpha)
    
    initial = subdict(settings.initial, ['A', 'alpha'])
    initial_ = {'T': 1.0, 'eshift': 0.0} | initial
    settings = settings.update(initial=initial_)

    # Define the pure JAX optimization function
    def normalize_fn(rho_values: jnp.ndarray,  key, pbar_id=None):
        """Pure JAX function that can be vmapped.
        
        Args:
            rho_values: Raw rho values (will be log-transformed)
            init: Initial guess for ((A, alpha), (T, eshift))
            pbar_id: Progress bar ID for vmap context (use PBar if not None)
            
        Returns:
            tuple: (theta, loss_history, normalized_rho_values)
        """
        A, alpha = initial_gauges(rho_values)
        initial = ((A, alpha), (initial_['T'], initial_['eshift']))

        # Convert to log space
        rho_log = jnp.log(rho_values + 1e-5)


        if settings.sample_rho_at_sn:
            z = jax.random.normal(key)
            rho_at_sn_val_ = rho_at_sn_val + z * rho_at_sn_err
        else:
            rho_at_sn_val_ = rho_at_sn_val


        rho_at_sn_log = jnp.log(rho_at_sn_val_)
        rho_at_sn_err_log = jnp.log(rho_at_sn_err)
        
        # Create loss function bound to this rho
        lossfn = jax.value_and_grad(
            partial(
                loss_fn,
                E=E,
                rho=rho_log,
                discrete=discrete_values,
                discrete_mask=discrete_mask,
                model_mask=model_mask,
                sn=sn,
                rho_at_sn=rho_at_sn_log,
                rho_at_sn_err=rho_at_sn_err_log,
                center_E=settings.center_E,
            )
        )
        
        # Initialize optimizer
        opt_state = opt.init(initial)
        
        # Optimization loop with scan_tqdm
        @scan_tqdm(settings.iterations, leave=settings.leave_tqdm, disable=settings.disable_tqdm)
        def body(carry, _):
            theta, state = carry
            loss, grad = lossfn(theta)
            updates, state = opt.update(grad, state)
            theta = optax.apply_updates(theta, updates)
            return (theta, state), loss
        
        # Wrap initial value in PBar if pbar_id is provided (vmap context)
        if pbar_id is not None:
            init_carry = PBar(id=pbar_id, carry=(initial, opt_state))
        else:
            init_carry = (initial, opt_state)
        
        # Run optimization
        final_carry, loss_history = jax.lax.scan(
            body,
            init_carry,
            jnp.arange(settings.iterations),
            length=settings.iterations,
        )
        
        # Extract theta from PBar if needed
        if pbar_id is not None:
            theta, _ = final_carry.carry
        else:
            theta, _ = final_carry
        
        # Apply normalization
        (A, alpha), (T, eshift) = theta
        E_centered = E - E.mean() if settings.center_E else E
        rho_normalized = jnp.exp(A + rho_log + alpha * E_centered)
        
        return theta, loss_history, rho_normalized
    
    return normalize_fn


def normalize(
    rho: Vector,
    discrete: Vector,
    rho_at_sn: tuple[float, float, float],
    opt=optax.adam(1e-4),
    settings: NormalizationSettings = NormalizationSettings(),
    **kwargs,
) -> NormalizationResult:
    """Normalize a single rho vector.
    
    This function fits the level density to discrete levels and a model at high energies,
    optimizing normalization parameters A and alpha.
    
    Args:
        rho: Level density vector to normalize
        discrete: Discrete level density from known levels
        rho_at_sn: Tuple of (sn, rho_value, rho_err) at neutron separation energy
        init: Initial guess for parameters ((A, alpha), (T, eshift))
        opt: Optax optimizer (default: Adam with lr=1e-4)
        settings: Normalization settings
        **kwargs: Additional settings to update
        
    Returns:
        NormalizationResult containing normalized rho, parameters, and loss history
    """
    # Handle settings and masks
    if kwargs:
        settings = settings.update(**kwargs)
    
    # Set default masks if not provided
    if settings.discrete_mask is None:
        settings = settings.update(discrete_mask=jnp.ones(discrete.shape, bool))
    if settings.model_mask is None:
        settings = settings.update(model_mask=jnp.ones(rho.shape, bool))
    
    # Convert to standard units and extract values
    rho_converted = rho.to_mid().to_unit("MeV")
    discrete_converted = discrete.to_mid().to_unit("MeV")
    
    E, rho_values = rho_converted.unpack()
    E_discrete, _ = discrete_converted.unpack()
    
    if not np.allclose(E, E_discrete):
        raise ValueError("Index values do not match between rho and discrete")
    
    # Create normalizer using factory
    normalize_fn = make_normalizer(discrete_converted, rho_at_sn, settings, opt)
    
    # Run normalization (single call, no vmap)
    theta, loss_history, normalized_values = normalize_fn(rho_values, key=jax.random.PRNGKey(0), pbar_id=None)
    
    # Package results into NormalizationResult
    (A, alpha), (T, eshift) = theta
    
    rho_result = rho_converted.clone(
        values=np.asarray(normalized_values),
        name=r"Normalized $\rho$",
        vlabel="density",
        xlabel="$E$",
    )
    
    loss = Vector(
        i=np.asarray(range(settings.iterations)),
        values=np.asarray(loss_history),
        name="Loss",
        vlabel="loss",
        xlabel="iteration",
        unit=''
    )
    
    theta_dict = {
        'A': float(A),
        'alpha': float(alpha),
        'T': float(T),
        'eshift': float(eshift),
    }
    
    return NormalizationResult(
        rho=rho_result,
        theta=theta_dict,
        loss=loss,
        discrete=discrete_converted,
        settings=settings,
        rho_at_sn=rho_at_sn,
    )


def normalize_vmap(
    rhos: list[Vector],
    discrete: Vector,
    rho_at_sn: tuple[float, float, float],
    opt=optax.adam(1e-4),
    settings: NormalizationSettings = NormalizationSettings(disable_tqdm=True),
    **kwargs,
) -> list[NormalizationResult]:
    """Normalize multiple rho vectors in parallel using vmap.
    
    This function efficiently normalizes a batch of level density vectors in parallel
    using JAX's vmap, with individual progress bars for each normalization.
    
    Args:
        rhos: List of Vector objects (all must have same energy grid)
        discrete: Discrete level density (same for all)
        rho_at_sn: Tuple of (sn, rho_value, rho_err) at neutron separation energy (same for all)
        init: Initial guess for parameters ((A, alpha), (T, eshift)), same for all
        opt: Optax optimizer (default: Adam with lr=1e-4)
        settings: Normalization settings
        **kwargs: Additional settings to update
        
    Returns:
        List of NormalizationResult objects, one per input rho
        
    Example:
        >>> results = normalize_vmap(
        ...     rho_list, 
        ...     discrete, 
        ...     rho_at_sn=(7.646, 926.7, 190),
        ...     settings=NormalizationSettings(iterations=1000)
        ... )
        >>> # Each result has individual parameters and normalized rho
        >>> print(results[0].theta)  # {'A': ..., 'alpha': ..., 'T': ..., 'eshift': ...}
    """
    # Handle settings
    if kwargs:
        settings = settings.update(**kwargs)
    
    # Set default masks if not provided
    if settings.discrete_mask is None:
        settings = settings.update(discrete_mask=jnp.ones(discrete.shape, bool))
    if settings.model_mask is None:
        settings = settings.update(model_mask=jnp.ones(rhos[0].shape, bool))
    
    # Extract rho values from Vectors
    template_rho = rhos[0].to_mid().to_unit("MeV")
    discrete_converted = discrete.to_mid().to_unit("MeV")
    
    rho_values_list = [rho.to_mid().to_unit("MeV").values for rho in rhos]
    batch_rho_values = jnp.stack(rho_values_list)
    
    # Create normalizer
    normalizer = make_normalizer(discrete_converted, rho_at_sn, settings, opt)
    
    # Vmap over batch dimension with PBar indexing
    vmapped_normalizer = jax.vmap(
        normalizer,
        in_axes=(0, 0, 0)  # (rho_values, key, pbar_id)
    )
    
    # Create progress bar indices
    pbar_ids = jnp.arange(len(batch_rho_values))

    keys = jax.random.split(jax.random.PRNGKey(0), len(batch_rho_values))
    # Run vmapped normalization
    print("Running vmapped normalization. You will not see any progress bars...")
    batch_theta, batch_loss, batch_normalized = vmapped_normalizer(
        batch_rho_values,
        keys,
        pbar_ids
    )
    print("Done with vmapped normalization.")
    
    # Package results into list of NormalizationResult
    # Note: batch_theta is structured as ((batch_A, batch_alpha), (batch_T, batch_eshift))
    # due to how vmap handles pytrees
    (batch_A, batch_alpha), (batch_T, batch_eshift) = batch_theta
    
    results = []
    for i in range(len(batch_rho_values)):
        rho_result = template_rho.clone(
            values=np.asarray(batch_normalized[i]),
            name=r"Normalized $\rho$",
            vlabel="density",
            xlabel="$E$",
        )
        
        loss = Vector(
            i=np.asarray(range(settings.iterations)),
            values=np.asarray(batch_loss[i]),
            name="Loss",
            vlabel="loss",
            xlabel="iteration",
            unit=''
        )
        
        theta_dict = {
            'A': float(batch_A[i]),
            'alpha': float(batch_alpha[i]),
            'T': float(batch_T[i]),
            'eshift': float(batch_eshift[i])
        }
        
        result = NormalizationResult(
            rho=rho_result,
            theta=theta_dict,
            loss=loss,
            discrete=discrete_converted,
            settings=settings,
            rho_at_sn=rho_at_sn,
        )
        results.append(result)
    
    return results

#@jax.jit
def estimate_A_alpha_leastsq(
    rho_u,            # unnormalized Oslo NLD on E
    E,                # 1D array of bin centers for rho_u
    mask_disc,        # boolean mask: where discrete spectrum is complete
    Sn,               # neutron separation energy (scalar)
    rho_Sn,           # target rho at Sn (per MeV) (scalar)
    rho_disc,         # density estimate from discrete levels on same bins as E (per MeV)
    w_sn: float = 5.0,
    eps: float = 1e-12,
) -> tuple[jnp.ndarray, jnp.ndarray]:

    # Interpolate rho_u at Sn
    rho_u_Sn = jnp.interp(Sn, E, rho_u)

    # Build regression targets: y = ln rho_target - ln rho_u ≈ ln A + alpha * E
    def safe(x): return jnp.clip(x, eps, jnp.inf)
    y_disc = jnp.log(safe(rho_disc[mask_disc])) - jnp.log(safe(rho_u[mask_disc]))
    E_disc = E[mask_disc]

    # Add Sn anchor
    y_sn = jnp.log(safe(rho_Sn)) - jnp.log(safe(rho_u_Sn))
    E_sn = jnp.array(Sn)

    E_fit = jnp.concatenate([E_disc, E_sn[None]])
    y_fit = jnp.concatenate([y_disc, y_sn[None]])

    # Weights (heavier on Sn)
    w = jnp.ones_like(E_fit)
    w = w.at[-1].set(w_sn)

    W  = jnp.sum(w)
    Ew = jnp.sum(w * E_fit) / W
    yw = jnp.sum(w * y_fit) / W

    varE  = jnp.sum(w * (E_fit - Ew) ** 2)
    covEy = jnp.sum(w * (E_fit - Ew) * (y_fit - yw))

    alpha = covEy / jnp.clip(varE, eps, jnp.inf)
    lnA   = yw - alpha * Ew
    A     = jnp.exp(lnA)
    return A, alpha


def subdict(d, exclude):
    exclude = set(exclude)   # ensure O(1) lookups
    return {k: v for k, v in d.items() if k not in exclude}
