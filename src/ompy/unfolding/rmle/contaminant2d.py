from __future__ import annotations
from .stubs import GegMatrix, GegDMatrix, ExpectationParameter2D, Nu2D, GexMatrix
from .utils import pytree_dataclass, relaxed_one_hot, var_penalty, into_array, bounded_param
import jax
from dataclasses import dataclass
from ... import Index, Matrix
from jaxtyping import Bool, Array
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Self
import numpy as np
@pytree_dataclass
class Contaminant2D:
    Eg: Index
    Ein: Index
    G_eg: GegMatrix
    GegD: GegDMatrix
    G_ein: GexMatrix
    initial: tuple[int, int, float]  # (index, index, amplitude)
    central_bounds_eg: tuple[int, int]  # (lower, upper)
    central_bounds_ein: tuple[int, int]  # (lower, upper)
    central_mask: Bool[Array, "Ein Eg"] = None
    temperature: float = 0.01
    amplitude_mu_penalty: float = 0.0
    amplitude_eta_penalty: float = 0.0
    amplitude_nu_penalty: float = 0.0
    amplitude_mu_bounds: tuple[float, float] | None = None
    amplitude_eta_bounds: tuple[float, float] | None = None
    amplitude_nu_bounds: tuple[float, float] | None = None

    @classmethod
    def from_(cls, mat: Matrix,
            initial,
            central_bounds_eg,
            central_bounds_ein,
            G_eg,
            GegD,
            G_ein,
            **kwargs
        ) -> Self:
        # The user can specify the central bounds as indexable expressions
        Ein, Eg = mat.X_index, mat.Y_index
        i = Eg.index_expression(central_bounds_eg[0])
        j = Eg.index_expression(central_bounds_eg[1])
        central_bounds_eg = (i, j)
        i = Ein.index_expression(central_bounds_ein[0])
        j = Ein.index_expression(central_bounds_ein[1])
        central_bounds_ein = (i, j)
        central_mask = jnp.ones((len(Ein), len(Eg)), dtype=bool)
        central_mask = central_mask.at[
            central_bounds_ein[0]:central_bounds_ein[1],
            central_bounds_eg[0]:central_bounds_eg[1]
        ].set(False)

        # Same with the central value for the initial guess
        k = Eg.index_expression(initial[0])
        l = Ein.index_expression(initial[1])
        initial = (k, l, initial[2])
        # i.e. matrix[k, l] = initial[2]
        # Check the bounds
        G_eg = into_array(G_eg)
        GegD = into_array(GegD)
        G_ein = into_array(G_ein)
        return cls(Eg, Ein, G_eg, GegD, G_ein, initial, central_bounds_eg, central_bounds_ein, central_mask, **kwargs)

        
    def __post_init__(self):
        initial = self.setup_initial()
        initial = initial.at[self.central_mask].set(0.0)

        if self.amplitude_mu_penalty != 0:
            if self.amplitude_mu_bounds is None:
                raise ValueError(
                    "amplitude_mu_bounds must be set if amplitude_mu_penalty != 0.0"
                )
            mu_max = jnp.max(initial)
            if mu_max > self.amplitude_mu_bounds[1] or mu_max < self.amplitude_mu_bounds[0]:
                raise ValueError(
                    "Initial mu is outside the bounds. Please adjust the initial guess or the bounds."
                    f"mu_max: {mu_max}, bounds: {self.amplitude_mu_bounds}"
                )
            
        if self.amplitude_eta_penalty != 0:
            if self.amplitude_eta_bounds is None:
                raise ValueError(
                    "amplitude_eta_bounds must be set if amplitude_eta_penalty != 0.0"
                )
            eta_max = jnp.max(self.G_ein@initial@self.G_eg)
            if eta_max > self.amplitude_eta_bounds[1] or eta_max < self.amplitude_eta_bounds[0]:
                raise ValueError(
                    "Initial eta is outside the bounds. Please adjust the initial guess or the bounds."
                    f"eta_max: {eta_max}, bounds: {self.amplitude_eta_bounds}"
                )
        if self.amplitude_nu_penalty != 0:
            if self.amplitude_nu_bounds is None:
                raise ValueError(
                    "amplitude_nu_bounds must be set if amplitude_nu_penalty != 0.0"
                )
            nu = self.G_ein@initial@self.GegD
            nu_max = jnp.max(jnp.where(self.central_mask, 0, self.G_ein@initial@self.GegD))
            if nu_max > self.amplitude_nu_bounds[1] or nu_max < self.amplitude_nu_bounds[0]:
                raise ValueError(
                    "Initial nu is outside the bounds. Please adjust the initial guess or the bounds."
                    f"nu_max: {nu_max}, bounds: {self.amplitude_nu_bounds}")

    def loss(self, mu: ExpectationParameter2D) -> tuple[Nu2D, float]:
        # Enforce the central bounds
        mu = mu.at[self.central_mask].set(0.0)
        # The one-hot removes the amplitude, so we must reapply it
        amplitude_mu = jnp.max(mu)
        # One-hot encoding ensures a single non-zero element
        mu = amplitude_mu * relaxed_one_hot(mu, temperature=self.temperature)
        # I think we can optimize this a lot by exploting the fact
        # that just a single element of mu is non-zero. 
        # then A@mu@B = v*outer(A[:, i], B[:, j]) with v=amplitude
        Gexmu = self.G_ein@mu
        nu = Gexmu @ self.GegD

        if self.amplitude_mu_penalty != 0:
            mu_cost = self.amplitude_mu_penalty * var_penalty(
                amplitude_mu, self.amplitude_mu_bounds[0], self.amplitude_mu_bounds[1]
            )
        else:
            mu_cost = 0.0

        if self.amplitude_eta_penalty != 0:
            eta = Gexmu@self.Geg
            amplitude_eta = jnp.max(eta)
            eta_cost = self.amplitude_eta_penalty * var_penalty(
                amplitude_eta, self.amplitude_eta_bounds[0], self.amplitude_eta_bounds[1]
            )
        else:
            eta_cost = 0.0

        if self.amplitude_nu_penalty != 0:
            amplitude_nu = jnp.max(jnp.where(self.central_mask, 0, nu))
            nu_cost = self.amplitude_nu_penalty * var_penalty(
                amplitude_nu, self.amplitude_nu_bounds[0], self.amplitude_nu_bounds[1]
            )
        else:
            nu_cost = 0.0

        return nu, mu_cost + eta_cost + nu_cost

        
    def setup_initial(self, initial: jnp.ndarray | None = None) -> jnp.ndarray:
        if initial is None:
            mu = jnp.zeros(self.shape)
        else:
            if initial.shape != self.shape:
                raise ValueError(f"Initial must be of shape {self.shape}, got {initial.shape}")
            mu = jnp.zeros_like(initial)

        mu = mu.at[self.initial[1], self.initial[0]].set(self.initial[2])
        return mu

    def plot(self, ax=None, space='eta', **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        mu = self.setup_initial()
        match space:
            case 'mu':
                mat = Matrix(values=mu, Ein=self.Ein, Eg=self.Eg)
            case 'eta':
                mat = Matrix(values=self.G_ein@mu@self.G_eg, Ein=self.Ein, Eg=self.Eg)
            case 'nu':
                mat = Matrix(values=self.G_ein@mu@self.GegD, Ein=self.Ein, Eg=self.Eg)
            case _:
                raise ValueError(f"Invalid space: {space}. Must be one of 'mu', 'eta', 'nu'.")
        mask = mat.clone(values=np.asarray(self.central_mask).astype(float))
        mat.plot(ax=ax, **kwargs)
        mask.plot(ax=ax, alpha=0.3, cmap='gray', add_cbar=False)
        eg_min = self.Eg[self.central_bounds_eg[0]]
        eg_max = self.Eg[self.central_bounds_eg[1]]
        ein_min = self.Ein[self.central_bounds_ein[0]]
        ein_max = self.Ein[self.central_bounds_ein[1]]
        ax.axvline(eg_min, color='red')
        ax.axvline(eg_max, color='red')
        ax.axhline(ein_min, color='red')
        ax.axhline(ein_max, color='red')
        return ax
        
    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.Ein), len(self.Eg))

    
    def _repr_html_(self):
        """Generates an HTML representation of the object for Jupyter notebooks."""
        table_rows = []

        # Define the attributes to be displayed
        i, j = self.initial[:2]
        eg, ex = self.Eg[i], self.Ein[j]
        initial = (*self.initial, f"{eg:.1f}", f"{ex:.1f}")
        attributes = [
            ("Eg (Index)", repr(self.Eg)),
            ("Ein (Index)", repr(self.Ein)),
            ("Initial (index, index, amplitude)", initial),
            ("Central Bounds Eg (lower, upper)", self.central_bounds_eg),
            ("Central Bounds Ein (lower, upper)", self.central_bounds_ein),
            ("Temperature", self.temperature),
            ("Amplitude μ Penalty", self.amplitude_mu_penalty),
            ("Amplitude η Penalty", self.amplitude_eta_penalty),
            ("Amplitude ν Penalty", self.amplitude_nu_penalty),
            ("Amplitude μ Bounds", self.amplitude_mu_bounds),
            ("Amplitude η Bounds", self.amplitude_eta_bounds),
            ("Amplitude ν Bounds", self.amplitude_nu_bounds),
            ("Shape", self.shape),
        ]

        for key, value in attributes:
            value_str = str(value)
            table_rows.append(
                f"<tr><th style='text-align:left; padding:5px;'>{key}</th><td style='padding:5px;'>{value_str}</td></tr>"
            )

        return f"""
        <table border="1" cellpadding="4" cellspacing="0" style="border-collapse: collapse; border: 1px solid black;">
            <thead style="background-color: #f2f2f2;">
                <tr>
                    <th style="text-align:left; padding:5px;">Attribute</th>
                    <th style="text-align:left; padding:5px;">Value</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        """

        
@pytree_dataclass
class SimpleContaminant2D:
    Eg: Index
    Ein: Index
    initial: tuple[float, float, float]  # (index, index, amplitude)
    central_bounds_eg: tuple[float, float]  # (lower, upper)
    central_bounds_ein: tuple[float, float]  # (lower, upper)
    mesh: jnp.ndarray
    temperature: float = 0.01
    amplitude_mu_penalty: float = 0.0
    amplitude_eta_penalty: float = 0.0
    amplitude_nu_penalty: float = 0.0
    amplitude_mu_bounds: tuple[float, float] | None = None
    amplitude_eta_bounds: tuple[float, float] | None = None
    amplitude_nu_bounds: tuple[float, float] | None = None
    sigma_eg: float = 5.0
    sigma_ein: float = 5.0

    def __post_init__(self):
        if self.amplitude_mu_penalty != 0:
            if self.amplitude_mu_bounds is None:
                raise ValueError(
                    "amplitude_mu_bounds must be set if amplitude_mu_penalty != 0.0"
                )


    @classmethod
    def from_(
        cls,
        matrix: Matrix,
        central_bounds_eg: tuple[int | float, int | float],
        central_bounds_ein: tuple[int | float, int | float],
        initial: tuple[int | float, int | float, float],
        **kwargs
    ) -> Self:
        Ein, Eg = matrix.X_index, matrix.Y_index
        i = Eg.index_expression(central_bounds_eg[0])
        j = Eg.index_expression(central_bounds_eg[1])
        central_bounds_eg = (Eg[i], Eg[j])
        i = Ein.index_expression(central_bounds_ein[0])
        j = Ein.index_expression(central_bounds_ein[1])
        central_bounds_ein = (Ein[i], Ein[j])
        i = Eg.index_expression(initial[0])
        j = Ein.index_expression(initial[1])
        initial = (Eg[i], Ein[j], initial[2])
        mesh = jnp.meshgrid(Eg.bins, Ein.bins)
        return cls(Eg, Ein, initial, central_bounds_eg, central_bounds_ein, mesh=mesh, **kwargs)

    def loss(self, params: tuple[float, float, float]):
        mu_x, mu_y, A = self.transform_out(params)
        # Soft indexing to make the gradient smooth
        mu = soft_peak(mu_x, mu_y, A, self.mesh[0], self.mesh[1], self.sigma_eg, self.sigma_ein)

        #if self.amplitude_mu_penalty != 0:
        #    mu_cost = self.amplitude_mu_penalty * var_penalty(
        #        A, self.amplitude_mu_bounds[0], self.amplitude_mu_bounds[1]
        #    )
        #else:
        #    mu_cost = 0.0

        return mu, 0.0

        
    def setup_initial(self) -> tuple[float, float, float]:
        #mu_x = bounded_param_invert(self.initial[0], self.central_bounds_eg[0], self.central_bounds_eg[1])
        #mu_y = bounded_param_invert(self.initial[1], self.central_bounds_ein[0], self.central_bounds_ein[1])
        mu_x, mu_y = 0.0, 0.0
        A = 0.0
        return (mu_x, mu_y, A)
        

    def shape(self) -> tuple[int, int]:
        return (len(self.Ein), len(self.Eg))

    def transform_out(self, params: tuple[float, float, float]) -> tuple[float, float, float]:
        mu_x, mu_y, A = params
        mu_x = bounded_param(mu_x, self.central_bounds_eg[0], self.central_bounds_eg[1])
        mu_y = bounded_param(mu_y, self.central_bounds_ein[0], self.central_bounds_ein[1])
        A = bounded_param(A, self.amplitude_mu_bounds[0], self.amplitude_mu_bounds[1])
        return (mu_x, mu_y, A)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        mu_x, mu_y, A = self.setup_initial()
        mu_x, mu_y, A = self.transform_out((mu_x, mu_y, A))
        mu = soft_peak(mu_x, mu_y, A, self.mesh[0], self.mesh[1], self.sigma_eg, self.sigma_ein)
        mat = Matrix(values=mu, Ein=self.Ein, Eg=self.Eg)
        mat.plot(ax=ax, **kwargs)

        ax.axvline(self.central_bounds_eg[0]  , color='red')
        ax.axvline(self.central_bounds_eg[1], color='red')
        ax.axhline(self.central_bounds_ein[0], color='red')
        ax.axhline(self.central_bounds_ein[1], color='red')
        return ax
        

def soft_peak(x, y, amplitude, grid_x, grid_y, sigma_eg, sigma_ein):
    dist_eg = (grid_x - x)**2 / (2 * sigma_eg**2)
    dist_ein = (grid_y - y)**2 / (2 * sigma_ein**2)
    dist = jnp.exp(-dist_eg - dist_ein)
    dist /= jnp.max(dist)
    return amplitude * dist



def bounded_param_invert(x, lower, upper):
    # Solve for the input that would produce x in bounded_param
    # x = lower + (upper - lower) * sigmoid(y)
    # (x - lower)/(upper - lower) = sigmoid(y) 
    # logit((x - lower)/(upper - lower)) = y
    return jax.scipy.special.logit((x - lower)/(upper - lower))