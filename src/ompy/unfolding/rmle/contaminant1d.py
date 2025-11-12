from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Self

from .tau import from_tau, to_tau
from .utils import pytree_dataclass, bounded_param

from .stubs import  ExpectationParameter1D, Xi1D, Nu1D, GegMatrix, GegDMatrix, ContaminantLossFn1D
from ... import Index, Vector


@pytree_dataclass
class Contaminant1D:
    """A helper class to specify contaminants
    TODO:
    - Allow the user to specify D_xi, G_egD_xi, G_exD_xi
    """

    E: Index
    #initial: tuple[float, float]  # (mu_x, amplitude)
    central_bounds: tuple[float, float]  # (lower, upper)
    amplitude_mu_bounds: tuple[float, float]
    sigma: float = 5.0  # Sigma of the soft peak, not the true resolution
    amplitude_mu_penalty: float = 0.0
    amplitude_eta_penalty: float = 0.0
    amplitude_nu_penalty: float = 0.0
    amplitude_eta_bounds: tuple[float, float] | None = None
    amplitude_nu_bounds: tuple[float, float] | None = None

    @classmethod
    def from_(cls, E: Index, central_bounds, **kwargs) -> Self:
        # The user can specify the central bounds as indexable expressions
        i = E.index_expression(central_bounds[0])
        j = E.index_expression(central_bounds[1])
        central_bounds = (E[i], E[j])
        E = jnp.asarray(E.bins)
        return cls(E, central_bounds, **kwargs)

    def __post_init__(self):
        # Check the bounds
        if self.amplitude_mu_penalty != 0 and self.amplitude_mu_bounds is None:
            raise ValueError(
                "amplitude_mu_bounds must be set if amplitude_mu_penalty != 0.0"
            )
        if self.amplitude_eta_penalty != 0 and self.amplitude_eta_bounds is None:
            raise ValueError(
                "amplitude_eta_bounds must be set if amplitude_eta_penalty != 0.0"
            )
        if self.amplitude_nu_penalty != 0 and self.amplitude_nu_bounds is None:
            raise ValueError(
                "amplitude_nu_bounds must be set if amplitude_nu_penalty != 0.0"
            )
        if self.amplitude_mu_bounds is not None:
            if self.amplitude_mu_bounds[0] > self.amplitude_mu_bounds[1]:
                raise ValueError(
                    "amplitude_mu_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of mu"
                )
        if self.amplitude_eta_bounds is not None:
            if self.amplitude_eta_bounds[0] > self.amplitude_eta_bounds[1]:
                raise ValueError(
                    "amplitude_eta_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of eta"
                )
        if self.amplitude_nu_bounds is not None:
            if self.amplitude_nu_bounds[0] > self.amplitude_nu_bounds[1]:
                raise ValueError(
                    "amplitude_nu_bounds must be a tuple of two numbers, the lower and upper bounds of the amplitude of nu"
                )
    
    def loss(self, mu: ExpectationParameter1D) -> tuple[Nu1D, float]:
        mu_x, A = self.transform_out(mu)
        mu = soft_peak(self.E, mu_x, A, self.sigma)
        return mu, 0.0

    def setup_initial(self) -> tuple[float, float]:
        mu_x, A = 0.0, 0.0
        return (mu_x, A)

    def transform_out(self, params: tuple[float, float]) -> tuple[float, float]:
        mu_x, A = params
        mu_x = bounded_param(mu_x, self.central_bounds[0], self.central_bounds[1])
        A = bounded_param(A, self.amplitude_mu_bounds[0], self.amplitude_mu_bounds[1])
        return (mu_x, A)

    def into_vector[T: Vector](self, vector: T | None = None,
                               params: tuple[float, float] | None = None) -> T:
        if params is None:
            mu_x, A = self.transform_out(self.setup_initial())
        else:
            mu_x, A = self.transform_out(params)

        if vector is not None:
            x = vector.X
            y = soft_peak(x, mu_x, A, self.sigma)
            return vector.clone(values=y, name='Contaminant')
        else:
            x = self.E
            y = soft_peak(x, mu_x, A, self.sigma)
            return Vector(values=y, E=np.asarray(x), name='Contaminant')
            
    def __len__(self):
        return len(self.E)

    def _repr_html_(self):
        """Generates an HTML representation of the object for Jupyter notebooks."""
        table_rows = []

        # Define the attributes to be displayed
        attributes = [
            ("E (Index)", repr(self.E)),
            ("Central Bounds (lower, upper)", self.central_bounds),
            ("Amplitude μ Penalty", self.amplitude_mu_penalty),
            ("Amplitude η Penalty", self.amplitude_eta_penalty),
            ("Amplitude ν Penalty", self.amplitude_nu_penalty),
            ("Amplitude μ Bounds", self.amplitude_mu_bounds),
            ("Amplitude η Bounds", self.amplitude_eta_bounds),
            ("Amplitude ν Bounds", self.amplitude_nu_bounds),
            ("Total Length", len(self)),
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

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        initial = self.transform_out(self.setup_initial())
        y = soft_peak(self.E, initial[0], initial[1], self.sigma)
        ax.plot(self.E, y, label="Initial in $\\mu$ space")
        return ax


def soft_peak(x, mu_x, A, sigma):
    dist = jnp.exp(-(x - mu_x)**2 / (2 * sigma**2))
    # Might sometimes want max instead of sum
    dist /= jnp.sum(dist)
    return A * dist
    

def var_penalty(param: float, lower: float, upper: float) -> float:
    """Calculate a quadratic penalty for values outside a specified interval.

    This function computes a quadratic penalty that grows as the parameter moves
    outside the specified bounds. Inside the bounds, the penalty is zero.

    Args:
        param: The parameter value to check
        lower: Lower bound of the allowed interval
        upper: Upper bound of the allowed interval

    Returns:
        float: The total penalty, which is the sum of penalties for violating
              the lower and upper bounds. Returns 0.0 if param is within bounds.
    """
    # Quadratic penalty outside the [lower, upper] interval.
    lower_penalty = jnp.where(param < lower, (param - lower) ** 2, 0.0)
    upper_penalty = jnp.where(param > upper, (param - upper) ** 2, 0.0)
    return lower_penalty + upper_penalty

def relaxed_one_hot(logits, temperature=0.01):
    """Compute a continuous relaxation of a one-hot vector using softmax.

    This function takes logits and returns a "soft" one-hot vector by applying
    the softmax function with a temperature parameter. As temperature approaches 0,
    the output approaches a discrete one-hot vector.

    Args:
        logits: Input logits tensor to be converted to probabilities
        temperature: Temperature parameter controlling the sharpness of the distribution.
                    Lower values make the output more discrete. Default is 0.01.

    Returns:
        A tensor of the same shape as logits containing probabilities that sum to 1.
    """
    return jax.nn.softmax(logits / temperature)

    

@pytree_dataclass
class ContaminantModel1D:
    # TODO: Elide common foldings
    loss: ContaminantLossFn1D
    G_eg: GegMatrix | None = None
    GegD: GegDMatrix | None = None

    @jax.jit
    def cost(self, tau: ExpectationParameter1D) -> tuple[Nu1D, float]:
        mu = from_tau(tau)
        return self.loss(mu, self.G_eg, self.GegD)
    
    def set_matrices(self, G_eg: GegMatrix, GegD: GegDMatrix) -> Self:
        if self.G_eg is None:
            self.G_eg = G_eg
        if self.GegD is None:
            self.GegD = GegD
        return self
