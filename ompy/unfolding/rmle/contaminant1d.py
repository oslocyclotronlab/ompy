
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .tau import from_tau, to_tau

from ... import Index

@dataclass(kw_only=True)
class Contaminant1D:
    """A helper class to specify contaminants
    TODO:
    - Allow the user to specify D_xi, G_egD_xi, G_exD_xi
    """

    E: Index
    initial: tuple[int, float]  # (index, amplitude)
    central_bounds: tuple[int, int]  # (lower, upper)
    temperature: float = 0.01
    amplitude_mu_penalty: float = 0.0
    amplitude_eta_penalty: float = 0.0
    amplitude_nu_penalty: float = 0.0
    amplitude_mu_bounds: tuple[float, float] | None = None
    amplitude_eta_bounds: tuple[float, float] | None = None
    amplitude_nu_bounds: tuple[float, float] | None = None

    def __post_init__(self):
        # The user can specify the central bounds as indexable expressions
        i = self.E.index_expression(self.central_bounds[0])
        j = self.E.index_expression(self.central_bounds[1])
        self.central_bounds = (i, j)
        # Same with the central value for the initial guess
        k = self.E.index_expression(self.initial[0])
        self.initial = (k, self.initial[1])
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

    def closure(self) -> Callable[[jnp.ndarray], tuple[jnp.ndarray, float]]:
        lower, upper = self.central_bounds
        T = self.temperature
        amplitude_mu_penalty = self.amplitude_mu_penalty
        amplitude_eta_penalty = self.amplitude_eta_penalty
        amplitude_nu_penalty = self.amplitude_nu_penalty

        # Validate and extract bounds for each amplitude penalty
        def get_bounds(penalty, bounds, name):
            if penalty != 0.0 and bounds is None:
                raise ValueError(f"{name}_bounds must be set if {name}_penalty != 0.0")
            if bounds is not None:
                return bounds
            # Default bounds when not used, for JAX tracer
            return (-1.0, -1.0)

        lower_mu, upper_mu = get_bounds(
            amplitude_mu_penalty, self.amplitude_mu_bounds, "amplitude_mu"
        )

        lower_eta, upper_eta = get_bounds(
            amplitude_eta_penalty, self.amplitude_eta_bounds, "amplitude_eta"
        )

        lower_nu, upper_nu = get_bounds(
            amplitude_nu_penalty, self.amplitude_nu_bounds, "amplitude_nu"
        )

        @jax.jit
        def func(
            mu: jnp.ndarray,  # mu of xi, not of the data
            G_eg: jnp.ndarray,
            G_egD: jnp.ndarray,
        ) -> tuple[jnp.ndarray, float]:
            # Enforce the central bounds
            mu = mu.at[:lower].set(0.0)
            mu = mu.at[upper:].set(0.0)
            # The one-hot removes the amplitude, so we must reapply it
            amplitude_mu = jnp.max(mu)
            # One-hot encoding ensures a single non-zero element
            mu = amplitude_mu * relaxed_one_hot(mu, temperature=T)

            def identity_penalty(_):
                return 0.0

            def mu_penalty(_):
                return amplitude_mu_penalty * var_penalty(
                    amplitude_mu, lower_mu, upper_mu
                )

            def eta_penalty(_):
                eta = mu @ G_eg
                amplitude_eta = jnp.max(eta)
                return amplitude_eta_penalty * var_penalty(
                    amplitude_eta, lower_eta, upper_eta
                )

            def nu_penalty(_):
                nu = mu @ G_egD
                # Here we only care about the amplitude within FE,
                # which is equivalent to being within the bounds
                amplitude_nu = jnp.max(nu[lower:upper])
                return amplitude_nu_penalty * var_penalty(
                    amplitude_nu, lower_nu, upper_nu
                )

            mu_cost = jax.lax.cond(
                amplitude_mu_penalty == 0.0, identity_penalty, mu_penalty, None
            )

            eta_cost = jax.lax.cond(
                amplitude_eta_penalty == 0.0, identity_penalty, eta_penalty, None
            )

            nu_cost = jax.lax.cond(
                amplitude_nu_penalty == 0.0, identity_penalty, nu_penalty, None
            )

            return mu, mu_cost + eta_cost + nu_cost

        return func

    def setup_initial(self, initial: jnp.ndarray | None = None) -> jnp.ndarray:
        if initial is None:
            mu = jnp.zeros(self.E.bins.shape)
        else:
            if len(initial) != len(self.E):
                raise ValueError(
                    f"Initial must be of length of data, got {len(initial)}"
                )
            mu = jnp.zeros_like(initial)
        # We map the amplitude to tau space since it will be inverted in the loop
        mu = mu.at[self.initial[0]].set(to_tau(self.initial[1]))
        return mu

    def __len__(self):
        return len(self.E)

    def _repr_html_(self):
        """Generates an HTML representation of the object for Jupyter notebooks."""
        table_rows = []

        # Define the attributes to be displayed
        attributes = [
            ("E (Index)", repr(self.E)),
            ("Initial (index, amplitude)", self.initial),
            ("Central Bounds (lower, upper)", self.central_bounds),
            ("Temperature", self.temperature),
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
        initial = from_tau(self.setup_initial())
        ax.plot(initial, label="Initial in $\\mu$ space")
        return ax


def setup_contaminants(
    contaminants: list[Contaminant1D], initial: jnp.ndarray, mask: jnp.ndarray
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Set up initial values and masks for contaminants.

    Take a list of contaminants and concatenates their initial values
    and masks into arrays suitable for optimization. For each contaminant, it:
    1. Gets the initial values using the contaminant's setup_initial method
    2. Creates a zero mask for the contaminant parameters
    3. Concatenates these with the existing arrays

    Args:
        contaminants: List of Contaminant1D objects to set up
        initial: Initial parameter values for the main spectrum
        mask: Mask array for the main spectrum parameters

    Returns:
        tuple containing:
            - Combined array of initial values for main spectrum and contaminants
            - Combined array of masks for main spectrum and contaminants
    """
    x = initial
    mask = mask
    for contaminant in contaminants:
        xi_initial = contaminant.setup_initial(initial)
        x = jnp.concatenate([x, xi_initial])
        xi_mask = jnp.zeros_like(mask)
        mask = jnp.concatenate([mask, xi_mask])
    return x, mask

def bound_variable(u, a, b):
    """Map an unbounded variable to a bounded interval using sigmoid.

    This function maps a variable u from (-∞, ∞) to the interval [a, b] using
    the sigmoid function. This is useful for constrained optimization where we
    want to optimize an unconstrained variable while ensuring the result lies
    within specified bounds.

    Args:
        u: Input variable to be bounded (can be any real number)
        a: Lower bound of the target interval
        b: Upper bound of the target interval (must be > a)

    Returns:
        The input mapped to the interval [a, b]. As u approaches -∞, the output
        approaches a. As u approaches ∞, the output approaches b.
    """
    return a + (b - a) * jax.nn.sigmoid(u)

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