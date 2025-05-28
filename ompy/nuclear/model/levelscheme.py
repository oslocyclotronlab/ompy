import numpy as np
import xarray as xr
from typing import Literal
from numba import njit
from dataclasses import dataclass
from pathlib import Path

from .stubs import Levels, Population, DiscreteLevels

"""
TODO:
    - AC dislikes using TALYS population as a source for the constructed level scheme.
      Instead, use experimental single particle spectra with a model for spin
      distribution. An alternative is just a flat population. Choice of spin
      distribution is heavily experiment dependent.
      
    - Do I take a population and discretize it by the level scheme, or do I take the level scheme and
      populate it?
      I want a population distribution to sample from.
"""


@dataclass
class PopulationDistribution:
    ex: np.ndarray
    J: np.ndarray
    pi: np.ndarray
    population: np.ndarray

    def sample(self, *args, **kwargs):
        pass

    def save(self, path: Path):
        #xarray netcdf
        path = Path(path)
        raise NotImplementedError("Saving of population distributions is not implemented yet")

    @staticmethod
    def from_path(self, path: Path):
        raise NotImplementedError("Loading of population distributions is not implemented yet")


@dataclass
class LevelScheme:
    discrete: DiscreteLevels
    constructed: Levels


def populate_flat(levels: LevelScheme, *, J_dist, pi_dist,
                  ex_min: float | None = None, ex_max: float | None = None) -> Population:
    """ Populate a constructed level scheme with a flat energy distribution

    Parameters
    ----------
    levels : Levels
        The constructed level scheme
    J_dist : SamplingFunction
        The distribution of spins
    pi_dist : SamplingFunction
        The distribution of parities
    ex_min : float | None, optional
        The minimum excitation energy, by default None uses the minimum in the levels
    ex_max : float | None, optional
        The maximum excitation energy, by default None uses the maximum in the levels

    Returns
    -------
    PopulationDistribution
        The population distribution
    """
    pass


def populate_from_dists(levels: Levels, ex_dist, J_dist, pi_dist) -> Population:
    pass


def discretize_population(population: Population, levels: LevelScheme) -> PopulationDistribution:
    pass

def sample_population_as_level(population: Population,
                               discrete: DiscreteLevels,
                               constructed: Levels,
                               e_crit: float,
                               eps: float = 1e-3,
                               equiparity: bool = True) -> tuple[float, float, Literal[-1, 1]]:
    """
    Selects a random level from the population (weighted by the population) and
    returns the closest matching level in either the discrete level scheme or the constructed level scheme,
    depending on whether the energy is above or below the critical energy.

    The spins of the constructed level scheme and discrete level scheme must at least contain the
    spins of the population.

    TODO: Artificial distinction between discrete and constructed levels. Should be abstracted.
          Should be optimized for jax as this is in the hot loop.
          Normalize TALYS population
          This can be split into two functions: 1) Sample from the population 2) Map it to a level
          As long as the population is not updated, the levels can be drawn by the cpu outside the loop,
          or in batches.
          There is perhaps an even better approach. What we want is to discretize the TALYS population,
          then sample for each event. The sampling is fast, the discretization is slow and can be done
          once. I am confused. Why does the population need to be discretized for the constructed
          scheme? Is talys coarser or finer in binning?

    Parameters
    ----------
    population : Population
        The population to sample from
    discrete : DiscreteLevels
        The discrete level scheme
    constructed : Levels
        The constructed level scheme
    e_crit : float
        The critical energy, below which the discrete level scheme is used
    eps : float
        The maximum energy difference to consider a discrete level a match
    equiparity : bool
        Whether to sample the parity from the population

    Returns
    -------
    tuple[float, float, Literal[-1, 1]]
        The excitation energy, spin, and parity of the drawn level
    """
    J_diff = set(population.J.values) - set(constructed.J.values)
    if J_diff:
        raise ValueError("The constructed level scheme must have at least the same spins as the population.\n"
                         f"Constructed spins: {constructed.J}.\nPopulation spins: {population.J}\n"
                         f"Missing spins: {J_diff}")

    if discrete.Ex.values.max() < e_crit:
        raise ValueError("The discrete level scheme must have levels above the critical energy. "
                         f"Highest level {discrete.Ex.max()} < {e_crit} e_crit.")

    i_ex, i_spin, i_pi = sample_3d_histogram(population)

    Ex = population.Ex.values
    ex = Ex[i_ex].item()
    dex = binwidth_at(i_ex, Ex)
    elow, ehigh = ex - dex / 2, ex + dex / 2
    # RAINIER uses elow < e_crit, but that seems wrong to me
    if ehigh < e_crit:
        # Discrete
        matching_level = closest_index(discrete.Ex, ex)
        level_ex = discrete.Ex[matching_level]
        if abs(level_ex - ex) > eps:
            raise ValueError(f"Could not find a matching level for {ex} MeV "
                             f"closer than {eps} MeV. Closest was level {matching_level} at {level_ex} MeV."
                             " You should ensure `e_crit` is properly selected.")
        level_j = discrete.J[matching_level]
        level_pi = discrete.pi[matching_level]

    else:
        # constructed
        level_ex = np.random.uniform(elow, ehigh)
        # The populated level in the constructed scheme must have the same
        # spin and parity as the populated level in the population
        #level_pi = np.random.choice(('-', '+')) if equiparity else population.pi.values[i_pi]
        level_pi_i = np.random.choice((0, 1))
        level_j = population.J.values[i_spin]

        # Do a search for matching level
        # v--- This takes unnecessary long time. About 50% of the time is spent here
        #subset = constructed.sel(J=level_j, pi=level_pi)
        # Assuming population J = constructed J and J = [0, 1, ...]
        subset = constructed.values[:, i_spin, level_pi_i]
        # Find the closest index that has non-zero population
        # RAINIER uses a random walk, but it is faster to do a "clever" search.
        level_index = find_index(constructed.Ex.values, subset, level_ex)
        level_ex = constructed.Ex.values[level_index].item()
        level_pi = -1 if level_pi_i == 0 else 1 #-1 if level_pi == '-' else 1

    return level_ex, int(level_j), level_pi


def sample_3d_histogram(hist: xr.DataArray) -> tuple[int, int, int]:
    """ Sample a 3d histogram

    Parameters
    ----------
    hist : xr.DataArray
        The 3D histogram to sample from. Must be normalized

    Returns
    -------
    tuple[int, int, int]
        The indices of the sampled bin
    """
    # Convert the DataArray to a numpy array and flatten it
    probabilities = hist.values.flatten()

    # Normalize the probabilities to ensure they sum to 1 (if not already)
    probabilities /= probabilities.sum()

    # Use numpy.random.choice to draw a sample from the flattened array
    flat_index = np.random.choice(a=len(probabilities), size=1, p=probabilities)

    # Convert the flat index back to 3D indices
    x, y, z = np.unravel_index(flat_index, shape=hist.shape)
    return x, y, z


def binwidth_at(i, X):
    l = len(X)
    if i == 0:
        return X[1] - X[0]
    elif i == l - 1:
        return X[-1] - X[-2]
    else:
        return X[i] - X[i-1]


def closest_index(arr: np.ndarray, value: float) -> int:
    idx = np.searchsorted(arr, value)

    # Determine the closest value
    if idx == 0:
        return idx
    elif idx == len(arr):
        return idx
    else:
        # Find the closest of the two surrounding values
        before = arr[idx - 1]
        after = arr[idx]
        return idx-1 if abs(value - before) <= abs(value - after) else idx


#@njit
def find_index(Ex, population, ex):
    # Find the index where 'ex' would be inserted to maintain order
    idx = np.searchsorted(Ex, ex)

    # Since 'searchsorted' will give us the index where 'ex' should be inserted
    # to maintain order, 'idx' could be equal to len(Ex); we check for this case
    if idx == len(Ex):
        idx -= 1  # Use the last index if 'ex' is larger than any value in 'Ex'
    elif idx > 0 and (ex - Ex[idx - 1]) < (Ex[idx] - ex):
        # If the previous index is closer in value, use that one
        idx -= 1

    # Now we check if the population at this index is not zero
    if population[idx] != 0:
        return idx  # Found the index with the closest 'Ex' and non-zero population

    # If the population at the closest index is zero, we need to look for the nearest non-zero population
    # We check the indices before and after the found index
    lower_indices = np.arange(idx - 1, -1, -1)  # Indices lower than 'idx'
    upper_indices = np.arange(idx + 1, len(Ex), 1)  # Indices higher than 'idx'

    for lower, upper in zip(lower_indices, upper_indices):
        # Check lower index if it's valid and has a non-zero population
        if lower >= 0 and population[lower] != 0:
            return lower
        # Check upper index if it's valid and has a non-zero population
        if upper < len(Ex) and population[upper] != 0:
            return upper

    # If no non-zero population is found, return None or raise an error
    return None