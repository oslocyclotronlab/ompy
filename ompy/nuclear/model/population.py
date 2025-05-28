from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from .levels import LevelScheme, DiscreteLevels, ContinuousLevels
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path


@dataclass
class Population:
    discrete: pd.DataFrame  # Population of discrete levels
    continuous: xr.DataArray  # Population of continuous levels
    e_crit: float  # Critical energy separating discrete and continuous regions

    @classmethod
    def from_level_scheme(cls, level_scheme: LevelScheme, 
                          population_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Population:
        """
        Factory method to create a LevelSchemePopulation from a LevelScheme and a population function.

        Args:
            level_scheme (LevelScheme): The level scheme to populate
            population_function (Callable): Function to generate population from Ex, J, and pi

        Returns:
            LevelSchemePopulation: Populated level scheme
        """
        # Populate discrete levels
        discrete_pop = cls._populate_discrete(level_scheme.discrete, population_function)

        # Populate continuous levels
        continuous_pop = cls._populate_continuous(level_scheme.continuous, population_function)

        return cls(discrete=discrete_pop, continuous=continuous_pop, e_crit=level_scheme.e_crit)

    @staticmethod
    def _populate_discrete(discrete_levels: DiscreteLevels, 
                           population_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> pd.DataFrame:
        """Populate discrete levels"""
        population = population_function(discrete_levels.Ex, discrete_levels.J, discrete_levels.pi)
        return pd.DataFrame({
            'Ex': discrete_levels.Ex,
            'J': discrete_levels.J,
            'pi': discrete_levels.pi,
            'population': population
        })

    @staticmethod
    def _populate_continuous(continuous_levels: ContinuousLevels, 
                             population_function: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> xr.DataArray:
        """Populate continuous levels"""
        pi = np.array([0, 1])  # Assuming 0 for negative parity, 1 for positive
        Ex, J, pi = np.meshgrid(continuous_levels.Ex, continuous_levels.J, pi, indexing='ij')
        Ex_flat, J_flat, pi_flat = Ex.flatten(), J.flatten(), pi.flatten()

        population = population_function(Ex_flat, J_flat, pi_flat).reshape(Ex.shape)
        
        return xr.DataArray(
            population,
            coords={'Ex': continuous_levels.Ex, 'J': continuous_levels.J, 'pi': ['-', '+']},
            dims=['Ex', 'J', 'pi']
        )

    @classmethod
    def from_normal_distribution(cls, level_scheme: LevelScheme, 
                                 mean_ex: float, std_ex: float, 
                                 mean_j: float, std_j: float) -> Population:
        """
        Factory method to create a LevelSchemePopulation using a normal distribution.

        Args:
            level_scheme (LevelScheme): The level scheme to populate
            mean_ex (float): Mean of the normal distribution for excitation energy
            std_ex (float): Standard deviation of the normal distribution for excitation energy
            mean_j (float): Mean of the normal distribution for angular momentum
            std_j (float): Standard deviation of the normal distribution for angular momentum

        Returns:
            LevelSchemePopulation: Populated level scheme
        """
        def normal_population(Ex, J, pi):
            ex_prob = norm.pdf(Ex, loc=mean_ex, scale=std_ex)
            j_prob = norm.pdf(J, loc=mean_j, scale=std_j)
            # Assuming equal probability for both parities
            return ex_prob * j_prob * 0.5

        return cls.from_level_scheme(level_scheme, normal_population)

    def integrate_to(self, total: int) -> Population:
        """
        Integrate the population to a total population of `total`.

        Args:
            total (int): The desired total population.

        Returns:
            Population: A new Population instance with integrated population.
        """
        total = int(total)
        # Calculate the current total population
        discrete_total = self.discrete['population'].sum()
        continuous_total = self.continuous.sum().item()
        current_total = discrete_total + continuous_total

        # Calculate the scaling factor
        scale_factor = total / current_total

        # Scale discrete levels
        new_discrete = self.discrete.copy()
        new_discrete['population'] = (new_discrete['population'] * scale_factor).round().astype(int)

        # Scale continuous levels
        new_continuous = (self.continuous* scale_factor).round().astype(int)

        # Adjust for rounding errors
        total_after_scaling = new_discrete['population'].sum() + new_continuous.sum().item()
        difference = total - total_after_scaling

        if difference != 0:
            # Add or subtract the difference from the highest populated level
            if self.discrete['population'].max() > self.continuous.max().item():
                idx = new_discrete['population'].idxmax()
                new_discrete.loc[idx, 'population'] += difference
            else:
                max_indices = np.unravel_index(new_continuous.argmax(), new_continuous.shape)
                new_continuous[max_indices] += difference

        return Population(new_discrete, new_continuous, self.e_crit)

    def plot(self,
                    cmap: str = 'viridis',
                    norm: colors.Normalize | None = None,
                    cbar_kwargs: dict | None = None) -> plt.Figure:
        """
        Plot the discrete and continuous populations combined.

        Ex is along the y-axis.
        Spin +- along the x-axis, runs from [Jmax-, (Jmax-1)-, ..., Jmin-, Jmin+, (Jmin+1)+, ..., Jmax+].

        Args:
            cmap (str, optional): Colormap to use for the plot. Default is 'viridis'.
            norm (colors.Normalize, optional): Normalization for the colormap. If None, LogNorm is used when applicable.
            cbar_kwargs (dict, optional): Keyword arguments for the colorbar.

        Returns:
            plt.Figure: The matplotlib figure containing the plot.
        """

        # Create figure and gridspec to accommodate the histogram on the side
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, width_ratios=[8, 1], wspace=0.05)
        ax_main = fig.add_subplot(gs[0, 0])
        ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_main)
        
        # Prepare colormap and normalization
        if norm is None:
            # Combine discrete and continuous populations to determine vmin and vmax
            pop_discrete = self.discrete['population'].values
            pop_continuous = self.continuous.values.flatten()
            combined_population = np.concatenate([pop_discrete, pop_continuous])
            # Remove zeros to avoid log(0) issues
            combined_population_nonzero = combined_population[combined_population > 0]
            if len(combined_population_nonzero) > 0:
                vmin = combined_population_nonzero.min()
                vmax = combined_population_nonzero.max()
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=0, vmax=1)  # Default normalization
        # Adjust the colormap to make zero population transparent/white
        cmap = plt.get_cmap(cmap)
        cmap.set_under('white')

        # Prepare J axis with spin and parity labels
        J_cont = self.continuous.coords['J'].values
        J_discrete = self.discrete['J'].values
        J_all = np.union1d(J_cont, J_discrete)
        J_max = np.max(J_all)
        J_min = np.min(J_all)

        # Create J axis for negative and positive parities
        J_values_neg = np.arange(J_max, J_min - 1, -1)
        J_values_pos = np.arange(J_min, J_max + 1)
        J_axis = np.concatenate([J_values_neg, J_values_pos])

        # Generate labels for the x-axis
        J_ticks = []
        J_labels = []
        for J in J_values_neg:
            J_ticks.append(-J)
            J_labels.append(f'{int(J)}-')
        for J in J_values_pos:
            J_ticks.append(J)
            J_labels.append(f'{int(J)}+')

        # Prepare discrete data
        # Map parities to -1 and +1 for plotting on x-axis
        parity_map = {'-': -1, '+': 1}
        discrete_df = self.discrete.copy()
        discrete_df['parity_num'] = discrete_df['pi']
        discrete_df['J_plot'] = discrete_df['J'] * discrete_df['parity_num']

        # Plot discrete levels as horizontal marks
        for idx, row in discrete_df.iterrows():
            Ex = row['Ex']
            J_plot = row['J_plot']
            pop = row['population']
            if pop <= 0:
                continue  # Skip zero or negative population
            color = cmap(norm(pop))
            ax_main.hlines(Ex, J_plot - 0.4, J_plot + 0.4, colors=color, linewidth=1)

        # Prepare continuous data
        Ex_cont = self.continuous.coords['Ex'].values
        pi_cont = self.continuous.coords['pi'].values  # Should be ['-', '+']
        population_cont = self.continuous.values  # Shape: (Ex, J, pi)

        # Flatten and prepare continuous data for negative and positive parities
        # We need to match J_plot similar to discrete data
        J_indices = np.arange(len(J_cont))
        Ex_indices = np.arange(len(Ex_cont))
        pi_indices = np.arange(len(pi_cont))

        # Create meshgrid for indices
        Ex_idx_mesh, J_idx_mesh, pi_idx_mesh = np.meshgrid(Ex_indices, J_indices, pi_indices, indexing='ij')

        # Map parity to -1 and +1
        pi_num = np.array([parity_map[pi] for pi in pi_cont])

        # Compute J_plot for continuous data
        J_plot_cont = J_cont[J_idx_mesh] * pi_num[pi_idx_mesh]

        # Flatten arrays
        Ex_flat = Ex_cont[Ex_idx_mesh.flatten()]
        J_plot_flat = J_plot_cont.flatten()
        pop_flat = population_cont.flatten()

        # Filter data above e_crit
        mask = Ex_flat >= self.e_crit
        Ex_flat = Ex_flat[mask]
        J_plot_flat = J_plot_flat[mask]
        pop_flat = pop_flat[mask]

        # Create a 2D histogram for pcolormesh
        x_edges = np.unique(J_plot_flat)
        y_edges = Ex_cont

        # Create 2D grid of population
        pop_grid = np.zeros((len(y_edges), len(x_edges)))

        # Map J_plot_flat to indices in x_edges
        J_indices = np.searchsorted(x_edges, J_plot_flat)
        Ex_indices = np.searchsorted(y_edges, Ex_flat)

        # Accumulate population into grid
        for xi, yi, pi in zip(J_indices, Ex_indices, pop_flat):
            if pi > 0:
                pop_grid[yi, xi] += pi  # Accumulate population

        # Mask zero population to make them transparent
        pop_grid_masked = np.ma.masked_where(pop_grid <= 0, pop_grid)

        # Plot continuous population with pcolormesh
        X_mesh, Y_mesh = np.meshgrid(x_edges, y_edges)
        mesh = ax_main.pcolormesh(X_mesh, Y_mesh, pop_grid_masked, cmap=cmap, norm=norm, shading='auto')

        # Set axis labels and title
        ax_main.set_xlabel('Spin J (-/+)')
        ax_main.set_ylabel('Excitation Energy Ex (MeV)')
        ax_main.set_title('Discrete and Continuous Populations Combined')

        # Set x-ticks and labels
        ax_main.set_xticks(J_ticks)
        ax_main.set_xticklabels(J_labels, rotation=90)

        # Add colorbar
        cbar_kw = {'label': 'Population'} | (cbar_kwargs or {})
        plt.colorbar(mesh, ax=ax_main, **cbar_kw)

        # Histogram along y-axis showing population density
        # Combine Ex and population from discrete and continuous
        Ex_discrete = discrete_df['Ex'].values
        pop_discrete = discrete_df['population'].values

        Ex_cont_flat = Ex_flat
        pop_cont_flat = pop_flat

        # Combine Ex and population
        Ex_all = np.concatenate([Ex_discrete, Ex_cont_flat])
        pop_all = np.concatenate([pop_discrete, pop_cont_flat])

        # Plot histogram
        sns.histplot(y=Ex_all, weights=pop_all, bins=50, ax=ax_hist, orientation='horizontal', color='gray', edgecolor=None)

        # Plot rug marks for each level, colored by population density
        # For discrete levels
        xmin, xmax = ax_hist.get_xlim()
        span = xmax - xmin
        offset = span * 0.2
        for Ex, pop in zip(Ex_discrete, pop_discrete):
            if pop <= 0:
                continue
            color = cmap(norm(pop))
            ax_hist.hlines(Ex, xmin+offset, xmax-offset, color=color, linewidth=0.5)

        # For continuous levels (optionally, can be omitted if too dense)
        # We will plot rug marks at selected intervals to avoid overcrowding
        # num_rugs = 100  # Adjust as needed
        # if len(Ex_cont_flat) > num_rugs:
        #     indices = np.linspace(0, len(Ex_cont_flat) - 1, num_rugs).astype(int)
        # else:
        #     indices = np.arange(len(Ex_cont_flat))
        # for idx in indices:
        #     Ex = Ex_cont_flat[idx]
        #     pop = pop_cont_flat[idx]
        #     if pop <= 0:
        #         continue
        #     color = cmap(norm(pop))
        #     ax_hist.axhline(Ex, color=color, linewidth=0.5)

        # Invert y-axis to have lower Ex at the bottom
        ax_main.invert_yaxis()
        ax_hist.invert_yaxis()

        # Hide y-axis labels for histogram
        ax_hist.yaxis.set_visible(False)
        ax_hist.set_xlabel('Population Density')

        # Adjust layout
        #plt.tight_layout()

        return ax_main, ax_hist

    def combine(self, E: np.ndarray, normalize: bool = False):
        # convert discrete into a xarray with same spin as continuous
        # and same Ex as E. Parity is combined.
        # Convert discrete DataFrame to xarray DataArray
        J = self.continuous.coords['J'].values
        discrete_xr = xr.DataArray(
            data=np.zeros((len(E), len(J))),
            coords={'Ex': E, 'J': J},
            dims=['Ex', 'J']
        )

        # Populate the discrete xarray
        for _, row in self.discrete.iterrows():
            ex_idx = np.argmin(np.abs(E - row['Ex']))
            j_idx = np.argmin(np.abs(J - row['J']))
            discrete_xr[ex_idx, j_idx] += row['population']

        # Rebin continuous data to use E. Combine both parities
        # Rebin continuous data to match E grid
        continuous_rebinned = self.continuous.interp(Ex=E, method='linear')
        
        # Combine parities by summing over the 'pi' dimension
        continuous_combined = continuous_rebinned.sum(dim='pi')
        
        # Combine discrete and continuous populations
        total_population = discrete_xr + continuous_combined
        
        # Ensure non-negative values
        total_population = total_population.clip(min=0)
        if normalize:
            total_population = total_population / total_population.sum()
        return total_population

    def write(self, path: Path, E: np.ndarray):
        path = Path(path)

        
        population = self.combine(E, normalize=True)
        J = population.coords['J'].values
        
        # Write to file
        with open(path, 'w') as f:
            # Write header
            f.write(f" bin    Ex     Popul.    ")
            for j in J:
                f.write(f"J= {j:.1f}    ")
            f.write("\n\n")

            # Write data
            pop = population.sum(dim='J')
            for i, ex in enumerate(E):
                f.write(f"{i:3d} {ex:8.3f} {pop.values[i]:.3e}")
                for j in J:
                    f.write(f" {population.sel(J=j).values[i]:.3e}")
                f.write("\n")

            
