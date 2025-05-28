from __future__ import annotations
from dataclasses import dataclass
from typing import TypeAlias, Literal
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from .stubs import DensityFunction, VectorizedFunction, ParityDistributionFunction, SpinDensityFunction
from .nld import ericson_spin_distribution, equiparity_distribution, nld_at_Sn_from_D0
from ..base.elements import Element
from ..readers.ripl3.reader import get_RIPL3_levels
from typing import TYPE_CHECKING, Iterator, Callable, Self
from .stubs import Spin, Parity, Energy
from .nld import NLDModel, SpincutModel, NLDatSnParameters, density
import warnings
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
import json

if TYPE_CHECKING:
    from ..readers.ripl3.reader import RIPL3Record, GammaRecord

""" 
TODO:
- It is difficult to extend this with new sources for data.
  Have a @register_provider decorator to add new data sources.
"""

@dataclass(frozen=True, slots=True)
class GammaBranch:
    final: int  # Sequential number of the final state
    Eg: float  # Gamma-ray energy in MeV
    Pg: float  # Probability of the level decaying through photon (gamma ray) emission
    Pem: float  # Probability of the electromagnetic transition (photon, conversion electron, pair creation)
    ICC: float  # Internal conversion coefficient of the transition

    @property
    def br(self) -> float:
        return self.Pg

    @classmethod
    def from_gamma_record(cls, record: GammaRecord) -> GammaBranch:
        return cls(final=record.Nf - 1, # 0-indexed
                   Eg=record.Eg,
                   Pg=record.Pg,
                   Pem=record.Pe,
                   ICC=record.ICC)

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks"""
        html = f"""
        <div style="font-family: Arial, sans-serif; margin: 10px 0;">
            <h4 style="margin: 0; color: #2c3e50;">GammaBranch</h4>
            <table style="width: 100%; border-collapse: collapse; margin-top: 5px;">
                <tr style="background-color: #f2f2f2;">
                    <th style="text-align: left; padding: 6px; border: 1px solid #ddd;">Attribute</th>
                    <th style="text-align: left; padding: 6px; border: 1px solid #ddd;">Value</th>
                    <th style="text-align: left; padding: 6px; border: 1px solid #ddd;">Description</th>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #ddd;">final</td>
                    <td style="padding: 6px; border: 1px solid #ddd; font-family: monospace;">{self.final}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">Sequential number of the final state</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 6px; border: 1px solid #ddd;">Eg</td>
                    <td style="padding: 6px; border: 1px solid #ddd; font-family: monospace;">{self.Eg:.4f} MeV</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">Gamma-ray energy</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #ddd;">Pg</td>
                    <td style="padding: 6px; border: 1px solid #ddd; font-family: monospace;">{self.Pg:.4f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">Photon emission probability</td>
                </tr>
                <tr style="background-color: #f2f2f2;">
                    <td style="padding: 6px; border: 1px solid #ddd;">Pem</td>
                    <td style="padding: 6px; border: 1px solid #ddd; font-family: monospace;">{self.Pem:.4f}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">Electromagnetic transition probability</td>
                </tr>
                <tr>
                    <td style="padding: 6px; border: 1px solid #ddd;">ICC</td>
                    <td style="padding: 6px; border: 1px solid #ddd; font-family: monospace;">{self.ICC:.4e}</td>
                    <td style="padding: 6px; border: 1px solid #ddd;">Internal conversion coefficient</td>
                </tr>
            </table>
        </div>
        """
        return html
    
    
@dataclass(frozen=True, slots=True)
class DiscreteLevels:
    """ Discrete levels of a nucleus

    The discrete levels are read from file, and are defined by their
    excitation energy, spin, and parity. 
    It is isomorphic to a 3d histogram.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    Ex: np.ndarray  # Excitation energy in MeV
    J: np.ndarray  # Spin
    pi: np.ndarray  # Parity
    T1_2: np.ndarray  # Half-life of the level in seconds
    gammas: list[list[GammaBranch]]  # Gamma branches

    def __post_init__(self):
        # Ensure all arrays are of the same length
        n_levels = len(self.Ex)
        for attr in ['J', 'pi', 'T1_2', 'gammas']:
            if not len(getattr(self, attr)) == n_levels:
                raise ValueError(f"All arrays must be of the same length."
                             f" {attr} has length {len(getattr(self, attr))}"
                             f" while the others have length {n_levels}")

    def __len__(self):
        return len(self.Ex)

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Level": np.arange(len(self)),
            "Ex": self.Ex,
            "J": self.J,
            "pi": self.pi,
            "T1_2": self.T1_2
        })

    @classmethod
    def from_df(cls, df: pd.DataFrame, cols: dict[str, str] = None,
                gamma_branches: list[list[GammaBranch]] | None = None) -> DiscreteLevels:
        cols = {**{"Ex": "Ex", "J": "J", "pi": "pi", "T1_2": "T1_2"}, **(cols or {})}
        if gamma_branches is None:
            gamma_branches = [[] for _ in range(len(df))]
        return cls(
            Ex=df[cols["Ex"]].to_numpy(),
            J=df[cols["J"]].to_numpy(),
            pi=df[cols["pi"]].to_numpy(),
            T1_2=df[cols["T1_2"]].to_numpy(),
            gammas=gamma_branches
        )

    def query(self, expression: str) -> DiscreteLevels:
        """Query the discrete levels using a pandas-like expression."""
        df = self.to_df()
        df_query = df.query(expression)
        indices = df_query.index.to_numpy()
        filtered_gamma_branches = [self.gammas[i] for i in indices]
        return DiscreteLevels.from_df(df_query, gamma_branches=filtered_gamma_branches)

    @classmethod
    def from_element(cls, element: str | Element, source: str = "ripl3") -> DiscreteLevels:
        match element:
            case str():
                element = Element.from_str(element)

        match source:
            case 'ripl3':
                return cls.from_ripl3(get_RIPL3_levels(element))
            case _:
                raise ValueError(f"Unsupported source {source}")

    @classmethod
    def from_ripl3(cls, data: RIPL3Record, **kwargs) -> DiscreteLevels:
        levels = data.levels_to_df(**kwargs)
        # Some levels might have been filtered out
        # We check if the level exists in the levels dataframe
        energies = levels.Ex.to_numpy()

        branches: list[list[GammaBranch]] = []
        for record in data.levels:
            if record.level.Elv not in energies:
                continue

            branches_for_level: list[GammaBranch] = []
            for gamma in record.gammas:
                gamma_branch = GammaBranch.from_gamma_record(gamma)
                branches_for_level.append(gamma_branch)
            branches.append(branches_for_level)

        return cls.from_df(data.levels_to_df(), gamma_branches=branches)

    def plot_decay(self, level: int, ax: plt.Axes | None = None,
                level_width: float = 1.0,
                arrow_width: float = 0.5,
                show_energy: bool = True,
                show_spin: bool = True,
                show_branching: bool = True,
                energy_format: str = '.3f',
                level_kwargs: dict | None = None,
                arrow_kwargs: dict | None = None,
                text_kwargs: dict | None = None) -> plt.Axes:
        """
        Plot the decay from a single energy level to lower levels.
        
        Parameters:
        -----------
        level : int
            Index of the level to plot decay from (0 is ground state, 1 is first excited state, etc.)
        ax : plt.Axes, optional
            Axes to plot on. If None, a new figure is created.
        level_width : float, optional
            Width of the horizontal lines representing energy levels.
        arrow_width : float, optional
            Width of the arrows representing transitions.
        show_energy : bool, optional
            Whether to show energy values for levels.
        show_spin : bool, optional
            Whether to show spin and parity for levels.
        show_branching : bool, optional
            Whether to show branching ratios for transitions.
        energy_format : str, optional
            Format string for energy labels.
        level_kwargs : dict, optional
            Additional styling for energy levels.
        arrow_kwargs : dict, optional
            Additional styling for transition arrows.
        text_kwargs : dict, optional
            Additional styling for text labels.
            
        Returns:
        --------
        plt.Axes
            The axes containing the plot.
        """
        import matplotlib.patches as patches
        
        # Check if the level is valid
        if level < 0 or level >= len(self):
            raise ValueError(f"Level index {level} is out of range (0-{len(self)-1})")
        
        # Initialize default styling kwargs
        level_kwargs = level_kwargs or {}
        arrow_kwargs = arrow_kwargs or {}
        text_kwargs = text_kwargs or {}
        
        # Merge with defaults
        _level_kwargs = dict(color='black', linewidth=1)
        _level_kwargs.update(level_kwargs)
        
        _arrow_kwargs = dict(color='red', linewidth=3, arrowstyle='-|>', mutation_scale=5)
        _arrow_kwargs.update(arrow_kwargs)
        
        _text_kwargs = dict(fontsize=8, ha='left', va='center')
        _text_kwargs.update(text_kwargs)
        
        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get the initial level properties
        initial_energy = self.Ex[level]
        
        # Determine which levels we need to plot
        # Include the initial level and all levels it decays to
        levels_to_plot = set([level])  # Initial level
        
        # Get all branches from this level
        branches = self.gammas[level]
        
        # If we have no branches, just plot the level itself
        if not branches:
            ax.text(0, initial_energy, f"No decay from level {level} (E={initial_energy:{energy_format}} MeV)",
                    **_text_kwargs)
            return ax
        
        # Add all final levels to the set
        for branch in branches:
            levels_to_plot.add(branch.final)
        
        # Convert to sorted list
        levels_to_plot = sorted(levels_to_plot)
        
        # Calculate x positions
        x_center = 0
        x_min = x_center - level_width/2
        x_max = x_center + level_width/2
        
        # Plot each relevant energy level as a horizontal line
        for i in levels_to_plot:
            ex, j, p = self.Ex[i], self.J[i], self.pi[i]
            ax.hlines(y=ex, xmin=x_min, xmax=x_max, **_level_kwargs)
            
            # Add level labels
            label_parts = []
            if show_energy:
                label_parts.append(f"{ex:{energy_format}} MeV")
            if show_spin:
                parity_str = '+' if p > 0 else '-'
                label_parts.append(r"$"+f"{int(j*2)}" + "/2" +f"{parity_str}$")
            
            if label_parts:
                label = ", ".join(label_parts)
                ax.text(x_max + 0.3, ex, label, **_text_kwargs)
        
        # Calculate horizontal offsets for arrows if there are multiple branches
        n_branches = len(branches)
        if n_branches > 1:
            offsets = np.linspace(-arrow_width/2, arrow_width/2, n_branches)
        else:
            offsets = [0]
        
        # Draw arrows for each branch
        for j, (branch, offset) in enumerate(zip(branches, offsets)):
            final_energy = self.Ex[branch.final]
            max_linewidth = _arrow_kwargs['linewidth']
            br = branch.br * 10
            linewidth = np.clip(br, 0.01, max_linewidth)
            __arrow_kwargs = _arrow_kwargs | dict(linewidth=linewidth)
            
            # Draw the arrow
            arrow = patches.FancyArrowPatch(
                (x_center + offset, initial_energy),
                (x_center + offset, final_energy),
                connectionstyle="arc3,rad=0",
                **__arrow_kwargs
            )
            ax.add_patch(arrow)
            
            # Add transition energy and branching ratio label if requested
            if show_energy or show_branching:
                mid_point = (initial_energy + final_energy) / 2
                label_parts = []
                
                if show_energy:
                    label_parts.append(f"{branch.Eg:{energy_format}} MeV")
                if show_branching:
                    label_parts.append(f"BR: {branch.br*100:.1f}%")
                    
                if label_parts:
                    label = "\n".join(label_parts)
                    ax.text(x_center + offset + 0.05, mid_point, label,
                            rotation=90, **_text_kwargs)
        
        # Highlight the initial level with a different color
        ax.hlines(y=initial_energy, xmin=x_min, xmax=x_max, 
                color='blue', linewidth=2, zorder=10)
        
        # Set axis properties
        ax.set_ylabel('Energy (MeV)')
        ax.set_title(f'Decay from Level {level} (E={initial_energy:{energy_format}} MeV)')
        
        # Set reasonable y limits to fit the levels
        energies = [self.Ex[i] for i in levels_to_plot]
        y_min = min(energies) - 0.1 * (max(energies) - min(energies))
        y_max = max(energies) + 0.1 * (max(energies) - min(energies))
        ax.set_ylim(y_min, y_max)
        
        # Set reasonable x limits
        x_padding = max(1.0, level_width * 3)  # Enough space for labels
        ax.set_xlim(x_min - arrow_width, x_max + x_padding)
        
        # Remove x-axis ticks and unnecessary spines
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        
        return ax

    def plot_hist(self, ax: plt.Axes | None = None, split_J: bool = False, 
                bins: int | list | np.ndarray = 20, colormap: str = 'turbo',
                **kwargs) -> plt.Axes:
        """
        Plot a histogram of the energy levels, optionally split by spin.
        
        Parameters:
        -----------
        ax : plt.Axes, optional
            Axes to plot on. If None, a new figure is created.
        split_J : bool, optional
            If True, separate levels by spin value.
        bins : int or array-like, optional
            Binning for the histogram.
        colormap : str, optional
            Colormap to use when splitting by spin.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib histogram functions.
            
        Returns:
        --------
        plt.Axes
            The axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        # Set default histogram styling
        hist_kwargs = {'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 0.5}
        hist_kwargs.update(kwargs)
        
        # If not splitting by spin, create a simple histogram
        if not split_J:
            ax.hist(self.Ex, bins=bins, **hist_kwargs)
        else:
            # Get unique spin values and sort them
            unique_spins = np.sort(np.unique(self.J))
            
            # Create a dictionary to group energies by spin
            spin_energies = {spin: self.Ex[self.J == spin] for spin in unique_spins}
            
            # Get a colormap
            cmap = plt.cm.get_cmap(colormap, len(unique_spins))
            
            # Create stacked histogram
            n_spins = len(unique_spins)
            
            # Plot each spin group separately
            for i, (spin, energies) in enumerate(spin_energies.items()):
                if len(energies) > 0:  # Only plot if there are levels with this spin
                    # Format spin as J/2
                    if spin.is_integer():
                        label = f'J = {int(spin)}'
                    else:
                        j_num = int(spin * 2)
                        label = f'J = {j_num}/2'
                    ax.hist(energies, bins=bins, label=label, 
                        color=cmap(i/max(1, n_spins-1)),
                        **hist_kwargs)
                        
            # Add a legend
            ax.legend(title='Spin', loc='upper right')
        
        # Set labels and title
        ax.set_ylabel('Number of levels')
        ax.set_xlabel(r'$E_x$ [MeV]')
        
        title = 'Discrete level density'
        if split_J:
            title += ' (split by spin)'
        ax.set_title(title)
        
        return ax


    def plot_hist_3d(self, ax: plt.Axes | None = None, 
                    energy_bins: int | list | np.ndarray = 20,
                    spin_bins: int | list | np.ndarray = None,
                    colormap: str = 'viridis', 
                    view_angle: tuple[float, float] = (30, -50),
                    show_side_histograms: bool = True,
                    **kwargs) -> plt.Figure:
        """
        Create a 3D histogram of energy levels split by spin, with optional 1D histograms along the sides.
        
        Parameters:
        -----------
        ax : plt.Axes, optional
            3D Axes to plot on. If None, a new figure with 3D axes is created.
        energy_bins : int or array-like, optional
            Binning for the energy axis.
        spin_bins : int or array-like, optional
            Binning for the spin axis. If None, use each unique spin value.
        figsize : tuple, optional
            Figure size if creating a new figure.
        colormap : str, optional
            Colormap to use for the bars.
        view_angle : tuple, optional
            Initial view angle (elevation, azimuth).
        show_side_histograms : bool, optional
            Whether to show 1D histograms along the sides.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib bar3d function.
            
        Returns:
        --------
        plt.Figure
            The figure containing the plots.
        """
        if ax is None:
            fig = plt.figure()
            ax_3d = fig.add_subplot(111, projection='3d')
        else:
            ax_3d = ax
            fig = ax.figure
        
        # Prepare energy bins
        if isinstance(energy_bins, int):
            energy_min = np.min(self.Ex)
            energy_max = np.max(self.Ex)
            energy_edges = np.linspace(energy_min, energy_max, energy_bins + 1)
        else:
            energy_edges = np.asarray(energy_bins)
        
        # Prepare spin bins
        unique_spins = np.sort(np.unique(self.J))
        if spin_bins is None:
            # Use each unique spin value with a small buffer
            spin_edges = np.concatenate([
                unique_spins - 0.25,
                [unique_spins[-1] + 0.25]
            ])
        elif isinstance(spin_bins, int):
            spin_min = np.min(unique_spins) - 0.5
            spin_max = np.max(unique_spins) + 0.5
            spin_edges = np.linspace(spin_min, spin_max, spin_bins + 1)
        else:
            spin_edges = np.asarray(spin_bins)
        
        # Calculate the 2D histogram
        H, energy_edges, spin_edges = np.histogram2d(
            self.Ex, self.J, 
            bins=[energy_edges, spin_edges]
        )
        
        # Get the centers of the bins
        energy_centers = (energy_edges[:-1] + energy_edges[1:]) / 2
        spin_centers = (spin_edges[:-1] + spin_edges[1:]) / 2
        
        # Create meshgrid for plotting
        energy_mesh, spin_mesh = np.meshgrid(energy_centers, spin_centers)
        
        # Calculate bar widths
        energy_width = energy_edges[1] - energy_edges[0]
        spin_width = spin_edges[1] - spin_edges[0]
        
        # Set default bar styling
        bar_kwargs = {'alpha': 0.7, 'edgecolor': 'black', 'linewidth': 0.0}
        bar_kwargs.update(kwargs)
        
        # Transpose H to match the meshgrid orientation
        H = H.T
        
        # Get colormap
        cmap = plt.cm.get_cmap(colormap)
        
        # Normalize colors based on bar heights
        max_height = np.max(H)
        
        # Plot the bars
        for i in range(len(spin_centers)):
            for j in range(len(energy_centers)):
                if H[i, j] > 0:  # Only plot non-zero bars
                    height = H[i, j]
                    # Color based on height
                    color = cmap(height / max_height)
                    
                    ax_3d.bar3d(
                        spin_centers[i] - spin_width/2,  # Swapped from energy_centers
                        energy_centers[j] - energy_width/2,  # Swapped from spin_centers
                        0,
                        spin_width,  # Swapped from energy_width
                        energy_width,  # Swapped from spin_width
                        height,
                        color=color,
                        **bar_kwargs
                    )
        
        # Set labels and title for 3D plot
        ax_3d.set_xlabel('Spin (ℏ)')  # Swapped from Energy
        ax_3d.set_ylabel('Energy (MeV)')  # Swapped from Spin
        ax_3d.set_zlabel('Number of levels')
        
        # Format spin tick labels as J/2
        spin_ticks = []
        spin_labels = []
        for spin in unique_spins:
            spin_ticks.append(spin)
            if spin.is_integer():
                spin_labels.append(f'{int(spin)}')
            else:
                j_num = int(spin * 2)
                spin_labels.append(f'{j_num}/2')
        
        ax_3d.set_xticks(spin_ticks)  # Changed from yticks
        ax_3d.set_xticklabels(spin_labels, rotation=90, fontsize=8)  # Changed from yticklabels
        
        # Set view angle
        ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # If showing side histograms, add them
        
        # Adjust layout
        if show_side_histograms:
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
        
        return fig

    def plot_spin_distribution(self, ax: plt.Axes | None = None,
                            sort_by: str = 'spin',
                            show_parity: bool = True,
                            colormap: str = 'coolwarm',
                            show_counts: bool = True,
                            **kwargs) -> plt.Axes:
        """
        Plot the distribution of nuclear levels by spin and optionally parity.
        
        Parameters:
        -----------
        ax : plt.Axes, optional
            Axes to plot on. If None, a new figure is created.
        sort_by : str, optional
            How to sort the data ('spin' or 'count').
        show_parity : bool, optional
            Whether to separate levels by parity.
        colormap : str, optional
            Colormap to use for bars.
        show_counts : bool, optional
            Whether to display the count values on top of the bars.
        **kwargs : dict
            Additional keyword arguments passed to matplotlib bar function.
            
        Returns:
        --------
        plt.Axes
            The axes containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        
        spins = self.J
        parities = self.pi
        energies = self.Ex
        energy_label = f"E ≤ {np.max(energies):.3f} MeV"
        
        # Get unique spin values
        unique_spins = np.sort(np.unique(spins))
        
        # Count levels by spin (and optionally parity)
        if show_parity:
            # Count separately for positive and negative parity
            pos_parity_counts = np.array([np.sum((spins == spin) & (parities > 0)) for spin in unique_spins])
            neg_parity_counts = np.array([np.sum((spins == spin) & (parities < 0)) for spin in unique_spins])
            
            # Sort spins if requested
            if sort_by == 'count':
                # Sort by total count
                total_counts = pos_parity_counts + neg_parity_counts
                sort_indices = np.argsort(total_counts)[::-1]  # descending
                unique_spins = unique_spins[sort_indices]
                pos_parity_counts = pos_parity_counts[sort_indices]
                neg_parity_counts = neg_parity_counts[sort_indices]
        else:
            # Count all levels by spin regardless of parity
            counts = np.array([np.sum(spins == spin) for spin in unique_spins])
            
            # Sort if requested
            if sort_by == 'count':
                sort_indices = np.argsort(counts)[::-1]  # descending
                unique_spins = unique_spins[sort_indices]
                counts = counts[sort_indices]
        
        # Format spin labels
        spin_labels = []
        for spin in unique_spins:
            if spin.is_integer():
                spin_labels.append(f'{int(spin)}')
            else:
                j_num = int(spin * 2)
                spin_labels.append(f'{j_num}/2')
        
        # Set bar width
        bar_width = 0.8 if not show_parity else 0.4
        
        # Plot the bars
        if show_parity:
            # Positions for side-by-side bars
            x = np.arange(len(unique_spins))
            
            # Create the stacked/grouped bars
            pos_bars = ax.bar(x - bar_width/2, pos_parity_counts, bar_width, 
                            label='Positive Parity (π = +)', 
                            color='skyblue', edgecolor='black', alpha=0.7)
            neg_bars = ax.bar(x + bar_width/2, neg_parity_counts, bar_width,
                            label='Negative Parity (π = -)', 
                            color='salmon', edgecolor='black', alpha=0.7)
            
            # Add count labels if requested
            if show_counts:
                for i, (pos_count, neg_count) in enumerate(zip(pos_parity_counts, neg_parity_counts)):
                    if pos_count > 0:
                        ax.text(x[i] - bar_width/2, pos_count, str(pos_count), 
                            ha='center', va='bottom', fontsize=9)
                    if neg_count > 0:
                        ax.text(x[i] + bar_width/2, neg_count, str(neg_count),
                            ha='center', va='bottom', fontsize=9)
        else:
            # Use a colormap for a gradient effect
            cmap = plt.cm.get_cmap(colormap)
            max_count = np.max(counts) if len(counts) > 0 else 1
            
            # Create the bars with color gradient
            bars = ax.bar(np.arange(len(unique_spins)), counts, bar_width,
                        color=[cmap(c/max_count) for c in counts],
                        edgecolor='black', alpha=0.8)
            
            # Add count labels if requested
            if show_counts:
                for i, count in enumerate(counts):
                    if count > 0:
                        ax.text(i, count, str(count), ha='center', va='bottom', fontsize=9)
        
        # Set axis properties
        ax.set_xticks(np.arange(len(unique_spins)))
        ax.set_xticklabels(spin_labels)
        ax.set_xlabel('Spin (ℏ)')
        ax.set_ylabel('Number of Levels')
        
        # Add legend if showing parity
        if show_parity:
            ax.legend()
        
        # Set title 
        ax.set_title(f'Nuclear Level Spin Distribution ({energy_label})')
        
        # Add summary statistics in a text box
        total_levels = len(spins)
        unique_parity_values = np.unique(parities)
        parity_counts = {p: np.sum(parities == p) for p in unique_parity_values}
        
        # Calculate average spin (weighted by level count)
        average_spin = np.sum(spins) / len(spins) if len(spins) > 0 else 0
        
        # Create statistics text
        stats_text = f"Total levels: {total_levels}\n"
        if average_spin > 0:
            if average_spin.is_integer():
                avg_spin_text = f"{int(average_spin)}"
            else:
                avg_spin_num = int(average_spin * 2)
                avg_spin_text = f"{avg_spin_num}/2"
            stats_text += f"Average spin: {avg_spin_text}\n"
        
        if show_parity and len(unique_parity_values) > 1:
            pos_count = parity_counts.get(1, 0)
            neg_count = parity_counts.get(-1, 0)
            stats_text += f"Positive parity: {pos_count} ({pos_count/total_levels*100:.1f}%)\n"
            stats_text += f"Negative parity: {neg_count} ({neg_count/total_levels*100:.1f}%)"
        
        # Add a text box with statistics
        ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        return ax
        
    def drop_empty_gammas(self) -> DiscreteLevels:
        """ Drop all levels that has no gammas """
        non_empty_gammas_idx = [i for i, gammas in enumerate(self.gammas) if gammas]
        return self.from_df(self.to_df().iloc[non_empty_gammas_idx],
                            gamma_branches=[self.gammas[i] for i in non_empty_gammas_idx])

    def __len__(self) -> int:
        return len(self.Ex)

    def __getitem__(self, index: int) -> tuple[Energy, Spin, Parity]:
        return self.Ex[index], self.J[index], self.pi[index]

    def __iter__(self) -> Iterator[tuple[Energy, Spin, Parity]]:
        return iter(zip(self.Ex, self.J, self.pi))

    def _repr_html_(self) -> str:
        """HTML representation with embedded JavaScript for DiscreteLevels.
        
        Creates a comprehensive, interactive view with collapsible arrays and nested structure
        for gamma branches. This representation mimics the GammaBranch _repr_html_ style while
        adding interactive elements.
        
        Returns:
            str: HTML string with embedded JavaScript
        """
        # Create a unique ID to avoid conflicts
        import uuid
        unique_id = str(uuid.uuid4()).replace('-', '')
        
        # Format arrays as JSON strings for JavaScript
        import json
        ex_json = json.dumps(self.Ex.tolist())
        j_json = json.dumps(self.J.tolist())
        pi_json = json.dumps(self.pi.tolist())
        t12_json = json.dumps(self.T1_2.tolist())
        
        # Prepare gamma branches data
        gamma_data = []
        for level_idx, branches in enumerate(self.gammas):
            level_branches = []
            for branch in branches:
                level_branches.append({
                    'final': branch.final,
                    'Eg': float(branch.Eg),
                    'Pg': float(branch.Pg),
                    'Pem': float(branch.Pem),
                    'ICC': float(branch.ICC)
                })
            gamma_data.append(level_branches)
        
        gammas_json = json.dumps(gamma_data)
        
        # Build HTML with embedded script
        html = f"""
        <div id="discrete-levels-{unique_id}" class="discrete-levels-container">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">DiscreteLevels</h3>
            <div class="loading">Loading...</div>
        </div>
        
        <script>
        (function() {{
            // Get container by ID
            const container = document.getElementById('discrete-levels-{unique_id}');
            container.style.fontFamily = 'Arial, sans-serif';
            container.style.margin = '10px 0';
            container.style.padding = '12px';
            container.style.border = '1px solid #e0e0e0';
            container.style.borderRadius = '5px';
            container.style.backgroundColor = '#f9f9f9';
            
            // Remove loading message
            const loading = container.querySelector('.loading');
            if (loading) {{
                container.removeChild(loading);
            }}
            
            // Parse the data
            const Ex = {ex_json};
            const J = {j_json};
            const pi = {pi_json};
            const T1_2 = {t12_json};
            const gammas = {gammas_json};
            
            // Helper function to format numbers nicely
            function formatNumber(num, precision = 4) {{
                if (Math.abs(num) < 0.0001 || Math.abs(num) >= 10000) {{
                    return num.toExponential(precision);
                }}
                return num.toFixed(precision);
            }}
            
            // Helper function to create collapsible array displays
            function createArrayDisplay(name, array, description) {{
                const wrapper = document.createElement('div');
                wrapper.style.margin = '8px 0';
                wrapper.style.backgroundColor = '#ffffff';
                wrapper.style.border = '1px solid #e0e0e0';
                wrapper.style.borderRadius = '4px';
                
                const titleRow = document.createElement('div');
                titleRow.style.display = 'flex';
                titleRow.style.alignItems = 'center';
                titleRow.style.cursor = 'pointer';
                titleRow.style.padding = '8px';
                titleRow.style.backgroundColor = '#f2f2f2';
                titleRow.style.borderTopLeftRadius = '4px';
                titleRow.style.borderTopRightRadius = '4px';
                
                // Toggle button
                const toggleBtn = document.createElement('span');
                toggleBtn.innerHTML = '▶';
                toggleBtn.style.marginRight = '10px';
                toggleBtn.style.fontSize = '12px';
                toggleBtn.style.transition = 'transform 0.2s';
                toggleBtn.style.display = 'inline-block';
                toggleBtn.style.width = '12px';
                
                // Label
                const label = document.createElement('span');
                label.style.fontWeight = 'bold';
                label.textContent = `${{name}}`;
                
                // Length indicator
                const lengthIndicator = document.createElement('span');
                lengthIndicator.style.marginLeft = '8px';
                lengthIndicator.style.color = '#0366d6';
                lengthIndicator.style.fontSize = '0.9em';
                lengthIndicator.textContent = `[${{array.length}} elements]`;
                
                // Preview
                const preview = document.createElement('span');
                preview.style.marginLeft = '12px';
                preview.style.fontFamily = 'monospace';
                preview.style.fontSize = '0.9em';
                preview.style.color = '#6c757d';
                
                // Get preview of the array
                if (array.length <= 5) {{
                    preview.textContent = `[${{array.map(v => formatNumber(v)).join(', ')}}]`;
                }} else {{
                    preview.textContent = `[${{array.slice(0, 3).map(v => formatNumber(v)).join(', ')}}, ..., ${{formatNumber(array[array.length-1])}}]`;
                }}
                
                // Description
                if (description) {{
                    const desc = document.createElement('span');
                    desc.style.marginLeft = '12px';
                    desc.style.color = '#6c757d';
                    desc.style.fontSize = '0.9em';
                    desc.textContent = description;
                    titleRow.appendChild(desc);
                }}
                
                titleRow.appendChild(toggleBtn);
                titleRow.appendChild(label);
                titleRow.appendChild(lengthIndicator);
                titleRow.appendChild(preview);
                
                // Content area
                const content = document.createElement('div');
                content.style.display = 'none';
                content.style.padding = '10px';
                content.style.backgroundColor = '#ffffff';
                content.style.borderBottomLeftRadius = '4px';
                content.style.borderBottomRightRadius = '4px';
                content.style.maxHeight = '300px';
                content.style.overflowY = 'auto';
                content.style.fontFamily = 'monospace';
                
                // Create a table for the array
                const table = document.createElement('table');
                table.style.width = '100%';
                table.style.borderCollapse = 'collapse';
                
                // Add header row
                const thead = document.createElement('thead');
                const headerRow = document.createElement('tr');
                
                const indexHeader = document.createElement('th');
                indexHeader.textContent = 'Index';
                indexHeader.style.padding = '6px';
                indexHeader.style.textAlign = 'left';
                indexHeader.style.borderBottom = '1px solid #e0e0e0';
                indexHeader.style.backgroundColor = '#f8f9fa';
                
                const valueHeader = document.createElement('th');
                valueHeader.textContent = 'Value';
                valueHeader.style.padding = '6px';
                valueHeader.style.textAlign = 'right';
                valueHeader.style.borderBottom = '1px solid #e0e0e0';
                valueHeader.style.backgroundColor = '#f8f9fa';
                
                headerRow.appendChild(indexHeader);
                headerRow.appendChild(valueHeader);
                thead.appendChild(headerRow);
                table.appendChild(thead);
                
                // Add data rows
                const tbody = document.createElement('tbody');
                for (let i = 0; i < array.length; i++) {{
                    const row = document.createElement('tr');
                    row.style.backgroundColor = i % 2 === 0 ? '#ffffff' : '#f8f9fa';
                    
                    const indexCell = document.createElement('td');
                    indexCell.textContent = i;
                    indexCell.style.padding = '4px 6px';
                    indexCell.style.borderBottom = '1px solid #e0e0e0';
                    indexCell.style.color = '#6c757d';
                    indexCell.style.textAlign = 'left';
                    
                    const valueCell = document.createElement('td');
                    valueCell.textContent = formatNumber(array[i]);
                    valueCell.style.padding = '4px 6px';
                    valueCell.style.textAlign = 'left';
                    valueCell.style.fontFamily = 'monospace';
                    valueCell.style.borderBottom = '1px solid #e0e0e0';
                    
                    row.appendChild(indexCell);
                    row.appendChild(valueCell);
                    tbody.appendChild(row);
                }}
                
                table.appendChild(tbody);
                content.appendChild(table);
                
                // Add event listener for toggling
                titleRow.addEventListener('click', function() {{
                    if (content.style.display === 'none') {{
                        content.style.display = 'block';
                        toggleBtn.innerHTML = '▼';
                        toggleBtn.style.transform = 'rotate(0deg)';
                    }} else {{
                        content.style.display = 'none';
                        toggleBtn.innerHTML = '▶';
                        toggleBtn.style.transform = 'rotate(-90deg)';
                    }}
                }});
                
                wrapper.appendChild(titleRow);
                wrapper.appendChild(content);
                return wrapper;
            }}
            
            // Create displays for each array
            const exDisplay = createArrayDisplay('Ex', Ex, 'Excitation energy (MeV)');
            const jDisplay = createArrayDisplay('J', J, 'Spin');
            const piDisplay = createArrayDisplay('pi', pi, 'Parity');
            const t12Display = createArrayDisplay('T1_2', T1_2, 'Half-life (s)');
            
            // Add all arrays to container
            container.appendChild(exDisplay);
            container.appendChild(jDisplay);
            container.appendChild(piDisplay);
            container.appendChild(t12Display);
            
            // Create gamma branches section
            const gammaSection = document.createElement('div');
            gammaSection.style.margin = '8px 0';
            gammaSection.style.backgroundColor = '#ffffff';
            gammaSection.style.border = '1px solid #e0e0e0';
            gammaSection.style.borderRadius = '4px';
            
            const gammaHeader = document.createElement('div');
            gammaHeader.style.display = 'flex';
            gammaHeader.style.alignItems = 'center';
            gammaHeader.style.cursor = 'pointer';
            gammaHeader.style.padding = '8px';
            gammaHeader.style.backgroundColor = '#f2f2f2';
            gammaHeader.style.borderTopLeftRadius = '4px';
            gammaHeader.style.borderTopRightRadius = '4px';
            
            const gammaToggle = document.createElement('span');
            gammaToggle.innerHTML = '▶';
            gammaToggle.style.marginRight = '10px';
            gammaToggle.style.fontSize = '12px';
            gammaToggle.style.transition = 'transform 0.2s';
            gammaToggle.style.display = 'inline-block';
            gammaToggle.style.width = '12px';
            
            const gammaLabel = document.createElement('span');
            gammaLabel.style.fontWeight = 'bold';
            gammaLabel.textContent = 'gammas';
            
            const gammaLengthIndicator = document.createElement('span');
            gammaLengthIndicator.style.marginLeft = '8px';
            gammaLengthIndicator.style.color = '#0366d6';
            gammaLengthIndicator.style.fontSize = '0.9em';
            gammaLengthIndicator.textContent = `[${{gammas.length}} levels]`;
            
            const gammaDesc = document.createElement('span');
            gammaDesc.style.marginLeft = '12px';
            gammaDesc.style.color = '#6c757d';
            gammaDesc.style.fontSize = '0.9em';
            gammaDesc.textContent = 'Gamma branches';
            
            gammaHeader.appendChild(gammaToggle);
            gammaHeader.appendChild(gammaLabel);
            gammaHeader.appendChild(gammaLengthIndicator);
            gammaHeader.appendChild(gammaDesc);
            
            const gammaContent = document.createElement('div');
            gammaContent.style.display = 'none';
            gammaContent.style.padding = '0px';
            gammaContent.style.backgroundColor = '#ffffff';
            gammaContent.style.borderBottomLeftRadius = '4px';
            gammaContent.style.borderBottomRightRadius = '4px';
            gammaContent.style.maxHeight = '500px';
            gammaContent.style.overflowY = 'auto';
            
            // Create each level's display
            for (let i = 0; i < gammas.length; i++) {{
                const levelWrapper = document.createElement('div');
                levelWrapper.style.margin = '8px';
                levelWrapper.style.backgroundColor = '#f8f9fa';
                levelWrapper.style.border = '1px solid #e0e0e0';
                levelWrapper.style.borderRadius = '4px';
                
                const levelHeader = document.createElement('div');
                levelHeader.style.display = 'flex';
                levelHeader.style.alignItems = 'center';
                levelHeader.style.cursor = 'pointer';
                levelHeader.style.padding = '6px 8px';
                levelHeader.style.backgroundColor = '#edf2f7';
                levelHeader.style.borderTopLeftRadius = '4px';
                levelHeader.style.borderTopRightRadius = '4px';
                
                const levelToggle = document.createElement('span');
                levelToggle.innerHTML = '▶';
                levelToggle.style.marginRight = '8px';
                levelToggle.style.fontSize = '10px';
                levelToggle.style.transition = 'transform 0.2s';
                levelToggle.style.display = 'inline-block';
                levelToggle.style.width = '10px';
                
                const levelNumber = document.createElement('span');
                levelNumber.style.fontWeight = 'bold';
                levelNumber.textContent = `Level ${{i}}`;
                
                const energyValue = document.createElement('span');
                energyValue.style.marginLeft = '10px';
                energyValue.style.fontFamily = 'monospace';
                energyValue.style.color = '#0366d6';
                energyValue.textContent = `E = ${{formatNumber(Ex[i])}} MeV`;
                
                const branchCount = document.createElement('span');
                branchCount.style.marginLeft = '10px';
                branchCount.style.color = '#6c757d';
                branchCount.style.fontSize = '0.9em';
                branchCount.textContent = `${{gammas[i].length}} branches`;
                
                levelHeader.appendChild(levelToggle);
                levelHeader.appendChild(levelNumber);
                levelHeader.appendChild(energyValue);
                levelHeader.appendChild(branchCount);
                
                const levelContent = document.createElement('div');
                levelContent.style.display = 'none';
                levelContent.style.padding = '8px';
                levelContent.style.backgroundColor = '#ffffff';
                levelContent.style.borderBottomLeftRadius = '4px';
                levelContent.style.borderBottomRightRadius = '4px';
                
                // Create branch cards for this level
                for (let j = 0; j < gammas[i].length; j++) {{
                    const branch = gammas[i][j];
                    
                    const branchCard = document.createElement('div');
                    branchCard.style.margin = '6px 0';
                    branchCard.style.padding = '8px';
                    branchCard.style.border = '1px solid #e0e0e0';
                    branchCard.style.borderRadius = '4px';
                    branchCard.style.backgroundColor = '#f8f9fa';
                    
                    const branchTitle = document.createElement('div');
                    branchTitle.style.fontWeight = 'bold';
                    branchTitle.style.marginBottom = '5px';
                    branchTitle.textContent = `Branch ${{j}}: Level ${{i}} → Level ${{branch.final}}`;
                    
                    const branchDetails = document.createElement('div');
                    branchDetails.style.marginLeft = '10px';
                    branchDetails.style.fontFamily = 'monospace';
                    branchDetails.style.fontSize = '0.9em';
                    
                    // Create a table for branch properties
                    const branchTable = document.createElement('table');
                    branchTable.style.width = '100%';
                    branchTable.style.borderCollapse = 'collapse';
                    
                    // Add rows for each property
                    const properties = [
                        ['Eg', formatNumber(branch.Eg) + ' MeV', 'Gamma-ray energy'],
                        ['Pg', formatNumber(branch.Pg), 'Photon emission probability'],
                        ['Pem', formatNumber(branch.Pem), 'Electromagnetic transition probability'],
                        ['ICC', formatNumber(branch.ICC, 4), 'Internal conversion coefficient']
                    ];
                    
                    for (let k = 0; k < properties.length; k++) {{
                        const [name, value, desc] = properties[k];
                        
                        const row = document.createElement('tr');
                        row.style.backgroundColor = k % 2 === 0 ? '#f8f9fa' : '#ffffff';
                        
                        const nameCell = document.createElement('td');
                        nameCell.textContent = name;
                        nameCell.style.padding = '3px 6px';
                        nameCell.style.width = '15%';
                        nameCell.style.fontWeight = 'bold';
                        
                        const valueCell = document.createElement('td');
                        valueCell.textContent = value;
                        valueCell.style.padding = '3px 6px';
                        valueCell.style.width = '25%';
                        
                        const descCell = document.createElement('td');
                        descCell.textContent = desc;
                        descCell.style.padding = '3px 6px';
                        descCell.style.color = '#6c757d';
                        descCell.style.width = '60%';
                        
                        row.appendChild(nameCell);
                        row.appendChild(valueCell);
                        row.appendChild(descCell);
                        branchTable.appendChild(row);
                    }}
                    
                    branchDetails.appendChild(branchTable);
                    branchCard.appendChild(branchTitle);
                    branchCard.appendChild(branchDetails);
                    levelContent.appendChild(branchCard);
                }}
                
                // Add event listener for toggling level
                levelHeader.addEventListener('click', function() {{
                    if (levelContent.style.display === 'none') {{
                        levelContent.style.display = 'block';
                        levelToggle.innerHTML = '▼';
                        levelToggle.style.transform = 'rotate(0deg)';
                    }} else {{
                        levelContent.style.display = 'none';
                        levelToggle.innerHTML = '▶';
                        levelToggle.style.transform = 'rotate(-90deg)';
                    }}
                }});
                
                levelWrapper.appendChild(levelHeader);
                levelWrapper.appendChild(levelContent);
                gammaContent.appendChild(levelWrapper);
            }}
            
            // Add event listener for toggling gamma section
            gammaHeader.addEventListener('click', function() {{
                if (gammaContent.style.display === 'none') {{
                    gammaContent.style.display = 'block';
                    gammaToggle.innerHTML = '▼';
                    gammaToggle.style.transform = 'rotate(0deg)';
                }} else {{
                    gammaContent.style.display = 'none';
                    gammaToggle.innerHTML = '▶';
                    gammaToggle.style.transform = 'rotate(-90deg)';
                }}
            }});
            
            gammaSection.appendChild(gammaHeader);
            gammaSection.appendChild(gammaContent);
            container.appendChild(gammaSection);
            
            // Add class info
            const classInfo = document.createElement('div');
            classInfo.style.marginTop = '10px';
            classInfo.style.padding = '8px';
            classInfo.style.backgroundColor = '#f0f7ff';
            classInfo.style.border = '1px solid #cce5ff';
            classInfo.style.borderRadius = '4px';
            classInfo.style.fontSize = '0.9em';
            classInfo.style.color = '#0c5460';
            classInfo.innerHTML = '<strong>DiscreteLevels:</strong> Discrete levels of a nucleus. The discrete levels are defined by their excitation energy, spin, and parity.';
            container.appendChild(classInfo);
            
            // Add a note about interactivity
            const note = document.createElement('div');
            note.style.marginTop = '8px';
            note.style.fontSize = '0.85em';
            note.style.color = '#6c757d';
            note.innerHTML = '<strong>Note:</strong> Click on section headers to expand/collapse details.';
            container.appendChild(note);
        }})();
        </script>
        """
        
        return html


@dataclass(frozen=True, slots=True)
class ContinuousLevels:
    """ Continuous levels of a nucleus

    The continuous levels are defined by their excitation energy, spin, and parity.
    The density is defined for each bin.
    The density has shape (Ex, J, pi).
    It is a discretized version of the nuclear level density function.
    """
    density: np.ndarray
    Ex: np.ndarray
    J: np.ndarray

    def __post_init__(self):
        if self.density.shape != (len(self.Ex), len(self.J), 2):
            raise ValueError(f"Density must be of shape (len(self.Ex), len(self.J), 2). Got {self.density.shape}")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> ContinuousLevels:
        inputs = tuple(input_.density if isinstance(input_, ContinuousLevels) else input_ for input_ in inputs)
        if method == '__call__':
            return ContinuousLevels(density=ufunc(*inputs, **kwargs),
                                    Ex=self.Ex,
                                    J=self.J)
        raise NotImplementedError(f"Unsupported ufunc {ufunc} with method {method}")


    @classmethod
    def from_density(cls, *, Ex: np.ndarray, J: np.ndarray, pi: np.ndarray, density: DensityFunction) -> ContinuousLevels:
        # Test for constant bin width
        de = np.diff(Ex)
        if not np.allclose(de, de[0]):
            raise ValueError("Excitation energies must have constant bin width")

        # Requires constant bin width
        de = Ex[1] - Ex[0]
        # Generate levels for each bin, spin, and parity
        levels = de * density(Ex, J, pi)
        return cls(density=levels, Ex=Ex, J=J)

    def shape(self) -> tuple[int, int, int]:
        return self.density.shape

    def to_xarray(self) -> xr.DataArray:
        return xr.DataArray(self.density, dims=["Ex", "J", "pi"],
                           coords={"Ex": self.Ex,
                                    "J": self.J,
                                    "pi": ["+", "-"]})

    def from_xarray(self, da: xr.DataArray, dims: dict[str, str] | None = None) -> ContinuousLevels:
        dims = {"Ex": "Ex", "J": "J", "pi": "pi"} | (dims or {})
        return ContinuousLevels(density=da.values,
                                Ex=da.coords[dims["Ex"]].values,
                                J=da.coords[dims["J"]].values,
                                pi=da.coords[dims["pi"]].values)

    def apply(self, f: VectorizedFunction) -> ContinuousLevels:
        return ContinuousLevels(density=f(self.density),
                                Ex=self.Ex,
                                J=self.J)



@dataclass(frozen=True, slots=True)
class LevelScheme:
    """ Level scheme of a nucleus

    The level scheme is known discrete levels combined with a discretized version of a level density function.
    """
    discrete: DiscreteLevels
    continuous: ContinuousLevels
    e_crit: float

    def __post_init__(self):
        assert self.e_crit >= 0, "e_crit must be non-negative"

    def make_realization(self, sampling = np.random.poisson) -> LevelScheme:
        return LevelScheme(discrete=self.discrete,
                           continuous=self.continuous.apply(sampling),
                           e_crit=self.e_crit)

    def drop_nan_density(self) -> 'LevelScheme':
        # Create a mask for rows of Ex where all values in the corresponding density slice are NaN
        mask_Ex = ~np.isnan(self.continuous.density).all(axis=(1, 2))
        
        # Create a mask for columns of J where all values in the corresponding density slice are NaN
        mask_J = ~np.isnan(self.continuous.density).all(axis=(0, 2))
        
        # Filter Ex and J based on the masks
        Ex_filtered = self.continuous.Ex[mask_Ex]
        J_filtered = self.continuous.J[mask_J]
        
        # Warn if the minimum excitation energy exceeds the critical energy
        if Ex_filtered.min() > self.e_crit:
            warnings.warn(f"Minimum excitation energy {Ex_filtered.min()} is larger than the critical energy {self.e_crit}.\n")
        
        # Filter the density matrix to remove rows and columns corresponding to NaNs in Ex and J
        density_filtered = self.continuous.density[mask_Ex][:, mask_J]

        # Return a new LevelScheme with the filtered values
        return LevelScheme(
            discrete=self.discrete,
            continuous=ContinuousLevels(
                density=density_filtered,
                Ex=Ex_filtered,
                J=J_filtered
            ),
            e_crit=self.e_crit
        )

    def plot(self, ax: plt.Axes | None = None,
             realizations: int = 0,
             normalize: bool = True,
             discrete_kwargs: dict | None = None,
             continuous_kwargs: dict | None = None,
             realization_kwargs: dict | None = None,
             crit_kwargs: dict | None = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        kw = {} if discrete_kwargs is None else discrete_kwargs
        kw = {'label': 'discrete levels'} | kw
        self.discrete.plot_hist(ax=ax, **kw)

        if normalize:
            de = self.continuous.Ex[1] - self.continuous.Ex[0]
        else:
            de = 1.0

        kw = {} if realization_kwargs is None else realization_kwargs
        kw = {'alpha': 0.2, 'color': 'k'} | kw
        for i in range(realizations):
            if i == realizations - 1:
                kw = {'label': 'realization'} | kw
            (self.make_realization().continuous.to_xarray().sum(['J', 'pi']) / de).plot.step(ax=ax, **kw)

        kw = {} if continuous_kwargs is None else continuous_kwargs
        kw = {'label': 'model mean',  'lw':0.1} | kw
        (self.continuous.to_xarray().sum(['J', 'pi']) / de).plot.step(ax=ax, **kw)
        
        kw = {} if crit_kwargs is None else crit_kwargs
        add_text = kw.pop('add_text', True)
        kw = {'label': 'critical energy', 'color': 'r', 'linestyle': '--'} | kw
        ax.axvline(self.e_crit, **kw)
        if add_text:
            ax.text(self.e_crit, ax.get_ylim()[1]/2, f'$e_{{crit}} = {self.e_crit:.2f}$ MeV',
                    ha='right', va='top', color='r', rotation=90)

        ax.set_yscale('log')
        ax.set_xlabel(r'$E_x$ [MeV]')
        ax.set_ylabel(r'$\rho(E)$')
        return ax


@dataclass(slots=True)
class LevelDensityModel:
    """ Level density model

    The level density model is a function that approximates the level density function.
    It is defined by a set of parameters that are fitted to experimental data.
    Discrete levels are assumed known, and the level density function is fitted to experimental data.

    TODO:
    - Add txt and html nice informative summary of the model.
    - For normalization, this would represent the outcome of a fit.
      A model *specification* would be models without parameters specified

    Attributes:
        discrete: DiscreteLevels
            The discrete levels of the nucleus.
        nld: NLDModel
            The total level density function rho(Ex).
        spincut: SpincutModel
            The spincut function sigma^2(Ex).
        parity_distribution: ParityDistributionFunction
            The parity distribution function Pi(Ex, pi).
            Default is equiparity distribution.
        spin_density: SpinDensityFunction
            The spin distribution function g(Ex, J, sigma^2).
            Default is the Ericson spin distribution.
        nld_at_sn_parameters: NLDatSnParameters | None
            The parameters of the level density function at the spincut Sn.
            Only for convenience, not used in the RAINIEST simulation.
    """
    discrete: DiscreteLevels
    nld: NLDModel
    spincut: SpincutModel
    parity_distribution: ParityDistributionFunction = equiparity_distribution
    spin_density: SpinDensityFunction = ericson_spin_distribution
    nld_at_sn_parameters: NLDatSnParameters | None = None

    @classmethod
    def from_element(cls, element: Element | str, 
                     nld: NLDModel | type[NLDModel], 
                     spincut: SpincutModel | type[SpincutModel],
                     discrete: DiscreteLevels | None = None,
                     parity_distribution: ParityDistributionFunction = equiparity_distribution,
                     spin_density: SpinDensityFunction = ericson_spin_distribution,
                     nld_at_sn_parameters: NLDatSnParameters | None = None,
                     source: Literal["ct", "bsfg"] = "ct") -> Self:
        if isinstance(nld, type):
            nld = nld.from_element(element)
        if isinstance(spincut, type):
            spincut = spincut.from_element(element, source)
        if nld_at_sn_parameters is None:
            nld_at_sn_parameters = NLDatSnParameters.from_element(element, source)
        if discrete is None:
            discrete = DiscreteLevels.from_element(element)
        
        return cls(discrete=discrete,
                   nld=nld,
                   spincut=spincut,
                   parity_distribution=parity_distribution,
                   spin_density=spin_density,
                   nld_at_sn_parameters=nld_at_sn_parameters)

    def partial_nld(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """ Partial level density function

        rho(Ex, J, pi) = rho(Ex) Pi(Ex, pi) g(J, sigma^2(Ex))

        The doc of the created function:

            Parameters
            ----------
            Ex : np.ndarray
                The excitation energies.
            J : np.ndarray
                The spins.
            pi : np.ndarray
                The parities.

            Returns
            -------
            np.ndarray
                The partial level density function.
        
        Returns
        -------
        Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
            The partial level density function.
        """
        return lambda Ex, J, pi: density(Ex, J, pi, self.nld, self.spincut,
                                self.parity_distribution, self.spin_density)

    def rasterize_continuous(self, Ex: np.ndarray, J: np.ndarray, pi: np.ndarray | None = None) -> ContinuousLevels:
        """ Rasterize the continuous level density function

        Parameters
        ----------
        Ex : np.ndarray
            The excitation energies.
        J : np.ndarray
            The spins.
        pi : np.ndarray | None
            The parities. Default is all parities.

        Returns
        -------
        ContinuousLevels
            The continuous level density function.
        """
        if pi is None:
            pi = np.array([0, 1])
        return ContinuousLevels.from_density(Ex=Ex, J=J, pi=pi, density=self.partial_nld())

    def make_level_scheme(self, Ex: np.ndarray, J: np.ndarray, pi: np.ndarray,
                          e_crit: float) -> LevelScheme:
        # The bins of the rasterization and the discrete levels
        # are independent, since they are simulated differently.
        # We also need to sample the rasterization to get a realization.
        Ex = np.atleast_1d(Ex)
        J = np.atleast_1d(J)
        pi = np.atleast_1d(pi)

        discrete_J = self.discrete.J.max()
        if J.max() > discrete_J:
            warnings.warn(f"Maximum spin {J.max()} is larger than the maximum spin {discrete_J} in the discrete levels.\n"
                          "If this is not intentional, drop the discrete levels with higher J before calling this method.")
        continuous = self.rasterize_continuous(Ex, J, pi)
        return LevelScheme(discrete=self.discrete, continuous=continuous,
                           e_crit=e_crit)

    def plot(self, *, ax: plt.Axes | None = None, emin: float | None = None, emax: float | None = None,
             Ex: np.ndarray | None = None, 
             discrete_kwargs: dict | None = None, continuous_kwargs: dict | None = None,
             sn_kwargs: dict | None = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        if emin is None:
            emin = self.discrete.Ex.min() #0.9*self.discrete.Ex.max()
        if emax is None:
            if self.nld_at_sn_parameters is None:
                emax = 2*self.discrete.Ex.max()
            else:
                emax = 1.1*self.Sn
        if Ex is None:
            Ex = np.linspace(emin, emax, 1000)
        # Histogram of discrete levels
        kw = ({} if discrete_kwargs is None else {}) | {}
        self.discrete.plot_hist(ax=ax, **kw)
        ax.set_title(f'Discrete level density with {self.nld.__class__.__name__}')

        # Level density function is the curve

        kw = ({} if continuous_kwargs is None else {}) | {}
        nld = self.nld(Ex)
        ax.plot(Ex, nld, **kw)

        # nld(Sn) given D0
        kw = ({} if sn_kwargs is None else {}) | {}
        if self.nld_at_sn_parameters is not None:
            nld_at_Sn = self.nld_at_Sn_from_D0()
            ax.scatter(self.Sn, nld_at_Sn, **kw)

        ax.set_ylabel(r'$\rho(E)$')
        ax.set_xlabel(r'$E [MeV]$')
        ax.set_yscale('log')
        return ax

    @property
    def Sn(self) -> float | None:
        return self.nld_at_sn_parameters.Sn if self.nld_at_sn_parameters is not None else None

    @property
    def D0(self) -> float | None:
        return self.nld_at_sn_parameters.D0 if self.nld_at_sn_parameters is not None else None

    @property
    def Jtarget(self) -> float | None:
        return self.nld_at_sn_parameters.Jtarget if self.nld_at_sn_parameters is not None else None



    def nld_at_Sn_from_D0(self) -> float | None:
        if self.nld_at_sn_parameters is None:
            return None
        sigma2 = self.spincut(self.Sn)
        g = lambda j: self.spin_density(j, sigma2)
        return nld_at_Sn_from_D0(self.D0, self.Jtarget, g)

