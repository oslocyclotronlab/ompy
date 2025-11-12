import matplotlib.pyplot as plt
from .stubs import DiscreteLevels, Levels
import numpy as np
from matplotlib.colors import LogNorm
import xarray as xr


def is_notebook():
    try:
        # This will succeed if running under IPython
        shell = get_ipython().__class__.__name__
        # ZMQInteractiveShell is used by Jupyter Notebook and qtconsole.
        if shell == 'ZMQInteractiveShell':
            return True  # Running in a notebook
        elif shell == 'TerminalInteractiveShell':
            return False  # Running in a terminal (IPython)
        else:
            return False  # Other type (possibly still not a notebook)
    except NameError:
        # get_ipython() is not defined, so definitely not in an IPython environment.
        return False


def plot_compare(discrete: DiscreteLevels, constructed: Levels, ax=None,
                 cumsum: bool = False, **kwargs):
    constructed = constructed.sum(['J', 'pi'])
    edges = constructed.Ex.values
    discrete, _ = np.histogram(discrete.Ex, bins=edges)
    if cumsum:
        constructed = constructed.cumsum('Ex')
        discrete = np.cumsum(discrete)

    if ax is None:
        fig, ax = plt.subplots()
    kw = kwargs | dict(where='mid', lw=0.5)
    ax.step(edges, constructed, label='Constructed NLD', **kw)
    ax.step(edges[:-1], discrete, label='Discrete NLD', **kw)
    ax.set_yscale('log')
    ax.set_ylabel('cumulative nld' if cumsum else 'nld')
    ax.legend()
    return ax


def plot_scheme(levels: DiscreteLevels, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    kw = kwargs | dict(color='k')
    if isinstance(levels, DiscreteLevels):
        for (_, level) in levels.iterrows():
            ax.axhline(level.Ex, **kw)
    ax.set_ylabel(r'$E_x$')
    return ax


def plot_constructed_density(constructed: Levels, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    # Shortened variable names for clarity and brevity
    df = constructed.to_dataframe(name="nld").reset_index()
    df['parity'] = df['pi'].map({'+': 1, '-': -1})
    df['sp'] = df['J'] * df['parity']  # spin_parity

    # Pivot the data for pcolormesh
    pv_data = df.pivot_table(index='Ex', columns='sp', values='nld', aggfunc='sum', fill_value=0)

    # Generate the meshgrid using the index and columns of the pivoted data
    X, Y = np.meshgrid(pv_data.columns, pv_data.index)

    # Plot the pcolormesh with logarithmic color scale
    pcm = ax.pcolormesh(X, Y, pv_data, norm=LogNorm(), shading='auto')
    ax.set_xlabel(r'$J\pi$')
    ax.set_ylabel('$E_x$ (MeV)')
    fig.colorbar(pcm, ax=ax, label='Level Density')
    fig.tight_layout()
    return ax