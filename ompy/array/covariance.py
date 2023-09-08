from .matrix import Matrix
from ..stubs import Axes, Figure, QuadMesh, Colorbar
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


class CorrelationMatrix(Matrix):
    def plot(self, ax: Axes | None = None,
             **kwargs) -> tuple[Axes, tuple[QuadMesh, Colorbar | None]]:
        """ Plots the matrix with the energy along the axis

        Args:
            ax: A matplotlib axis to plot onto
            title: Defaults to the current matrix state
            scale: Scale along the z-axis. Can be either "log"
                or "linear". Defaults to logarithmic
                if number of counts > 1000
            vmin: Minimum value for coloring in scaling
            vmax Maximum value for coloring in scaling
            add_cbar: Whether to add a colorbar. Defaults to True.
            **kwargs: Additional kwargs to plot command.

        Returns:
            The ax used for plotting

        Raises:
            ValueError: If scale is unsupported
        """
        if ax is None:
            fig, ax = plt.subplots()
        kw = dict(cmap='RdBu_r', vmin=-1, vmax=1) | kwargs
        return super().plot(ax=ax, **kw)
