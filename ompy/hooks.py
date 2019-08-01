import inspect
from .matrix import Matrix
from typing import Callable, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def plot_hook(func, condition: Callable[[Any], bool] = lambda i: True):
    """ Hook which plots all plotable returned values

    Args:
        condition: Is checked at each call. If true,
            performs plotting. The function call is
            always performed. The single argument is
            the current call number. Starts at 0.
    """
    def inner(*args, **kwargs):
        res = func(*args, **kwargs)

        if not condition(inner.calls):
            inner.calls += 1
            return res

        names = return_var_names(func)
        for i, item in enumerate(res):
            if isinstance(item, Matrix):
                item.plot()
            elif isinstance(item, np.ndarray):
                fig, ax = plt.subplots()
                if item.sum() > 1000:
                    norm = LogNorm()
                else:
                    norm = Normalize()

                if item.ndim == 1:
                    ax.plot(item)
                elif item.ndim == 2:
                    mesh = ax.pcolormesh(item, norm=norm)
                    ax.set_title(names[i])
                    fig.colorbar(mesh, ax=ax)
                else:
                    continue
        inner.calls += 1
        return res
    inner.calls = 0
    return inner


def return_var_names(func: Any) -> List[str]:
    """ Returns the variable names of each variable in the return
    """
    return_vars = inspect.getsourcelines(func)[0][-1]
    return_vars = return_vars.strip().rstrip().split(' ')[1:]
    return_vars = " ".join(return_vars).split(',')
    return_vars = [var.strip().rstrip() for var in return_vars]
    return return_vars
