from .stubs import Axes
from typing import Callable
import matplotlib.pyplot as plt


def make_axes(func: Callable[..., Axes]) -> Callable[..., Axes]:
    def wrapper(*args, ax: Axes | None = None, **kwargs) -> Axes:
        if ax is None:
            _, ax = plt.subplots()
        assert ax is not None
        result = func(*args, ax=ax, **kwargs)
        return result
    return wrapper
