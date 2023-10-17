from .stubs import Axes, Line2D
from typing import Callable, Any, Iterable, TypeVar
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from functools import wraps
from pathlib import Path
from typing import Callable
import inspect
from datetime import timedelta
from warnings import warn
import numpy as np

T = TypeVar('T')
def make_axes(func: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, ax: Axes | None = None, **kwargs) -> T:
        if ax is None:
            _, ax_ = plt.subplots()
            assert isinstance(ax_, Axes)
            ax = ax_
        assert ax is not None
        result = func(*args, ax=ax, **kwargs)
        return result
    return wrapper

def make_ax(ax: Axes | None = None) -> Axes:
    if ax is None:
        _, ax_ = plt.subplots()
        assert isinstance(ax_, Axes)
        ax = ax_
    return ax


def combine_legend(headings: list[tuple[str, Line2D | Iterable[Line2D | Iterable[Line2D]]]], *misc):
    """ Combine legend entries from multiple subplots

    The headings have the format
    [(title1, line1, line2, ...), (title2, line1, line2, ...), optional_untitled_lines]
    where line_i is a Line2D or a nested iterable of Line2D objects.
    """
    handles = []
    labels = []
    for name, *line_collection in headings:
        # The string is handled by LegendTitle
        handles.append(name)
        # The associated label is discarded
        labels.append('')
        # Add all other lines under this subheading
        rec_add(handles, labels, line_collection)
    if misc is not None:
        for line in misc:
            if isinstance(line, tuple):
                labels.append(line[0].get_label())
                handles.append(line)
            else:
                labels.append(line.get_label())
                handles.append(line)
    return handles, labels

def rec_add(handles, labels, collection):
    """ Recursively add legend entries to handles and labels

    Walks down nested lists and adds the Line2D objects it finds. Tuples
    are treated as a single entry.
    """
    for elem in collection:
        if isinstance(elem, tuple):
            labels.append(elem[0].get_label())
            handles.append(elem)
        else:
            if hasattr(elem, 'get_label'):
                labels.append(elem.get_label())
                handles.append(elem)
            else:
                rec_add(handles, labels, elem)

def make_combined_legend(ax, headings, *misc, **kwargs):
    handles, labels = combine_legend(headings, *misc)
    handler_map = {str: LegendTitle()} | kwargs.pop('handler_map', {})
    return ax.legend(handles, labels, handler_map=handler_map, **kwargs)


class LegendTitle:
    def __init__(self, text_props=None):
        self.text_props = text_props or {}

    def legend_artist(self, legend, orig_handle, fontsize: int, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title

def maybe_set(ax: Axes, **kwargs) -> list[Any]:
    ret = []
    for attr, value in kwargs.items():
        old = getattr(ax, 'get_' + attr)()
        if not old:
            ret.append(getattr(ax, 'set_'+attr)(value))
    return ret

def ensure_path(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the function's signature
        sig = inspect.signature(func)
        # Convert args to kwargs for easier handling
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Check and modify kwargs
        for param, annotation in sig.parameters.items():
            if annotation.annotation == Path:
                value = bound_args.arguments[param]
                match value:
                    case Path():
                        pass
                    case str():
                        bound_args.arguments[param] = Path(value)
                    case _:
                        raise TypeError(f"Argument {param} must be a Path or a string")
        return func(*bound_args.args, **bound_args.kwargs)
    return wrapper


def print_readable_time(elapsed) -> None:
    print(readable_time(elapsed))

def readable_time(elapsed) -> str:
    delta = timedelta(seconds=elapsed)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds, microseconds = divmod(delta.microseconds, 1000)

    time_units = []
    if hours > 0:
        time_units.append(f"{int(hours)}h")
    if minutes > 0:
        time_units.append(f"{int(minutes)}m")
    if seconds > 0:
        time_units.append(f"{int(seconds)}s")
    if milliseconds > 0:
        time_units.append(f"{int(milliseconds)}ms")
    if microseconds > 0 or not time_units:
        time_units.append(f"{microseconds}µs")

    formatted_time = " ".join(time_units)
    return formatted_time


def bytes_to_readable(num):
    """Convert bytes to a human-readable string."""
    for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"


def warn_memory(memory, msg: str, limit=None):
    limit = 10e9 if limit is None else limit
    if memory > limit:
        warn(f"Memory usage of `{msg}` is {bytes_to_readable(memory)}", RuntimeWarning)


def estimate_memory_usage(shape, dtype=np.float64):
    """Estimate memory usage of a numpy array before allocation."""
    return np.dtype(dtype).itemsize * np.prod(shape)


def append_label(label: str, kwargs: dict[str, Any] | str | None = None) -> str:
    if isinstance(kwargs, str):
        return kwargs + ': ' + label
    if kwargs is None:
        return label
    if 'label' in kwargs:
        label = kwargs.pop('label') + ': ' + label
    return label


