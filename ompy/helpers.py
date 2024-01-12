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
from scipy.stats import gaussian_kde
from matplotlib.colorbar import Colorbar
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import matplotlib.colorbar as cbar
import matplotlib as mpl
from time import time


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

def make_ax(ax: Axes | None = None, **kwargs) -> Axes:
    if ax is None:
        _, ax = plt.subplots(**kwargs)
        return ax
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

def is_running_in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True


class ReverseNormalization(mcolors.Normalize):
    def __init__(self, original_norm):
        # Initialize with the original normalization's vmin and vmax
        super().__init__(vmin=original_norm.vmin, vmax=original_norm.vmax)
        self.original_norm = original_norm

    def __call__(self, value, clip=None):
        # Normalize using the original normalization, then reverse
        return 1 - self.original_norm(value, clip)

def cmap_complement(cmap):
    """
    Create a complementary colormap based on the given colormap.

    :param cmap: The original colormap (matplotlib.colors.Colormap instance).
    :return: A new colormap with complementary colors.
    """
    # Evaluate the original colormap
    c = cmap(np.arange(256))

    # Calculate the complementary colors
    complementary_colors = 1 - c[:, :3]  # Invert RGB, keep alpha unchanged
    complementary_colors = np.hstack((complementary_colors, c[:, -1:]))  # Reattach alpha channel

    # Create a new colormap from the complementary colors
    return mcolors.ListedColormap(complementary_colors)

def cmap_contrast(cmap):
    """
    Create a contrast-maximized grayscale colormap based on the given colormap.

    :param cmap: The original colormap (matplotlib.colors.Colormap instance).
    :return: A new grayscale colormap with maximized contrast.
    """
    # Evaluate the original colormap
    colors = cmap(np.arange(256))

    # Convert RGB to grayscale using luminosity method
    grayscale_values = np.dot(colors[:, :3], [0.2989, 0.5870, 0.1140])

    # Stretch the grayscale values to cover the full range [0, 1]
    min_gray = np.min(grayscale_values)
    max_gray = np.max(grayscale_values)
    stretched_gray = 1 - (grayscale_values - min_gray) / (max_gray - min_gray)

    # Create a new colormap from the stretched grayscale values
    new_colors = np.vstack((stretched_gray, stretched_gray, stretched_gray, colors[:, 3])).T
    return mcolors.ListedColormap(new_colors)


class AnnotatedColorbar(Colorbar):
    """ A colorbar with annotations

    This class is a wrapper around matplotlib.colorbar.Colorbar that adds
    annotations to the colorbar. The annotations are the percentage of values
    that are outside the colorbar limits.

    It handles the `extend` by setting it depending on values outside the
    limits.
    """
    def __init__(self, mappable, ax, cax=None, lower: bool = True,
                 higher: bool = True, nans: bool = True, use_gridspec=True,
                 linewidth=1, extend=None, color_by: str = 'complement',
                 draw_kde: bool = True, draw_histogram: bool = False,
                 **kwargs):
        # Code copied from matplotlib.Figure.colorbar
        if cax is None:
            fig = (  # Figure of first axes; logic copied from make_axes.
                [*ax.flat] if isinstance(ax, np.ndarray)
                else [*ax] if np.iterable(ax)
                else [ax])[0].figure
            current_ax = fig.gca()
            if (fig.get_layout_engine() is not None and
                    not fig.get_layout_engine().colorbar_gridspec):
                use_gridspec = False
            if (use_gridspec
                    and isinstance(ax, mpl.axes._base._AxesBase)
                    and ax.get_subplotspec()):
                cax, kwargs = cbar.make_axes_gridspec(ax, **kwargs)
            else:
                cax, kwargs = cbar.make_axes(ax, **kwargs)
            # make_axes calls add_{axes,subplot} which changes gca; undo that.
            fig.sca(current_ax)
            cax.grid(visible=False, which='both', axis='both')
        NON_COLORBAR_KEYS = [  # remove kws that cannot be passed to Colorbar
            'fraction', 'pad', 'shrink', 'aspect', 'anchor', 'panchor']

        # Set extends
        if extend is None:
            array = mappable.get_array()
            any_lower = np.any(array < mappable.norm.vmin)
            any_higher = np.any(array > mappable.norm.vmax)
            if any_lower and any_higher:
                extend = 'both' 
            elif any_lower:
                extend = 'min'
            elif any_higher:
                extend = 'max'
        kwargs['extend'] = extend
        
        super().__init__(cax, mappable, **kwargs)
        cb = cbar.Colorbar(cax, mappable, **{
            k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS})
        self.lower = lower
        self.higher = higher
        self.nans = nans
        self.linewidth = linewidth
        self.lower_text = None
        self.higher_text = None
        self.nans_text = None
        self.color_by = color_by
        self.do_draw_kde = draw_kde
        self.do_draw_histogram = draw_histogram
        self.annotate(mappable)
        cax.figure.stale = True

    def annotate(self, mappable):
        if self.do_draw_kde:
            try:
                self.draw_kde(mappable)
            except:
                pass
        if self.do_draw_histogram:
            try :
                self.draw_histogram(mappable)
            except:
                pass
        self.set_text(mappable)

    def draw_kde(self, mappable):
        data = mappable.get_array()
        norm = self.norm

        # We don't want outliers to affect the KDE
        vmin, vmax = IQR_range(data, factor=1.5)

        y = data[np.isfinite(data)].ravel()
        y = y[(y > vmin) & (y < vmax)]
        kde = gaussian_kde(y)
        x = np.linspace(0, 1, 100)
        x = norm.inverse(x)  # [0, 1] -> [vmin, vmax]
        y = kde(x)
        y /= y.max()
        y*= 0.7  # move it slightly away from the edges
        y = 1-y  # To make it look like it follows the ticks

        points = np.array([y, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        if self.color_by == 'complement':
            cmap = cmap_complement(self.cmap)
        elif self.color_by == 'contrast':
            cmap = cmap_contrast(self.cmap)
        else:
            cmap = self.color_by
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=self.linewidth)
        lc.set_array(x)

        self.ax.add_collection(lc)
        self.set_text(mappable)

    def draw_histogram(self, mappable):
        # TODO: Doesn't work
        data = mappable.get_array()
        norm = self.norm

        # We don't want outliers to affect the histogram
        vmin, vmax = IQR_range(data, factor=1.5)

        y = data[np.isfinite(data)].ravel()
        y = y[(y > vmin) & (y < vmax)]

        y_norm = norm(y)

        # Create histogram
        hist, bin_edges = np.histogram(y_norm, bins=200, range=(0, 1),
                                       density=True)
        hist = hist.astype(float)
        hist /= hist.max()  # Normalize the histogram
        hist *= 0.7
        hist = 1 - hist

        # Create points for line collection
        x = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute midpoints of bins
        x = norm.inverse(x)
        points = np.array([hist, x]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Choose color map
        if self.color_by == 'complement':
            cmap = cmap_complement(self.cmap)
        elif self.color_by == 'contrast':
            cmap = cmap_contrast(self.cmap)
        else:
            cmap = self.color_by

        # Create line collection and add to the axis
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=self.linewidth)
        lc.set_array(x)

        self.ax.add_collection(lc)
        self.set_text(mappable)


    def set_text(self, mappable):
        data = mappable.get_array()

        mask = np.isfinite(data)
        nans = np.sum(~mask)

        # If it is a masked array, let it decide
        # "good" bins. Else exclude nans and 0.
        if isinstance(data, np.ma.MaskedArray):
            data = data.compressed()
        else:
            data = data[mask & (data != 0)]
        N = data.size
        
        lower = (data < self.norm.vmin).sum() / N
        higher = (data > self.norm.vmax).sum() / N

        # This mess sets the text if necessary, and creates it if it doesn't exist
        if self.lower:
            if lower > 0:
                text = f'{lower*100:.1f}%'
                if self.lower_text is not None:
                    self.lower_text.set_text(text)
                else:
                    self.lower_text = self.ax.text(0.5, -0.05, text, transform=self.ax.transAxes, ha='center', va='top')
            elif self.lower_text is not None:
                self.lower_text.set_text('')
        if self.higher:
            if higher > 0:
                text = f'{higher*100:.1f}%'
                if self.higher_text is not None:
                    self.higher_text.set_text(text)
                else:
                    self.higher_text = self.ax.text(0.5, 1.05, text, transform=self.ax.transAxes, ha='center', va='bottom')
            elif self.higher_text is not None:
                self.higher_text.set_text('')
        if self.nans:
            if nans > 0:
                text = f'{nans} nan'
                if self.nans_text is not None:
                    self.nans_text.set_text(text)
                else:
                    self.nans_text = self.ax.text(0.5, -0.10, text, transform=ax.transAxes, ha='center', va='top')
            elif self.nans_text is not None:
                self.nans_text.set_text('')

        self.ax.figure.stale = True

    def update_normal(self, mappable):
        super().update_normal(mappable)
        self.set_text(mappable)


def IQR_range(data, factor=1.5) -> tuple[float, float]:
    # Calculate the IQR
    if isinstance(data, np.ma.masked_array):
        data = data.compressed()
    else:
        data = data[np.isfinite(data)]

    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    
    # Define color limits to be a certain factor beyond the IQR
    vmin = q25 - factor * iqr
    vmax = q75 + factor * iqr
    vmin = max(vmin, data.min())
    vmax = min(vmax, data.max())
    return vmin, vmax


def robust_z_score(data):
    """
    Calculate the robust Z-scores for a given dataset.

    :param data: A list or numpy array of data points.
    :return: A numpy array of robust Z-scores.
    """
    # Calculate the median of the data
    median = np.median(data)

    # Calculate the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(data - median))

    # To avoid division by zero, add a small epsilon where MAD is zero
    mad = mad if mad else np.finfo(data.dtype).eps

    # Calculate the robust Z-scores
    robust_z_scores = (data - median) / mad

    return robust_z_scores


def robust_z_score_i(z, data):
    """ Invert the robust Z-score transformation

    Return the data value of z, given the data.
    """
    mad = np.median(np.abs(data - np.median(data)))
    mad = mad if mad else np.finfo(data.dtype).eps
    return z * mad + np.median(data)

