import inspect
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Iterable, TypeVar
from typing import Callable
from warnings import warn

import matplotlib as mpl
import matplotlib.colorbar as cbar
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib.colors import Normalize, BoundaryNorm, to_rgba
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde
from matplotlib.widgets import Cursor, SpanSelector
from ipywidgets import Button, VBox
from IPython.display import display

from .stubs import Axes, Line2D

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
            if isinstance(line, (tuple, list)):
                labels.append(line[0].get_label())
                if len(line) > 1:
                    handles.append(line)
                else:
                    handles.append(line[0])
            else:
                print(line, line.get_label())
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
            ret.append(getattr(ax, 'set_' + attr)(value))
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
            # BUG: There is a weird bug here where Path doesn't equal itself.
            if annotation.annotation == Path or str(annotation.annotation) == 'Path':
                bound_args.arguments[param] = Path(bound_args.arguments[param])
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


class ColorScheme:
    def __init__(self, iterable, cmap='viridis', padding=0.1):
        self.iterable = list(iterable)
        self.cmap_name = cmap
        self.padding = padding
        self.colors = self._generate_colors(cmap, padding)
        self.value_to_index = {value: idx for idx, value in enumerate(self.iterable)}
        self._cmap = None
        self._norm = None

    def _generate_colors(self, cmap_name, padding):
        cmap = plt.get_cmap(cmap_name)
        num_colors = len(self.iterable)
        color_values = np.linspace(padding, 1, num_colors)
        return [cmap(value) for value in color_values]

    def __getitem__(self, index):
        return self.colors[index]

    def __call__(self, value):
        idx = self.value_to_index.get(value)
        if idx is not None:
            return self.colors[idx]
        else:
            raise KeyError(f"Value {value} not found in the iterable")

    def mappable(self, discrete=False):
        norm = Normalize(vmin=min(self.iterable), vmax=max(self.iterable))
        cmap = plt.get_cmap(self.cmap_name)

        if discrete:
            # Create discrete color levels
            boundaries = np.linspace(min(self.iterable), max(self.iterable), len(self.iterable) + 1)
            norm = BoundaryNorm(boundaries, cmap.N, clip=True)
        self._cmap = cmap
        self._norm = norm
        sm = ScalarMappable(norm=norm, cmap=self.cmap_name)
        sm.set_array([])
        return sm

    def set_ticks(self, cbar, adjust_color=True):
        # Create discrete color levels
        boundaries = np.linspace(min(self.iterable), max(self.iterable), len(self.iterable) + 1)
        # Set the ticks to be in the middle of each segment
        tick_locs = (boundaries[:-1] + boundaries[1:]) / 2
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(self.iterable)
        if adjust_color:
            self._adjust_tick_colors(cbar)
            cbar.ax.figure.canvas.draw()

    def _adjust_tick_colors(self, cbar):
        for tick in cbar.ax.get_yticklines():
            color = self._get_color_for_tick(tick)
            tick.set_markeredgecolor(color)

    def _get_color_for_tick(self, tick):
        tick_value = tick.get_ydata()[0]
        color = self._cmap(self._norm(tick_value))
        r, g, b, _ = to_rgba(color)
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        # Calculate the gray level based on luminance
        return 'white' if luminance < 0.5 else 'black'




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
    
    TODO:
        `draw_kde` works, but handles extreme values poorly.
        Drawing fails when it is done through a callback (such as `mappable.callback`). I have
        no idea why, debugging is difficult. 
        The colorbar should update when the user zooms on the main matrix.
    """

    def __init__(self, mappable, ax, cax=None, lower: bool = True, higher: bool = True, nans: bool = True,
                 use_gridspec=True, linewidth=1, extend=None, color_by: str = 'complement', draw_kde: bool = False,
                 draw_histogram: bool = True, n_hist_bins: int = 100,
                 hist_norm=np.log10,  **kwargs):
        # Code copied from matplotlib.Figure.colorbar
        if cax is None:
            fig = (  # Figure of first axes; logic copied from make_axes.
                [*ax.flat] if isinstance(ax, np.ndarray) else [*ax] if np.iterable(ax) else [ax])[0].figure
            current_ax = fig.gca()
            if (fig.get_layout_engine() is not None and not fig.get_layout_engine().colorbar_gridspec):
                use_gridspec = False
            if (use_gridspec and isinstance(ax, mpl.axes._base._AxesBase) and ax.get_subplotspec()):
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
        self.n_hist_bins = n_hist_bins
        self.hist_norm = hist_norm
        # Super can call methods defined later, so the attributed must already be defined
        super().__init__(cax, mappable, **kwargs)
        cb = cbar.Colorbar(cax, mappable, **{k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS})
        self.annotate(mappable)
        cax.figure.stale = True  # ax.callbacks.connect('xlim_changed', lambda ev: print(ev))  # ax.callbacks.connect('ylim_changed')

    def annotate(self, mappable):
        if self.do_draw_kde:
            try:
                self.draw_kde(mappable)
            except Exception as e:
                raise e
        if self.do_draw_histogram:
            try:
                self.draw_histogram(mappable)
            except Exception as e:
                raise e
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
        x = norm.inverse(x)  # [0, 1] -> [vmin, vmax] # this is false
        y = kde(x)
        y /= y.max()
        y *= 0.7  # move it slightly away from the edges
        y = 1 - y  # To make it look like it follows the ticks

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
        data = mappable.get_array()
        norm = self.norm

        # We don't want outliers to affect the histogram
        # This can conflict with the vmin, vmax of the main plot
        # Should be, given data in the range [vmin, vmax] of the main plot
        vmin, vmax = IQR_range(data, factor=1.5)
        vmin = max(vmin, norm.vmin)
        vmax = min(vmax, norm.vmax)

        y = data[np.isfinite(data)].ravel()
        y = y[(y > vmin) & (y < vmax)]

        # Create histogram
        bins = np.linspace(0, 1, self.n_hist_bins+1)
        bins = norm.inverse(bins)
        counts, bin_edges = np.histogram(y, bins=bins, density=True)

        # width = 1/self.n_hist_bins
        #norm_edges = norm(bin_edges)
        norm_edges = bin_edges

        # Choose color map
        if self.color_by == 'complement':
            cmap = cmap_complement(self.cmap)
        elif self.color_by == 'contrast':
            cmap = cmap_contrast(self.cmap)
        else:
            cmap = self.color_by
        assert isinstance(cmap, mcolors.Colormap)

        for count, left, right in zip(counts, norm_edges[:-1], norm_edges[1:]):
            # Height of the rectangle is proportional to the count, scaled down to fit within the colorbar
            height = count / counts.max() * 0.7  # Scale factor for height; adjust as needed
            width = right - left
            xpos = 1 - height
            #xpos = self.hist_norm(xpos)

            # Create and add the rectangle patch to the colorbar's axis
            rect = Rectangle((xpos, left), width=height, height=width, color=cmap(norm(left + width/2)))
            self.ax.add_patch(rect)

        self.set_text(mappable)

    def set_text(self, mappable):
        # This can be called by super()
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
                text = f'{lower * 100:.1f}%'
                if self.lower_text is not None:
                    self.lower_text.set_text(text)
                else:
                    self.lower_text = self.ax.text(0.5, -0.05, text, transform=self.ax.transAxes, ha='center', va='top')
            elif self.lower_text is not None:
                self.lower_text.set_text('')
        if self.higher:
            if higher > 0:
                text = f'{higher * 100:.1f}%'
                if self.higher_text is not None:
                    self.higher_text.set_text(text)
                else:
                    self.higher_text = self.ax.text(0.5, 1.05, text, transform=self.ax.transAxes, ha='center',
                                                    va='bottom')
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
        # self.set_text(mappable)
        self.annotate(mappable)


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


def mc_boxplot(X: np.ndarray, Y: np.ndarray, ax: Axes | None = None,
               scale: str | None = None,
               **kwargs) -> tuple[Axes, Any]:
    ax = make_ax(ax)
    match scale:
        case 'log':
            X = np.log10(X)
    width = 0.5
    default = dict(widths=width)
    kwargs = default | kwargs
    lines = ax.boxplot(Y, positions=X, **kwargs)
    return ax, lines


def mc_errplot(X: np.ndarray, Y: np.ndarray, ax: Axes | None = None,
               mean: bool = True,
               median: bool = True,
               fill: bool = False,
               bars: bool = True,
               fillkwargs = None,
               errkwargs = None,
               **kwargs) -> tuple[Axes, Any]:
    ax = make_ax(ax)
    plot_mean = mean
    plot_median = median

    # Calculate mean, median and standard deviation
    mean = np.mean(Y, axis=0)
    median = np.median(Y, axis=0)
    std = np.std(Y, axis=0)

    lines = []
    # Plot mean and median
    if plot_mean:
        l0 = ax.plot(X, mean, label='mean')
        lines.append(l0)

    if plot_median:
        l1 = ax.plot(X, median, label='median')
        lines.append(l1)

    if fill:
        default = dict(alpha=0.2)
        fillkwargs = {} if not fillkwargs else fillkwargs
        kwargs = default | fillkwargs
        l = ax.fill_between(X, mean - std, mean + std, **kwargs)
        lines.append(l)

    # Plot error bars
    if bars:
        default = dict()
        errkwargs = {} if not errkwargs else errkwargs
        kwargs = default | errkwargs
        l2 = ax.errorbar(X, mean, yerr=std, **kwargs)
        lines.append(l2)

    return ax, lines



def setup_interactive_plot(matrix, ax, enable_button=True, enable_hotkey=True, hotkey='a'):
    """
    Sets up an interactive plot with a matrix, allowing selection of a range along the x-axis to sum columns.

    Parameters:
    - matrix: 2D numpy array to be plotted
    - ax: Matplotlib axis to which the matrix is already plotted
    - enable_button: Boolean, if True a button will be displayed to activate the range selector
    - enable_hotkey: Boolean, if True a hotkey can be used to activate the range selector
    - hotkey: Character, the key to be used as a hotkey for activating the range selector
    """

    # Function to sum and plot selected columns
    def plot_selected_columns(x_min, x_max):
        x_min, x_max = int(np.floor(x_min)), int(np.ceil(x_max))
        x_min = max(x_min, 0)
        x_max = min(x_max, matrix.shape[1])

        selected_region = matrix[:, x_min:x_max]
        summed_values = np.sum(selected_region, axis=1)

        plt.figure(figsize=(10, 4))
        plt.plot(summed_values, marker='o')
        plt.title(f"Summed Values from Column {x_min} to {x_max}")
        plt.xlabel("Y-axis")
        plt.ylabel("Sum")
        plt.grid(True)
        plt.show()

    # Span selector
    def on_select(x_min, x_max):
        plot_selected_columns(x_min, x_max)

    span_selector = SpanSelector(ax, on_select, 'horizontal', useblit=True, minspan=1, interactive=True)

    # Function to activate selector via button or hotkey
    def activate_selector(event=None):
        span_selector.set_active(True)
        print(f"Range selector activated by {('button' if event else 'hotkey')}")

    if enable_hotkey:
        def on_key(event):
            if event.key == hotkey:
                activate_selector()
        ax.figure.canvas.mpl_connect('key_press_event', on_key)

    # Button to activate the selector
    if enable_button:
        activate_button = Button(description="Activate Range Selector")
        activate_button.on_click(activate_selector)
        display(VBox([activate_button]))
