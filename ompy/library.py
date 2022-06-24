import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.stats import truncnorm
import matplotlib
from itertools import product
from typing import Optional, Tuple, Iterator, Any, Union
import inspect
import re
from matplotlib import patches
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from .header import ArrayFloat, arraylike, Unitlike, ArrayKeV


def div0(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    # Check whether a or b (or both) are numpy arrays. If not, we don't
    # use the fancy function.
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~ np.isfinite(c )] = 0  # -inf inf NaN
    else:
        if b == 0:
            c = 0
        else:
            c = a / b
    return c


def i_from_E(E, E_array):
    # Returns index of the E_array value closest to given E
    assert(E_array.ndim == 1), "E_array has more then one dimension"
    return np.argmin(np.abs(E_array - E))


def line(x, points):
    """ Returns a line through coordinates [x1,y1,x2,y2]=points
    """
    a = (points[3]-points[1])/float(points[2]-points[0])
    b = points[1] - a*points[0]
    return a*x + b


def make_mask(Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2):
    # Make masking array to cut away noise below Eg=Ex+dEg diagonal
    # Define cut   x1    y1    x2    y2
    cut_points = [i_from_E(Eg1, Eg_array), i_from_E(Ex1, Ex_array),
                  i_from_E(Eg2, Eg_array), i_from_E(Ex2, Ex_array)]
    i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis
    j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
    return np.where(i_mesh > line(j_mesh, cut_points), 1, 0)


def E_array_from_calibration(a0: float, a1: float, *,
                             N: Optional[int] = None,
                             E_max: Optional[float] = None) -> np.ndarray:
    """
    Return an array of mid-bin energy values corresponding to the
    specified calibration.

    Args:
        a0, a1: Calibration coefficients; E = a0 + a1*i
        either
            N: Number of bins
        or
            E_max: Max energy. Array is constructed to ensure last bin
                 covers E_max. In other words,
                 E_array[-1] >= E_max - a1
    Returns:
        E: Array of mid-bin energy values
    """
    if E_max is not None and N is not None:
        raise ValueError("Cannot give both N and E_max -- must choose one")

    if N is not None:
        return np.linspace(a0, a0+a1*(N-1), N)
    elif E_max is not None:
        N = int(np.round((E_max - a0)/a1)) + 1
        return np.linspace(a0, a0+a1*(N-1), N)
    else:
        raise ValueError("Either N or E_max must be given")


def fill_negative_max(array: np.ndarray,
                      window_size: Union[int, np.array]) -> np.ndarray:
    """
    Fill negative channels with positive counts from neighboring channels.

    The idea is that the negative counts are somehow connected to the (γ-ray)
    resolution and should thus be filled from a channel within the resolution.

    This implementation loops through the closest channels with maximum number
    of counts to fill the channle(s) with negative counts. Note that it can
    happen that some bins will remain with negative counts (if not enough bins
    with possitive counts are available within the window_size) .

    The routine is performed for each Ex row independently.

    Args:
        array: Input array, ordered as [Ex, Eg]
        window_size (Union[int, np.array]): Window_size, eg. FWHM. If `int`
            `float`, the same FWHM will be applied for all `Eg` bins.
            Otherwise, provide an array with the FWHM for each `Eg` bin.

    Returns:
        array with negative counts filled, where possible
    """
    if isinstance(window_size, int):
        window_size = np.full(array.shape[1], window_size)
    else:
        assert len(window_size) == array.shape[1], "Array length incompatible"
        assert window_size.dtype == np.integer, "Check input"

    array = np.copy(array)
    N_Ex = array.shape[0]
    N_Eg = array.shape[1]
    for i_Ex in range(N_Ex):
        row = array[i_Ex, :]
        for i_Eg in np.where(row < 0)[0]:
            window_size_Eg = window_size[i_Eg]
            max_distance = window_size_Eg
            max_distance = int(np.ceil((window_size_Eg - 1) / 2))
            i_Eg_low = max(0, i_Eg - max_distance)
            i_Eg_high = min(N_Eg, i_Eg + max_distance)
            while row[i_Eg] < 0:
                i_max = np.argmax(row[i_Eg_low:i_Eg_high + 1])
                i_max = i_Eg_low + i_max
                if row[i_max] <= 0:
                    break
                shuffle_counts(row, i_max, i_Eg)

    return array


def fill_negative_gauss(array: np.ndarray, Eg: np.ndarray,
                        window_size: Union[int, float, np.array],
                        n_trunc: float = 3) -> np.ndarray:
    """
    Fill negative channels with positive counts from weighted neighbor chnls.

    The idea is that the negative counts are somehow connected to the (γ-ray)
    resolution and should thus be filled from a channel within the resolution.

    This implementation loops through channels with the maximum "weight", where
    the weight is given by
        weight = gaussian(i, loc=i, scale ~ window_size) * counts(i),
    to fill the channle(s) with negative counts. Note that it can
    happen that some bins will remain with negative counts (if not enough bins
    with possitive counts are available within the window_size) .

    The routine is performed for each Ex row independently.

    Args:
        array: Input array, ordered as [Ex, Eg]
        Eg: Gamma-ray energies
        window_size: FWHM for the gaussian. If `int` or
            `float`, the same FWHM will be applied for all `Eg` bins.
            Otherwise, provide an array with the FWHM for each `Eg` bin.
        n_trun (float, optional): Truncate gaussian for faster calculation.
            Defaults to 3.

    Returns:
        array with negative counts filled, where possible
    """
    if isinstance(window_size, (int, float)):
        window_size = np.full(array.shape[1], window_size)
        sigma = window_size/2.355  # convert FWHM to sigma
    else:
        assert len(window_size) == array.shape[1], "Array length incompatible"
        sigma = window_size/2.355  # convert FWHM to sigma

    # generate truncated gauss for each Eg bin, format [Eg-bin, gauss-values]
    lower, upper = Eg - n_trunc*sigma, Eg + n_trunc*sigma
    a = (lower - Eg) / sigma
    b = (upper - Eg) / sigma
    gauss = [truncnorm(a=a, b=b, loc=p, scale=sig).pdf(Eg)
             for p, sig in zip(Eg, sigma)]
    gauss = np.array(gauss)

    array = np.copy(array)
    N_Ex = array.shape[0]
    for i_Ex in range(N_Ex):
        row = array[i_Ex, :]
        for i_Eg in np.nonzero(row < 0)[0]:
            positives = np.where(row < 0, 0, row)
            weights = positives * gauss[i_Eg, :]

            for i_from in np.argsort(weights):
                if row[i_from] < 0:
                    break
                shuffle_counts(row, i_from, i_Eg)
                if row[i_Eg] >= 0:
                    break

    return array


def shuffle_counts(row: np.ndarray, i_from: int, i_to: int):
    """Shuffles counts in `row` from bin `i_from` to `i_to`

    Transfers at maximum row[i_from] counts, so that row[i_from] cannot be
    negative after the shuffling.

    Note:
        Assumes that row[i_from] > 0 and row[i_to] < 0.

    Args:
        row: Input array
        i_from: Index of bin to take counts from
        i_to: Index of bin to fill

    """
    positive = row[i_from]
    negative = row[i_to]
    fill = min(0, positive + negative)
    rest = max(0, positive + negative)
    row[i_to] = fill
    row[i_from] = rest


def cut_diagonal(matrix, Ex_array, Eg_array, E1, E2):
        """
        Cut away counts to the right of a diagonal line defined by indices

        Args:
            matrix (np.ndarray): The matrix of counts
            Ex_array: Energy calibration along Ex
            Eg_array: Energy calibration along Eg
            E1 (list of two floats): First point of intercept, ordered as Ex,Eg
            E2 (list of two floats): Second point of intercept
        Returns:
            The matrix with counts above diagonal removed
        """
        Ex1, Eg1 = E1
        Ex2, Eg2 = E2
        mask = make_mask(Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2)
        matrix_out = np.where(mask, matrix, 0)
        return matrix_out


def interpolate_matrix_1D(matrix_in, E_array_in, E_array_out, axis=1):
    """ Does a one-dimensional interpolation of the given matrix_in along axis
    """
    if axis not in [0, 1]:
        raise IndexError("Axis out of bounds. Must be 0 or 1.")
    if not (matrix_in.shape[axis] == len(E_array_in)):
        raise Exception("Shape mismatch between matrix_in and E_array_in")

    if axis == 1:
        N_otherax = matrix_in.shape[0]
        matrix_out = np.zeros((N_otherax, len(E_array_out)))
        f = interp1d(E_array_in, matrix_in, axis=1,
                     kind="linear",
                     bounds_error=False, fill_value=0)
        matrix_out = f(E_array_out)
    elif axis == 0:
        N_otherax = matrix_in.shape[1]
        matrix_out = np.zeros((N_otherax, len(E_array_out)))
        f = interp1d(E_array_in, matrix_in, axis=0,
                     kind="linear",
                     bounds_error=False, fill_value=0)
        matrix_out = f(E_array_out)
    else:
        raise IndexError("Axis out of bounds. Must be 0 or 1.")

    return matrix_out


def interpolate_matrix_2D(matrix_in, E0_array_in, E1_array_in,
                          E0_array_out, E1_array_out):
    """Does a two-dimensional interpolation of the given matrix to the _out
    axes
    """
    if not (matrix_in.shape[0] == len(E0_array_in)
            and matrix_in.shape[1] == len(E1_array_in)):
        raise Exception("Shape mismatch between matrix_in and energy arrays")

    # Do the interpolation using splines of degree 1 (linear):
    f = RectBivariateSpline(E0_array_in, E1_array_in, matrix_in,
                            kx=1, ky=1)
    matrix_out = f(E0_array_out, E1_array_out)

    # Make a rectangular mask to set values outside original bounds to zero
    mask = np.ones(matrix_out.shape, dtype=bool)
    mask[E0_array_out <= E0_array_in[0], :] = 0
    mask[E0_array_out >= E0_array_in[-1], :] = 0
    mask[:, E1_array_out <= E1_array_in[0]] = 0
    mask[:, E1_array_out >= E1_array_in[-1]] = 0
    matrix_out = np.where(mask,
                          matrix_out,
                          0
                          )

    return matrix_out


def log_interp1d(xx, yy, **kwargs):
    """ Interpolate a 1-D function.logarithmically """
    logy = np.log(yy)
    lin_interp = interp1d(xx, logy, kind='linear', **kwargs)
    log_interp = lambda zz: np.exp(lin_interp(zz))  # noqa
    return log_interp


def call_model(fun,pars,pars_req):
    """ Call `fun` and check if all required parameters are provided """

    # Check if required parameters are a subset of all pars given
    if pars_req <= set(pars):
        pcall = {p: pars[p] for p in pars_req}
        return fun(**pcall)
    else:
        raise TypeError("Error: Need following arguments for this method:"
                        " {0}".format(pars_req))


def annotate_heatmap(im, matrix, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.
    Parameters
    ----------
    im
        The AxesImage to be labeled.
    # iex, ieg = list(zip(*mat.diagonal_elements()))
    # ieg = list(ieg)
    # for i, v in enumerate(ieg):
    #     if v >= len(mat.Eg):
    #         ieg[i] = len(mat.Eg) - 1
    # eg, ex = [mat.Eg[i] for i in ieg], [mat.Ex[i] for i in iex]
    # args, _kwargs = cut.get_args('trapezoid')
    # arg = lambda i: args[i] if i < len(args) else None
    # Ex_min = _kwargs.get('Ex_min', arg(0))
    # Ex_max = _kwargs.get('Ex_max', arg(1))
    # Eg_min = _kwargs.get('Eg_min', arg(2))
    # if len(args) >= 4:
    #     Eg_max = _kwargs.get('Eg_max', arg(3))
    # else:
    #     Eg_max = _kwargs.get('Eg_max', arg(3))

    # if Eg_max is None:
    #     lastEx = mat[-1, :]
    #     try:
    #         iEg_max = np.nonzero(lastEx)[0][-1]
    #     except IndexError():
    #         raise ValueError("Last Ex column has no non-zero elements")
    #     Eg_max = mat.Eg[iEg_max]

    # Eg_min = matrix.to_same_Eg(Eg_min)
    # Eg_max = matrix.to_same_Eg(Eg_max)
    # Ex_min = matrix.to_same_Ex(Ex_min)
    # Ex_max = matrix.to_same_Ex(Ex_max)

    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(matrix.values.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for j, i in product(*map(range, matrix.shape)):
        x = matrix.Eg[i]
        y = matrix.Ex[j]
        kw.update(color=textcolors[int(im.norm(matrix.values[i, j]) > threshold)])
        text = im.axes.text(y, x, valfmt(matrix.values[i, j], None), **kw)
        texts.append(text)

    return texts


def diagonal_elements(matrix: np.ndarray) -> Iterator[Tuple[int, int]]:
    """ Iterates over the last non-zero elements

    Args:
        mat: The matrix to iterate over
    Yields:
        Indicies (i, j) over the last non-zero (=diagonal)
        elements.
    """
    Ny = matrix.shape[1]
    for i, row in enumerate(matrix):
        for j, col in enumerate(reversed(row)):
            if col != 0.0:
                yield i, Ny-j
                break


def self_if_none(instance: Any, variable: Any, nonable: bool = False) -> Any:
    """ Sets `variable` from instance if variable is None.

    Note: Has to be imported as in the normalizer class due to class stack
          name retrieval

    Args:
        instance: instance
        variable: The variable to check
        nonable: Does not raise ValueError if
            variable and self.<variable_name> is None
            (where <variable_name> is replace by the variable's name).
    Returns:
        The value of variable or instance.variable
    Raises:
        ValueError: Nonable is True and if both variable and
            self.variable are None
    """
    name = _retrieve_name(variable)
    if variable is None:
        self_variable = getattr(instance, name)
        if not nonable and self_variable is None:
            raise ValueError(f"`{name}` must be set")
        return self_variable
    return variable


def _retrieve_name(var: Any) -> str:
    """ Finds the source-code name of `var`

        NOTE: Only call from self.reset.

     Args:
        var: The variable to retrieve the name of.
    Returns:
        The variable's name.
    """
    # Retrieve the line of the source code of the third frame.
    # The 0th frame is the current function, the 1st frame is the
    # calling function and the second is the calling function's caller.
    line = inspect.stack()[3].code_context[0].strip()
    match = re.search(r".*\((\w+).*\).*", line)
    assert match is not None, "Retrieving of name failed"
    name = match.group(1)
    return name


def only_one_not_none(*args):
    x = only_one(*[arg is not None for arg in args])
    return x


def only_one(*args):
    """ One and exactly one of the arguments evaluate to true """
    already_true = False
    for arg in args:
        if arg and not already_true:
            already_true = True
        elif arg and already_true:
            return False
    return already_true


def plot_trapezoid(cut: 'Action', matrix: 'Matrix',
                   ax=None, plot_matrix=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()

    mat = cut.act_on(matrix, inplace=False)[0]
    Eg_max = mat.Eg.max()
    Eg_min = mat.Eg.min()
    Ex_max = mat.Ex.max()
    Ex_min = mat.Ex.min()

    dEg = Eg_max - Ex_max
    if dEg > 0:
        binwidth = mat.Eg[1]-mat.Eg[0]
        dEg = np.ceil(dEg/binwidth) * binwidth
    x = [Eg_min, Ex_min+dEg, Ex_max+dEg, Eg_min]
    y = [Ex_min, Ex_min, Ex_max, Ex_max]
    if plot_matrix:
        matrix.plot(ax=ax)
    ax.add_patch(patches.Polygon(xy=list(zip(x, y)), fill=False, **kwargs))
    return ax


def ascii_plot(matrix: 'Matrix', shape=(5, 5)):
    """Plots a rebinned ascii version of the matrix

    Args:
        shape (tuple, optional): Shape of the rebinned matrix
    """
    values = np.unique(np.sort(matrix.values.flatten()))
    values = values[values > 0]
    N = len(values)/4

    def block(count):
        i = np.argmin(np.abs(count - values))
        b = int(i // N)
        return ['░░', '▒▒', '▓▓', '██'][b]

    for row in reversed(range(matrix.shape[0])):
        print('│', end='')
        for col in range(matrix.shape[1]):
            elem = matrix[row, col]
            if elem == 0:
                print('  ', end='')
            else:
                print(block(elem), end='')
        print('')
    print('└', end='')
    for col in range(matrix.shape[1]):
        print('──', end='')
    print('')

def contains_zeroes_patches(mat: 'Matrix', threshold: float = 0.1,
                            kernel_size: Union[str, int,
                                               Tuple[int, int]] = 'auto'):
    """ Check if matrix contains too many clustered zeroes

    Only bins above the diagonal are checked. Note that the diagonal
    is assumed to begin at the last non-zero bin. The zeroes are found
    by running a counting convolution over a normalized matrix, counting
    how many zeroes that are within the (NxN) kernel. If this is larger than
    threshold %, returns true.

    Args:
        mat: The matrix
        threshold: Number between 0 and 1. Controls how many zeroes must be
                   present within the kernel matrix to count as too many.
                   Increasing the threshold makes the test less sensitive.
                   Defaults to 0.1, that is, 10% zero bins within the kernel.
        kernel_size: The size of the kernel. If a single number,
                     creates a square kernel.
                     Increasing the kernel will detect more un-clustered zeroes.
                     Can be either 'auto' or int. If 'auto', defaults to 1/5th
                     the size of the smallest dimension of the matrix.
    Returns:
        True if clustered zeroes are detected.
    """
    if isinstance(kernel_size, str):
        if kernel_size != 'auto':
            raise ValueError("Kernel size must be 'auto' or int.")
        width = max(int(0.2*mat.shape[0]), 2)
        height = max(int(0.2*mat.shape[1]), 2)
    elif isinstance(kernel_size, int):
        width = height = kernel_size
    else:
        width, height = kernel_size
    #print(width, height)
    kernel = np.array(np.ones((width, height)))
    v = mat.values
    vv = v/v
    vv[np.isnan(vv)] = 0
    # Size we convolve in 2D, each pixel in the convolution
    # will contain information about a subarray of the matrix.
    # However, we know that beyond the diagonal is noise
    # so we must skip all pixels that have been convolved
    # with bins from beyond the diagonal. This is done by
    # setting them to NaN and discarding all NaNs in the
    # convolution.
    for i, j in mat.diagonal_elements():
        vv[i, j:] = np.NaN
    conv = nd.convolve(vv, kernel)
    m2 = mat.clone(values=conv)
    #m2.plot()
    mask = conv < (1 - threshold)*width*height
    mask = np.array(mask, dtype=float)
    mask[np.isnan(conv)] = np.nan
    #fig, ax = plt.subplots()

    found = False
    for i, row in enumerate(mask):
        for j, elem in enumerate(row):
            if np.isnan(elem):
                break
            if elem:
                #print(elem)
                #print(i, j)
                mask[i, j] = -5
                found = True
                break
        if found:
            break
    #ax.pcolormesh(mask)
    return found

def plot_projection_rectangle(ax: 'Axes', matrix: 'Matrix', axis: int | str,
                              Emin: float | None = None,
                              Emax: float | None = None,
                              **kwargs):
        axis = to_plot_axis(axis)
        if axis not in (0, 1):
            raise ValueError(f"Axis must be 0 or 1, got: {axis}")

        isEx = axis == 1
        index = matrix.index_Ex if isEx else matrix.index_Eg
        E = matrix.Ex if isEx else matrix.Eg
        E2 = matrix.Eg if isEx else matrix.Ex
        emin = E[index(Emin)] if Emin is not None else E[0]
        emax = E[index(Emax)] if Emax is not None else E[-1]
        start = (emin, 0) if isEx else (0, emin)
        w = emax - emin if isEx else E[-1]
        h = emax - emin if not isEx else E2[-1]
        kwargs |= {'fill': False}
        kwargs |= {'lw': 1}
        kwargs |= {'edgecolor': 'r'}
        rectangle = patches.Rectangle(start, width=w, height=h, **kwargs)
        ax.add_patch(rectangle)
        return ax


def to_plot_axis(axis: int | str) -> int:
    """Maps axis to 0, 1 or 2 according to which axis is specified

    Args:
        axis: Can be either of (0, 'Eg', 'x'), (1, 'Ex', 'y'), or
              (2, 'both', 'egex', 'exeg', 'xy', 'yx')
    Returns:
        An int describing the axis in the basis of the plot,
        _not_ the values' dimension.

    Raises:
        ValueError if the axis is not supported
    """
    try:
        axis = axis.lower()
    except AttributeError:
        pass

    if axis in (0, 'eg', 'x'):
        return 0
    elif axis in (1, 'ex', 'y'):
        return 1
    elif axis in (2, 'both', 'egex', 'exeg', 'xy', 'yx'):
        return 2
    else:
        raise ValueError(f"Unrecognized axis: {axis}")


def handle_rebin_arguments(*, bins: ArrayKeV, transform, LOG,
                           newbins: arraylike | None = None,
                           factor: float | None = None,
                           binwidth: Unitlike | None = None,
                           numbins: int | None = None) -> ArrayFloat:
    if not only_one_not_none(newbins, factor, binwidth, numbins):
        print(newbins, factor, binwidth, numbins)
        raise ValueError("Either 'bins', 'factor', `numbins` or 'binwidth' must be"
                            " specified, but not more than one.")

    binwidth_ = transform(binwidth) if binwidth is not None else None
    newbins_ = transform(newbins) if newbins is not None else None

    if factor is not None:
        if factor <= 0:
            raise ValueError("`factor` must be positive")
        numbins = int(len(bins)/factor)

    if numbins is not None:
        newbins_, step = np.linspace(bins[0], bins[-1],
                                 num=numbins, retstep=True)
        LOG.debug("Rebinning with factor %g, giving %g mids",
                    factor, len(newbins_))
        LOG.debug("Old step size: %g\nNew step size: %g",
                    bins[1] - bins[0], step)

    if binwidth_ is not None:
        newbins_ = np.arange(bins[0], bins[-1], binwidth_, dtype=float)

    assert newbins_ is not None

    if not np.isclose(newbins_[-1], bins[-1]) or not np.isclose(newbins_[0], bins[0]):
        LOG.warning("The rebinning resizes the energy vector.")
    if len(newbins_) > len(bins):
        raise ValueError("Can not rebin to more bins")

    return newbins_

