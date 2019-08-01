# -*- coding: utf-8 -*-
"""
Library of utility classes and functions for the Oslo method.

---

This file is part of oslo_method_python, a python implementation of the
Oslo method.

Copyright (C) 2018 Jørgen Eriksson Midtbø
Oslo Cyclotron Laboratory
jorgenem [0] gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from .constants import DE_PARTICLE, DE_GAMMA_8MEV, DE_GAMMA_1MEV
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d, RectBivariateSpline
import matplotlib
from itertools import product

import ompy as om


def mama_read(filename):
    # Reads a MAMA matrix file and returns the matrix as a numpy array,
    # as well as a list containing the four calibration coefficients
    # (ordered as [bx, ax, by, ay] where Ei = ai*channel_i + bi)
    # and 1-D arrays of lower-bin-edge calibrated x and y values for plotting
    # and similar.
    matrix = np.genfromtxt(filename, skip_header=10, skip_footer=1)
    cal = {}
    with open(filename, 'r') as datafile:
        calibration_line = datafile.readlines()[6].split(",")
        # a = [float(calibration_line[2][:-1]), float(calibration_line[3][:-1]), float(calibration_line[5][:-1]), float(calibration_line[6][:-1])]
        # JEM update 20180723: Changing to dict, including second-order term for generality:
        # print("calibration_line =", calibration_line, flush=True)
        cal = {
            "a0x": float(calibration_line[1]),
            "a1x": float(calibration_line[2]),
            "a2x": float(calibration_line[3]),
            "a0y": float(calibration_line[4]),
            "a1y": float(calibration_line[5]),
            "a2y": float(calibration_line[6])
        }
    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny - 1, Ny)
    x_array = np.linspace(0, Nx - 1, Nx)
    # Make arrays in center-bin calibration:
    x_array = cal["a0x"] + cal["a1x"] * x_array + cal["a2x"] * x_array**2
    y_array = cal["a0y"] + cal["a1y"] * y_array + cal["a2y"] * y_array**2
    # Then correct them to lower-bin-edge:
    y_array = y_array - cal["a1y"] / 2
    x_array = x_array - cal["a1x"] / 2
    # Matrix, Eg array, Ex array
    return matrix, x_array, y_array


def mama_write(mat, filename, comment=""):
    # Calculate calibration coefficients.
    calibration = mat.calibration()
    cal = {
        "a0x": calibration['a00'],
        "a1x": calibration['a01'],
        "a2x": 0,
        "a0y": calibration['a10'],
        "a1y": calibration['a11'],
        "a2y": 0
    }
    # Convert from lower-bin-edge to centre-bin as this is what the MAMA file
    # format is supposed to have:
    cal["a0x"] += cal["a1x"] / 2
    cal["a0y"] += cal["a1y"] / 2

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Spectrum \n'
    header_string += '!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string += '!EXPERIMENT= oslo_method_python \n'
    header_string += '!COMMENT={:s} \n'.format(comment)
    header_string += '!TIME=DATE:' + time.strftime("%d-%b-%y %H:%M:%S",
                                                   time.localtime()) + '   \n'
    header_string += (
        '!CALIBRATION EkeV=6, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E \n'
        % (
            cal["a0x"],
            cal["a1x"],
            cal["a2x"],
            cal["a0y"],
            cal["a1y"],
            cal["a2y"],
        ))
    header_string += '!PRECISION=16 \n'
    header_string += "!DIMENSION=2,0:{:4d},0:{:4d} \n".format(
        mat.shape[1] - 1, mat.shape[0] - 1)
    header_string += '!CHANNEL=(0:%4d,0:%4d) ' % (mat.shape[1] - 1,
                                                  mat.shape[0] - 1)

    footer_string = "!IDEND=\n"

    # Write matrix:
    np.savetxt(
        filename,
        mat.values,
        fmt="%-17.8E",
        delimiter=" ",
        newline="\n",
        header=header_string,
        footer=footer_string,
        comments="")


def read_response(fname_resp_mat, fname_resp_dat):
    # Import response matrix
    R, cal_R, Eg_array_R, tmp = mama_read(fname_resp_mat)
    # We also need info from the resp.dat file:
    resp = []
    with open(fname_resp_dat) as file:
        # Read line by line as there is crazyness in the file format
        lines = file.readlines()
        for i in range(4,len(lines)):
            try:
                row = np.array(lines[i].split(), dtype="double")
                resp.append(row)
            except:
                break


    resp = np.array(resp)
    # Name the columns for ease of reading
    FWHM = resp[:,1]#*6.8 # Correct with fwhm @ 1.33 MeV?
    eff = resp[:,2]
    pf = resp[:,3]
    pc = resp[:,4]
    ps = resp[:,5]
    pd = resp[:,6]
    pa = resp[:,7]

    return R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_R


def div0(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    # Check whether a or b (or both) are numpy arrays. If not, we don't
    # use the fancy function.
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b )
            c[ ~ np.isfinite(c )] = 0  # -inf inf NaN
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


def E_array_from_calibration(a0, a1, N=None, E_max=None):
    """
    Return an array of lower-bin-edge energy values corresponding to the
    specified calibration.

    Args:
        a0, a1 (float): Calibration coefficients; E = a0 + a1*i
        either
            N (int): Number of bins
        or
            E_max (float): Max energy. Array is constructed to ensure last bin
                           covers E_max. In other words,
                           E_array[-1] >= E_max - a1
    Returns:
        E_array (np.ndarray): Array of lower-bin-edge energy values
    """
    E_array = None
    if E_max is not None and N is not None:
        raise Exception("Cannot give both N and E_max -- must choose one")
    if N is not None:
        E_array = np.linspace(a0, a0+a1*(N-1), N)
    elif E_max is not None:
        N = int(np.ceil((E_max - a0)/a1))
        E_array = np.linspace(a0, a0+a1*(N-1), N)
    else:
        raise Exception("Either N or E_max must be given")

    return E_array


def fill_negative(matrix, window_size):
    """
    Fill negative channels with positive counts from neighbouring channels

    The MAMA routine for this is very complicated. It seems to basically
    use a sliding window along the Eg axis, given by the FWHM, to look for
    neighbouring bins with lots of counts and then take some counts from there.
    Can we do something similar in an easy way?

    Todo: Debug me!
    """
    warnings.warn("Hello from the fill_negative() function. Please debug me.")
    matrix_out = np.copy(matrix)
    # Loop over rows:
    for i_Ex in range(matrix.shape[0]):
        for i_Eg in np.where(matrix[i_Ex, :] < 0)[0]:
            # print("i_Ex = ", i_Ex, "i_Eg =", i_Eg)
            # window_size = 4  # Start with a constant window size.
            # TODO relate it to FWHM by energy arrays
            i_Eg_low = max(0, i_Eg - window_size)
            i_Eg_high = min(matrix.shape[1], i_Eg + window_size)
            # Fill from the channel with the larges positive count
            # in the neighbourhood
            i_max = np.argmax(matrix[i_Ex, i_Eg_low:i_Eg_high])
            # print("i_max =", i_max)
            if matrix[i_Ex, i_max] <= 0:
                pass
            else:
                positive = matrix[i_Ex, i_max]
                negative = matrix[i_Ex, i_Eg]
                fill = min(0, positive + negative)  # Don't fill more than to 0
                rest = positive
                # print("fill =", fill, "rest =", rest)
                matrix_out[i_Ex, i_Eg] = fill
                # matrix_out[i_Ex, i_max] = rest
    return matrix_out


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
    log_interp = lambda zz: np.exp(lin_interp(zz))
    return log_interp


def call_model(fun,pars,pars_req):
    """ Call `fun` and check if all required parameters are provided """

    # Check if required parameters are a subset of all pars given
    if pars_req <= set(pars):
        return fun(**pars)
    else:
        raise TypeError("Error: Need following arguments for this method: {0}".format(pars_req))


def get_discretes(Emids, fname, resolution=0.1):
    """ Get discrete levels, and smooth by some resolution [MeV]
    and the bins centers [MeV]
    For now: Assume linear binning """
    energies = np.loadtxt(fname)
    energies /= 1e3  # convert to MeV

    # Emax = energies[-1]
    # nbins = int(np.ceil(Emax/binsize))
    # bins = np.linspace(0,Emax,nbins+1)
    binsize = Emids[1] - Emids[0]
    bin_edges = np.append(Emids, Emids[-1] + binsize)
    bin_edges -= binsize / 2

    hist, _ = np.histogram(energies, bins=bin_edges)
    hist = hist.astype(float) / binsize  # convert to levels/MeV

    from scipy.ndimage import gaussian_filter1d
    hist_smoothed = gaussian_filter1d(hist, sigma=resolution / binsize)
    return hist_smoothed, hist


def tranform_nld_gsf(samples: dict, nld=None, gsf=None,
                     N_max: int = 100,
                     random_state=np.random.RandomState(65489)):
    """
    Use a list(dict) of samples of `A`, `B`, and `alpha` parameters from
    multinest to transform a (list of) nld and/or gsf sample(s). Can be used
    to normalize the nld and/or gsf

    Args:
        samples (dict): Multinest samples.
        nld (om.Vector or list/array[om.Vector], optional):
            nld ("unnormalized")
        gsf (om.Vector or list/array[om.Vector], optional):
            gsf ("unnormalized")
        N_max (int, optional): Maximum number of samples returned if `nld`
                               and `gsf` is a list/array
        random_state (optional): random state, set by default such that
                                 a repeted use of the function gives the same
                                 results.

    Returns:
        `nld_trans` and/or `gsf_trans`: Transformed `nld` and or `gsf`,
                                        depending on what input is given.

<<<<<<< variant A
    Parameters
    ----------
    im
        The AxesImage to be labeled.
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
>>>>>>> variant B
======= end
    """

    # Need to sweep though multinest samples in random order
    # as they are ordered with decreasing likelihood by default
    for key, value in samples.items():
        N_multinest = len(value)
        break
    randlist = np.arange(N_multinest)
    random_state.shuffle(randlist)  # works in-place

    if nld is not None:
        A = samples["A"]
        alpha = samples["alpha"]
        if type(nld) is om.Vector:
            N = min(N_multinest, N_max)
        else:
            N = len(nld)
        nld_trans = []

    if gsf is not None:
        B = samples["B"]
        alpha = samples["alpha"]
        if type(gsf) is om.Vector:
            N = min(N_multinest, N_max)
        else:
            N = len(gsf)
        gsf_trans = []

    # transform the list
    for i in range(N):
        i_multi = randlist[i]
        # nld loop
        try:
            if type(nld) is om.Vector:
                nld_tmp = nld
            else:
                nld_tmp = nld[i]
            nld_tmp = nld_tmp.transform(alpha=alpha[i_multi],
                                        const=A[i_multi])
            nld_trans.append(nld_tmp)
        except:
            pass
        # gsf loop
        try:
            if type(gsf) is om.Vector:
                gsf_tmp = gsf
            else:
                gsf_tmp = gsf[i]
            gsf_tmp = gsf_tmp.transform(alpha=alpha[i_multi],
                                        const=B[i_multi])
            gsf_trans.append(gsf_tmp)
        except:
            pass

    if nld is not None and gsf is not None:
        return nld_trans, gsf_trans
    elif gsf is not None:
        return gsf_trans
    elif nld is not None:
        return nld_trans
