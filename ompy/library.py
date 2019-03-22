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
import numpy as np
import time
import warnings
from scipy.interpolate import interp1d, RectBivariateSpline


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
        for i in range(4, len(lines)):
            try:
                row = np.array(lines[i].split(), dtype="double")
                resp.append(row)
            except:
                break

    resp = np.array(resp)
    # Name the columns for ease of reading
    FWHM = resp[:, 1]  #*6.8 # Correct with fwhm @ 1.33 MeV?
    eff = resp[:, 2]
    pf = resp[:, 3]
    pc = resp[:, 4]
    ps = resp[:, 5]
    pd = resp[:, 6]
    pa = resp[:, 7]

    return R, FWHM, eff, pc, pf, ps, pd, pa, Eg_array_R


def div0(a, b):
    """ division function designed to ignore / 0, i.e. div0([-1, 0, 1], 0 ) -> [0, 0, 0] """
    # Check whether a or b (or both) are numpy arrays. If not, we don't
    # use the fancy function.
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(a, b)
            c[~np.isfinite(c)] = 0  # -inf inf NaN
    else:
        if b == 0:
            c = 0
        else:
            c = a / b
    return c


def i_from_E(E, E_array):
    # Returns index of the E_array value closest to given E
    return np.argmin(np.abs(E_array - E))



def shift_and_smooth3D(array, Eg_array, FWHM, p, shift, smoothing=True):
    # Updated 201807: Trying to vectorize so all Ex bins are handled simultaneously.
    # Takes a 2D array of counts, shifts it (downward only!) with energy 'shift'
    # and smooths it with a gaussian of specified 'FWHM'.
    # This version is vectorized to shift, smooth and scale all points
    # of 'array' individually, and then sum together and return.

    # TODO: FIX ME! There is a bug here, it does not do Compton subtraction right.

    # The arrays from resp.dat are missing the first channel.
    p = np.append(0, p)
    FWHM = np.append(0, FWHM)

    a1_Eg = (Eg_array[1] - Eg_array[0])  # bin width
    N_Ex, N_Eg = array.shape

    # Shift is the same for all energies
    if shift == "annihilation":
        # For the annihilation peak, all channels should be mapped on E = 511 keV. Of course, gamma channels below 511 keV,
        # and even well above that, cannot produce annihilation counts, but this is taken into account by the fact that p
        # is zero for these channels. Thus, we set i_shift=0 and make a special dimensions_shifted array to map all channels of
        # original array to i(511).
        i_shift = 0
    else:
        i_shift = i_from_E(shift, Eg_array) - i_from_E(
            0, Eg_array)  # The number of indices to shift by

    N_Eg_sh = N_Eg - i_shift
    indices_original = np.linspace(i_shift, N_Eg - 1, N_Eg - i_shift).astype(
        int
    )  # Index array for original array, truncated to shifted array length
    if shift == "annihilation":  # If this is the annihilation peak then all counts should end up with their centroid at E = 511 keV
        # indices_shifted = (np.ones(N_Eg-i_from_E(511, Eg_array))*i_from_E(511, Eg_array)).astype(int)
        indices_shifted = (np.ones(N_Eg) * i_from_E(511, Eg_array)).astype(int)
    else:
        indices_shifted = np.linspace(0, N_Eg - i_shift - 1,
                                      N_Eg - i_shift).astype(
                                          int)  # Index array for shifted array

    if smoothing:
        # Scale each Eg count by the corresponding probability
        # Do this for all Ex bins at once:
        array = array * p[0:N_Eg].reshape(1, N_Eg)
        # Shift array down in energy by i_shift indices,
        # so that index i_shift of array is index 0 of array_shifted.
        # Also flatten array along Ex axis to facilitate multiplication.
        array_shifted_flattened = array[:, indices_original].ravel()
        # Make an array of N_Eg_sh x N_Eg_sh containing gaussian distributions
        # to multiply each Eg channel by. This array is the same for all Ex bins,
        # so it will be repeated N_Ex times and stacked for multiplication
        # To get correct normalization we multiply by bin width
        pdfarray = a1_Eg * norm.pdf(
            np.tile(Eg_array[0:N_Eg_sh], N_Eg_sh).reshape((N_Eg_sh, N_Eg_sh)),
            loc=Eg_array[indices_shifted].reshape(N_Eg_sh, 1),
            scale=FWHM[indices_shifted].reshape(N_Eg_sh, 1) / 2.355)

        # Remove eventual NaN values:
        pdfarray = np.nan_to_num(pdfarray, copy=False)
        # print("Eg_array[indices_shifted] =", Eg_array[indices_shifted], flush=True)
        # print("pdfarray =", pdfarray, flush=True)
        # Repeat and stack:
        pdfarray_repeated_stacked = np.tile(pdfarray, (N_Ex, 1))

        # Multiply array of counts with pdfarray:
        multiplied = pdfarray_repeated_stacked * array_shifted_flattened.reshape(
            N_Ex * N_Eg_sh, 1)

        # Finally, for each Ex bin, we now need to sum the contributions from the smoothing
        # of each Eg bin to get a total Eg spectrum containing the entire smoothed spectrum:
        # Do this by reshaping into 3-dimensional array where each Eg bin (axis 0) contains a
        # N_Eg_sh x N_Eg_sh matrix, where each row is the smoothed contribution from one
        # original Eg pixel. We sum the columns of each of these matrices:
        array_out = multiplied.reshape((N_Ex, N_Eg_sh, N_Eg_sh)).sum(axis=1)
        # print("array_out.shape =", array_out.shape)
        # print("array.shape[0],array.shape[1]-N_Eg_sh =", array.shape[0],array.shape[1]-N_Eg_sh)

    else:
        # array_out = np.zeros(N)
        # for i in range(N):
        #     try:
        #         array_out[i-i_shift] = array[i] #* p[i+1]
        #     except IndexError:
        #         pass

        # Instead of above, vectorizing:
        array_out = p[indices_original].reshape(
            1, N_Eg_sh) * array[:, indices_original]

    # Append zeros to the end of Eg axis so we match the length of the original array:
    if i_shift > 0:
        array_out = np.concatenate((array_out, np.zeros(
            (N_Ex, N_Eg - N_Eg_sh))),
                                   axis=1)
    return array_out



def EffExp(Eg_array):
    # Function from MAMA which makes an additional efficiency correction based on discriminator thresholds etc.
    # Basically there is a hard-coded set of energies and corresponding efficiencies in MAMA, and it should be
    # zero below and 1 above this range.
    Egs = np.array([30., 80., 122., 183., 244., 294., 344., 562., 779., 1000])
    Effs = np.array([0.0, 0.0, 0.0, 0.06, 0.44, 0.60, 0.87, 0.99, 1.00, 1.000])
    EffExp_array = np.zeros(len(Eg_array))
    for iEg in range(len(Eg_array)):
        if Eg_array[iEg] < Egs.min():
            EffExp_array[iEg] = 0
        elif Eg_array[iEg] >= Egs.max():
            EffExp_array[iEg] = 1
        else:
            EffExp_array[iEg] = Effs[np.argmin(np.abs(Egs - Eg_array[iEg]))]

    return EffExp_array


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
        E_array = np.linspace(a0, a0 + a1 * (N - 1), N)
    elif E_max is not None:
        N = int(np.ceil((E_max - a0) / a1))
        E_array = np.linspace(a0, a0 + a1 * (N - 1), N)
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

def make_mask(Ex_array, Eg_array, Ex1, Eg1, Ex2, Eg2):
    # Make masking array to cut away noise below Eg=Ex+dEg diagonal
    # Define cut   x1    y1    x2    y2
    cut_points = [i_from_E(Eg1, Eg_array), i_from_E(Ex1, Ex_array),
                  i_from_E(Eg2, Eg_array), i_from_E(Ex2, Ex_array)]
    i_array = np.linspace(0,len(Ex_array)-1,len(Ex_array)).astype(int) # Ex axis
    j_array = np.linspace(0,len(Eg_array)-1,len(Eg_array)).astype(int) # Eg axis
    i_mesh, j_mesh = np.meshgrid(i_array, j_array, indexing='ij')
    return np.where(i_mesh > line(j_mesh, cut_points), 1, 0)


def line(x, points):
    """
    Returns a line through coordinates [x1,y1,x2,y2]=points


    """
    a = (points[3]-points[1])/float(points[2]-points[0])
    b = points[1] - a*points[0]
    return a*x + b


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
