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




class OldMatrix():
    """
    The matrix class stores matrices along with calibration and energy axis
    arrays.

    """
    def __init__(self, matrix=None, E0_array=None, E1_array=None,
                 std=None):
        """
        Initialise the class. There is the option to initialise
        it in an empty state. In that case, all class variables will be None.
        It can be filled later using the load() method.
        """
        # Sanity checks:
        if matrix is not None and E0_array is not None:
            if matrix.shape[0] != len(E0_array):
                raise ValueError("Shape mismatch between matrix and E0_array.")
        if matrix is not None and E1_array is not None:
            if matrix.shape[1] != len(E1_array):
                raise ValueError("Shape mismatch between matrix and E1_array.")
        if matrix is not None and std is not None:
            if matrix.shape != std.shape:
                raise ValueError("Shape mismatch between matrix and std.")

        # Fill class variables:
        self.matrix = matrix
        self.E0_array = E0_array
        self.E1_array = E1_array


    def calibration(self):
        """Calculate and return the calibration coefficients of the energy axes
        """
        calibration = None
        if (self.matrix is not None and self.E0_array is not None
                and self.E1_array is not None):
            calibration = {
                           # Formatted as "a{axis}{power of E}"
                           "a00": self.E0_array[0],
                           "a01": self.E0_array[1]-self.E0_array[0],
                           "N0": len(self.E0_array),
                           "a10": self.E1_array[0],
                           "a11": self.E1_array[1]-self.E1_array[0],
                           "N1": len(self.E1_array)
                          }
        else:
            raise Exception("calibration() called on empty Matrix instance")
        return calibration

    def plot(self, ax=None, title="", zscale="log", zmin=None, zmax=None,
             **kwargs):
        cbar = None
        # Add one more element to the array for plotting purposes - this is the
        # upper bin edge of the last bin:
        E0_array_plot = np.append(self.E0_array, self.E0_array[-1]
                                  + self.E0_array[1]-self.E0_array[0])
        E1_array_plot = np.append(self.E1_array, self.E1_array[-1]
                                  + self.E1_array[1]-self.E1_array[0])
        if ax is None:
            f, ax = plt.subplots(1, 1)
        if zscale == "log":
            # z axis shall have log scale
            # Check whether z limits were given:
            if (zmin is not None and zmax is None):
                # zmin only,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     norm=LogNorm(vmin=zmin),
                                     **kwargs
                                     )
            elif (zmin is None and zmax is not None):
                # or zmax only,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     norm=LogNorm(vmax=zmax),
                                     **kwargs
                                     )
            elif (zmin is not None and zmax is not None):
                # or both,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     norm=LogNorm(vmin=zmin, vmax=zmax),
                                     **kwargs
                                     )
            else:
                # or finally, no limits:
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     norm=LogNorm(),
                                     **kwargs
                                     )
        elif zscale == "linear":
            # z axis shall have linear scale
            # Check whether z limits were given:
            if (zmin is not None and zmax is None):
                # zmin only,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     vmin=zmin,
                                     **kwargs
                                     )
            elif (zmin is None and zmax is not None):
                # or zmax only,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     vmax=zmax,
                                     **kwargs
                                     )
            elif (zmin is not None and zmax is not None):
                # or both,
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     vmin=zmin,
                                     vmax=zmax,
                                     **kwargs
                                     )
            else:
                # or finally, no limits.
                cbar = ax.pcolormesh(E1_array_plot,
                                     E0_array_plot,
                                     self.matrix,
                                     **kwargs
                                     )
        else:
            raise Exception("Unknown zscale type", zscale)
        assert cbar is not None  # cbar should be filled at this point
        ax.set_title(title)

        # == Change tick marks ==
        # Find a nice spacing between ticks
        # E1_array_middle_bin = (self.E1_array
        #                        + (self.E1_array[1] - self.E1_array[0])/2)
        # ax.set_xticks(E1_array_middle_bin)
        # E0_array_middle_bin = (self.E0_array
        #                        + (self.E0_array[1] - self.E0_array[0])/2)
        # ax.set_yticks(E0_array_middle_bin)

        # if ax is None:
            # f.colorbar(cbar, ax=ax)
            # plt.show()
        return cbar  # Return the colorbar to allow it to be plotted outside

    def plot_projection(self, E_limits, axis, ax=None, normalize=False,
                        **kwargs):
        """Plots the projection of the matrix along axis

        Args:
            axis (int, 0 or 1): The axis to project onto.
            E_limits (list of two floats): The energy limits for the
                                           projection.
            ax (matplotlib axes object, optional): The axes object to put
                                                   the plot in.
        """
        if ax is None:
            f, ax = plt.subplots(1, 1)
        else:
            pass

        if axis == 0:
            i_E_low = i_from_E(E_limits[0], self.E1_array)
            i_E_high = i_from_E(E_limits[1], self.E1_array)
            if normalize:
                projection = np.mean(
                                div0(
                                    self.matrix[:, i_E_low:i_E_high],
                                    np.sum(self.matrix[:, i_E_low:i_E_high],
                                           axis=0
                                           )
                                    ),
                                axis=1
                                )
            else:
                projection = self.matrix[:, i_E_low:i_E_high].sum(axis=1)
            E0_array_middle_bin = (self.E0_array
                                   + (self.E0_array[1] - self.E0_array[0])/2)
            ax.step(E0_array_middle_bin,
                    projection,
                    where="mid",  # Corresponds to lower bin edge
                    **kwargs
                    )
        elif axis == 1:
            i_E_low = i_from_E(E_limits[0], self.E0_array)
            i_E_high = i_from_E(E_limits[1], self.E0_array)
            if normalize:
                projection = np.mean(
                                div0(
                                    self.matrix[i_E_low:i_E_high, :],
                                    (np.sum(self.matrix[i_E_low:i_E_high, :],
                                     axis=1)
                                     * self.calibration()["a01"])[:, None]
                                    ),
                                axis=0
                                )
            else:
                projection = self.matrix[i_E_low:i_E_high, :].sum(axis=0)
            E1_array_middle_bin = (self.E1_array
                                   + (self.E1_array[1] - self.E1_array[0])/2)
            ax.step(E1_array_middle_bin,
                    projection,
                    where="mid",  # Corresponds to lower bin edge
                    **kwargs
                    )
        else:
            raise ValueError("Variable axis must be one of (0, 1) but is",
                             axis)

    def plot_projection_x(self, E_limits, ax=None, normalize=False, **kwargs):
        """ Wrapper to call plot_projection(axis=1) to project on x axis"""
        self.plot_projection(E_limits=E_limits, axis=1, ax=ax,
                             normalize=normalize, **kwargs)

    def plot_projection_y(self, E_limits, ax=None, normalize=False, **kwargs):
        """ Wrapper to call plot_projection(axis=0) to project on y axis"""
        self.plot_projection(E_limits=E_limits, axis=0, ax=ax,
                             normalize=normalize, **kwargs)

    def save(self, fname):
        """
        Save matrix to mama file
        """
        write_mama_2D(self, fname,
                      comment="Made by oslo_method_python")

    def load(self, fname, suppress_warning=False):
        """
        Load matrix from mama file
        """
        if self.matrix is not None and not suppress_warning:
            print("Warning: load() called on non-empty matrix", flush=True)

        # Load matrix from file:
        # matrix, calibration, Ex_array, Eg_array = read_mama_2D(fname)
        matrix_object = read_mama_2D(fname)
        self.matrix = matrix_object.matrix
        self.E0_array = matrix_object.E0_array
        self.E1_array = matrix_object.E1_array

    def cut_rect(self, axis, E_limits, inplace=True):
        """
        Cuts the matrix (and std, if present) to the sub-interval E_limits.

        Args:
            axis (int): Which axis to apply the cut to.
            E_limits (list): [E_min, E_max], where
                E_min, E_max (float): Upper and lower energy limits for cut
            inplace (bool): Whether to make the cut in place or not

        Returns:
            None if inplace==False
            cut_matrix (Matrix): The cut version of the matrix
        """
        if axis in (0, 1):
            if not len(E_limits) == 2:
                raise ValueError("E_limits must be a list of length 2")
        elif axis == "both":
            if not len(E_limits) == 4:
                raise ValueError("E_limits must be a list of length 4")
        else:
            raise ValueError("axis must be 0, 1 or \"both\"")

        if not (E_limits[1] >= E_limits[0]):  # Sanity check
            raise ValueError("Upper limit must be >= lower limit")
        if axis == "both":
            if not (E_limits[1] >= E_limits[0] and E_limits[3] >= E_limits[2]):
                raise ValueError("Upper limits must be >= lower limits")
        matrix_cut = None
        std_cut = None
        out = None
        if axis == 0:
            i_E_min = np.argmin(np.abs(self.E0_array-E_limits[0]))
            i_E_max = np.argmin(np.abs(self.E0_array-E_limits[1]))+1
            matrix_cut = self.matrix[i_E_min:i_E_max, :]
            E0_array_cut = self.E0_array[i_E_min:i_E_max]
            if inplace:
                self.matrix = matrix_cut
                self.E0_array = E0_array_cut
            else:
                out = Matrix(matrix_cut, E0_array_cut, E1_array)

        elif axis == 1:
            i_E_min = np.argmin(np.abs(self.E1_array-E_limits[0]))
            i_E_max = np.argmin(np.abs(self.E1_array-E_limits[1]))+1
            matrix_cut = self.matrix[:, i_E_min:i_E_max]
            E1_array_cut = self.E1_array[i_E_min:i_E_max]
            if inplace:
                self.matrix = matrix_cut
                self.E1_array = E1_array_cut
            else:
                out = Matrix(matrix_cut, E0_array, E1_array_cut)
        elif axis == "both":
            i_E0_min = np.argmin(np.abs(self.E0_array-E_limits[0]))
            i_E0_max = np.argmin(np.abs(self.E0_array-E_limits[1]))+1
            i_E1_min = np.argmin(np.abs(self.E1_array-E_limits[2]))
            i_E1_max = np.argmin(np.abs(self.E1_array-E_limits[3]))+1
            matrix_cut = self.matrix[i_E0_min:i_E0_max, i_E1_min:i_E1_max]
            E0_array_cut = self.E0_array[i_E0_min:i_E0_max]
            E1_array_cut = self.E1_array[i_E1_min:i_E1_max]
            if inplace:
                self.matrix = matrix_cut
                self.E0_array = E0_array_cut
                self.E1_array = E1_array_cut
            else:
                out = Matrix(matrix_cut, E0_array_cut, E1_array_cut)
        else:
            raise ValueError("Axis must be one of (0, 1), but is", axis)

        return out

    def cut_diagonal(self, E1, E2):
        self.matrix = cut_diagonal(self.matrix, self.E0_array,
                                   self.E1_array, E1, E2)

    def fill_negative(self, window_size):
        self.matrix = fill_negative(self.matrix, window_size)

    def remove_negative(self):
        self.matrix = np.where(self.matrix > 0, self.matrix, 0)


class Vector():
    def __init__(self, vector=None, E_array=None):
        if not isinstance(vector, np.ndarray):
            raise TypeError("vector must be numpy array")
        if not isinstance(E_array, np.ndarray):
            raise TypeError("E_array must be numpy array")
        if len(vector) != len(E_array):
            raise ValueError("shape mismatch between input arrays")
        self.vector = vector
        self.E_array = E_array

    def calibration(self):
        """Calculate and return the calibration coefficients of the energy axes
        """
        calibration = None
        if (self.vector is not None and self.E_array is not None):
            calibration = {
                           # Formatted as "a{axis}{power of E}"
                           "a0": self.E_array[0],
                           "a1": self.E_array[1]-self.E_array[0],
                          }
        else:
            raise Exception("calibration() called on empty Vector instance")
        return calibration

    def plot(self, ax=None, yscale="linear", ylim=None, xlim=None,
             title=None, **kwargs):
        """
        Plot self.vector against self.E_array.

        Args:
            ax (matplotlib.axes.Axes, optional): The matplotlib Axes object to
                plot into.
            yscale (str): Scaling on y axis. One of {"linear", "log", "symlog",
                "logit"}
            ylim (list, optional): [min, max] limits on y axis
            xlim (list, optional): [min, max] limits on x axis
            title (str, optional): Title of the plot (passed to
                ax.set_title(title))
            **kwargs: Additional arguments, passed on to pyplot.plot(**kwargs)

        Todo:
            Consider if the function should return the ax.
        """
        if ax is None:
            f, ax = plt.subplots(1, 1)

        # Plot with middle-bin energy values:
        E_array_midbin = self.E_array + self.calibration()["a1"]/2
        ax.plot(E_array_midbin, self.vector, **kwargs)

        ax.set_yscale(yscale)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xlim is not None:
            ax.set_xlim(xlim)
        if title is not None:
            ax.set_title(title)
        if ax is None:
            plt.show()

    def save(self, fname):
        """
        Save vector to mama file
        """
        raise Exception("Not implemented yet")

        return None

    def load(self, fname):
        """
        Load vector from mama file
        """
        raise Exception("Not implemented yet")

        return None

    def transform(self, const=1, alpha=0, inplace=False):
        """
        Return a transformed version of the vector:
        vector -> const * vector * exp(alpha*E_array)
        """
        E_array_midbin = self.E_array + self.calibration()["a1"]/2
        vector_transformed = (const * self.vector
                              * np.exp(alpha*E_array_midbin)
                              )
        if inplace:
            self.vector = vector_transformed
        else:
            return Vector(E_array=self.E_array, vector=vector_transformed)


def read_mama_2D(filename):
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
        cal = {"a0x":float(calibration_line[1]), "a1x":float(calibration_line[2]), "a2x":float(calibration_line[3]), 
             "a0y":float(calibration_line[4]), "a1y":float(calibration_line[5]), "a2y":float(calibration_line[6])}
    Ny, Nx = matrix.shape
    y_array = np.linspace(0, Ny-1, Ny)
    x_array = np.linspace(0, Nx-1, Nx)
    # Make arrays in center-bin calibration:
    x_array = cal["a0x"] + cal["a1x"]*x_array + cal["a2x"]*x_array**2
    y_array = cal["a0y"] + cal["a1y"]*y_array + cal["a2y"]*y_array**2
    # Then correct them to lower-bin-edge:
    y_array = y_array - cal["a1y"]/2
    x_array = x_array - cal["a1x"]/2
    out = Matrix(matrix=matrix, E0_array=y_array, E1_array=x_array)
    return out


def write_mama_2D(mat, filename, comment=""):
    # Calculate calibration coefficients.
    cal = {"a0x": mat.E1_array[0],
           "a1x": mat.E1_array[1]-mat.E1_array[0],
           "a2x": 0,
           "a0y": mat.E0_array[0],
           "a1y": mat.E0_array[1]-mat.E0_array[0],
           "a2y": 0}
    # Convert from lower-bin-edge to centre-bin as this is what the MAMA file
    # format is supposed to have:
    cal["a0x"] += cal["a1x"]/2
    cal["a0y"] += cal["a1y"]/2

    # Write mandatory header:
    header_string = '!FILE=Disk \n'
    header_string += '!KIND=Spectrum \n'
    header_string += '!LABORATORY=Oslo Cyclotron Laboratory (OCL) \n'
    header_string += '!EXPERIMENT= oslo_method_python \n'
    header_string += '!COMMENT={:s} \n'.format(comment)
    header_string += '!TIME=DATE:'+time.strftime("%d-%b-%y %H:%M:%S", time.localtime())+'   \n'
    header_string += ('!CALIBRATION EkeV=6, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E, %12.6E \n'
                      % (cal["a0x"], cal["a1x"], cal["a2x"],
                         cal["a0y"], cal["a1y"], cal["a2y"],
                         )
                      )
    header_string += '!PRECISION=16 \n'
    header_string += "!DIMENSION=2,0:{:4d},0:{:4d} \n".format(
                        mat.matrix.shape[1]-1, mat.matrix.shape[0]-1)
    header_string += '!CHANNEL=(0:%4d,0:%4d) ' % (
                        mat.matrix.shape[1]-1, mat.matrix.shape[0]-1)

    footer_string = "!IDEND=\n"

    # Write matrix:
    np.savetxt(filename, mat.matrix, fmt="%-17.8E", delimiter=" ",
               newline="\n", header=header_string, footer=footer_string,
               comments=""
               )



def read_response(fname_resp_mat, fname_resp_dat):
    # Import response matrix
    R, cal_R, Eg_array_R, tmp = read_mama_2D(fname_resp_mat)
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
    return np.argmin(np.abs(E_array - E))


def line(x, points):
    """
    Returns a line through coordinates [x1,y1,x2,y2]=points


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



def EffExp(Eg_array):
    # Function from MAMA which makes an additional efficiency correction based on discriminator thresholds etc.
    # Basically there is a hard-coded set of energies and corresponding efficiencies in MAMA, and it should be 
    # zero below and 1 above this range.
    Egs = np.array([30.,80.,122.,183.,244.,294.,344.,562.,779.,1000])
    Effs = np.array([0.0,0.0,0.0,0.06,0.44,0.60,0.87,0.99,1.00,1.000])
    EffExp_array =  np.zeros(len(Eg_array))
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


def diagonal_resolution(Ex: np.ndarray) -> np.ndarray:
    """ Calculate Ex-dependent detector resolution (sum of sqroot)

    Args:
        Ex: Excitation energy bin array
    """
    # Assume constant particle resolution:
    dE_particle = DE_PARTICLE
    # Interpolate the gamma resolution linearly:
    # Eg = Ex + self.bin_width_out/2
    dE_gamma = ((DE_GAMMA_8MEV - DE_GAMMA_1MEV) / (8000 - 1000)
                * (Ex - 1000))  + DE_GAMMA_1MEV

    dE_resolution = np.sqrt(dE_particle**2 + dE_gamma**2)
    #dE_max_res = np.max(dE_resolution)
    return dE_resolution
