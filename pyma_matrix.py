"""
Class matrix(), for matrix storage in pyma.

---

This is pyma, the python implementation of the Oslo method.
It handles two-dimensional matrices of event count spectra, and
implements detector response unfolding, first generation method
and other manipulation of the spectra.

It is heavily inspired by MAMA, written by Magne Guttormsen and others,
available at https://github.com/oslocyclotronlab/oslo-method-software

Copyright (C) 2018 J{\o}rgen Eriksson Midtb{\o}
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

class matrix():
    """ 
    The matrix class stores matrices along with calibration and energy axis arrays.

    """
    def __init__(self, matrix=None, Ex_array=None, Eg_array=None):
        """
        Initialise the class. There is the option to initialise 
        it in an empty state. In that case, all class variables will be None.
        It can be filled later using the load() method.
        """
        self.matrix = matrix
        self.Ex_array = Ex_array
        self.Eg_array = Eg_array

        if matrix is not None and Ex_array is not None and Eg_array is not None:
            # Calculate calibration based on energy arrays, assuming linear calibration:
            self.calibration = {"a0x":Eg_array[0], "a1x":Eg_array[1]-Eg_array[0], "a2x":0, 
              "a0y":Ex_array[0], "a1y":Eg_array[1]-Eg_array[0], "a2y":0}

    def plot(self, title="", norm="log"):
        import matplotlib.pyplot as plt
        plot_object = None
        if norm == "log":
            from matplotlib.colors import LogNorm
            plot_object = plt.pcolormesh(self.Eg_array, self.Ex_array, self.matrix, norm=LogNorm(vmin=1e-1))
        else:
            plot_object = plt.pcolormesh(self.Eg_array, self.Ex_array, self.matrix)
        plt.title(title)
        plt.show()
        return True

    def save(self, fname):
        """
        Save matrix to mama file
        """
        pml.write_mama_2D(self.matrix, fname, self.Ex_array, self.Eg_array, comment="Made by pyma")
        return True

    def load(self, fname):
        """
        Load matrix from mama file
        """
        if self.matrix is not None:
            print("Warning: load() called on non-empty matrix", flush=True)

        # Load matrix from file:
        matrix, calibration, Ex_array, Eg_array = pml.read_mama_2D(fname)
        self.matrix = matrix
        self.Ex_array = Ex_array
        self.Eg_array = Eg_array
        self.calibration = calibration

        return True