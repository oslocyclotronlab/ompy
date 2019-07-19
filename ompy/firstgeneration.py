#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Implementation of the first generation method
(Guttormsen, Ramsøy and Rekstad, Nuclear Instruments and Methods in
Physics Research A 255 (1987).)

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
import copy
import logging
import numpy as np
from .matrix import Matrix

LOG = logging.getLogger(__name__)
logging.captureWarnings(True)


class FirstGeneration:
    def __init__(self):
        self.threshold_statistical = 430.0
        self.threshold_total = 200.0
        self.threshold_ratio = 0.3
        self.Ex_entry_0t = 0.0
        self.Ex_entry_0s = 300.0

    def apply(self, unfolded: Matrix):
        matrix = copy.deepcopy(unfolded)
        # We don't want negative energies
        matrix.cut('Ex', Emin=0.0)

        # Get some numbers:
        Nx, Ny = matrix.shape
        calib_in = matrix.calibration()
        bx = calib_in["a10"]
        ax = calib_in["a11"]
        by = calib_in["a00"]
        ay = calib_in["a01"]

        # Remove all counts above Ex_max
        cut[cut.Ex > self.Ex_max, :] = 0

        self.Eg_max = matrix.Ex + self.dEg

        multiplicity = self.multiplicity()

    def multiplicty(self, matrix: Matrix):
        Eg_mesh, Ex_mesh = np.meshgrid(matrix.Eg, matrix.Ex)
        slide = None
