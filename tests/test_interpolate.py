# -*- coding: utf-8 -*-
from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

matrix_in = np.array([[1, 2, 3], [4, 5, 6]])
E0_array = np.array([4.5, 6])
E1_array = np.array([5, 6, 7])

E0_array_out = np.linspace(4, 8, 12)
E1_array_out = np.linspace(4, 9, 10)

matrix_1D_E0 = om.interpolate_matrix_1D(matrix_in, E0_array, E0_array_out,
                                        axis=0)
matrix_1D_E1 = om.interpolate_matrix_1D(matrix_in, E1_array, E1_array_out,
                                        axis=1)
matrix_1D_both = om.interpolate_matrix_1D(matrix_1D_E0, E1_array, E1_array_out,
                                          axis=1)

matrix_2D = om.interpolate_matrix_2D(matrix_in, E0_array, E1_array,
                                     E0_array_out, E1_array_out)

# Plot
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.pcolormesh(matrix_in)
ax1.set_title("orig")
ax2.pcolormesh(matrix_1D_E0)
ax2.set_title("1D E0")
ax3.pcolormesh(matrix_1D_E1)
ax3.set_title("1D E1")
ax4.pcolormesh(matrix_1D_both)
ax4.set_title("1D both")
ax5.pcolormesh(matrix_2D)
ax5.set_title("2D")

plt.tight_layout()
plt.show()
