# Compare rebin functions between Python and Cython

from context import oslo_method_python as om
import numpy as np
import time
import matplotlib.pyplot as plt
import sys

N = 1000

# Test the calc_overlap function
overlap = om.calc_overlap(0, 1, 0, 2)
print("overlap =", overlap)
# Seems to work!
# sys.exit(0)

counts_in = np.random.uniform(size=N)
print("counts_in.dtype =", counts_in.dtype, flush=True)
E_array_in = np.linspace(0, 5000, N)

# E_array_out = np.linspace(200, 5000, 300)
E_array_out = np.linspace(0, 5000, int(N/2))

t1 = time.time()
counts_out_python = om.rebin_python(counts_in, E_array_in, E_array_out)
t2 = time.time()
print("python time =", t2-t1)

t1 = time.time()
counts_out_cython = om.rebin_cython(counts_in, E_array_in, E_array_out)
t2 = time.time()
print("cython time =", t2-t1)

# Plot them all
f, ax = plt.subplots(1, 1)
ax.step(E_array_in, counts_in, where="post", label="in")
ax.step(E_array_out, counts_out_python, where="post", label="out (python)")
ax.step(E_array_out, counts_out_cython, where="post", label="out (cython)")

ax.legend()
plt.show()


# Related test: What is sufficient to get complete overlap between 
# initial and final bins?
# Get calibration coefficients and number of elements from array:
Nin = len(E_array_in)
a0_in, a1_in = E_array_in[0], E_array_in[1]-E_array_in[0]
Nout = len(E_array_out)
a0_out, a1_out = E_array_out[0], E_array_out[1]-E_array_out[0]

# for i in range(Nout):
#     jmin = max(0, int((a0_out + a1_out*(i-1) - a0_in)/a1_in))
#     jmax = min(Nin-1, int((a0_out + a1_out*(i+1) - a0_in)/a1_in))
#     print("E_array_out[i] =", E_array_out[i])
#     for j in range(jmin, jmax+1):
#         print("E_array_in[j] =", E_array_in[j])
#         if E_array_in[jmin] > E_array_out[i]:
#             print("E_array_in[jmin] > E_array_out[i]")
#         if E_array_in[jmax]+a1_in < E_array_out[i]+a1_out:
#             print("E_array_in[jmax]+a1_in = E_array_in[jmax+1] < E_array_out[i]+a1_out=E_array_out[i+1]")

# Conclusion: Looks like it's only a problem on the edges, 
# and that's always going to be tricky.