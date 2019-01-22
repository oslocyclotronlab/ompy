# -*- coding: utf-8 -*-
# Test that the Vector class is behaving correctly, 
# including plotting
# TODO convert asserts to unittest system

from context import oslo_method_python as om
import matplotlib.pyplot as plt
import numpy as np

# Set precision for element-wise array comparison
decimal = 2

# Set up test array
E_array = np.linspace(-501, 3020, 100)
counts = np.random.normal(loc=0.01*E_array, size=len(E_array))

# Test rebinning on a simple case where neighbouring
# bins are joined
E_array_out = E_array[::2]
counts_rebinned = om.rebin(counts, E_array, E_array_out)
counts_manually_rebinned = counts.reshape(int(len(counts)/2), 2).sum(axis=1)
np.testing.assert_array_almost_equal(counts_rebinned,
                                     counts_manually_rebinned,
                                     decimal=decimal
                                     )

# Also test a complex case by comparing with a pure-python rebinning function:
# Look at this for inspiration, delete after:


def TEST_rebin_python(counts_in, Ein_array, Eout_array):
    """
    Rebins, i.e. redistributes the numbers from vector counts
    (which is assumed to have calibration given by Ein_array)
    into a new vector, which is returned.
    The total number of counts is preserved.
    In case of upsampling (smaller binsize out than in),
    the distribution of counts between bins is done by simple
    proportionality.
    Inputs:
    counts: Numpy array of counts in each bin
    Ein_array: Numpy array of energies, assumed to be linearly spaced,
               corresponding to the lower-bin-edge energies of counts
    Eout_array: Numpy array of energies corresponding to the desired
                rebinning of the output vector, also lower-bin-edge
    """

    # Get calibration coefficients and number of elements from array:
    Nin = len(Ein_array)
    a0_in, a1_in = Ein_array[0], Ein_array[1]-Ein_array[0]
    Nout = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]

    # Replace the arrays by bin-edge energy arrays of length N+1
    # (so that all bins are covered on both sides).
    Ein_array = np.linspace(a0_in, a0_in + a1_in*Nin, Nin+1)
    Eout_array = np.linspace(a0_out, a0_out + a1_out*Nout, Nout+1)

    # Allocate vector to fill with rebinned counts
    counts_out = np.zeros(Nout)
    # Loop over all indices in both arrays. Maybe this can be speeded up?
    for i in range(Nout):
        # for j in range(Nin):
        # Can we make it faster by looping j over a subset? What is a
        # sufficient subset?
        jmin = max(0, int((a0_out + a1_out*(i-1) - a0_in)/a1_in))
        jmax = min(Nin-1, int((a0_out + a1_out*(i+1) - a0_in)/a1_in))
        for j in range(jmin, jmax+1):
            # Calculate proportionality factor based on current overlap:
            overlap = (np.minimum(Eout_array[i+1], Ein_array[j+1])
                       - np.maximum(Eout_array[i], Ein_array[j]))
            overlap = overlap if overlap > 0 else 0
            counts_out[i] += counts_in[j] * overlap / a1_in

    return counts_out


# Change to larger binsize:
E_array_out = np.linspace(100, 4000, 111)
counts_python_rebinned = TEST_rebin_python(counts, E_array, E_array_out)
counts_rebinned = om.rebin(counts, E_array, E_array_out)

np.testing.assert_array_almost_equal(counts_rebinned,
                                     counts_python_rebinned,
                                     decimal=decimal)


# And to smaller binsize:
E_array_out = np.linspace(100, 2000, 11)
counts_python_rebinned = TEST_rebin_python(counts, E_array, E_array_out)
counts_rebinned = om.rebin(counts, E_array, E_array_out)

np.testing.assert_array_almost_equal(counts_rebinned,
                                     counts_python_rebinned,
                                     decimal=decimal)
