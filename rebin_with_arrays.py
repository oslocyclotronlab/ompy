import numpy as np 
import matplotlib.pyplot as plt 




def rebin_by_arrays(counts_in, Ein_array, Eout_array):
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
               corresponding to the middle-bin energies of counts
    Eout_array: Numpy array of energies corresponding to the desired
                rebinning of the output vector, also middle-bin
    """

    # Get calibration coefficients and number of elements from array:
    Nin = len(Ein_array)
    a0_in, a1_in = Ein_array[0], Ein_array[1]-Ein_array[0]
    Nout = len(Eout_array)
    a0_out, a1_out = Eout_array[0], Eout_array[1]-Eout_array[0]

    # Replace the arrays by bin-edge energy arrays of length N+1 
    # (so that all bins are covered on both sides).
    Ein_array = np.linspace(a0_in - a1_in/2, a0_in - a1_in/2 + a1_in*Nin, Nin+1)
    Eout_array = np.linspace(a0_out - a1_out/2, a0_out - a1_out/2 + a1_out*Nout, Nout+1)

    print("Ein_array bin-edge =", Ein_array)
    print("Eout_array bin-edge =", Eout_array)



    # Allocate vector to fill with rebinned counts
    counts_out = np.zeros(Nout)
    # Loop over all indices in both arrays. Maybe this can be speeded up?
    for i in range(Nout):
        for j in range(Nin):
            # Calculate proportionality factor based on current overlap:
            overlap = np.minimum(Eout_array[i+1], Ein_array[j+1]) - np.maximum(Eout_array[i], Ein_array[j])
            overlap = overlap if overlap > 0 else 0
            counts_out[i] += counts_in[j] * overlap / a1_in
            if i == 2:
                print("i =", i, ", j =", j, "overlap =", overlap, "prop =", overlap/a1_in, flush=True)
                print("Eout_array[i+1], Ein_array[j+1], Eout_array[i], Ein_array[j] =", Eout_array[i+1], Ein_array[j+1], Eout_array[i], Ein_array[j], flush=True)

    return counts_out







a0_in, a1_in = 10, 5
Nbins_in = 20
Ein_array = np.linspace(a0_in, a0_in + a1_in*(Nbins_in-1), Nbins_in )
print("Ein_array =", Ein_array)

np.random.seed(2)
counts_in = np.random.uniform(low=0, high=100, size=Nbins_in)


a0_out, a1_out = 1, 2
# a0_out, a1_out = 0, 4
# a0_out, a1_out = 0, 10
# a0_out, a1_out = a0_in, a1_in # This makes no change, as it should be.
Nbins_out = 40
Eout_array = np.linspace(a0_out, a0_out + a1_out*(Nbins_out-1), Nbins_out )
print("Eout_array =", Eout_array)

counts_out = rebin_by_arrays(counts_in, Ein_array, Eout_array)


print("counts_in.sum() =", counts_in.sum(), ", counts_out.sum() =", counts_out.sum(), flush=True)



f, ax = plt.subplots(1,1)

# ax.step(Ein_array, counts_in, label="in", where="mid")
# ax.step(Eout_array, counts_out, label="out", where="mid")
ax.bar(Eout_array, height=counts_out, width=a1_out, edgecolor="k", label="out", alpha=0.8)
ax.bar(Ein_array, height=counts_in, width=a1_in, edgecolor="k", label="in", alpha=0.8)
ax.legend()

plt.show()



