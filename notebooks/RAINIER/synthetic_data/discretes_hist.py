import numpy as np
import matplotlib.pyplot as plt

# create histogram over discrete levels of levels.dat

levels = np.loadtxt("counting.dat")
levels /= 1e3 # kev to Mev
binwitdth = 0.125 # MeV
Emax = levels[-1]
print Emax
nbins = int(np.ceil(Emax/binwitdth))
Emax_adjusted = binwitdth*nbins # Trick to get an integer number of bins
bins = np.linspace(0,Emax_adjusted,nbins+1)

res = np.histogram(levels,bins=bins)
hist,_ =res
for i in range(nbins):
	print bins[i], hist[i]/binwitdth

print bins-binwitdth/2

f_rho, ax_rho = plt.subplots(1,1)
ax_rho.step(bins-binwitdth/2, np.append(0,hist/binwitdth),where="pre")
ax_rho.set_yscale('log')
plt.show()
