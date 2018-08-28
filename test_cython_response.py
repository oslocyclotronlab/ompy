from response_cython import *

from unfold import *

folderpath = "oscar2017_scale1.15"
a0_resp = 0
a1_resp = 7.0 # keV
E_resp_max = 1.4e4
N_resp = int((E_resp_max-a0_resp)/a1_resp + 0.5)
E_resp_array = np.linspace(a0_resp, a0_resp + a1_resp*(N_resp-1),N_resp)
print("E_resp_array =", E_resp_array)
# import sys
# sys.exit(0)
FWHM = 6.8 # keV - value at 1.33 MeV
R, FWHM_rel, Eff_tot, pcmp, pFE, pSE, pDE, p511 = response(folderpath, E_resp_array, FWHM)

write_mama_2D(R, "response_interpolation_test.m", E_resp_array, E_resp_array, comment="Made by JEM using response.pyx August 2018.")

import matplotlib.pyplot as plt 
f, ax = plt.subplots(1,1)

for i_plt in [np.linspace(0,N_resp-1,10).astype(int)]:
        ax.plot(E_resp_array, R[i_plt,:], label="interpolated, E_g = {:.0f}".format(E_resp_array[i_plt]), linestyle="--")


ax.legend()
plt.show()