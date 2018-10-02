from unfold import *
import matplotlib.pyplot as plt 
from matplotlib.colors import LogNorm


fname_resp = 'data/resp-Si28-14keV.dat'
fname_resp_mat = 'data/response_matrix-Si28-14keV.m'
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)
# R_2D = div0(R_2D , R_2D.sum(rebin_axis=1))
E_thres = 100
i_thres = np.argmin(np.abs(E_resp_array - E_thres))
R_2D[:,:i_thres] = 0
for i in range(R_2D.shape[0]):
	try:
		R_2D[i,:] = R_2D[i,:] / R_2D[i,:].sum()
	except:
		R_2D[i,:] = 0


# f_cmp, ax_cmp = plt.subplots(1,1)
# ax_cmp.plot(E_resp_array, R_2D[400,:])



resp = []
with open(fname_resp) as file:
    # Read line by line as there is crazyness in the file format
    lines = file.readlines()
    for i in range(4,len(lines)):
        try:
            row = np.array(lines[i].split(), dtype="double")
            resp.append(row)
        except:
            break




# plt.pcolormesh(E_resp_array, E_resp_array, R_2D, norm=LogNorm())
# sys.exit(0)


Emin = 100
Emax = 10*1e3

Nbins = len(E_resp_array)
matrix = np.zeros((Nbins, Nbins))

N_draws = 100
N_resp_draws = 1000
M = 2

np.random.seed(2)
# E1s = np.random.uniform(low=Emin, high=Emax, size=N_draws)
E1s = np.random.triangular(left=Emin, mode=(Emax+Emin)/2, right=Emax, size=N_draws)
# E1s = 5*1e3*np.ones(N_draws)
E2s = Emax - E1s
# E2s = np.random.uniform(low=Emin, high=Emax, size=N_draws)

indices_E_resp_array = np.linspace(0,len(E_resp_array)-1, len(E_resp_array)).astype(int)

for i_draw in range(N_draws):
	E1 = E1s[i_draw]
	E2 = E2s[i_draw]
	index_E1 = np.argmin(np.abs(E_resp_array - E1))
	index_E2 = np.argmin(np.abs(E_resp_array - E2))
	# print("E1 =", E1, " index_E1 =", index_E1, " E_resp_array[{:d}] =".format(index_E1), E_resp_array[index_E1])


	# print("R_2D[index_E1,:].sum() =", R_2D[index_E1,:].sum())
	E1_folded = np.random.choice(E_resp_array, size=N_resp_draws, p=R_2D[index_E1,:])
	E2_folded = np.random.choice(E_resp_array, size=N_resp_draws, p=R_2D[index_E2,:])

	Ex_folded = E1_folded + E2_folded

	# print("E1_folded =", E1_folded, flush=True)

	for i_resp_draws in range(N_resp_draws):
		i_Ex = np.argmin(np.abs(E_resp_array - Ex_folded[i_resp_draws]))
		i_E1 = np.argmin(np.abs(E_resp_array - E1_folded[i_resp_draws]))
		i_E2 = np.argmin(np.abs(E_resp_array - E2_folded[i_resp_draws]))
		matrix[i_Ex,i_E1] += 1
		matrix[i_Ex,i_E2] += 1



# print(matrix[matrix>0], flush=True)

N_final = int(len(E_resp_array)/6)
matrix_rebinned, E_resp_array_rebinned = rebin_and_shift(rebin_and_shift(matrix, E_resp_array, N_final=N_final, rebin_axis=0), E_resp_array, N_final=N_final, rebin_axis=1)

f_max, ax_mat = plt.subplots(1,1)
ax_mat.pcolormesh(E_resp_array, E_resp_array, matrix, norm=LogNorm())
plt.show()




