from unfold import *


fname_resp = 'resp.dat'
fname_resp_mat = 'response-si28-20171112.m'
R_2D, cal_resp, E_resp_array, tmp = read_mama_2D(fname_resp_mat)


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