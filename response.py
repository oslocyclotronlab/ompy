import sys, os
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import matplotlib.pyplot as plt

from firstgen import *
from unfold import *




def response(folderpath, Eg_array):
    """
    Function to make response matrix and related arrays from 
    source files. 
    Assumes the source files are in the folder "folderpath",
    and that they are formatted in a certain standard way.

    The function interpolates the data to give a response matrix with
    the desired energy binning specified by Eg_array.
    """
    # Read resp.dat file, which gives information about the energy bins 
    # and discrete peaks
    resp = []
    Nlines = -1
    with open(os.path.join(folderpath, "resp.dat")) as file:
        while True:
            line = file.readline()
            if not line: 
                break
            if line[0:22] == "# Next: Numer of Lines":
                line = file.readline()
                Nlines = int(line)
                print("Nlines =", Nlines)
                break

        line = file.readline()
        print("line =", line)
        if not line:
            raise Exception("Error reading resp.dat")

        for i in range(Nlines):
            line = file.readline()
            print("line =", line)
            row = np.array(line.split(), dtype="double")
            resp.append(row)
    
    
    resp = np.array(resp)
    print("resp =", resp)
    print("energies = ", resp[:,0])





    # return R, FWHM, eff, pf, pc, ps, pd, pa
    return True



if __name__ == "__main__":
    folderpath = "oscar2017_scale1.0"
    Eg_array = np.linspace(0,2000, 10)
    print("response(\"{:s}\") =".format(folderpath), response(folderpath, Eg_array))