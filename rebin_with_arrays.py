import numpy as np 
import matplotlib.pyplot as plt 


a0_in, a1_in = 10, 5
Nbins_in = 20
Ein_array = np.linspace(a0_in, a0_in + a1_in*(Nbins_in-1), Nbins_in )
print("Ein_array =", Ein_array)

counts_in = np.random.uniform(low=0, high=100, size=Nbins_in)


a0_out, a1_out = 0, 2
Nbins_out = 200
Eout_array = np.linspace(a0_out, a0_out + a1_out*(Nbins_out-1), Nbins_out )
print("Eout_array =", Eout_array)







f, ax = plt.subplots(1,1)

ax.step(Ein_array, counts_in, where="post")

plt.show()



