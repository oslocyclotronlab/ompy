import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import norm

# Import functions from unfold.py -- in the end assemble all functions into library file:
from unfold import *


# === Utility functions ===

def first_generation_spectrum(matrix, Ex_range_mat, Egamma_range, N_Exbins, Ex_max, dE_gamma, N_iterations=1):
	"""
	Function implementing the first generation method from Guttormsen et al. (NIM 1987)
	The code is heavily influenced by the original implementation by Magne in MAMA.
	Mainly written autumn 2016 at MSU
	"""


	Ny = len(matrix[:,0])
	Nx = len(matrix[0,:])
	# Extract / calculate calibration coefficients
	bx = Egamma_range[0]
	ax = Egamma_range[1] - Egamma_range[0]
	by = Ex_range_mat[0]
	ay = Ex_range_mat[1] - Ex_range_mat[0]

	statistical_or_total = 1
	ThresSta = 430.0
	# AreaCorr = 1
	ThresTot = 	200.000000
	ThresRatio = 0.3000000
	# ExH = 7520.00000
	ExEntry0s = 300.000000
	ExEntry0t = 0.00000000
	apply_area_correction = True


	# Ex_max = 7500 # keV - maximum excitation energy
	# Ex_min = 300 # keV - minimal excitation energy, effectively moving the ground-state energy up because we cannot resolve the low-energy yrast gamma lines. This is weighed up by also using an effective multiplicity which is lower than the real one, again not considering the low-energy yrast gammas.
	# dE_gamma = 300 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
	# Ex_binsize = 40 # keV - bin size that we want on y axis
	# N_Exbins = 120 # Number of excitation energy bins (NB! It will only rebin in whole multiples, so a slight change in N_Exbins might only result in getting some more empty bins on top.)
	# N_Exbins_original = (Ex_max+dE_gamma)/ay # The number of bins between 0 and Ex_max + dE_gamma in the original matrix
	# grouping = int(np.ceil(len(y_array[np.logical_and(0 < y_array, y_array < Ex_max + dE_gamma)])/N_Exbins)) # The integer number of bins that need to be grouped to have approximately N_Exbins bins between Ex_min and Ex_max after compression (rounded up)

	# Make arrays of Ex and Egamma axis values
	Ex_range = np.linspace(by, Ex_max + dE_gamma, N_Exbins)
	Egamma_range = np.linspace(0,Nx-1,Nx)*ax + bx # Range of Egamma values
	
	# Compress matrix along Ex
	#matrix_ex_compressed = matrix[0:int(N_Exbins*grouping),:].reshape(N_Exbins, grouping, Nx).sum(axis=1)
	matrix_ex_compressed = rebin(matrix[0:int((Ex_max+dE_gamma)/Ex_range_mat.max()*Ny),:], N_Exbins, rebin_axis = 0) # This seems crazy. Does it cut away anything at all?
	# print Ny, N_Exbins, N_Exbins_original	
	# plt.pcolormesh(Egamma_range, Ex_range, matrix_ex_compressed, norm=LogNorm(vmin=0.001, vmax=matrix_ex_compressed.max()))
	# plt.matshow(matrix_ex_compressed)
	# plt.colorbar()
	# plt.show()

	# Remove counts in matrix for Ex higher than Ex_max:
	matrix_ex_compressed[Ex_range>Ex_max, :] = 0
	# plt.matshow(matrix_ex_compressed)
	# plt.colorbar()
	# plt.show()


	# ==== Calculate multiplicities: ====

	# Setup meshgrids for making boolean indexing arrays
	Egamma_mesh, Ex_mesh = np.meshgrid(Egamma_range, Ex_range)
	Egamma_max = Ex_range + dE_gamma # Maximal Egamma value for each Ex bin
	Egamma_max_grid = np.meshgrid(np.ones(Nx), Egamma_max)[1]
	if statistical_or_total == 1:
		# Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
		slide = np.minimum( np.maximum(ThresRatio*Ex_mesh, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.
		# plt.figure(5)
		# plt.plot(slide[:,0])
		# plt.show()
		# sys.exit(0)
		# good_indices = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid) , True, False)
		matrix_ex_compressed_cut = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), matrix_ex_compressed, 0)
	elif statistical_or_total == 2:
		# Total multiplicity calculation
		# good_indices = np.where(Egamma_mesh < Egamma_max_grid, True, False)
		matrix_ex_compressed_cut = np.where(Egamma_mesh < Egamma_max_grid, matrix_ex_compressed, 0)
	# for i in range(len(good_indices[:,0])):
	# 	print len(good_indices[i,good_indices[i,:]]) # OK, it actually works.
	
	# Cut away counts higher than Egamma = Ex + dE_gamma
	# matrix_ex_compressed_cut = np.where(good_indices, matrix_ex_compressed, 0)
	# plt.figure(1)
	# plt.pcolormesh(Egamma_range, Ex_range, matrix_ex_compressed_cut, norm=LogNorm(vmin=0.01, vmax=matrix_ex_compressed.max()))
	# plt.show()
	# sys.exit(0)

	# Calculate average multiplicity for each Ex channel
	area_matrix_ex_compressed_cut = np.sum(matrix_ex_compressed_cut, axis=1)
	Egamma_average = div0( np.sum(Egamma_mesh * matrix_ex_compressed_cut, axis =1) , area_matrix_ex_compressed_cut )
	if statistical_or_total == 1:
		# Statistical multiplicity - use the effective Ex0 value
		multiplicity = div0( Ex_range - np.maximum( np.minimum(Ex_range - 200, ExEntry0s), 0), Egamma_average)
	elif statistical_or_total == 2:
		# Total multiplicity - use actual Ex0 = 0
		multiplicity = div0( Ex_range, Egamma_average )

	# plt.figure(2)
	# plt.step(Ex_range, multiplicity) # This rises like a straight line from 0 to about 3-4 - seems very right!
	# plt.show()
	# sys.exit(0)

	# Set up dummy first-generation matrix to start iterations, made of normalized boxes:
	H = np.zeros((N_Exbins, Nx))
	for i in range(N_Exbins):
		Ni = len(Egamma_range[Egamma_range<Ex_range[i] + dE_gamma])
		# print Ni
		H[i, Egamma_range < Ex_range[i] + dE_gamma] = 1/Ni
	# print np.sum(H0, axis=1) # Seems to work!

	# Set up normalization matrix N
	area = np.sum(matrix_ex_compressed_cut, axis=1) # Get total number of counts in each Ex bin
	# plt.plot(Ex_range, area)
	# plt.show()

	area_grid = np.tile(area, (N_Exbins, 1)) # Copy the array N_Exbins times down to make a square matrix
	# print area_grid.shape
	multiplicity_grid = np.tile(multiplicity, (N_Exbins, 1)) 
	# print multiplicity_grid.shape
	normalization_matrix = div0(( np.transpose(multiplicity_grid) * area_grid ) , (multiplicity_grid * np.transpose(area_grid) )).T # The transpose gives the right result. Haven't twisted my head around exactly why.
	# normalization_matrix_check = np.zeros((N_Exbins, N_Exbins))
	# for i in range(N_Exbins):
	# 	for j in range(N_Exbins):
	# 		normalization_matrix_check[i, j] = multiplicity[i]*area[j]/(multiplicity[j]*area[i])
	normalization_matrix[np.isnan(normalization_matrix)] = 0
	# plt.matshow(normalization_matrix, origin='lower', norm=LogNorm(vmin=0.01, vmax=normalization_matrix.max()))
	# plt.show()
	# plt.matshow(normalization_matrix_check, origin='lower') # There is a difference of a transposition, check which is the right one
	# plt.show()

	# Set up compression parameters for Egamma axis to be used by H below:
	i_Egamma_max = np.where(Egamma_range > Ex_max+ dE_gamma)[0][0] # Get the maximal allowed gamma energy (need to make H square, thus Egamma <= Ex + dE_gamma, since that's the maximal Ex channel in the compressed matrix)
	# print i_Egamma_max, Egamma_range[i_Egamma_max], N_Exbins, int(i_Egamma_max/N_Exbins)
	# i_Egamma_max = i_Egamma_max + N_Exbins - i_Egamma_max%N_Exbins # Make sure the number of indices is a whole multiple of N_Exbins (rounded up)
	# print i_Egamma_max
	grouping_Egamma = int(np.ceil(i_Egamma_max/N_Exbins))
	# print grouping_Egamma
	# Egamma_range_compressed = Egamma_range[0:i_Egamma_max]*grouping_Egamma
	Egamma_range_compressed = Ex_range

	# plt.matshow(H[:,0:i_Egamma_max])
	# plt.show()
	# H_extended = np.insert(H[:,0:i_Egamma_max], np.linspace(0,i_Egamma_max, N_Exbins - i_Egamma_max%N_Exbins), H[:,(np.linspace(0,i_Egamma_max, N_Exbins - i_Egamma_max%N_Exbins).astype(int))], axis=1)
	# H_extended[:,grouping_Egamma:-1:grouping_Egamma] /= 2
	# H_extended[:,grouping_Egamma+1:-2:grouping_Egamma] /= 2
	# H_extended = H[:,0:i_Egamma_max].repeat(N_Exbins).reshape(len(H[:,0]),N_Exbins,i_Egamma_max).sum(axis=2)/N_Exbins
	# plt.matshow(H_extended)
	# plt.show()
	# H_compressed = H[:,0:i_Egamma_max+ N_Exbins - i_Egamma_max%N_Exbins].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
	# plt.matshow(H_compressed)
	# plt.show()
	# H_compressed = rebin(H[:,0:i_Egamma_max], N_Exbins, 1)
	# plt.matshow(H_compressed)
	# plt.show()

	# H_compressed_extended = H_extended.reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
	# plt.matshow(H_compressed_extended)
	# plt.show()

	# sys.exit(0)

	# Declare variables which will define the limits for the diff spectrum colorbar (for plotting purposes)
	vmin_spec = -200
	vmax_spec = 200
	vmin_diff = -100
	vmax_diff = 100

	# Perform the iterative subtraction:
	for iteration in range(N_iterations):
	# convergence_criterion = 1
	# max_diff = 100
	# while max_diff > convergence_criterion:
		# Store H from previous iteration to compare at the end
		H_old = H
		# Compress the H matrix along gamma axis to facilitate conversion to excitation energy
		# H_compressed = H[:,0:i_Egamma_max].reshape(N_Exbins, N_Exbins, grouping_Egamma).sum(axis=2)
		H_compressed = rebin(H[:,0:i_Egamma_max], N_Exbins, rebin_axis=1)

		# plt.pcolormesh(Egamma_range_compressed, Ex_range, H_compressed)
		# plt.show()

		# Convert first-generation spectra H into weights W
		W = np.zeros((N_Exbins, N_Exbins))
		for i in range(0,N_Exbins):
			# print H_compressed[i,i:0:-1].shape
			W[i,0:i] = H_compressed[i,i:0:-1]
		# plt.matshow(W, origin='lower', vmin=W.min(), vmax=W.max())
		# plt.colorbar()
		# plt.title('Before')
		# plt.show()
		# Remove negative weights
		W[W<0] = 0
		# Normalize each Ex channel to unity
		# W = np.where(np.invert(np.isnan(W/W.sum(axis=1).astype(float))),  W/W.sum(axis=1).astype(float), 0)
		# Remove Inf and NaN
		W = div0(W, W.sum(axis=1).reshape(N_Exbins,1))
		# W = np.nan_to_num(W) 
		# plt.matshow(W, origin='lower', vmin=W.min(), vmax=W.max())
		# plt.colorbar()
		# plt.title('After')
		# plt.show()

		# sys.exit(0)

		# print "W = "
		# print W
		# print "matrix_ex_compressed = "
		# print matrix_ex_compressed
		# print "product ="
		# plt.matshow(np.dot(W, matrix_ex_compressed), origin='lower', norm=LogNorm())
		# plt.show()

		# Calculate product of normalization matrix, weight matrix and raw count matrix
		G = np.dot( (normalization_matrix * W), matrix_ex_compressed) # Matrix of weighted sum of spectra below
		
		# Apply area correction
		if apply_area_correction:
			# Setup meshgrids for making boolean indexing arrays
			# Egamma_mesh_compressed, Ex_mesh_compressed = np.meshgrid(Egamma_range_compressed, Ex_range)
			# Egamma_max = Ex_range + dE_gamma # Maximal Egamma value for each Ex bin
			# Egamma_max_grid_compressed = np.meshgrid(np.ones(N_Exbins), Egamma_max)[1]
			# print "Egamma_mesh_compressed, Egamma_max, Egamma_max_grid"
			# print Egamma_mesh_compressed.shape, Egamma_max.shape, Egamma_max_grid.shape
			if statistical_or_total == 1:
				# Statistical multiplicity calculation (i.e. trying to use statistical/continuum region only)
				# slide_compressed = np.minimum( np.maximum(ThresRatio*Ex_mesh_compressed, ThresTot), ThresSta ) # The sliding lower limit for Egamma integral - sliding between ThresTot and ThresSta.
				# print "slide_compressed"
				# print slide_compressed.shape
				# plt.figure(5)
				# plt.plot(slide[:,0])
				# plt.show()
				# sys.exit(0)
				# good_indices_G = np.where(np.logical_and(slide_compressed < Egamma_mesh_compressed, Egamma_mesh_compressed < Egamma_max_grid) , True, False)
				G_area = np.where(np.logical_and(slide < Egamma_mesh, Egamma_mesh < Egamma_max_grid), G, 0).sum(axis=1)
			elif statistical_or_total == 2:
				# Total multiplicity calculation
				# good_indices_G = np.where(Egamma_mesh_compressed < Egamma_max_grid, True, False)
				G_area = np.where(Egamma_mesh_compressed < Egamma_max_grid, G, 0).sum(axis=1)
			# G_area = np.where(good_indices_G, G, 0).sum(axis=1)
			# print "print G_area.shape"
			# print G_area.shape
			# print "print G_area"
			# print G_area
			alpha = np.where(G_area > 0, (1 - div0(1,multiplicity)) * div0( area_matrix_ex_compressed_cut, G_area ), 1)
			alpha[alpha < 0.85] = 0.85
			alpha[alpha > 1.15] = 1.15
			# print "alpha.shape"
			# print alpha.shape
			# print "alpha"
			# print alpha
		else:
			alpha = np.ones(N_Exbins)


		# The actual subtraction
		H = matrix_ex_compressed - alpha.reshape((len(alpha), 1))*G
		# print H.shape
		# Plotting:
		# vmin_diff = (H-H_old).min()
		# vmax_diff = (H-H_old).max()
		# vmin_spec = H.min()
		# vmax_spec = H.max()

		# plt.figure(10)
		# plt.subplot(1,2,1)
		# plt.title('First gen spectrum, current')
		# plt.pcolormesh(Egamma_range, Ex_range, H, norm=LogNorm(vmin=0.01, vmax=vmax_spec))
		# plt.colorbar()
		# plt.subplot(1,2,2)
		

		# plt.title('Diff with previous')
		# plt.pcolormesh(Egamma_range, Ex_range, H-H_old, vmin=vmin_diff, vmax=vmax_diff)
		# plt.colorbar()
		# plt.show()



		# Check convergence
		max_diff = np.max(np.power(H-H_old,2))
		# print max_diff

	
	# Remove negative counts
	H[H<0] = 0
	# Return
	return H, H-H_old, Egamma_range, Ex_range

def rebin(array, N_final, rebin_axis=0):
	# Function to rebin an M-dimensional array either to larger or smaller binsize.
	# Rebinning is done with simple proportionality. E.g. for down-scaling rebinning (N_final < N_initial): 
	# if a bin in the original spacing ends up between two bins in the reduced spacing, 
	# then the counts of that bin is split proportionally between adjacent bins in the 
	# rebinned array. 
	# Upward binning (N_final > N_initial) is done in the same way, dividing the content of bins
	# equally among adjacent bins.

	# Technically it's done by repeating each element of array N_final times and dividing by N_final to 
	# preserve total number of counts, then reshaping the array from M dimensions to M+1 before flattening 
	# along the new dimension of length N_initial, resulting in an array of the desired dimensionality.
	indices = np.insert(array.shape, rebin_axis, N_final) # Indices to reshape 
	return array.repeat(N_final, axis=rebin_axis).reshape(indices).sum(axis=(rebin_axis+1))/float(N_final)




# === run ===
if __name__=="__main__":
	
	fname_unf = "unfolded-28Si.m"
	unfolded, cal_unf, Ex_array_unf, Eg_array_unf = read_mama(fname_unf)

	Ex_max = 12000 # keV - maximum excitation energy
	dE_gamma = 300 # keV - allow gamma energy to exceed excitation energy by this much, to account for experimental resolution
	N_Exbins = 300
	firstgen, diff, Eg_array_fg, Ex_array_fg = first_generation_spectrum(unfolded, Eg_array_unf, Ex_array_unf, N_Exbins, Ex_max, dE_gamma, N_iterations=20)



	write_mama(firstgen, 'firstgen-28Si.m', Eg_array_fg, Ex_array_fg, comment="Made using firstgen.py by JEM, during development of pyma, summer 2018")	
	
	# Diagnostic plots:
	fig, (ax_unf, ax_fg) = plt.subplots(2,1)
	ax_unf.pcolormesh(Eg_array_unf, Ex_array_unf, unfolded, norm=LogNorm(vmin=1), cmap="jet")
	ax_fg.pcolormesh(Eg_array_fg, Ex_array_fg, firstgen, norm=LogNorm(vmin=1), cmap="jet")
	
	
	plt.show()
