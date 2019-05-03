Concept DOI: [![DOI](https://zenodo.org/badge/141709973.svg)](https://zenodo.org/badge/latestdoi/141709973)

If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above.


NB! OMpy is currently in beta. Use at your own risk!

# OMpy

This is `ompy`, the Oslo method in python. It contains all the functionality needed to go from a raw coincidence matrix, via unfolding and the first-generation method, to fitting a level density and gamma-ray strength function. It also supports uncertainty propagation by Monte Carlo.


## How to use this package
First off, make sure to compile the Cython modules by doing (in the main repo folder)
```
python setup.py build_ext --inplace
```

All the functions and classes in the package are available in the main module. You get everything with the import

```
import ompy
```

The overarching philosophy is that the package shall be flexible and transparent to use and modify. We would rather open up for possible misuse of the code than hinder the user from doing something. Therefore, there are several ways to use the package, which will now be outlined:

## The Matrix() and Vector() classes
The two most important utility classes in the package are `ompy.Matrix()` and `ompy.Vector()`. They are used to store matrices (2D) or vectors (1D) of numbers (typically spectra of counts) along with energy calibration information. These are just numpy arrays, and the structure is like this:
```
mat = ompy.Matrix()
mat.matrix # A 2D numpy array
mat.E0_array # Array of lower-bin-edge energy values for axis 0 (i.e. the row axis, or y axis)
mat.E1_array # Array of lower-bin-edge energy values for axis 1 (i.e. the column axis, or x axis)

vec = ompy.Vector()
vec.vector # A 1D numpy array
vec.E_array # Array of lower-bin-edge energy values for the single axis
```
 They also contain some other methods:
```
mat.load(filename) # Load a spectrum from a 2D MAMA file (similar, but 1D, for vec)
mat.save(filename) # Save spectrum to MAMA file

```

## Matrix manipulation
The core of the Oslo method involves working with two-dimensional spectra, the excitation-energy-gamma-ray-energy-matrices, often called alfna matrices for obscure reasons.<sup>1</sup> Starting with a raw matrix of $E_x$-$E_\gamma$ coincidences, you typically want to unfold the counts along the gamma-energy axis and then apply the first-generation method to obtain the matrix of first-generation, or primary, gamma rays from the decaying nucleus. 


### The unfold() function
An implementation of the unfolding method presented in Guttormsen et al., Nuclear Instruments and Methods in Physics Research A 374 (1996).

It may contain bugs. Please report any bugs you encounter to the issue tracker, or if you like, fix them and make a pull request.

It also currently lacks some functionality. Notably, the Compton subtraction method has some issues that we are working on.

#### How to make a response matrix
At present, the module doesn't contain a method for interpolating response functions. We have written one, but it needs to be cythonized because it's very slow. 

Because of this, if you want to do unfolding then you need to specify one or two file paths in the keywords `fname_resp_mat` and `fname_resp_dat`, respectively. The first is the response matrix. It needs to have the same energy calibration as the gamma-energy axis of the spectrum you're unfolding, and therefore it must be interpolated by MAMA. The second is only needed if you want to use the Compton subtraction method (the keyword `use_comptonsubtraction = True`). It is a list of probabilities for full-energy, single escape, double escape, etc., for each gamma-ray energy.

To make the response matrix: Open you raw matrix with MAMA. Type `rm` to make the response matrix to your specifications. Then type `gr` to load the response matrix into view, and `wr` to write it to disk.

If you want to do the Compton subtraction method (which doesn't work as of February 2019), you need the response parameters. They are automatically written to the file `resp.dat` when you run the `rm` command, but MAMA by default only prints a subset of the energy data points. To fix this, you need to edit the MAMA source file `folding.f` and comment out the line

```     IF(RDIM.GT.50)iStep=RDIM/50                  !steps for output```

which in the MAMA version I have is located at about line 1980. Then recompile MAMA.

### The first_generation_method() function
An implementation of the first generation method present in Guttormsen, Ramsøy and Rekstad, Nuclear Instruments and Methods in Physics Research A 255 (1987). 

Like the unfolding function, please be on the lookout for bugs.

### The MatrixAnalysis class
The steps described above make up the core matrix manipulation routines of `OMpy`. We therefore provide a class which helps to streamline the process, called `MatrixAnalysis`.
```
ma = ompy.MatrixAnalysis()
# Load raw matrix:
ma.raw.load(fname_raw)
# Unfold the raw matrix and place it
# in the variable ma.unfolded:
ma.unfold(*args)
# Apply first generation method 
# and get result in ma.firstgen:
ma.first_generation_method(*args)
```
The class methods `ma.unfold()` and `ma.first_generation_method()` accept the same input arguments as their respective stand-alone functions. The advantage of the `MatrixAnalysis` class is that all parameter choices for the unfolding and first-generation method are stored within the class instance. For this reason, the `MatrixAnalysis` class is necessary when one wants to do uncertainty propagation, to ensure that the same settings are applied to each member of the ensemble. 

## The ErrorPropagation class
To perform the uncertainty propagation, instantiate the class `ErrorPropagation` with the instance of `MatrixAnalysis` as an input argument:
```
ep = ompy.ErrorPropagation(ma)
```
Then, an ensemble of `N_ensemble_members` is generated by the command
```
ep.generate_ensemble(N_ensemble_members)
```
The standard deviations matrices for `raw`, `unfolded` and `firstgen` are automatically calculated and put in the variables `ep.std_raw`, `ep.std_unfolded` and `ep.std_firstgen`, respectively.

## The FitRhoT class
To fit $\rho$ and $\mathcal{T}$ to the first-generation matrix (also requiring `ep.std_firstgen`), do
```
# Instantiate the fitting class:
fitting = ompy.FitRhoT(ma.firstgen,
                       ep.std_firstgen,
                       bin_width_out,
                       Ex_min, Ex_max,
                       Eg_min
                       )
# Perform the fit:
fitting.fit()
# The fitted functions are available 
# as Vector instances:
fitting.rho, fitting.T
```
Here, `bin_width_out` gives the energy bin width of the resulting fit, and the other arguments determine the region of the first-generation matrix to fit to. To get the uncertainty on the fitted quantities, one can run the fitting for each member in the ensemble:
```
# Allocate arrays to store each ensemble
# member fit:
rho_ens = np.zeros((N_ensemble_fit,
                    len(rho.vector)))
T_ens = np.zeros((N_ensemble_fit,
                  len(T.vector)))
# As a trick, we copy the instance ma
# and replace its matrix every iteration:
import copy
ma_curr = copy.deepcopy(ma)
# Loop through all and perform fit:
for i_ens in range(N_ensemble_fit):
    ma_curr.firstgen.matrix = \
        ep.firstgen_ensemble[i_ens, :, :]
    fitter_curr = ompy.FitRhoT(
                    ma_curr.firstgen,
                    ep.std_firstgen,
                    bin_width_out,
                    Ex_min, Ex_max,
                    Eg_min
                    )
    rho_curr= fitter_curr.rho
    T_curr = fitter_curr.T
    rho_ens[i_ens, :] = rho_curr.vector
    T_ens[i_ens, :] = T_curr.vector
```

## Adding new features
OMpy is written with modularity in mind. We want it to be as easy as possible for the user to add custom functionality and interface OMpy with other Python packages. For example, it may be of interest to try other unfolding algorithms than the one presently implemented. To achieve this, one just has to write a wrapper function that has the same input and output structure as the function `unfold()`, found in the file `ompy/unfold.py`, and replace the calls to `unfold()` by the custom function. We hope to make such modifications even easier in a future version.

It is our hope and goal that `OMpy` will be used, and we are happy to provide support. Feedback and suggestions are also very welcome. We encourage users who implement new features to share them by opening a pull request in the Github repository.




<sup>1</sup>It stands for alfa-natrium, and comes from the fact that the outgoing projectile hitting the excitation energy detector SiRi used to be exclusively alpha particles and that the gamma-detector array CACTUS consisted of NaI scintillators.
