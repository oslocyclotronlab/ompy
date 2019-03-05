# Oslo Method Python - OMpy
![Travis CI](https://travis-ci.com/Caronthir/oslo_method_python.svg?token=oq37SFt93PBk5sCGgATJ&branch=master)

<div style="text-align:center"><img height="300px" align="center" src="resources/demo.png?raw=true"></div>

<p align="center">
<b><a href="#how-to-use-this-package">How to</a></b>
|
<b><a href="#the-unfold-function">Unfolding</a></b>
|
<b><a href="#credits">Credits</a></b>
|
<b><a href="#license">License</a></b>
</p>
<br>

NB! This repo is currently under development. Most things do not work correctly. Use at your own risk!

This is a python implementation of the Oslo method. The goal is that this will become a complete Oslo method ecosystem within Python, which can also be interfaced with other Python packages to do e.g. Bayesian inference.


## How to use this package
First off, make sure to compile the Cython modules by doing (in the main repo folder)
```console
python setup.py build_ext --inplace
pip install .
```

All the functions and classes in the package are available in the main module. You get everything with the import

```console
import ompy
```

The overarching philosophy is that the package shall be flexible and transparent to use and modify. I'd rather open up for possible misuse of the code than hinder the user from doing something. Therefore, there are several ways to use the package, which will now be outlined:

### The Matrix and Vector classes
The two most important utility classes in the package are `om.Matrix()` and `om.Vector()`. They are used to store matrices (2D) or vectors (1D) of numbers (typically spectra of counts) along with energy calibration information. These are just numpy arrays, and the structure is like this:
```py
mat = om.Matrix()
mat.matrix # A 2D numpy array
mat.E0_array # Array of lower-bin-edge energy values for axis 0 (i.e. the row axis, or y axis)
mat.E1_array # Array of lower-bin-edge energy values for axis 1 (i.e. the column axis, or x axis)

vec = om.Vector()
vec.vector # A 1D numpy array
vec.E_array # Array of lower-bin-edge energy values for the single axis
```
 They also contain some other methods:
```py
mat.load(filename) # Load a spectrum from a 2D MAMA file (similar, but 1D, for vec)
mat.save(filename) # Save spectrum to MAMA file

```

## Matrix manipulation
The core of the Oslo method involves working with two-dimensional spectra, the excitation-energy-gamma-ray-energy-matrices, often called alfna matrices for obscure reasons.<sup>1</sup> Starting with a raw matrix of $E_x$-$E_\gamma$ coincidences, you typically want to unfold the counts along the gamma-energy axis and then apply the first-generation method to obtain the matrix of first-generation, or primary, gamma rays from the decaying nucleus. 


## The unfold function
An implementation of the unfolding method presented in Guttormsen et al., Nuclear Instruments and Methods in Physics Research A 374 (1996).

It may contain bugs. Please report any bugs you encounter to the issue tracker, or if you like, fix them and make a pull request :)

It also currently lacks some functionality. Notably, the compton subtraction method has some major issues somewhere. Also, it differs slightly from MAMA in the unfolded result, particularly on the lowest gamma-ray energies. I need to study the MAMA code carefully to figure out how they're treated.

#### How to make a response matrix
At present, the module doesn't contain a method for interpolating response functions. I have written one, but it needs to be cythonized because it's very slow. 

Because of this, if you want to do unfolding then you need to specify one or two file paths in the keywords `fname_resp_mat` and `fname_resp_dat`, respectively. The first is the response matrix. It needs to have the same energy calibration as the gamma-energy axis of the spectrum you're unfolding, and therefore it must be interpolated by MAMA. The second is only needed if you want to use the Compton subtraction method (the keyword `use_comptonsubtraction = True`). It is a list of probabilities for full-energy, single escape, double escape, etc., for each gamma-ray energy.

To make the response matrix: Open you raw matrix with MAMA. Type `rm` to make the response matrix to your specifications. Then type `gr` to load the response matrix into view, and `wr` to write it to disk.

If you want to do the Compton subtraction method (which doesn't work as of February 2019), you need the response parameters. They are automatically written to the file `resp.dat` when you run the `rm` command, but MAMA by default only prints a subset of the energy data points. To fix this, you need to edit the MAMA source file `folding.f` and comment out the line

```     IF(RDIM.GT.50)iStep=RDIM/50                  !steps for output```

which in the MAMA version I have is located at about line 1980. Then recompile MAMA.

### The first_generation_method() function
An implementation of the first generation method present in Guttormsen, Ramsøy and Rekstad, Nuclear Instruments and Methods in Physics Research A 255 (1987). 

Like the unfolding function, please be on the lookout for bugs...

### The MatrixAnalysis class
The MatrixAnalysis class is a convenience wrapper for the `unfold()` and `first_generation_method()` functions, along with some utility functions to use on the spectra. You do not have to use it except if you want to do error propagation. In that case, the parameters to use for unfolding and first generation are stored in the MatrixAnalysis instance that you pass to the ErrorPropagation instance, to ensure that the ensemble of perturbed copies of the spectra are treated identically.

## The ErrorPropagation class
Propagate statistical errors through the Oslo method by making an ensemble of perturbed copies of the input spectrum. You can choose between gaussian or poisson errors. 

[Write something about the probablity theory and assumptions. Also about how the spectra are stored and how the variance matrix is calculated at the end. Should also have some examples throughout.]

<sup>1</sup>It stands for alfa-natrium, and comes from the fact that the outgoing projectile hitting the excitation energy detector SiRi used to be exclusively alpha particles and that the gamma-detector array CACTUS consisted of NaI scintillators.

## Credits
The contributors of this project are Jørgen Eriksson Midtbø, Fabio Zeiser and Erlend Lima.

## License
This project is licensed under the terms of the **GNU General Public License v3.0** license.
You can find the full license [here](LICENSE.md).
