# oslo_method_python

NB! This repo is currently in testing. Most things do not work correctly. Use at your own risk!

This is a python implementation of the Oslo method. The goal is that this will become a complete Oslo method ecosystem within Python, which can also be interfaced with other Python packages to do e.g. Bayesian inference.


## How to use this package
First off, make sure to compile the Cython modules by doing (in the main repo folder)
```
python setup.py build_ext --inplace
```

All the functions and classes in the package are available in the main module. You get everything with the import

```
import oslo_method_python as om
```

My philosophy is that the package shall be flexible and transparent to use and modify. I'd rather open up for possible misuse of the code than hinder the user from doing something. Therefore, there are several ways to use the package, which will now be outlined:

### The Matrix() and Vector() classes
The two most important utility classes in the package are `om.Matrix()` and `om.Vector()`. They are used to store matrices (2D) or vectors (1D) of numbers (typically spectra of counts) along with energy calibration information. These are just numpy arrays, and the structure is like this:
```
mat = om.Matrix()
mat.matrix # A 2D numpy array
mat.Ex_array # Array of lower-bin-edge energy values for axis 0
mat.Eg_array # Array of lower-bin-edge energy values for axis 1

vec = om.Vector()
vec.vector # A 1D numpy array
vec.E_array # Array of lower-bin-edge energy values for the single axis
```
 They also contain some other methods:
```
mat.load(filename) # Load a spectrum from a 2D MAMA file (similar, but 1D, for vec)
mat.save(filename) # Save spectrum to MAMA file

```

### The MatrixAnalysis class
The core of the Oslo method involves working with two-dimensional spectra, the excitation-energy-gamma-ray-energy-matrices, often called alfna matrices for obscure reasons.<sup>1</sup> Starting with a raw matrix of $E_x$-$E_\gamma$ coincidences, you typically want to unfold the counts along the gamma-energy axis and then apply the first-generation method to obtain the matrix of first-generation, or primary, gamma rays from the decaying nucleus. 

In this package, this functionality can be compactly accessed [to be continued]



<sup>1</sup>It stands for alfa-na, and comes from the fact that the outgoing projectile hitting the excitation energy detector SiRi used to be exclusively alpha particles and that the gamma-detector array CACTUS consisted of NaI scintillators.
