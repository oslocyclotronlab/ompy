# Oslo Method Python - OMpy
![Version](https://img.shields.io/badge/version-0.3-informational?style=flat-square)
![Travis (.org)](https://img.shields.io/travis/oslocyclotronlab/ompy?style=flat-square)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/oslocyclotronlab/ompy?style=flat-square)
[![DOI](https://zenodo.org/badge/141709973.svg)](https://zenodo.org/badge/latestdoi/141709973)
<div style="text-align:center"><img height="300px" align="center" src="resources/demo.png?raw=true"></div>

<p align="center">
<b><a href="#installation">Installation</a></b>
|
<b><a href="#unfolding">Unfolding</a></b>
|
<b><a href="#first-generation-method">First Generation</a></b>
|
<b><a href="#ensemble">Ensemble</a></b>
|
<b><a href="#nuclear-level-density-and-gamma-strength-function">Rhosig</a></b>
</p>

<p align="center">
<b><a href="#credits">Credits</a></b>
|
<b><a href="#license">License</a></b>
</p>
<br>

NB! This repo is currently under development. Many features do not work correctly.
=======

If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above; create a new version if necessary.
The DOI is to last *published* version; the *master* branch may be ahead of the *published* version.

# OMpy

This is `ompy`, the Oslo method in python. It contains all the functionality needed to go from a raw coincidence matrix, via unfolding and the first-generation method, to fitting a level density and gamma-ray strength function. It also supports uncertainty propagation by Monte Carlo.


## Installation
First off, make sure to compile the Cython modules by doing (in the main repo folder)
```console
python setup.py build_ext --inplace
pip install .
```

For development it will be more convenient to create symbolic link:
```console
pip3 install -e .
```

All the functions and classes in the package are available in the main module. You get everything by importing the package

```console
import ompy
```

The overarching philosophy is that the package shall be flexible and transparent to use and modify. All of the "steps" in the Oslo method are implemented as classes with a common structure and call signature. If you understand one class, you'll understand them all, making extending the code easy. 

As the Oslo method is a complex method involving dozen of variables which can be daunting for the uninitiated, many class attributes have default values that should give satisfying results. Attributes that _should_ be modified even though it is not strictly necessary to do so will give annoying warnings. The documentation and docstrings give in-depth explanation of each variable and its usage.

## Matrix manipulation
The core of the Oslo method involves working with two dimensional spectra often called alfna matrices for obscure reasons.
<sup>1</sup> Starting with a raw matrix of $E_x$-$E_\gamma$ coincidences, you typically want to unfold the counts 
along the gamma-energy axis and then apply the first-generation method to obtain the matrix of first-generation, or primary, gamma rays from the decaying nucleus. 

The two most important utility classes in the package are `Matrix()` and `Vector()`. They are used to store matrices (2D) or vectors (1D) of numbers, typically spectra of counts, along with energy calibration information. Their basic structure is
```py
mat = ompy.Matrix()
mat.values  # A 2D numpy array
mat.Ex      # Array of mid-bin-edge energy values for axis 0 (i.e. the row axis, or y axis)
mat.Eg      # Array of mid-bin-edge energy values for axis 1 (i.e. the column axis, or x axis)

vec = ompy.Vector()
vec.values  # A 1D numpy array
vec.E       # Array of mid-bin-edge energy values for the single axis
```

As these underpin the entire package, they contain many useful functions to make life easier. Loading and saving to several formats, plotting, projections, rebinning and cutting, to mention a few.
See the documentation for an exhaustive list.


## Unfolding
An implementation of the unfolding method presented in Guttormsen et al., Nuclear Instruments and Methods in Physics Research A 374 (1996).
The functionality is provided by `Unfolder()` whose basic usage is

```py
matrix = om.Matrix(...)
response = om.Matrix(path=...)
unfolder = om.Unfolder(response=response)
unfolded = unfolder(matrix)
```

### The response matrix
At present, the module doesn't contain a method for interpolating response functions. We have written one, but it needs to be cythonized because it's very slow. 

Because of this, if you want to do unfolding then you need to specify one or two matrices in the keywords `response` and `compton_response`. The first is the response matrix. 
It needs to have the same energy calibration as the gamma-energy axis of the spectrum you're unfolding, and therefore it must be interpolated by MAMA.
The second is only needed if you want to use the Compton subtraction method (the attribute `use_comptonsubtraction = True`). It is a list of probabilities for full-energy, single escape, double escape, etc., for each gamma-ray energy.

To make the response matrix: Open you raw matrix with MAMA. Type `rm` to make the response matrix to your specifications. Then type `gr` to load the response matrix into view, and `wr` to write it to disk.

If you want to do the Compton subtraction method (which doesn't work as of February 2019), you need the response parameters. They are automatically written to the file `resp.dat` when you run the `rm` command, but MAMA by default only prints a subset of the energy data points. To fix this, you need to edit the MAMA source file `folding.f` and comment out the line

```fortran
IF(RDIM.GT.50)iStep=RDIM/50                  !steps for output 
```

which in the MAMA version I have is located at about line 1980. Then recompile MAMA.

## First generation method
An implementation of the first generation method present in Guttormsen, Ramsøy and Rekstad, Nuclear Instruments and Methods in Physics Research A 255 (1987). 

The first generation method is implemented in `FirstGeneration()` whose basic usage is

```py
unfolded = unfolder(matrix)
firstgen = om.FirstGeneration()
primary = firstgen(unfolded)
```

## Ensemble

To estimate the uncertainty in the Oslo method, we can generate an ensemble of perturbated matrices from an initial matrix. The class `Ensemble()` provides this
feature. Its basic usage is

```py
matrix = om.Matrix(...)
response = om.Matrix(path=...)
unfolder = om.Unfolder(response=response)
firstgen = om.FirstGeneration()
ensemble = om.Ensemble(matrix)
ensemble.unfolder = unfolder
ensemble.first_generation_method = firstgen
ensemble.generate(100)    # Generates 100 perturbated members
```
the generated members are saved to disk and can be retrieved. Unfolded members can be retrieved as `ensemble.get_unfolded(i)`, for example. Their standard deviation is `ensemble.std_unfolded`.

## Nuclear level density and gamma strength function

After matrix has been cut, unfolded and firstgen'd, perhaps ensembled, its nuclear level density (nld) and gamma strength function ($\gamma$SF) can be extracted using the
`Extractor()` class. For a single matrix, its usage is

```py
primary = firstgen(unfolded)
cutout = primary.trapezoid(Ex_min=..., Ex_max=..., Eg_min=..., inplace=False)
extractor = om.Extractor()
nld, gsf = extractor.decompose(cutout)
```
When extracting nld and gsf from an ensemble, a trapezoidal cutout must be performed on each ensemble member. This is achieved by `Action()` which allows for delayed function calls on
matrices and vectors.
```py
ensemble.generate(100)
trapezoid_cut = om.Action('matrix')
trapezoid_cut.trapezoid(Ex_min=..., Ex_max=..., Eg_min=...)
extractor = Extractor()
extractor.trapezoid = trapezoid_cut
extractor.extract_from(ensemble)
```
The resulting nld and gsf are saved to disk and exposed as `extractor.nld` and `extractor.gsf`.

## Normalization

Not yet implemented.

## Validation and introspection

An important feature of physics programs is the ability to validate that the program works as intended. This can be achieved by either running the program on problems whose solutions are already known,
or by inspecting the program and confirming that each step is working as expected. OMpy uses both methods. Integration tests are performed both on artificial data satisfying the minimal assumptions required
of each method (unfold, first generation method, etc.), as well as experimental data which has already been analyzed using other programs (MAMA). 

In addition, the methods themselves are written in a way
which separates the uninteresting "book keeping" of each method, such as constructing arrays and normalizing rows, from the actual interesting steps performing the calculations. All parts of a method, its
initial set up, progression and tear down, can be separately inspected using the `ompy.hooks` submodule and `logging` framework. This allows the user to not only verify that each method works as intended, 
but also get a visual understanding of how they work beyond their mere equational forms. 

## Development 
OMpy is written with modularity in mind. We want it to be as easy as possible for the user to add custom functionality and interface OMpy with other Python packages. For example, 
it may be of interest to try other unfolding algorithms than the one presently implemented. To achieve this,
one just has to write a wrapper function that has the same input and output structure as the function `Unfolder.__call__()`,
found in the file `ompy/unfolder.py`. 

It is our hope and goal that `OMpy` will be used, and we are happy to provide support. Feedback and suggestions are also very welcome. We encourage users who implement new features to share them by opening a pull request in the Github repository.


<sup>1</sup>It stands for alfa-natrium, and comes from the fact that the outgoing projectile hitting the excitation energy detector SiRi used to be exclusively alpha particles and that the gamma-detector array CACTUS consisted of NaI scintillators.

## Credits
The contributors of this project are Jørgen Eriksson Midtbø, Fabio Zeiser and Erlend Lima.

## License
This project is licensed under the terms of the **GNU General Public License v3.0** license.
You can find the full license [here](LICENSE.md).
