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

# OMpy

This is `ompy`, the Oslo method in python. It contains all the functionality needed to go from a raw coincidence matrix, via unfolding and the first-generation method, to fitting a level density and gamma-ray strength function. It also supports uncertainty propagation by Monte Carlo.

This is a short introduction, see more at https://ompy.readthedocs.io/

### NB! This repo is currently under development. Many features do not work correctly.

## Citing
If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above; create a new version if necessary. The DOI is to last *published* version; the *master* branch may be ahead of the *published* version.

The full version (including the git commit) can also be obtained from `ompy.__full_version__` after installation.


## Installation
Start off by downloading ompy:
``` bash
git clone https://github.com/oslocyclotronlab/ompy/
```

### Dependencies
 - Get and compile MultiNest (use the cmake version from https://github.com/JohannesBuchner/MultiNest). The goal is to create lib/libmultinest.so
    ``` bash
    git clone https://github.com/JohannesBuchner/MultiNest
    cd MultiNest/build
    cmake ..
    make
    ```
 - We require `python>=3.7`. Make sure you use the correct python version and the correct `pip`.
   You may need to replace `python` by `python3` and `pip` by `pip3` in the examples below. Run
   `python --version` and `pip --version` to check whether you have a sufficient python version.
 - All other dependencies can be installed automatically by `pip` (see below). Alternatively,
   make sure to install all requirements listed in `requirements.txt`, eg. using `conda` or `apt-get`.
   You may try following in `conda` (untested)
    ``` bash
   conda install --file requirements.txt
   ```
 - Many examples are written with [jupyter notebooks](https://jupyter.org/install), so you probably want to install this, too.

### OMpy package

There are two main options on how to install OMpy. We will start off with our recommendation, that is with the `-e` flag is a local project in “editable” mode. This way, you will not have to reinstall ompy if you pull a new version from git or create any local changes yourself.

Note: If you change any of the `cython` modules, you will have to reinstall/recompile anyways.
```bash
pip install -e .
```

If you want to install at the system specific path instead, use
```bash
pip install .
```

For debugging, you might want to compile the `cython` modules "manually". The first line here is just to delete any existing cython modules in order to make sure that they will be recompiled.
```bash
rm ompy/*.so
rm ompy/*.c
python setup.py build_ext --inplace
```

### Troubleshooting
#### Python version
If you had some failed attempts, you might try to uninstall `ompy` before retrying the stepts above:
```bash
pip uninstall ompy
```
Note that we require python 3.7 or higher. If your standard `python` and `pip` link to python 2, you may have to use `python3` and `pip3`.

#### OpenMP / MAC
If you don't have OpenMP / have problems installing it, you can install without OpenMP. Type `export ompy_OpenMP=False` in the terminal before the setup above. For attempts to solve this issue, see also [#30](https://github.com/oslocyclotronlab/ompy/issues/30).

## General usage
All the functions and classes in the package are available in the main module. You get everything by importing the package

```py
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

### Response matrix
You have to read in a response matrix from disk. Convert it to have the incident energies on the `Ex` axis (y-axis) and the response on the `Eg` axis. In case you use one of the *usual* setups like OSCAR, files will be provided for you already.

Most accurate results, especially for low energies, are obtained if you provide the response matrix with same binning as the matrix that is to be unfolded. If this is not feasable, you may want to use the "Fan"-method (Guttormsen1996), that is implementet through the `interpolate_response` function. To get an idea of the quality of the interpolaton, have a look at `test_response_function_interpolation.ipynb`.

### Compton subtraction method
To use the Compton subtraction method, you need to provide a list of probabilities for full-energy, single escape, double escape, etc, for each gamma-ray energy. This can be obtained from the `interpolate_response` using with `return_table=True`. The unfolder with the compton subtraction method is then called  by

```
response, response_tab = om.interpolate_response(folderpath, raw.Eg, FWHM, return_table=True)

unfolder = om.Unfolder(response=response)
unfolder.use_compton_subtraction = True
unfolder.response_tab = response_tab
unfolded = unfolder(matrix)
```


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

Test implementation only through `norm_nld`and `norm_gsf` classes. Do not have the same calling signatures yet.

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
