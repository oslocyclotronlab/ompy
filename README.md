# Oslo Method Python - OMpy
[![Build Status](https://img.shields.io/travis/oslocyclotronlab/ompy/master?label=build%20%28master%29)](https://travis-ci.com/oslocyclotronlab/ompy)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/oslocyclotronlab/ompy?style=flat-square)
[![DOI](https://zenodo.org/badge/141709973.svg)](https://zenodo.org/badge/latestdoi/141709973)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.cpc.2020.107795-yellowgreen)](https://doi.org/10.1016/j.cpc.2020.107795)
<div style="text-align:center"><img height="300px" align="center" src="resources/demo.png?raw=true"></div>

<p align="center">
|
<b><a href="#installation">Installation</a></b>
|
<b><a href="#troubleshooting">Troubleshooting</a></b>
|
<b><a href="#general-usage">General usage</a></b>
|
</p>

<p align="center">
<b><a href="#package-structure-summary">Package structure summary</a></b>
|
<b><a href="#citing">Citing</a></b>
|
<b><a href="LICENSE.md">License</a></b>
</p>
<br>

# OMpy

This is `ompy`, the Oslo method in python. It contains all the functionality needed to go from a raw coincidence matrix, via unfolding and the first-generation method, to fitting a level density and gamma-ray strength function. It also supports uncertainty propagation by Monte Carlo.
If you want to try the package before installation, you may simply [click here](https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb) to launch it on Binder.

**This is a short introduction, see more at https://ompy.readthedocs.io/**

**NB! This repo is currently under development. Use it at your own risk.**

## Citing
Please cite the following (more info below):
- The code, by using the DOI documenting this github
- The article describing the implementation
- If the unfolding / first-generation methods are used, cite the corresponding articles
- If you are unfolding, make sure to document also the response, e.g. through the citation guide on [OCL_response_functions](https://github.com/oslocyclotronlab/OCL_response_functions/)
- For the decomposition / normalization, you *may* cite the previous implementation by Schiller (2000).

**The code**: If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above; create a new version if necessary. The DOI is to last *published* version; the *master* branch may be ahead of the *published* version.

The full version (including the git commit) can also be obtained from `ompy.__full_version__` after installation.

**The article**: The *article* describing the implementation is now published in Comp. Phys. Comm. (2021): *A new software implementation of the Oslo method with rigorous statistical uncertainty propagation* [DOI: 10.1016/j.cpc.2020.107795](https://doi.org/10.1016/j.cpc.2020.107795).

**Other methods**: We have reimplemented the unfolding [[Guttormsen (1996)]](https://doi.org/10.1016/0168-9002(96)00197-0) and first generation method [[Guttormsen (1987)]](https://doi.org/10.1016/0168-9002(87)91221-6), see also documentation in the corresponding classes. The decomposition/normalization is subject to the same degeneracy as shown in [[Schiller (2000)]](http://dx.doi.org/10.1016/s0168-9002(99)01187-0), but the minimizer and the normalization procedure are different, which is explained in detail in the OMpy article. 

## Installation
Start off by downloading ompy:
``` bash
git clone --recurse https://github.com/oslocyclotronlab/ompy/
```
where the `--recurse` flag specifies, that all submodules shall be downloaded as well.

### Dependencies
 - Get and compile MultiNest (use the cmake version from [github.com/JohannesBuchner/MultiNest](https://github.com/JohannesBuchner/MultiNest)). The goal is to create lib/libmultinest.so    
    ``` bash
    git clone https://github.com/JohannesBuchner/MultiNest
    cd MultiNest/build
    cmake ..
    make
    sudo make install
    ```
    :warning: If the `make` steps above fails it might be that you have gcc/gfortran version 10 or higher. To fix this issue use the following steps instead
    ``` bash
    git clone https://github.com/JohannesBuchner/MultiNest
    cd MultiNest/build
    cmake -DCMAKE_Fortran_FLAGS="-std=legacy" ..
    make
    sudo make install
    ```
    
    Multinest has following hard dependencies: `lapack` and `blas`. To use MPI, additionally openmp has to be installed (probably does not work for MAC users, see below.). With apt-get you may fix the dependencies by:
    ```bash
    sudo apt-get install liblapack-dev libblas-dev libomp-dev
    ```
    If you still get an error like:
    ``` bash
    OSError: libmultinest.so: cannot open shared object file: No such file or directory
    ```
    visit http://johannesbuchner.github.io/PyMultiNest/install .
 - We require `python>=3.7`. Make sure you use the correct python version and the correct `pip`.
   You may need to replace `python` by `python3` and `pip` by `pip3` in the examples below. Run
   `python --version` and `pip --version` to check whether you have a sufficient python version.
 - All other dependencies can be installed automatically by `pip` (see below). *Alternatively*,
   make sure to install all requirements listed in `requirements.txt`, eg. using `conda` or `apt-get`. 
   You may try following in `conda` (untested)
    ``` bash
   conda install --file requirements.txt
   ```
 - For openMP support (optional), install `libomp`. Easiest on linux/ubuntu: `sudo apt-get install libomp-dev` or MAC `brew install libomp`.
 - Many examples are written with [jupyter notebooks](https://jupyter.org/install), so you probably want to install this, too.

### OMpy package

There are two main options on how to install OMpy. We will start off with our recommendation, that is with the `-e` flag is a local project in “editable” mode. This way, you will *in principal* not have to reinstall ompy if you pull a new version from git or create any local changes yourself.

Note: If you change any of the `cython` modules (`*.pyx` files), you will have to reinstall/recompile anyways. As they may have changed upstream, the easiest is probably if you install again every time you pull.
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
#### Try to reinstall
If you changed / if after a `git pull` there have been any changes to one of the `cython` modules, you will have to reinstall/recompile anyways: `pip install -e .`.

#### Docker container
If you don't succeed with the above, we also provide a [Docker](https://www.docker.com/get-started) container via dockerhub, see https://hub.docker.com/r/oslocyclotronlab/ompy. However, for everyday usage, we think it's easier to install the package *normally* on your machine. The dockerfile is in the [.binder](.binder) folder.

#### Python version
If you had some failed attempts, you might try to uninstall `ompy` before retrying the stepts above:
```bash
pip uninstall ompy
```
Note that we require python 3.7 or higher. If your standard `python` and `pip` link to python 2, you may have to use `python3` and `pip3`.

#### OpenMP
If you don't have OpenMP / have problems installing it (see above), you can install without OpenMP. Type `export ompy_OpenMP=False` in the terminal before the setup above.

#### Cloned the repo before September 2019
**NB: Read this (only) if you have cloned the repo before October 2019:**
We cleaned the repository from old comits clogging the repo (big data files that should never have been there). Unfortunetely, this has the sideeffect that the history had to be rewritten: Previous commits now have a different SHA1 (git version keys). If you need anything from the previous repo, see [ompy_Archive_Sept2019](https://github.com/oslocyclotronlab/ompy_Archive_Sept2019). This will unfortunately also destroy references in issues.
The simplest way to get the new repo is to rerun the installation instructions below.

## General usage
All the functions and classes in the package are available in the main module. You get everything by importing the package

```py
import ompy
```
## Package structure summary

Below you can find a summary of the most important files and directories of this repository. You can find a full documentation of the packages functionality [here](https://ompy.readthedocs.io/en/latest/API/index.html).  

.`ompy` - main repository    
├── `.binder` - [Dockerfile](https://docs.docker.com/engine/reference/builder/) for easy & reproducible installation and hooks for [MyBinder](https://mybinder.org/) 
├── `Dockerfile` - [Dockerfile](https://docs.docker.com/engine/reference/builder/) link   
├── `docs` - documentation, rendered at https://ompy.readthedocs.io/  
│   └── ...    
├── `example_data` - some example data for calculations  
│   └── ...      
├── `LICENSE.md` - License file  
├── `notebooks` - usage example(s)  
│   └── ...  
├── `OCL_response_functions` - *submodule* facilitating the *unfolding* method   
│   └── ...    
├── `ompy` - **package code**, see also [packaging projects](https://packaging.python.org/tutorials/packaging-projects/)  
│   └── ...    
├── `README.md` - Readme with short instructions. See also the [online documentation](https://ompy.readthedocs.io/)  
├── `requirements.txt` - dependencies, see also [pip](https://pip.pypa.io/en/stable/user_guide/#requirements-files)  
├── `resources` - miscellaneous files  
│   └── ...    
├── `setup.py` - setup, see also [packaging projects](https://packaging.python.org/tutorials/packaging-projects/)  
└── `tests` unit test files, see also [packaging projects](https://packaging.python.org/tutorials/packaging-projects/)  
    └── ...    
    
