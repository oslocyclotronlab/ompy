# Oslo Method Python - OMpy
![Travis (.org)](https://img.shields.io/travis/oslocyclotronlab/ompy?style=flat-square)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb)
![Code Climate maintainability](https://img.shields.io/codeclimate/maintainability/oslocyclotronlab/ompy?style=flat-square)
[![DOI](https://zenodo.org/badge/141709973.svg)](https://zenodo.org/badge/latestdoi/141709973)
<div style="text-align:center"><img height="300px" align="center" src="resources/demo.png?raw=true"></div>

<p align="center">
<b><a href="#citing">Citing</a></b>
|
<b><a href="#installation">Installation</a></b>
|
<b><a href="#troubleshooting">Troubleshooting</a></b>
|
<b><a href="#general-usage">General usage</a></b>
|
</p>

<p align="center">
<b><a href="#credits">Credits</a></b>
|
<b><a href="#license">License</a></b>
</p>
<br>

# OMpy

This is `ompy`, the Oslo method in python. It contains all the functionality needed to go from a raw coincidence matrix, via unfolding and the first-generation method, to fitting a level density and gamma-ray strength function. It also supports uncertainty propagation by Monte Carlo.
If you want to try the package before installation, you may simply [click here](https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb) to launch it on Binder.

**This is a short introduction, see more at https://ompy.readthedocs.io/**

**NB! This repo is currently under development. Use it at your own risk.**

## Citing
If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above; create a new version if necessary. The DOI is to last *published* version; the *master* branch may be ahead of the *published* version.

The full version (including the git commit) can also be obtained from `ompy.__full_version__` after installation.

An article describing the implementation more detailled will follow shortly. A draft can be read on arXiv: [A new software implementation of the Oslo method with complete uncertainty propagation](https://arxiv.org/abs/1904.13248).


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
    Multinest had following hard dependencies: `lapack` and `blas`. To use MPI, additionally openmp has to be installed (probably does not work for MAC users, see below.). With apt-get you may fix the dependencies by:
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
#### Docker container
If you don't succeed with the above, we also provide a [Docker](https://www.docker.com/get-started) container via dockerhub, see https://hub.docker.com/r/oslocyclotronlab/ompy. However, for everyday usage, we think it's easier to install the package *normally* on your machine

#### Python version
If you had some failed attempts, you might try to uninstall `ompy` before retrying the stepts above:
```bash
pip uninstall ompy
```
Note that we require python 3.7 or higher. If your standard `python` and `pip` link to python 2, you may have to use `python3` and `pip3`.

#### OpenMP / MAC
If you don't have OpenMP / have problems installing it, you can install without OpenMP. Type `export ompy_OpenMP=False` in the terminal before the setup above. For attempts to solve this issue, see also [#30](https://github.com/oslocyclotronlab/ompy/issues/30).

#### Cloned the repo before September 2019
**NB: Read this (only) if you have cloned the repo before October 2019:**
We cleaned the repository from old comits clogging the repo (big data files that should never have been there). Unfortunetely, this has the sideeffect that the history had to be rewritten: Previous commits now have a different SHA1 (git version keys). If you need anything from the previous repo, see [ompy_Archive_Sept2019](https://github.com/oslocyclotronlab/ompy_Archive_Sept2019). This will unfortunately also destroy references in issues.
The simplest way to get the new repo is to rerun the installation instructions below.

## General usage
All the functions and classes in the package are available in the main module. You get everything by importing the package

```py
import ompy
```
