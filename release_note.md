# Release notes for OMpy

## [Unreleased]
Changes:
- Fixed a bug where the `std` attribute of `Vector` was not saved to file.
- Changed the way the `Extractor` class are called. The ensamble and trapezoid are no longer given in the
  constructor, but are mandatory arguments in the `extract_from()` method. The class also now uses the same
  convention for loading and saving from file as the `Vector` and `Matrix` classes.
- Initial work to unify the way objects are saved/loaded from disk.

## v.1.1.0
Most important changes:
- Changed response function interpolation between Compton edge and he chosen max. energy. Before, there was a 
  misunderstanding of the *bin-by-bin interpolation* in Guttormsen1996. It now used a fan-like interpolation,
  too. Most noticable for response functions with small fwhm (like 1/10 Magne recommends for unfolding). 

## v1.0.0
Several changes, most important:

**theoretical framework**:
- Corrected likelihood: Forgot non-constant K term when K=K(theta): c8c0046153b1eb269d00280b572f742b1a3cf4d7

**parameters and choices**:
- unfolding parameters: 0fcafe2ff7770be8c2bb107256201af79739cdb3
- unfolder and fg method use remove negatives only, no fill: 9edb48537cca1f88c3120a73fa8eb92f6ebb5177
- Randomize p0 for decomposition 77dec9db9a3a34d5fd6195752c84cfbca0c26c39

**implementation and convenience**:
- different save/load for vectors e5f7e52ce13cff04e8b23f50a00902be1d098bfc and parent commits
- Enable pickling of normalizer instances via dill: 896b352686594a8c7dbe52904645cc5b900ba800


## v0.9.1
Changes:

- corrected version number
(v 0.9.0 has still v.0.8.0 as the version number)

## v0.9
Many changes to the API have occured since v.0.2. Here a (incomplete) summary of the main changes:

- `Vector` and `Matrix` are now in mid-bin calibration. Most or all other functions have been adopted.
- Many changes (bugfix & readability) to the ensemble, decomposition and normalization classes.
- Normalization of nld and gsf ensembles working
- Parallelization, even though it could be more efficient for multinest (see #94 )
- Renamed response functions submodule; run `git submodule update --init --recursive` after `git pull` to get the new files 
- remember to run `pip install -e .` such that changes to the cython files will be recompiled
- Documentation now available via https://ompy.readthedocs.io
- Installation requirements are hopefully all specified; docker file is provided with integration at https://hub.docker.com/r/oslocyclotronlab/ompy and [mybinder](https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb) can be used to rund the examples.
- We have clean-up the history of the repo to downsize it. 
  Here the warning message: *NB: Read this (only) if you have cloned the repo before October 2019: We cleaned the repository from old comits clogging the repo (big data files that should never have been there). Unfortunetely, this has the sideeffect that the history had to be rewritten: Previous commits now have a different SHA1 (git version keys). If you need anything from the previous repo, see ompy_Archive_Sept2019. This will unfortunately also destroy references in issues. The simplest way to get the new repo is to rerun the installation instructions below.*

## v0.2-beta
This is the first public beta version of the OMpy library, the Oslo Method in Python.

**NB: Read this (only) if you have cloned the repo before October 2019 (which affects this release, v0.2-beta)**: 
We cleaned the repository from old comits clogging the repo (big data files that should never have been there). Unfortunetely, this has the sideeffect that the history had to be rewritten: Previous commits now have a different SHA1 (git version keys). If you need anything from the previous repo, see [ompy_Archive_Sept2019](https://github.com/oslocyclotronlab/ompy_Archive_Sept2019). This will unfortunately also destroy references in issues. The simplest way to get the new repo is to rerun the installation instructions below.

**In essence**: This tag does not work any longer; you have to download the version from https://zenodo.org/record/2654604
