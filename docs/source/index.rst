Welcome to OMpy's documentation!
================================


|Travis| |Binder| |Code Climate maintainability| |DOI|

This is ``ompy``, the Oslo method in python. It contains all the
functionality needed to go from a raw coincidence matrix, via unfolding
and the first-generation method, to fitting a level density and
gamma-ray strength function. It also supports uncertainty propagation by
Monte Carlo. If you want to try the package before installation, you may
simply `click here`_ to launch it on Binder.

.. _click here: https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb

Citing
******
If you cite OMpy, please use the version-specific DOI found by clicking the Zenodo badge above; create a new version if necessary. The DOI is to last *published* version; the *master* branch may be ahead of the *published* version.

The full version (including the git commit) can also be obtained from `ompy.__full_version__` after installation.

An article describing the implementation more detailled will follow shortly. A draft can be read on arXiv: [A new software implementation of the Oslo method with complete uncertainty propagation](https://arxiv.org/abs/1904.13248).


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   getting_started
   concepts
   API/index
   LICENSE


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
.. Include this to generate the api files to be included under API/index
.. include:: api


.. |Travis| image:: https://img.shields.io/travis/oslocyclotronlab/ompy?style=flat-square
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/oslocyclotronlab/ompy/master?filepath=ompy%2Fnotebooks%2Fgetting_started.ipynb
.. |Code Climate maintainability| image:: https://img.shields.io/codeclimate/maintainability/oslocyclotronlab/ompy?style=flat-square
.. |DOI| image:: https://zenodo.org/badge/141709973.svg
   :target: https://zenodo.org/badge/latestdoi/141709973
