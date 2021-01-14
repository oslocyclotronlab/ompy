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
------

Please cite the following (more info below):

-  The code, by using the DOI documenting this github
-  The article describing the implementation
-  If the unfolding / first-generation methods are used, cite the
   corresponding articles
-  If you are unfolding, make sure to document also the response, e.g.
   through the citation guide on
   `OCL\_response\_functions <https://github.com/oslocyclotronlab/OCL_response_functions/>`__
-  For the decomposition / normalization, you *may* cite the previous
   implementation by Schiller (2000).

**The code**: If you cite OMpy, please use the version-specific DOI
found by clicking the Zenodo badge above; create a new version if
necessary. The DOI is to last *published* version; the *master* branch
may be ahead of the *published* version.

The full version (including the git commit) can also be obtained from
``ompy.__full_version__`` after installation.

**The article**: The *article* describing the implementation is now
published in Comp. Phys. Comm. (2021): *A new software implementation of
the Oslo method with rigorous statistical uncertainty propagation* `DOI:
10.1016/j.cpc.2020.107795 <https://doi.org/10.1016/j.cpc.2020.107795>`__.

**Other methods**: We have reimplemented the unfolding `[Guttormsen
(1996)] <https://doi.org/10.1016/0168-9002(96)00197-0>`__ and first
generation method `[Guttormsen
(1987)] <https://doi.org/10.1016/0168-9002(87)91221-6>`__, see also
documentation in the corresponding classes. The
decomposition/normalization is subject to the same degeneracy as shown
in `[Schiller
(2000)] <http://dx.doi.org/10.1016/s0168-9002(99)01187-0>`__, but the
minimizer and the normalization procedure are different, which is
explained in detail in the OMpy article.


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
.. |DOI-article| image:: https://img.shields.io/badge/DOI-10.1016/j.cpc.2020.107795-yellowgreen
   :target: https://doi.org/10.1016/j.cpc.2020.107795
