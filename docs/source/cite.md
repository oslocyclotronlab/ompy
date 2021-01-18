## How to cite
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
