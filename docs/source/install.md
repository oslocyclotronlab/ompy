# Installation
First off, make sure to compile the Cython modules by doing (in the main repo folder)
```bash
python setup.py build_ext --inplace
pip install .
```

For development it will be more convenient to create symbolic link:
[You may also try this if you have an emtpy import of `ompy` when trying it from another directory.]
```bash
pip install -e .
```

If you had some failed attempts, you might try to uninstall `ompy` before retrying the stepts above:
```bash
pip uninstall ompy
```

Note that we require python 3.7 or higher. If your standard `python` and `pip` link to python 2, you may have to use `python3` and `pip3`. If you don't have OpenMP / have problems installing it, you can install without OpenMP. Type `export ompy_OpenMP=False` in the terminal before the setup above. For attempts to solve this issue, see also [#30](https://github.com/oslocyclotronlab/ompy/issues/30).

All the functions and classes in the package are available in the main module. You get everything by importing the package

```py
import ompy
```

The overarching philosophy is that the package shall be flexible and transparent to use and modify. All of the "steps" in the Oslo method are implemented as classes with a common structure and call signature. If you understand one class, you'll understand them all, making extending the code easy.

As the Oslo method is a complex method involving dozen of variables which can be daunting for the uninitiated, many class attributes have default values that should give satisfying results. Attributes that _should_ be modified even though it is not strictly necessary to do so will give annoying warnings. The documentation and docstrings give in-depth explanation of each variable and its usage.
