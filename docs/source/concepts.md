## Concepts and advanced usage
### General usage
All the functions and classes in the package are available in the main module. You get everything by importing the package

```py
import ompy
```

The overarching philosophy is that the package shall be flexible and transparent to use and modify. All of the "steps" in the Oslo method are implemented as classes with a common structure and call signature. If you understand one class, you'll understand them all, making extending the code easy.

As the Oslo method is a complex method involving dozen of variables which can be daunting for the uninitiated, many class attributes have default values that should give satisfying results. Attributes that _should_ be modified even though it is not strictly necessary to do so will give annoying warnings. The documentation and docstrings give in-depth explanation of each variable and its usage.

### Normalization

Still working on a nice interface for the `gsf` normalization implementation. Test implementation only through `norm_gsf` classes. Does not have the same calling signatures yet.

### Validation and introspection

An important feature of physics programs is the ability to validate that the program works as intended. This can be achieved by either running the program on problems whose solutions are already known,
or by inspecting the program and confirming that each step is working as expected. OMpy uses both methods. Integration tests are performed both on artificial data satisfying the minimal assumptions required
of each method (unfold, first generation method, etc.), as well as experimental data which has already been analyzed using other programs (MAMA).

In addition, the methods themselves are written in a way
which separates the uninteresting "book keeping" of each method, such as constructing arrays and normalizing rows, from the actual interesting steps performing the calculations. All parts of a method, its
initial set up, progression and tear down, can be separately inspected using the `ompy.hooks` submodule and `logging` framework. This allows the user to not only verify that each method works as intended,
but also get a visual understanding of how they work beyond their mere equational forms.

### Development
OMpy is written with modularity in mind. We want it to be as easy as possible for the user to add custom functionality and interface OMpy with other Python packages. For example,
it may be of interest to try other unfolding algorithms than the one presently implemented. To achieve this,
one just has to write a wrapper function that has the same input and output structure as the function `Unfolder.__call__()`,
found in the file `ompy/unfolder.py`.

It is our hope and goal that `OMpy` will be used, and we are happy to provide support. Feedback and suggestions are also very welcome. We encourage users who implement new features to share them by opening a pull request in the Github repository.
