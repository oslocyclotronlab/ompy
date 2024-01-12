from __future__ import annotations
from typing import TypeVar, Type, TYPE_CHECKING, Self
from sys import modules
import threading

"""
This code is taken from pymc.model.core
"""


T = TypeVar("T", bound="ContextMeta")


class ContextMeta(type):
    """Functionality for objects that put themselves in a context using
    the `with` statement.
    """

    def __new__(cls, name, bases, dct, **kwargs):  # pylint: disable=unused-argument
        """Add __enter__ and __exit__ methods to the class."""

        def __enter__(self):
            self.__class__.context_class.get_contexts().append(self)
            # self._pytensor_config is set in Model.__new__
            self._config_context = None
            #if hasattr(self, "_pytensor_config"):
            #    self._config_context = pytensor.config.change_flags(**self._pytensor_config)
            #    self._config_context.__enter__()
            return self

        def __exit__(self, typ, value, traceback):  # pylint: disable=unused-argument
            print(self.__class__.context_class.get_contexts())
            self.__class__.context_class.get_contexts().pop()
            # self._pytensor_config is set in Model.__new__
            if self._config_context:
                self._config_context.__exit__(typ, value, traceback)

        dct[__enter__.__name__] = __enter__
        dct[__exit__.__name__] = __exit__

        # We strip off keyword args, per the warning from
        # StackExchange:
        # DO NOT send "**kwargs" to "type.__new__".  It won't catch them and
        # you'll get a "TypeError: type() takes 1 or 3 arguments" exception.
        return super().__new__(cls, name, bases, dct)

    # FIXME: is there a more elegant way to automatically add methods to the class that
    # are instance methods instead of class methods?
    def __init__(
        cls, name, bases, nmspc, context_class: Type | None = None, **kwargs
    ):  # pylint: disable=unused-argument
        """Add ``__enter__`` and ``__exit__`` methods to the new class automatically."""
        if context_class is not None:
            cls._context_class = context_class
        super().__init__(name, bases, nmspc)

    def get_context(cls, error_if_none=True, allow_block_model_access=False) -> T | None:
        """Return the most recently pushed context object of type ``cls``
        on the stack, or ``None``. If ``error_if_none`` is True (default),
        raise a ``TypeError`` instead of returning ``None``."""
        try:
            candidate: T | None = cls.get_contexts()[-1]
        except IndexError:
            # Calling code expects to get a TypeError if the entity
            # is unfound, and there's too much to fix.
            if error_if_none:
                raise TypeError(f"No {cls} on context stack")
            return None
        #if isinstance(candidate, BlockModelAccess) and not allow_block_model_access:
        #    raise BlockModelAccessError(candidate.error_msg_on_access)
        return candidate

    def get_contexts(cls) -> list[T]:
        """Return a stack of context instances for the ``context_class``
        of ``cls``."""
        # This lazily creates the context class's contexts
        # thread-local object, as needed. This seems inelegant to me,
        # but since the context class is not guaranteed to exist when
        # the metaclass is being instantiated, I couldn't figure out a
        # better way. [2019/10/11:rpg]

        # no race-condition here, contexts is a thread-local object
        # be sure not to override contexts in a subclass however!
        context_class = cls.context_class
        assert isinstance(
            context_class, type
        ), f"Name of context class, {context_class} was not resolvable to a class"
        if not hasattr(context_class, "contexts"):
            context_class.contexts = threading.local()

        contexts = context_class.contexts

        if not hasattr(contexts, "stack"):
            contexts.stack = []
        return contexts.stack

    # the following complex property accessor is necessary because the
    # context_class may not have been created at the point it is
    # specified, so the context_class may be a class *name* rather
    # than a class.
    @property
    def context_class(cls) -> Type:
        def resolve_type(c: Type | str) -> Type:
            if isinstance(c, str):
                c = getattr(modules[cls.__module__], c)
            if isinstance(c, type):
                return c
            raise ValueError(f"Cannot resolve context class {c}")

        assert cls is not None
        if isinstance(cls._context_class, str):
            cls._context_class = resolve_type(cls._context_class)
        if not isinstance(cls._context_class, (str, type)):
            raise ValueError(
                f"Context class for {cls.__name__}, {cls._context_class}, is not of the right type"
            )
        return cls._context_class

    # Inherit context class from parent
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.context_class = super().context_class

    # Initialize object in its own context...
    # Merged from InitContextMeta in the original.
    def __call__(cls, *args, **kwargs):
        # We type hint Model here so type checkers understand that Model is a context manager.
        # This metaclass is only used for Model, so this is safe to do. See #6809 for more info.
        instance: Model = cls.__new__(cls, *args, **kwargs)
        with instance:  # appends context
            instance.__init__(*args, **kwargs)
        return instance


class Model(metaclass=ContextMeta):
    """Encapsulates the variables and likelihood factors of a model.

    Model class can be used for creating class based models. To create
    a class based model you should inherit from :class:`~pymc.Model` and
    override the `__init__` method with arbitrary definitions (do not
    forget to call base class :meth:`pymc.Model.__init__` first).

    Parameters
    ----------
    name: str
        name that will be used as prefix for names of all random
        variables defined within model
    check_bounds: bool
        Ensure that input parameters to distributions are in a valid
        range. If your model is built in a way where you know your
        parameters can only take on valid values you can set this to
        False for increased speed. This should not be used if your model
        contains discrete variables.

    Examples
    --------
    How to define a custom model

    .. code-block:: python

        class CustomModel(Model):
            # 1) override init
            def __init__(self, mean=0, sigma=1, name=''):
                # 2) call super's init first, passing model and name
                # to it name will be prefix for all variables here if
                # no name specified for model there will be no prefix
                super().__init__(name, model)
                # now you are in the context of instance,
                # `modelcontext` will return self you can define
                # variables in several ways note, that all variables
                # will get model's name prefix

                # 3) you can create variables with the register_rv method
                self.register_rv(Normal.dist(mu=mean, sigma=sigma), 'v1', initval=1)
                # this will create variable named like '{name::}v1'
                # and assign attribute 'v1' to instance created
                # variable can be accessed with self.v1 or self['v1']

                # 4) this syntax will also work as we are in the
                # context of instance itself, names are given as usual
                Normal('v2', mu=mean, sigma=sigma)

                # something more complex is allowed, too
                half_cauchy = HalfCauchy('sigma', beta=10, initval=1.)
                Normal('v3', mu=mean, sigma=half_cauchy)

                # Deterministic variables can be used in usual way
                Deterministic('v3_sq', self.v3 ** 2)

                # Potentials too
                Potential('p1', pt.constant(1))

        # After defining a class CustomModel you can use it in several
        # ways

        # I:
        #   state the model within a context
        with Model() as model:
            CustomModel()
            # arbitrary actions

        # II:
        #   use new class as entering point in context
        with CustomModel() as model:
            Normal('new_normal_var', mu=1, sigma=0)

        # III:
        #   just get model instance with all that was defined in it
        model = CustomModel()

        # IV:
        #   use many custom models within one context
        with Model() as model:
            CustomModel(mean=1, name='first')
            CustomModel(mean=2, name='second')

        # variables inside both scopes will be named like `first::*`, `second::*`
    """

    if TYPE_CHECKING:

        def __enter__(self: Self) -> Self:
            ...

        def __exit__(self, exc_type: None, exc_val: None, exc_tb: None) -> None:
            ...

    def __new__(cls, *args, **kwargs):
        # resolves the parent instance
        instance = super().__new__(cls)
        if kwargs.get("model") is not None:
            instance._parent = kwargs.get("model")
        else:
            instance._parent = cls.get_context(error_if_none=False)
        #instance._pytensor_config = kwargs.get("pytensor_config", {})
        return instance

    @staticmethod
    def _validate_name(name):
        if name.endswith(":"):
            raise KeyError("name should not end with `:`")
        return name

    def __init__(
        self,
        name="",
        check_bounds=True,
        *,
        model=None,
    ):
        del model  # used in __new__
        self.name = self._validate_name(name)
        self.check_bounds = check_bounds
        if self.parent is not None:
            self.named_vars = treedict(parent=self.parent.named_vars)
        else:
            self.named_vars = treedict()

    @property
    def model(self):
        return self

    @property
    def parent(self):
        return self._parent

    def add_named_variable(self, var):
        if var.name is None:
            raise ValueError("Variable is unnamed.")
        if self.named_vars.tree_contains(var.name):
            raise ValueError(f"Variable name {var.name} already exists.")

        self.named_vars[var.name] = var

# this is really disgusting, but it breaks a self-loop: I can't pass Model
# itself as context class init arg.
Model._context_class = Model

def withparent(meth):
    """Helper wrapper that passes calls to parent's instance"""

    def wrapped(self, *args, **kwargs):
        res = meth(self, *args, **kwargs)
        if getattr(self, "parent", None) is not None:
            getattr(self.parent, meth.__name__)(*args, **kwargs)
        return res

    # Unfortunately functools wrapper fails
    # when decorating built-in methods so we
    # need to fix that improper behaviour
    wrapped.__name__ = meth.__name__
    return wrapped

class treedict(dict):
    """A dict that passes mutable extending operations used in Model
    to parent dict instance.
    Extending treedict you will also extend its parent
    """

    def __init__(self, iterable=(), parent=None, **kwargs):
        super().__init__(iterable, **kwargs)
        assert isinstance(parent, dict) or parent is None
        self.parent = parent
        if self.parent is not None:
            self.parent.update(self)

    # typechecking here works bad
    __setitem__ = withparent(dict.__setitem__)
    update = withparent(dict.update)

    def tree_contains(self, item):
        # needed for `add_named_variable` method
        if isinstance(self.parent, treedict):
            return dict.__contains__(self, item) or self.parent.tree_contains(item)
        elif isinstance(self.parent, dict):
            return dict.__contains__(self, item) or self.parent.__contains__(item)
        else:
            return dict.__contains__(self, item)


