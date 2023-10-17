from __future__ import annotations
from typing import Any, TYPE_CHECKING, Type, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .visitor import Visitor
    from .treeconvertable import TreeConvertable

"""
TODO:
Only root and branches can have children. Rest are leaves
"""

@dataclass
class Node(ABC):
    children: list[BranchNode] = field(default_factory=list, init=False)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def append(self, node: Root | BranchNode) -> None:
        match node:
            case Root():
                for child in node.children:
                    self.append(child)
            case BranchNode():
                node.set_parent(self)
                self.children.append(node)
            case _:
                raise RuntimeError(f"Cannot append {type(node)} to {type(self)}")

    @abstractmethod
    def accept(self, visitor: Visitor, **kwargs): ...

    def primitive_children(self) -> Iterable[Primitive | PrimitiveComposite]:
        for child in self.children:
            if isinstance(child, (Primitive, PrimitiveComposite)):
                yield child

    def non_primitive_children(self) -> Iterable[BranchNode]:
        for child in self.children:
            if not isinstance(child, (Primitive, PrimitiveComposite)):
                yield child

    def array_children(self) -> Iterable[Array]:
        for child in self.children:
            if isinstance(child, Array):
                yield child

    def tree_convertable_children(self) -> Iterable[TreeConvertableNode]:
        for child in self.children:
            if isinstance(child, TreeConvertableNode):
                yield child

    def tree_composite_children(self) -> Iterable[TreeComposite]:
        for child in self.children:
            if isinstance(child, TreeComposite):
                yield child


class Root(Node):
    def __repr__(self):
        return f"Root()"

    def accept(self, visitor: Visitor):
        return visitor.visit_root(self)


@dataclass(kw_only=True)
class BranchNode(Node):
    label: str
    parent: Node
    annotation: str
    dtype: type | None = None

    def set_parent(self, parent: Node) -> None:
        self.parent = parent


@dataclass
class DataNode(BranchNode):
    data: Any = None

    def __post_init__(self):
        if self.dtype is None:
            self.dtype = type(self.data)


@dataclass(kw_only=True)
class Primitive(DataNode):
    """ Can be simply saved """
    def __repr__(self):
        return f"Primitive({self.label}, {self.dtype}, {self.data})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_primitive(self, **kwargs)


@dataclass
class DataContainer(DataNode):
    element_type: type | None = None
    def __post_init__(self):
        if self.element_type is None:
            if len(self.data) > 0:
                self.element_type = type(next(iter(self.data)))

@dataclass
class BranchContainer(BranchNode):
    element_type: Type[TreeConvertable]

@dataclass
class PrimitiveComposite(DataContainer):
    """ Container whose elements are primitives """
    def __repr__(self):
        return f"PrimitiveComposite({self.label}, {self.dtype}, {self.element_type}, len={len(self.data)})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_primitive_composite(self, **kwargs)


@dataclass
class TreeConvertableNode(BranchNode):
    dtype: Type[TreeConvertable]
    """ Can be converted into a tree """
    def __repr__(self):
        return f"TreeConvertable({self.label}, {self.dtype})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_tree_convertable(self, **kwargs)


@dataclass
class TreeComposite(BranchContainer):
    children: list[TreeConvertableNode] = field(default_factory=list, init=False)
    """ Container whose elements are tree convertable """
    def __repr__(self):
        return f"TreeComposite({self.label}, {self.element_type}, len={len(self.children)})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_tree_composite(self, **kwargs)


@dataclass
class Array(DataContainer):
    """ Numpy array or similar """
    shape: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self):
        super().__post_init__()
        if self.shape == ():
            self.shape = self.data.shape

    def __repr__(self):
        return f"Array({self.label}, {self.shape})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_array(self, **kwargs)

@dataclass
class Unsupported(DataNode):
    """ Nodes that could not be parsed """
    def __repr__(self):
        return f"Unsupported({self.label}, {self.dtype})"

    def accept(self, visitor: Visitor, **kwargs):
        return visitor.visit_unsupported(self, **kwargs)

