from __future__ import annotations
from dataclasses import dataclass, fields, Field
from abc import ABC, abstractmethod
from .tree import (Node, Root, Array, Primitive, PrimitiveComposite,
                    TreeConvertableNode, TreeComposite, Unsupported,
                    DataContainer, BranchNode, DataNode)

from typing import Self, Any, Iterable, Type
import numpy as np
from .visitor import Visitor

TREE_CONVERTABLES: dict[str, Type[TreeConvertable]] = {}

class TreeConvertable(ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TREE_CONVERTABLES[cls.__name__] = cls

    @abstractmethod
    def to_tree(self) -> Root: ...

    @classmethod
    @abstractmethod
    def from_tree(cls, node: Root | BranchNode) -> Self: ...


@dataclass
class TreeConvertableDC(TreeConvertable):
    def to_tree(self) -> Root:
        root = Root()
        top = TreeConvertableNode(label=type(self).__name__, parent=root,
                                    annotation='', dtype=type(self))
        for field in fields(self):
            self._value_to_node(top, field)
        root.append(top)
        return root

    def _value_to_node(self, node: BranchNode, field: Field) -> None:
        label = field.name
        ftype = field.type
        value = getattr(self, label)
        match value:
            case np.ndarray():
                a = Array(label=label, data=value, parent=node, annotation=ftype)
                node.append(a)
            case int() | float() | str() | bool() | None:
                p = Primitive(label=label, data=value, parent=node, annotation=ftype,
                              dtype=type(value))
                node.append(p)
            case list() | set() | tuple():
                if is_primitive_composite(value):
                    x = PrimitiveComposite(label=label, data=value,
                                           parent=node, annotation=ftype,
                                           dtype=type(value))
                    node.append(x)
                else:
                    etype = type(next(iter(value)))
                    x = TreeComposite(label=label, parent=node, 
                                      annotation=ftype, element_type=etype,
                                      dtype=type(value))
                    for i, v in enumerate(value):
                        x.append(v.to_tree())
                    node.append(x)
            case TreeConvertable():
                x = TreeConvertableNode(label=label, parent=node, annotation=ftype,
                                        dtype=type(value))
                y = value.to_tree()
                #            ROOT   FOO-v  Attributes
                for child in y.children[0].children:
                    x.append(child)
                node.append(x)
            case x:
                x = Unsupported(label=label, data=x, parent=node, 
                                annotation=ftype)
                node.append(x)


    @classmethod
    def from_tree(cls, node: Root | BranchNode) -> Self:
        if isinstance(node, Root):
            node = node.children[0]
        # TODO Check that type matches
        if not isinstance(node, TreeConvertableNode):
            raise ValueError(f"Expected TreeConvertableNode, got {type(node)}")
        if not str(cls.__name__) == str(node.label):
            raise ValueError(f"Expected node with label {cls.__name__}, got {node.label}")
        attr = {}
        for field in fields(cls):
            label = field.name
            for child in node.children:
                if child.label == label:
                    match child:
                        case Primitive():
                            attr[label] = child.data
                        case PrimitiveComposite():
                            attr[label] = child.data
                        case TreeConvertableNode():
                            attr[label] = child.dtype.from_tree(child)
                        case TreeComposite():
                            attr[label] = [c.dtype.from_tree(c) for c in child.children]
                        case Array():
                            attr[label] = child.data
                        case Unsupported():
                            attr[label] = child.data
                    break
            else:
                raise ValueError(f"Missing field {label} in tree")
        return cls(**attr)



def is_primitive(data: Any) -> bool: 
    return isinstance(data, (int, float, str, bool, np.ndarray)) or data is None


def is_primitive_composite(data: Iterable, homogenuous: bool = False) -> bool:
    it = iter(data)
    try:
        x0 = next(it)
        if not is_primitive(x0):
            return False
    except StopIteration:
        # Empty defaults to true
        return True
    base: type = type(x0)
    for x in it:
        if not is_primitive(x):
            return False
        if homogenuous and not isinstance(x, base):
            return False
    return True
    

