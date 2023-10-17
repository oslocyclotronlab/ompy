from __future__ import annotations
from typing import TYPE_CHECKING, Literal, Generator, overload
from .visitor import Visitor
from pathlib import Path
from contextlib import contextmanager
from .writer import Writer, PrimitiveWriter
from .numpywriter import NumpyWriter
from .jsonwriter import JSONWriter

if TYPE_CHECKING:
    from .tree import (Node, DataNode, Root, Primitive,
        PrimitiveComposite, TreeConvertableNode,
        TreeComposite, Array, Unsupported)


class DirectoryWriter(Visitor):
    """
    Primitives and primitive composites are written to the same
    file by `primitive_writer`.
    Arrays are written using `numpy_writer`.
    OMpy arrays are written using `array_writer`.
    The name of the array files are given by the node label.
    A new directory is created for each tree composite.

    Breadth first?
    Semi-breadth first? All primitives are written first, then all trees,
    then trees of trees

    Write nodes to the primitives file as
        {label: label, type: type, link?: true/false, value: value or a relative path}
    Write version to data files.

    """
    def __init__(self, path: Path | str,
                 numpy_writer: Writer = NumpyWriter(),
                 primitive_writer: PrimitiveWriter = JSONWriter()):
        self.path: Path = Path(path)
        self.primitive_writer: PrimitiveWriter = primitive_writer
        self.array_writer: Writer = None
        self.numpy_writer: Writer = numpy_writer 
        self.cwd: Path = self.path

    def visit_root(self, node: Root):
        # Save type of writers at root
        self.cwd = self.path  # Reset cwd
        self.handle_composite(node)  # Walk down the tree

    def handle_composite(self, node: Node):
        self.cwd.mkdir(parents=True, exist_ok=True)
        path = self.cwd / 'data'  # The writer sets the suffix
        with self.using_writer('primitive', path) as pwriter:
            for child in node.primitive_children():
                child.accept(self, writer=pwriter)
            pwriter.write_meta()
            # Need to write the rest

            for child in node.array_children():
                # i is only needed in composites
                path = self.cwd / child.label  # The writer sets the suffix
                with self.using_writer('numpy', path) as writer:
                    child.accept(self, writer=writer)
                pwriter.write_link(child, path)

            # Doesn't this also walk down?
            for child in node.tree_convertable_children():
                child.accept(self)

            # The following walks down the tree. The previous were flat.
            for child in node.tree_composite_children():
                child.accept(self)
        

    def visit_primitive(self, node: Primitive, writer: Writer):
        writer.write(node)

    def visit_primitive_composite(self, node: PrimitiveComposite, writer: Writer):
        writer.write(node)

    def visit_tree_convertable(self, node: TreeConvertableNode, element: bool = False):
        # TODO Handle special case of Matrix and Vector
        # + other exceptions
        if not element:
            self.cwd = self.cwd / node.label

        self.handle_composite(node) 

        if not element:
            self.cwd = self.cwd.parent

    def visit_tree_composite(self, node: TreeComposite):
        # "Enter" new directory
        for i, child in enumerate(node.children):
            self.cwd = self.cwd / f"{node.label}_{i}"

            # Always visit_tree_convertable
            child.accept(self, element=True)

            # "Exit" directory
            self.cwd = self.cwd.parent

    def visit_array(self, node: Array, writer: Writer):
        writer.write(node)

    def visit_unsupported(self, node: Unsupported, **kwargs):
        print(f"Unsupported node: {node}")

    @overload
    @contextmanager
    def using_writer(self, writer_type: Literal["primitive"],
                     path: Path) -> Generator[PrimitiveWriter, None, None]: ...

    @overload
    @contextmanager
    def using_writer(self, writer_type: Literal["numpy", "array"],
                     path: Path) -> Generator[Writer, None, None]: ...

    @contextmanager
    def using_writer(self, writer_type: Literal["primitive", "numpy", "array"],
                     path: Path) -> Generator[Writer, None, None]:
        """Context manager to set the active writer temporarily."""
        writer_attr_name = f"{writer_type}_writer"
        writer: Writer = getattr(self, writer_attr_name)
        
        with writer.open(path) as active_writer:
            yield active_writer
