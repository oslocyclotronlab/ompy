from __future__ import annotations
from typing import Any, TypeAlias, TypeVar, TypeGuard, overload, Self, Literal, Sequence, Iterable
from collections import deque
import numpy as np
from abc import ABC, abstractmethod
import json
from dataclasses import dataclass, asdict, fields
from rich import print
from pathlib import Path
import ROOT
from collections import defaultdict
from contextlib import contextmanager
import os
from enum import Enum, unique, auto


Primitive: TypeAlias = int | float | np.number | np.ndarray | str | bool | None
SimpleComposite: TypeAlias = dict[Any,
                                  Primitive] | list[Primitive] | tuple[Primitive] | set[Primitive]
ComplexComposite: TypeAlias = dict | list | tuple | set

VERSION = "0.0.1"

TVisit = TypeVar("TVisit", bound="DataVisitor")

""" TODO
- [ ] Very confusing what type of data a node contains. Add a type field
- [ ] A [Primitive] and [TreeConvertable] looks the same when empty. Must be distinguished
- [ ] DirectoyTreeWriter
- [ ] ROOTTreeWriter
- [ ] HDF5TreeWriter
- [ ] A DirectoryTreeWriter should take a class that defines how arrays are written,
      e.g. DirectoryTreeWriter('.').set_array_writer(NPZWriter)
- [ ] An array may use the same structure, something like Matrix.save_with(MAMAWriter)
- [ ] How should a Composite work? Take its parent name+counter, and type indicating element type?
- [ ] Move DataType into type level, i.e. subclass DataNode.
- [ ] Visitors need a lookup table TreeConvertables can specify so they can handle special cases

The node in question specifies for the visitor which function the visitor must use to read it
tree.accept(visitor) -> visitor.visit(tree) -> visitor.process_node(tree) -> tree.visit(visitor) -> visitor.visit_primitive(tree)
"""


@unique
class DataType(Enum):
    """ To tag a node about the structure of the data it contains

    """
    Primitive = auto()           # Can be simply saved
    TreeConvertable = auto()     # Is transversable
    PrimitiveComposite = auto()  # Can be tranversed, but contains only primitives
    TreeComposite = auto()       # Can be transversed, and contains TreeConvertables
    Unsupported = auto()         # Leftover
    # Element = auto()             # A single element of a composite
    Root = auto()                # The root node
    Array = auto()               # An arraylike


class DataNode:
    """Intermediate representation of a structure to be saved"""

    def __init__(self, name: str | None = None, data: Any = None,
                 children: list[DataNode] | None = None,
                 parent: DataNode | None = None,
                 dtype: DataType = DataType.Root):
        if data is not None and (name is None or name == ""):
            raise ValueError("DataNode must have a name if it has data")
        if data is not None and dtype == DataType.Root:
            raise ValueError("DataNode must have a dtype if it has data")
        self.name: str = "" if name is None else name
        self.data = data
        self.parent: DataNode | None = None
        self.children: list[DataNode] = [] if children is None else children
        self.dtype: DataType = dtype

    def append(self, node: DataNode) -> None:
        node.set_parent(self)
        self.children.append(node)

    def insert(self, name: str, data: Any = None) -> DataNode:
        node = DataNode(name, data)
        self.append(node)
        return node

    def accept(self, visitor: TVisit) -> TVisit:
        visitor.visit(self)
        return visitor

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def set_parent(self, parent: DataNode | None = None, overwrite: bool = False) -> None:
        if not overwrite and self.parent is not None:
            raise ValueError(
                "Cannot set parent of node that already has a parent")
        self.parent = parent

    def print(self) -> None:
        render_tree(self)

    def identifier(self) -> str:
        if self.is_root():
            if self.name == "":
                return '/'
            else:
                return '/[' + self.name + ']/'
        else:
            return self.parent.identifier() + self.name + '/'


class DataVisitor(ABC):
    """Visitor pattern for DataNode"""
    @abstractmethod
    def process_node(self, node: DataNode, depth: int) -> None: ...

    @abstractmethod
    def visit(self, tree: DataNode) -> None: ...


class BFS(DataVisitor):
    """Visitor pattern for DataNode"""

    def visit(self, tree: DataNode) -> None:
        if tree is None:
            return

        # Each item in the queue is a tuple of (node, depth)
        queue = deque([(tree, 0)])

        while queue:
            node, depth = queue.popleft()
            self.process_node(node, depth)

            for child in node.children:
                queue.append((child, depth + 1))


class DFS(DataVisitor):
    """Visitor pattern for DataNode"""

    def visit(self, tree: DataNode) -> None:
        self._dfs(tree, 0)  # Start the DFS traversal with depth 0

    def _dfs(self, node: DataNode, depth: int) -> None:
        if node is None:
            return
        self.process_node(node, depth)
        for child in node.children:
            self._dfs(child, depth + 1)


class TreeError(ValueError):
    pass


class JSONWriter(DFS):
    def __init__(self):
        super().__init__()
        self.json_structure = {}

    def visit(self, node: DataNode) -> None:
        self.json_structure = self._dfs(node)

    def _dfs(self, node: DataNode) -> dict:
        node_dict = {
            'name': node.name,
            'data': node.data,
            'class': node.data.__class__.__name__
        }
        if not node.is_leaf():
            node_dict['children'] = [
                self._dfs(child) for child in node.children]
        if node.is_root() or is_composite(node.data) or isinstance(node.data, TreeConvertable):
            if not is_composite(node.data):
                node_dict['version'] = VERSION
        else:
            if False and not node.is_root() and len(node.children) > 0:
                raise TreeError("Primitive nodes cannot have children.\n"
                                f"Node {node.identifier()} has children {node.children}")
        return node_dict

    def get_json(self) -> str:
        return json.dumps(self.json_structure, indent=4)

    def process_node(self, node: DataNode) -> None:
        # In this specific implementation, the process_node method is not utilized,
        # as the DFS traversal and JSON structure creation are handled within the _dfs method.
        pass


class TreeConvertable(ABC):
    """Interface for serializing into a tree"""
    @abstractmethod
    def dendrify(self) -> DataNode: ...

    def dedendrify(self, tree: DataNode) -> Self:
        raise NotImplementedError


@dataclass
class TreeConvertableDC(TreeConvertable):
    """Serializes a dataclass into a tree"""

    def dendrify(self) -> DataNode:
        root = DataNode()
        for label in fields(self):
            # The field contains the type definition
            value = getattr(self, label.name)
            self._handle_value(root, label.name, value)
        return root

    def _handle_value(self, tree: DataNode, label: str, value: Any) -> None:
        match value:
            case np.ndarray():
                # TODO: Handle
                tree.insert(label, value.tolist(), dtype=DataType.Array)
            case int() | float() | str() | bool() | None:
                tree.insert(label, value, dtype=DataType.Primitive)
            case list() | set() | tuple():
                if is_primitive_composite(value):
                    tree.insert(
                        label, value, dtype=DataType.PrimitiveComposite)
                    return

                branch = DataNode(label, {'type': type(
                    value).__name__}, dtype=DataType.TreeComposite)
                for i, x in enumerate(value):
                    if not isinstance(x, TreeConvertable):
                        raise TreeError(
                            f"Cannot serialize into a tree, too nested. {value}")
                    twig = x.dendrify()
                    twig.name = 'item_' + str(i)
                    branch.append(x.dendrify())
                tree.append(branch)
            case TreeConvertable():
                tree.append(value.dendrify())
            case x:
                raise TreeError(
                    f"Cannot serialize into tree, unsupported type: {type(x)}{x}")


class Writer(ABC):
    def __init__(self, path: str | Path):
        self.path: Path = Path(path)
        self.file: Any | None = None

    def open(self) -> Self:
        if self.file:
            raise ValueError("File already open")
        self.file = self._make_handle()
        return self

    @abstractmethod
    def _make_handle(self) -> Self: ...

    def cleanup(self) -> None:
        pass

    def close(self) -> None:
        self.cleanup()
        if self.file:
            if hasattr(self.file, 'Close'):
                self.file.Close()
            else:
                self.file.close()

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def write(self, obj: DataNode | TreeConvertable) -> None:
        if isinstance(obj, DataNode):
            obj.accept(self)
        elif isinstance(obj, TreeConvertable):
            obj.dendrify().accept(self)
        else:
            raise ValueError(f"Cannot write object of type {type(obj)}")


class ROOTWriter(DFS, Writer):
    def __init__(self, path: str | Path):
        DFS.__init__(self)
        Writer.__init__(self, Path(path).with_suffix('.root'))
        self.current_dir: ROOT.TDirectory | None = None
        self.trees: dict[int, ROOT.TTree] = {}
        self.dir_names: list[str] = []

    def _make_handle(self) -> Any:
        return ROOT.TFile(str(self.path), "RECREATE")

    def process_node(self, node: DataNode, depth: int) -> None:
        self._navigate_to_depth(depth)
        # Create a TTree for the node

        if not node.is_leaf():  # If the node is a branch
            # Create a new directory for the branch
            dir_name = node.identifier()
            dir_name = dir_name.replace('/', '_')
            if dir_name in self.dir_names:
                dir_name += str(len(self.dir_names))
            self.dir_names.append(dir_name)
            new_dir = self.current_dir.mkdir(dir_name)
            print(new_dir)
            new_dir.cd()
            self.current_dir = new_dir
        else:  # If the node is a leaf
            # Use (or create) a TTree for this depth
            tree = self.trees.get(depth)
            if not tree:
                tree_name = f"tree_depth_{depth}"
                tree = ROOT.TTree(tree_name, tree_name)
                self.trees[depth] = tree
            # Assuming node.data is a float for simplicity
            match node.data:
                case int():
                    val = ROOT.std.vector('int')()
                case float():
                    val = ROOT.std.vector('float')()
                case str():
                    val = ROOT.std.vector('string')()
                case x:
                    print(f"Skipping {node.name}: {x}")
                    return
            val.push_back(node.data)
            tree.Branch(node.name, val)

    def _navigate_to_depth(self, depth: int):
        # Navigate back to root directory
        self.file.cd()
        # Navigate to the desired depth, creating directories as needed
        for i in range(depth):
            dir_name = f"depth_{i}"
            if not self.file.GetDirectory(dir_name):
                self.file.mkdir(dir_name)
            self.file.cd(dir_name)

        self.current_dir = ROOT.gDirectory

    def cleanup(self) -> None:
        for tree in self.trees.values():
            tree.Write()
        self.trees = {}


class DirectoryWriter(DFS):
    """Directory and JSON file writer based on DFS traversal"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.new_dirs: dict[str, int] = defaultdict(int)
        self.cwd = self.base_path
        self.last_depth = 0

    @contextmanager
    def _chdir(self, path: Path):
        """Context manager to temporarily change directory"""
        old_dir = Path.cwd()
        path.mkdir(parents=True, exist_ok=True)
        os.chdir(path)
        yield
        old_dir.chdir()

    def process_node(self, node: DataNode, depth: int) -> None:
        if depth < self.last_depth:
            print("changing dir")
            self.cwd = self.cwd.parent
        self.last_depth = depth

        if node.data is None:
            # New directory
            if node.name == '':
                if node.is_root():
                    name = ''
                else:
                    name = node.parent.name
            else:
                name = node.name
            key = str(self.cwd / name)
            print("name of dir is ", key)
            counter = self.new_dirs[key]
            self.new_dirs[key] += 1
            print("newpath", self.new_dirs[key])
            # cwd.mkdir(exist_ok=True)
            cwd = key + str(counter)
            cwd = Path(cwd)
            self.cwd = cwd
            print(cwd, " is directory")
        else:
            print(self.cwd / node.name, " is file data",
                  len(node.children), node.data)

    def __enter__(self):
        self._original_path = Path.cwd()
        # self._chdir(self.base_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # self._original_path.chdir()
        pass


@dataclass
class Parameters(TreeConvertableDC):
    xindex: np.ndarray
    yindex: np.ndarray
    resolution: float
    quality: int
    offsets: tuple[float, float]
    label: str
    comments: list[str]
    connections: list[Parameters]


@overload
def is_primitive(data: Primitive) -> Literal[True]: ...


@overload
def is_primitive(data: Any) -> Literal[False]: ...


def is_primitive(data: Any) -> TypeGuard[Primitive]:
    return isinstance(data, (int, float, str, bool, np.ndarray)) or data is None


def is_composite(data: Any) -> TypeGuard[Iterable]:
    try:
        if isinstance(data, (str, np.ndarray)):
            return False
        _ = iter(data)
    except TypeError:
        return False
    return True


def is_primitive_composite(data: Sequence, homogenuous: bool = False) -> bool:
    if len(data) < 2:
        return True
    base: type = type(data[0])
    for x in data[1:]:
        if not is_primitive(x):
            return False
        if homogenuous and not isinstance(x, base):
            return False
    return True


def render_tree(root):
    if root is None:
        return

    # The queue will store nodes along with their depth level
    queue = deque([(root, 0)])
    current_depth = 0

    while queue:
        node, depth = queue.popleft()

        # Check if we're entering a new depth level
        if depth != current_depth:
            print("")  # Print a newline for the new depth
            print(" " * depth, end="└")
            current_depth = depth

        # Print the node data with spaces to indicate depth
        if not node.name:
            print(f"-({'√' if node.is_root() else ''})-", end="-")
        else:
            print(f"-{node.name}", end="-")

        for child in node.children:
            queue.append((child, depth + 1))
    print()


def check(tree: DataNode):
    for child in tree.children:
        assert child.parent == tree, f"Parent of {child.identifier()} is not {tree.identifier()}"
        check(child)


if __name__ == "__main__":
    d = DataNode()
    d.insert("a", 1)
    d.insert("b", 2)
    d.insert("c", 3).insert("d", {'a': 3, 'y': 6.7})
    # d.print()
    p = Parameters(np.array([1, 2, 3]), np.array(
        [4, 5, 6]), 0.1, 1, (0.1, 0.2), "test", ["comment1", "comment2"], [])
    # other values than for p
    p2 = Parameters(np.array([1, 2, 3])+2, 3*np.array([4, 5, 6]),
                    2.1, 2, (0.2, 22), "test2", ["c2omment1", "2comment2"], [p, p])
    check(p2.dendrify())
    tree = p2.dendrify()
    writer = JSONWriter()
    print(tree.accept(writer).get_json())
    # with ROOTWriter('test.root') as writer:
    #    tree.accept(writer)
    with DirectoryWriter("base") as writer:
        tree.accept(writer)
