from __future__ import annotations
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tree import (Node, DataNode, Root, Primitive,
        PrimitiveComposite, TreeConvertableNode,
        TreeComposite, Array, Unsupported)


class Visitor(ABC):
    def visit(self, node: Node, **kwargs):
        return node.accept(self, **kwargs)

    @abstractmethod
    def visit_root(self, node: Root, **kwargs):
        pass

    @abstractmethod
    def visit_primitive(self, node: Primitive, **kwargs):
        pass

    @abstractmethod
    def visit_primitive_composite(self, node: PrimitiveComposite, **kwargs):
        pass

    @abstractmethod
    def visit_tree_convertable(self, node: TreeConvertableNode, **kwargs):
        pass

    @abstractmethod
    def visit_tree_composite(self, node: TreeComposite, **kwargs):
        pass

    @abstractmethod
    def visit_array(self, node: Array, **kwargs):
        pass

    @abstractmethod
    def visit_unsupported(self, node: Unsupported, **kwargs):
        pass


class BFS(Visitor):
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


class DFS(Visitor):
    """Visitor pattern for DataNode"""

    def visit(self, tree: Node) -> None:
        self._dfs(tree, 0)  # Start the DFS traversal with depth 0

    def _dfs(self, node: Node, depth: int) -> None:
        if node is None:
            return
        self.process_node(node, depth)
        for child in node.children:
            self._dfs(child, depth + 1)
