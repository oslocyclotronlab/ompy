from .tree import *
from .visitor import DFS
from typing import Any
import numpy as np


class JSONVisitor(DFS):
    def visit(self, node: Node) -> None:
        return node.accept(self)

    def visit_root(self, node: Root) -> dict[str, Any]:
        d = {'label': 'ROOT',
             'node': 'root',
             'children': [self.visit(child) for child in node.children]}
        return d

    def visit_branch(self, node: BranchNode) -> dict[str, Any]:
        return {
            'label': node.label,
            'annotation': str(node.annotation),
            'dtype': str(node.dtype),
        }

    def visit_datanode(self, node: DataNode) -> dict[str, Any]:
        d = self.visit_branch(node)
        d['data'] = node.data
        return d

    def visit_container(self, node: DataContainer) -> dict[str, Any]:
        d = self.visit_datanode(node)
        d['etype'] = str(node.element_type)
        return d

    def visit_primitive(self, node: Primitive) -> dict[str, Any]:
        d = self.visit_datanode(node)
        d['node'] = 'primitive'
        return d

    def visit_primitive_composite(self, node: PrimitiveComposite) -> dict[str, Any]:
        d = self.visit_container(node)
        d['node'] = 'primitive_composite'
        if node.element_type is np.ndarray:
            d['data'] = [x.tolist() for x in node.data]
        return d

    def visit_tree_convertable(self, node: TreeConvertableNode) -> dict[str, Any]:
        d = self.visit_branch(node)
        d['node'] = 'tree_convertable'
        d['children'] = [self.visit(child) for child in node.children]
        return d

    def visit_tree_composite(self, node: TreeComposite) -> dict[str, Any]:
        d = self.visit_branch(node)
        d['etype'] = str(node.element_type)
        d['node'] = 'tree_composite'
        d['children'] = [self.visit(child) for child in node.children]
        return d

    def visit_array(self, node: Array) -> dict[str, Any]:
        d = self.visit_container(node)
        d['data'] = node.data.tolist()
        d['node'] = 'array'
        d['shape'] = node.shape
        return d

    def visit_unsupported(self, node: Unsupported) -> dict[str, Any]:
        d = self.visit_datanode(node)
        d['node'] = 'unsupported'
        return d
