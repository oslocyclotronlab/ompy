


class DictVisitor(Visitor):
    def visit(self, node: Node) -> dict[str, Any]:
        print(f"Visiting {node}")
        return node.accept(self)

    def visit_root(self, node: Root) -> dict[str, Any]:
        return self.visit(node.children[0])

    def visit_datanode(self, node: DataNode) -> dict[str, Any]:
        assert node.dtype is not None
        return {node.label: node.data if node.data is not None
               else node.dtype()}

    def visit_primitive(self, node: Primitive) -> dict[str, Any]:
        return self.visit_datanode(node)

    def visit_primitive_composite(self, node: PrimitiveComposite) -> dict[str, Any]:
        return self.visit_datanode(node)

    def visit_tree_convertable(self, node: TreeConvertableNode) -> dict[str, Any]:
        assert node.dtype is not None
        #obj = TREE_CONVERTABLES[node.dtype.__name__].from_tree(node)
        return {node.label: ''}

    def visit_tree_composite(self, node: TreeComposite) -> dict[str, Any]:
        assert node.dtype is not None
        assert node.element_type is not None
        return {node.label: node.dtype([self.visit(c) for c in node.children])}

    def visit_array(self, node: Array) -> dict[str, Any]:
        assert node.dtype is not None
        return {node.label: node.data}

    def visit_unsupported(self, node: Unsupported) -> dict[str, Any]:
        print(f"Unsupported node encountered: {node}")
        return {}
