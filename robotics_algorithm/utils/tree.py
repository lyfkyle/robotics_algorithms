import networkx as nx


class TreeNode:
    def __init__(self, nx_key, tree: nx.DiGraph):
        self.nx_key = nx_key

        self.children = []
        self.parent = None

        self._tree = tree

    @property
    def attr(self):
        return self._tree.nodes[self.nx_key]

    def transition_attr(self, child):
        return self._tree.edges[self.nx_key, child.nx_key]

    def add_child(self, child, **kwargs):
        self._tree.add_edge(self.nx_key, child.nx_key, **kwargs)
        self.children.append(child)
        child.parent = self

    def delete_child(self, child):
        self._tree.remove_node(child.nx_key)
        self.children.remove(child)


class Tree:
    def __init__(self):
        self._num_node = 0
        self._tree = nx.DiGraph()

    def add_node(self, **kwargs) -> TreeNode:
        # n_node = TreeNode(value)
        nx_key = self._tree.number_of_nodes()
        self._tree.add_node(nx_key, **kwargs)
        return TreeNode(nx_key, self._tree)

    def delete_node(self, node: TreeNode):
        def _delete_recur(n):
            for child in n.children:
                _delete_recur(child)

            self._tree.remove_node(n.nx_key)

        assert node.nx_key in self._tree.nodes

        # Delete subtree rooted at n
        _delete_recur(node)

    def add_child(self, parent: TreeNode, child: TreeNode, **kwargs):
        return parent.add_child(child, **kwargs)

    def delete_child(
        self,
        parent: TreeNode,
        child: TreeNode,
    ):
        return parent.delete_child(child)
