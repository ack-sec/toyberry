from graphviz import Digraph
import logging


class Visualizer:
    """
    Visualizes the MCTS tree using Graphviz.
    """

    def __init__(self, root_node):
        self.root = root_node
        self.dot = Digraph(comment="MCTS Tree")
        self.logger = logging.getLogger(self.__class__.__name__)

    def visualize(self, path="mcts_tree.gv", view=False):
        self._add_nodes_edges(self.root)
        self.dot.render(path, view=view)
        self.logger.info(f"MCTS tree visualized and saved to {path}")

    def _add_nodes_edges(self, node, parent_id=None):
        node_label = (
            f"ID: {node.id}\nAction: {node.action.name if node.action else 'Root'}"
        )
        self.dot.node(str(node.id), node_label)
        if parent_id is not None:
            self.dot.edge(str(parent_id), str(node.id))
        for child in getattr(node, "children", []):
            self._add_nodes_edges(child, node.id)
