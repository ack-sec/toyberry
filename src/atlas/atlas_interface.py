from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional
from core.memory import Memory


class AtlasNode(ABC):
    """
    A representation of a single reasoning state.
    Atlas search works by constructing a tree of these Nodes.
    """

    _node_count = 0

    def __init__(self) -> None:
        self.id = AtlasNode._node_count
        AtlasNode._node_count += 1
        self.rollout_id: Optional[int] = None

    def set_rollout_id(self, rollout_id: int) -> None:
        self.rollout_id = rollout_id

    @abstractmethod
    def find_children(self, rollout_id: int) -> List["AtlasNode"]:
        """
        All possible successors of this reasoning state.
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Returns True if the node has no children.
        """
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        """
        Assumes `self` is a terminal node. Returns a numerical reward.
        """
        pass

    @abstractmethod
    def skip_backprop(self) -> bool:
        """
        If True, the reward of this node will not be accumulated in the backpropagation step.
        """
        pass


class AtlasSearcher(ABC):
    """
    Abstract base class for Atlas searchers.
    """

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        discount: float,
        verbose: bool = False,
    ):
        self.Q: Dict[AtlasNode, float] = defaultdict(float)  # Total reward of each node
        self.N: Dict[AtlasNode, int] = defaultdict(
            int
        )  # Total visit count for each node
        self.parent2children: Dict[AtlasNode, List[AtlasNode]] = (
            dict()
        )  # Children of each node
        self.explored_nodes: set[AtlasNode] = set()
        self.exploration_weight = exploration_weight
        self.weight_scheduler = weight_scheduler
        self.num_rollouts = num_rollouts
        self.discount = discount
        self.verbose = verbose

    @abstractmethod
    def do_rollout(self, root_node: AtlasNode, rollout_id: int) -> None:
        """
        Perform a single rollout in the Atlas search tree.
        """
        pass

    @abstractmethod
    def find_children(self, node: AtlasNode, rollout_id: int) -> List[AtlasNode]:
        """
        Generate children nodes for a given node.
        """
        pass

    @abstractmethod
    def calculate_reward(self, node: AtlasNode) -> float:
        """
        Calculate the reward for a given node.
        """
        pass

    @abstractmethod
    def skip_backprop(self, node: AtlasNode) -> bool:
        """
        Determine if backpropagation should be skipped for a node.
        """
        pass

    @abstractmethod
    def create_node(
        self,
        state: str,
        llm_interface,
        memory: Memory,
        condition: str,
        injected_thoughts: str,
    ) -> AtlasNode:
        """
        Create a new node given the state and other parameters.
        """
        pass

    @abstractmethod
    def _select(self, node: AtlasNode, rollout_id: int) -> AtlasNode:
        """
        Select a node to expand based on the UCB1 algorithm.
        """
        pass

    @abstractmethod
    def _expand(self, node: AtlasNode, rollout_id: int) -> None:
        """
        Expand a node by generating its children.
        """
        pass

    @abstractmethod
    def _simulate(self, node: AtlasNode, rollout_id: int) -> List[AtlasNode]:
        """
        Simulate a rollout from the node to a terminal state.
        """
        pass

    @abstractmethod
    def _backpropagate(self, path: List[AtlasNode]) -> None:
        """
        Backpropagate the reward up the path.
        """
        pass
