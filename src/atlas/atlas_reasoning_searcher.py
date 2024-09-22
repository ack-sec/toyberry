from atlas.atlas_interface import AtlasSearcher, AtlasNode
from core.llm_interface import LLMInterface
from core.memory import Memory
from core.reward_function import RewardFunction
from core.discriminator import Discriminator
from typing import List
import math
import random
import logging
from atlas.atlas_concrete_node import ReasoningAtlasNode

class ReasoningAtlasSearcher(AtlasSearcher):
    """
    Concrete implementation of AtlasSearcher for reasoning tasks.
    """

    def __init__(
        self,
        exploration_weight: float,
        weight_scheduler: str,
        num_rollouts: int,
        discount: float,
        llm_interface: LLMInterface,
        memory: Memory,
        reward_function: RewardFunction,
        discriminator: Discriminator,
        verbose: bool = False,
    ):
        super().__init__(
            exploration_weight=exploration_weight,
            weight_scheduler=weight_scheduler,
            num_rollouts=num_rollouts,
            discount=discount,
            verbose=verbose,
        )
        self.llm = llm_interface
        self.memory = memory
        self.reward_function = reward_function
        self.discriminator = discriminator
        self.logger = logging.getLogger(self.__class__.__name__)

    def create_node(
        self, state: str, condition: str, injected_thoughts: str
    ) -> ReasoningAtlasNode:
        """
        Create a new ReasoningAtlasNode with the given state and parameters.
        """
        node = ReasoningAtlasNode(
            state=state,
            action=None,  # Root node has no action
            parent=None,  # Root node has no parent
            llm_interface=self.llm,
            memory=self.memory,
            condition=condition,
            injected_thoughts=injected_thoughts,
        )
        self.logger.debug(f"Created root node {node.id} with state: {node.state}")
        return node

    def _select(self, node: ReasoningAtlasNode, rollout_id: int) -> ReasoningAtlasNode:
        """
        Select a child node with the highest UCB1 value.
        """
        self.logger.debug(f"Selecting node {node.id} for rollout {rollout_id}")
        while not node.is_terminal():
            if node not in self.parent2children or not self.parent2children[node]:
                self.logger.debug(f"Node {node.id} is not fully expanded.")
                return node  # Node is not fully expanded
            # Select child with highest UCB1 score
            best_score = float('-inf')
            best_child = None
            for child in self.parent2children[node]:
                ucb1 = self._ucb1(child)
                self.logger.debug(f"Node {child.id} UCB1 Score: {ucb1}")
                if ucb1 > best_score:
                    best_score = ucb1
                    best_child = child
            if best_child is None:
                self.logger.debug(f"No suitable child found for node {node.id}.")
                return node
            self.logger.debug(
                f"Selected child node {best_child.id} with UCB1 score {best_score}."
            )
            node = best_child
        self.logger.debug(f"Selected node {node.id} is terminal.")
        return node

    def _ucb1(self, node: ReasoningAtlasNode) -> float:
        """
        Calculate the UCB1 score for a node.
        """
        if self.N[node] == 0:
            self.logger.debug(
                f"Node {node.id} has not been visited yet. Assigning UCB1 as infinity."
            )
            return float('inf')  # Ensure unvisited nodes are selected first
        exploitation = self.Q[node] / self.N[node]
        parent = node.parent
        if parent is None or self.N[parent] == 0:
            exploration = 0
        else:
            exploration = self.exploration_weight * math.sqrt(
                math.log(self.N[parent]) / self.N[node]
            )
        ucb1 = exploitation + exploration
        self.logger.debug(
            f"Node {node.id} UCB1: Exploitation={exploitation:.4f}, Exploration={exploration:.4f}, Total={ucb1:.4f}"
        )
        return ucb1

    def _expand(self, node: ReasoningAtlasNode, rollout_id: int):
        """
        Expand a node by generating its children.
        """
        self.logger.debug(f"Expanding node {node.id} for rollout {rollout_id}")
        children = node.find_children(rollout_id)
        self.parent2children[node] = children
        self.logger.debug(f"Node {node.id} has {len(children)} children.")
        for child in children:
            self.Q[child] = 0.0
            self.N[child] = 0

    def _simulate(
        self, node: ReasoningAtlasNode, rollout_id: int
    ) -> List[ReasoningAtlasNode]:
        """
        Simulate a rollout from the node to a terminal state.
        """
        path = []
        current_node = node
        steps = 0
        max_simulation_steps = self.memory.max_depth  # Prevent infinite simulations
        self.logger.debug(
            f"Starting simulation from node {current_node.id} with max steps {max_simulation_steps}"
        )

        while not current_node.is_terminal() and steps < max_simulation_steps:
            children = current_node.find_children(rollout_id)
            if not children:
                self.logger.debug(
                    f"Node {current_node.id} has no children. Ending simulation."
                )
                break
            current_node = random.choice(children)
            path.append(current_node)
            self.logger.debug(
                f"Simulation step {steps + 1}: Moved to node {current_node.id}"
            )
            steps += 1

        if current_node.is_terminal():
            self.logger.debug(
                f"Simulation reached terminal node {current_node.id} with state: {current_node.state}"
            )
        else:
            self.logger.debug(
                f"Simulation ended at node {current_node.id} after reaching max steps."
            )

        return path

    def _backpropagate(self, path: List[ReasoningAtlasNode]):
        """
        Backpropagate the rewards up the path.
        """
        if not path:
            self.logger.debug("No path to backpropagate.")
            return

        cumulative_reward = 0.0

        for node in reversed(path):
            if node.skip_backprop():
                self.logger.debug(
                    f"Skipping backpropagation for node {node.id} due to skip_backprop."
                )
                continue

            if node.action:
                reward = self.reward_function.get_reward(node.action)
                cumulative_reward += reward
                self.logger.debug(
                    f"Node {node.id} Action: {node.action.name}, Reward: {reward:.4f}, Cumulative Reward: {cumulative_reward:.4f}"
                )
            else:
                reward = 0.0  # No reward for root node
                self.logger.debug(f"Node {node.id} is root. Reward: {reward:.4f}")

            self.Q[node] += cumulative_reward
            self.N[node] += 1
            self.logger.debug(
                f"Updated node {node.id}: Q={self.Q[node]:.4f}, N={self.N[node]}"
            )

            cumulative_reward *= self.discount

    def do_rollout(self, root_node: ReasoningAtlasNode, rollout_id: int):
        """
        Perform a single rollout in the Atlas search tree.
        """
        self.logger.debug(f"Starting rollout {rollout_id}")
        selected_node = self._select(root_node, rollout_id)
        if not selected_node.is_terminal():
            self._expand(selected_node, rollout_id)
            children = self.parent2children.get(selected_node, [])
            if children:
                selected_node = random.choice(children)
                self.logger.debug(
                    f"After expansion, selected node {selected_node.id} for simulation."
                )
        simulation_path = self._simulate(selected_node, rollout_id)
        self._backpropagate(simulation_path)
        self.logger.debug(f"Completed rollout {rollout_id}")

    # Implement the abstract methods from AtlasSearcher
    def calculate_reward(self, node: AtlasNode) -> float:
        """
        Implementation of the abstract calculate_reward method.
        This method is no longer used as rewards are handled in _backpropagate.
        """
        return node.calculate_reward()

    def find_children(self, node: AtlasNode, rollout_id: int):
        """
        Implementation of the abstract find_children method.
        This method is handled within the node's find_children method.
        """
        pass  # Already handled in the node's find_children method

    def skip_backprop(self, node: AtlasNode) -> bool:
        """
        Implementation of the abstract skip_backprop method.
        This method is handled within the node's skip_backprop method.
        """
        return node.skip_backprop()
