from core.llm_interface import LLMInterface
from core.actions import Action
from core.reward_function import RewardFunction
from core.memory import Memory
from core.discriminator import Discriminator
from atlas.atlas_reasoning_searcher import ReasoningAtlasSearcher
from atlas.atlas_interface import AtlasNode
from core.calculate_confidence import ConfidenceCalculator
from typing import List, Dict, Optional
import logging


class IntegratedAtlas:
    """
    Integrates the ReasoningAtlasSearcher and ReasoningAtlasNode with the Atlas pipeline.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        discriminator: Discriminator,
        memory: Memory,
        reward_function: RewardFunction,
        exploration_weight: float = 1.4,
        weight_scheduler: str = "const",
        num_rollouts: int = 10,
        discount: float = 1.0,
        verbose: bool = False,
    ):
        self.searcher = ReasoningAtlasSearcher(
            exploration_weight=exploration_weight,
            weight_scheduler=weight_scheduler,
            num_rollouts=num_rollouts,
            discount=discount,
            llm_interface=llm_interface,
            memory=memory,
            reward_function=reward_function,
            discriminator=discriminator,
            verbose=verbose,
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def solve(
        self, question: str, injected_thoughts: str = "", condition: str = ""
    ) -> Optional[List[Dict[str, str]]]:
        """
        Solves the reasoning problem using integrated Atlas search.
        Returns the best trajectory.
        """
        self.logger.info(f"Starting Atlas search for question: {question}")
        # Initialize root node
        root_node = self.searcher.create_node(
            state=question, condition=condition, injected_thoughts=injected_thoughts
        )
        self.searcher.root_node = root_node  # Assign root node for visualization

        # Run Atlas rollouts
        for rollout_id in range(1, self.searcher.num_rollouts + 1):
            self.searcher.do_rollout(root_node, rollout_id)
            if rollout_id % 10 == 0:
                self.logger.info(f"Completed {rollout_id} rollouts.")

        # Collect all trajectories
        collected_trajectories = self.collect_trajectories(root_node)
        self.logger.info(
            f"Collected {len(collected_trajectories)} trajectories from Atlas search."
        )

        # Validate trajectories with discriminator
        validated_trajectories = [
            traj
            for traj in collected_trajectories
            if self.searcher.discriminator.verify_trajectory(question, traj)
        ]

        self.logger.info(
            f"{len(validated_trajectories)} trajectories passed mutual consistency check."
        )

        if not validated_trajectories:
            self.logger.warning("No valid trajectory found.")
            return None

        # Select the best trajectory based on reward and confidence
        best_trajectory = self.select_best_trajectory(validated_trajectories)
        self.logger.info("Selected best trajectory.")

        # Store the trajectory in memory
        self.searcher.memory.add_trajectory(question, best_trajectory)

        return best_trajectory

    def collect_trajectories(
        self,
        node: AtlasNode,
        path: List[AtlasNode] = None,
        trajectories: List[List[Dict[str, str]]] = None,
    ) -> List[List[Dict[str, str]]]:
        if trajectories is None:
            trajectories = []
        if path is None:
            path = []
        path = path + [node]
        if not self.searcher.parent2children.get(node):
            # Convert node path to trajectory format
            trajectory = [
                {"action": n.action, "state": n.state}
                for n in path
                if hasattr(n, "action") and n.action
            ]
            trajectories.append(trajectory)
            self.logger.debug(f"Collected trajectory: {trajectory}")
        else:
            for child in self.searcher.parent2children[node]:
                trajectories = self.collect_trajectories(child, path, trajectories)
        return trajectories

    def select_best_trajectory(
        self, trajectories: List[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Selects the best trajectory based on cumulative rewards and confidence.
        """
        confidence_calculator = ConfidenceCalculator()
        confidence = confidence_calculator.calculate_confidence(trajectories)
        self.logger.debug(f"Calculated confidence: {confidence:.2f}")

        best_score = float("-inf")
        best_traj = None
        for traj in trajectories:
            cumulative_reward = sum(
                self.searcher.reward_function.get_reward(step["action"])
                for step in traj
            )
            score = cumulative_reward * confidence
            self.logger.debug(
                f"Trajectory score: {score:.2f} (Cumulative Reward: {cumulative_reward:.2f}, Confidence: {confidence:.2f})"
            )
            if score > best_score:
                best_score = score
                best_traj = traj
        return best_traj