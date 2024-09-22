from atlas.atlas_interface import AtlasNode
from core.actions import Action
from core.llm_interface import LLMInterface
from core.memory import Memory
from typing import List, Dict, Optional
import logging


class ReasoningAtlasNode(AtlasNode):
    """
    Concrete implementation of AtlasNode for reasoning tasks.
    Each node represents a state in the reasoning trajectory.
    """

    def __init__(
        self,
        state: str,
        action: Optional[Action] = None,
        parent: Optional["ReasoningAtlasNode"] = None,
        llm_interface: Optional[LLMInterface] = None,
        memory: Optional[Memory] = None,
        condition: str = "",
        injected_thoughts: str = "",
    ):
        super().__init__()
        self.state = state  # Current reasoning steps as a string
        self.action = action  # Action taken to reach this state
        self.parent = parent  # Parent node
        self.children: List["ReasoningAtlasNode"] = (
            []
        )  # List of child ReasoningAtlasNodes
        self.llm = llm_interface  # Instance of LLMInterface
        self.memory = memory  # Instance of Memory
        self.condition = condition  # Conditioned input
        self.injected_thoughts = injected_thoughts  # Injected thoughts
        self.logger = logging.getLogger(self.__class__.__name__)

    def find_children(self, rollout_id: int) -> List["ReasoningAtlasNode"]:
        """
        Generate all possible children nodes by applying each possible action.
        """
        children = []
        for action in Action:
            messages = self.generate_messages(action)
            response = self.llm.call_llm(messages)
            self.logger.debug(f"LLM Response for Action {action.name}: {response}")
            new_state = f"{self.state} ⊕ {response}"
            child_node = ReasoningAtlasNode(
                state=new_state,
                action=action,
                parent=self,
                llm_interface=self.llm,
                memory=self.memory,
                condition=self.condition,
                injected_thoughts=self.injected_thoughts,
            )
            children.append(child_node)
            self.logger.debug(
                f"Created child node {child_node.id} with state: {child_node.state}"
            )
        return children

    def generate_messages(self, action: Action) -> List[Dict[str, str]]:
        """
        Generate the messages list based on the action and current state.
        """
        system_message = {
            "role": "system",
            "content": "You are an AI system that provides diverse and creative reasoning steps to solve problems. Ensure that each reasoning step explores a different perspective or method.",
        }
        user_prompt = f"Given the question: {self.get_question()}\nCurrent reasoning steps: {self.get_reasoning_steps()}\n"

        if action == Action.A1:
            user_prompt += "Propose the next one-step thought."
        elif action == Action.A2:
            user_prompt += (
                "Propose the remaining reasoning steps to reach the final answer."
            )
        elif action == Action.A3:
            user_prompt += "Propose the next sub-question along with its answer."
        elif action == Action.A4:
            user_prompt += (
                "Re-answer the last sub-question using a few-shot Chain-of-Thought."
            )
        elif action == Action.A5:
            user_prompt = f"Rephrase the question to clearly list all conditions provided in the problem statement: {self.get_question()}"
        elif action == Action.A6:
            user_prompt += (
                "Explore an alternative problem-solving method for this question."
            )
        elif action == Action.A7:
            user_prompt += "Summarize the reasoning steps taken so far."
        else:
            raise ValueError("Invalid action")

        # For terminal actions, ensure "ANSWER:" is included
        if action in [Action.A2, Action.A4, Action.A6, Action.A7]:
            user_prompt += (
                " Please conclude your reasoning with 'ANSWER: [your answer here]'."
            )

        user_message = {"role": "user", "content": user_prompt}

        messages = [system_message, user_message]
        self.logger.debug(f"Generated messages for Action {action.name}: {messages}")
        return messages

    def get_question(self) -> str:
        """
        Extracts the original question from the state.
        """
        question = self.state.split("⊕")[0] if "⊕" in self.state else self.state
        self.logger.debug(f"Extracted question: {question}")
        return question

    def get_reasoning_steps(self) -> str:
        """
        Extracts the reasoning steps from the state.
        """
        reasoning = " ⊕ ".join(self.state.split("⊕")[1:]) if "⊕" in self.state else ""
        self.logger.debug(f"Extracted reasoning steps: {reasoning}")
        return reasoning

    def is_terminal(self) -> bool:
        """
        Determine if the node is a terminal node.
        """
        terminal = (
            "ANSWER:" in self.state.upper()
            or len(self.state.split("⊕")) >= self.memory.max_depth
        )
        self.logger.debug(f"Node {self.id} is_terminal: {terminal}")
        return terminal

    def calculate_reward(self) -> float:
        """
        Calculate the reward for a terminal node.
        """
        if "ANSWER:" in self.state.upper():
            self.logger.debug(f"Node {self.id} contains ANSWER. Assigning reward 1.0")
            return 1.0  # Reward for reaching an answer
        else:
            self.logger.debug(
                f"Node {self.id} does not contain ANSWER. Assigning reward 0.0"
            )
            return 0.0  # No reward otherwise

    def skip_backprop(self) -> bool:
        """
        Decide whether to skip backpropagation for this node.
        """
        if self.action == Action.A5:
            self.logger.debug(f"Node {self.id} action is A5. Skipping backpropagation.")
            return True
        self.logger.debug(
            f"Node {self.id} action is {self.action.name}. Backpropagation allowed."
        )
        return False
