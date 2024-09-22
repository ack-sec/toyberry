from core.llm_interface import LLMInterface
from typing import List, Dict
from sentence_transformers import SentenceTransformer, util


class Discriminator:
    """
    Evaluates the mutual consistency and validity of reasoning trajectories.
    """

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        # Initialize a semantic similarity model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def verify_trajectory(
        self, question: str, trajectory: List[Dict[str, str]]
    ) -> bool:
        """
        Verifies the consistency and correctness of a given trajectory.
        Returns True if the trajectory is valid, False otherwise.
        """
        # Construct a prompt to evaluate the trajectory
        prompt = f"Evaluate the following reasoning trajectory for the question: '{question}'. Determine if the reasoning is consistent and leads to the correct answer.\n\n"
        for idx, step in enumerate(trajectory, 1):
            prompt += f"Step {idx}: Action: {step['action'].name}, Reasoning: {step['state']}\n"
        prompt += (
            "\nDoes this trajectory lead to a correct and consistent answer? Yes or No."
        )

        # Get the discriminator's evaluation
        response = self.llm.call_llm(
            [
                {
                    "role": "system",
                    "content": "You are an evaluator that assesses the consistency and correctness of reasoning trajectories.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        # Interpret the response
        is_valid = "yes" in response.lower()
        return is_valid

    def is_diverse(
        self,
        new_trajectory: List[Dict[str, str]],
        existing_trajectories: List[List[Dict[str, str]]],
        threshold: float = 0.7,
    ) -> bool:
        """
        Determines if the new trajectory is sufficiently different from existing trajectories.
        Returns True if diverse, False otherwise.
        """
        if not existing_trajectories:
            return True  # First trajectory is always diverse

        # Convert trajectories to concatenated reasoning strings
        new_reasoning = " ".join([step["state"] for step in new_trajectory])
        new_embedding = self.model.encode(new_reasoning, convert_to_tensor=True)

        for traj in existing_trajectories:
            existing_reasoning = " ".join([step["state"] for step in traj])
            existing_embedding = self.model.encode(
                existing_reasoning, convert_to_tensor=True
            )
            similarity = util.cos_sim(new_embedding, existing_embedding).item()
            if similarity > threshold:
                return False  # Not diverse enough
        return True  # Diverse
