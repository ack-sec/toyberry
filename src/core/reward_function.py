from core.actions import Action


class RewardFunction:
    """
    Assigns numerical rewards to actions based on their quality or correctness.
    """

    def get_reward(self, action: Action) -> float:
        """
        Returns a numerical reward for a given action.
        """
        rewards = {
            Action.A1: 0.5,  # Propose next one-step thought
            Action.A2: 1.0,  # Propose remaining reasoning steps
            Action.A3: 0.7,  # Propose next sub-question with answer
            Action.A4: 1.2,  # Re-answer with few-shot CoT
            Action.A5: 0.3,  # Rephrase the question
            Action.A6: 1.0,  # Explore alternative methods
            Action.A7: 0.8,  # Summarize reasoning
            
        }
        return rewards.get(action, 0.0)
