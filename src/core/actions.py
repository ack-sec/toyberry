from enum import Enum, auto


class Action(Enum):
    A1 = auto()  # Propose next one-step thought
    A2 = auto()  # Propose remaining reasoning steps
    A3 = auto()  # Propose next sub-question with answer
    A4 = auto()  # Re-answer with few-shot Chain-of-Thought
    A5 = auto()  # Rephrase the question
    A6 = auto()  # Explore alternative problem-solving methods
    A7 = auto()  # Summarize the reasoning so far


class ActionType:
    """
    Represents an action with a name and description.
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def __repr__(self):
        return f"ActionType(name='{self.name}', description='{self.description}')"
