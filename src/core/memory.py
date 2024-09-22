import pickle
import os
from core.config import MAX_DEPTH, MAX_TRAJECTORIES


class Memory:
    def __init__(self, filepath="memory.pkl", max_depth=10, max_trajectories=10):
        self.filepath = filepath
        self.max_depth = MAX_DEPTH
        self.max_trajectories = MAX_TRAJECTORIES
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "rb") as f:
                return pickle.load(f)
        return {"trajectories": {}, "injected_thoughts": {}, "conditions": {}}

    def save_memory(self):
        with open(self.filepath, "wb") as f:
            pickle.dump(self.memory, f)

    # Trajectory Methods
    def add_trajectory(self, question, trajectory):
        if question not in self.memory["trajectories"]:
            self.memory["trajectories"][question] = []
        if len(self.memory["trajectories"][question]) >= self.max_trajectories:
            self.memory["trajectories"][question].pop(0)  # Remove oldest trajectory
        self.memory["trajectories"][question].append(trajectory)
        self.save_memory()

    def get_trajectories(self, question):
        return self.memory["trajectories"].get(question, [])

    # Injected Thoughts Methods
    def inject_thoughts(self, identifier, thoughts):
        """
        Inject pre-defined thoughts with a unique identifier.
        """
        self.memory["injected_thoughts"][identifier] = thoughts
        self.save_memory()

    def get_injected_thoughts(self, identifier):
        return self.memory["injected_thoughts"].get(identifier, "")

    # Conditioned Inputs Methods
    def add_condition(self, identifier, condition):
        """
        Add conditioned inputs with a unique identifier.
        """
        self.memory["conditions"][identifier] = condition
        self.save_memory()

    def get_condition(self, identifier):
        return self.memory["conditions"].get(identifier, "")
