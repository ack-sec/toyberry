from sentence_transformers import SentenceTransformer, util
import re
import math
from typing import List, Dict, Union, Tuple


class ConfidenceCalculator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def calculate_confidence(
        self, trajectories: List[List[Dict[str, str]]], threshold: float = 0.7
    ) -> float:
        """
        Calculates a confidence score based on the diversity and agreement among trajectories.
        """
        if not trajectories:
            return 0.0

        embeddings = self._get_embeddings(trajectories)
        average_similarity = self._compute_average_similarity(embeddings)

        # Confidence is inversely related to average similarity
        confidence = (
            1.0 - average_similarity
        )  # Higher confidence for more diverse trajectories
        return max(0.0, min(1.0, confidence))

    def calculate_confidence1(
        self, trajectories: Union[List[Dict], List[List], List[str]]
    ) -> float:
        """
        Aggregate confidence scores from different metrics.
        """
        confidence_semantic = self.calculate_confidence_semantic(trajectories)
        confidence_variance = self.calculate_confidence_variance(trajectories)
        confidence = (0.7 * confidence_semantic) + (0.3 * confidence_variance)
        return max(0.0, min(1.0, confidence))

    def calculate_confidence_semantic(
        self, trajectories: Union[List[Dict], List[List], List[str]]
    ) -> float:
        """
        Calculate confidence based on semantic similarity among multiple trajectories.
        """
        if len(trajectories) < 2:
            return 0.0  # Not enough data to calculate confidence

        embeddings = self._get_embeddings(trajectories)
        average_similarity = self._compute_average_similarity(embeddings)
        return max(0.0, min(1.0, average_similarity))

    def calculate_confidence_variance(
        self, trajectories: Union[List[Dict], List[List], List[str]]
    ) -> float:
        """
        Calculate confidence based on the variance of numerical answers.
        """
        answers = self._extract_numerical_answers(trajectories)
        if not answers:
            return 0.0

        mean, variance = self._calculate_mean_and_variance(answers)
        std_dev = math.sqrt(variance)

        confidence_variance = max(0.0, 1 - (std_dev / (mean if mean != 0 else 1)))
        return confidence_variance

    def _get_embeddings(
        self,
        trajectories: Union[
            List[List[Dict[str, str]]], List[Dict], List[List], List[str]
        ],
    ) -> List:
        if isinstance(trajectories[0], dict) and "state" in trajectories[0]:
            texts = [traj["state"] for traj in trajectories]
        elif isinstance(trajectories[0], list) and isinstance(trajectories[0][0], dict):
            texts = [
                " ".join([step["state"] for step in traj]) for traj in trajectories
            ]
        elif isinstance(trajectories[0], list):
            texts = [" ".join(map(str, traj)) for traj in trajectories]
        else:
            texts = [str(traj) for traj in trajectories]

        return self.model.encode(texts, convert_to_tensor=True)

    def _compute_average_similarity(self, embeddings: List) -> float:
        cosine_scores = util.cos_sim(embeddings, embeddings)
        n = len(embeddings)
        similarity_sum = (
            cosine_scores.sum().item() - n
        ) / 2  # Exclude self-similarities
        return similarity_sum / (n * (n - 1) / 2) if n > 1 else 0.0

    def _extract_numerical_answers(
        self, trajectories: Union[List[Dict], List[List], List[str]]
    ) -> List[int]:
        answers = []
        for traj in trajectories:
            state = (
                traj["state"]
                if isinstance(traj, dict)
                else " ".join(map(str, traj)) if isinstance(traj, list) else str(traj)
            )
            match = re.search(r"\b\d+\b", state)
            if match:
                answers.append(int(match.group()))
        return answers

    def _calculate_mean_and_variance(self, numbers: List[int]) -> Tuple[float, float]:
        n = len(numbers)
        mean = sum(numbers) / n
        variance = sum((x - mean) ** 2 for x in numbers) / n
        return mean, variance
