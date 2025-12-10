from collections import Counter, defaultdict
from typing import List

from inspect_ai.scorer import SampleScore, Value, metric


@metric
def maj_at_k():
    """Majority voting metric across epochs."""

    def metric_fn(scores: List[SampleScore]) -> Value:
        if not scores:
            return 0.0

        grouped = defaultdict(list)
        for sample_score in scores:
            grouped[sample_score.sample_id].append(sample_score.score)

        correct = 0
        for sample_scores in grouped.values():
            answers = [score.answer for score in sample_scores if score.answer]
            if not answers:
                continue

            majority_answer = Counter(answers).most_common(1)[0][0]
            if any(score.value == 1 and score.answer == majority_answer for score in sample_scores):
                correct += 1

        return correct / len(grouped)

    return metric_fn

