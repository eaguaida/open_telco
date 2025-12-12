from collections import Counter, defaultdict

from inspect_ai.scorer import SampleScore, Value, metric


@metric
def maj_at_k():
    """Majority voting metric across epochs."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        if not scores:
            return 0.0

        grouped = defaultdict(list)
        for score in scores:
            grouped[score.sample_id].append(score.score)

        correct = 0
        for sample_scores in grouped.values():
            answers = [s.answer for s in sample_scores if s.answer]
            if not answers:
                continue

            majority = Counter(answers).most_common(1)[0][0]
            correct += any(
                s.value == 1 and s.answer == majority for s in sample_scores
            )

        return correct / len(grouped)

    return metric_fn