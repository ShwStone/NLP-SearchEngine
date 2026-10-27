"""Historical research metrics and seeded paired bootstrap."""

import math
import random
import statistics


def metrics(ids, relevance):
    relevant = {doc for doc, grade in relevance.items() if grade > 0}
    top = ids[:10]
    dcg = sum((2 ** relevance.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(top))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:10]))
    hits, ap = 0, 0.0
    for rank, doc in enumerate(ids[:100], 1):
        if doc in relevant:
            hits += 1
            ap += hits / rank
    return {"ndcg@10": dcg / ideal if ideal else 0.0,
            "recall@10": len(set(top) & relevant) / len(relevant) if relevant else 0.0,
            "map@100": ap / len(relevant) if relevant else 0.0}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    left = int(position)
    return ordered[left] + (ordered[min(left + 1, len(ordered) - 1)] - ordered[left]) * (position - left)


def bootstrap(differences, samples=2000):
    if not differences:
        return None
    rng = random.Random(42)
    means = [statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples)]
    return {"mean": statistics.mean(differences), "ci95": [percentile(means, .025), percentile(means, .975)]}
