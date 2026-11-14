"""Label-free matched-budget controls for optional CE40 reranking."""

from collections import defaultdict
from hashlib import sha256


def matched_random(
    selected: dict[str, bool], capacity: dict[str, int], seed: int = 2026
) -> dict[str, bool]:
    """Sample the same number of reranked queries at each positive pair capacity.

    Capacity is the number of CE pairs the query would score if selected, with
    zero reserved for ineligible or empty-pool queries. Keeping the count at
    each capacity fixes both the query count and the total number of CE pairs.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if selected.keys() != capacity.keys():
        raise ValueError("selected and capacity must cover the same query IDs")

    strata: dict[int, list[str]] = defaultdict(list)
    for query_id, pairs in capacity.items():
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query IDs must be nonempty strings")
        if type(selected[query_id]) is not bool:
            raise ValueError(f"selection for {query_id!r} must be boolean")
        if type(pairs) is not int or not 0 <= pairs <= 40:
            raise ValueError(f"capacity for {query_id!r} must be an integer from 0 to 40")
        if pairs == 0 and selected[query_id]:
            raise ValueError(f"selected query {query_id!r} has zero CE pair capacity")
        if pairs > 0:
            strata[pairs].append(query_id)

    matched = dict.fromkeys(selected, False)
    prefix = f"ce-budget:{seed}\0".encode("utf-8")
    for query_ids in strata.values():
        count = sum(selected[query_id] for query_id in query_ids)
        ordered = sorted(query_ids, key=lambda query_id: (
            sha256(prefix + query_id.encode("utf-8")).digest(), query_id
        ))
        for query_id in ordered[:count]:
            matched[query_id] = True
    return matched
