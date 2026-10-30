"""Exact score-descending/docid-ascending top-k without a full corpus sort."""
import math


def stable_dense_topk(scores, k):
    import torch
    if scores.ndim != 1 or type(k) is not int or k < 1 or scores.numel() < k:
        raise ValueError('expected a dense score vector and an available positive k')
    n = scores.numel()
    values, indices = torch.topk(scores, min(k + 1, n), sorted=True)
    vals, ids = values.cpu().tolist(), indices.cpu().tolist()
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('nonfinite dense retrieval score')
    if k < n and vals[k - 1] == vals[k]:
        cutoff = vals[k - 1]
        higher = [(i, v) for i, v in zip(ids, vals) if v > cutoff]
        # nonzero visits the original vector in docid order. All ties at the
        # boundary must be considered, including those omitted by torch.topk.
        ties = torch.nonzero(scores == cutoff, as_tuple=False).flatten()[:k - len(higher)].cpu().tolist()
        pairs = higher + [(i, cutoff) for i in ties]
    else:
        pairs = list(zip(ids[:k], vals[:k]))
    return sorted(pairs, key=lambda row: (-row[1], row[0]))
