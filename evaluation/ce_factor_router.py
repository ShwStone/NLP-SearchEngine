"""Fixed-quota RFC-0005 research allocator. Its inputs are pre-CE predictions."""
import argparse
import json
from pathlib import Path

import numpy as np


def _finite_array(value, shape, field):
    values = np.asarray(value, dtype=float)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{field} must be finite and have shape {shape}")
    return values


def route_batch(query_ids, prior_score, prior_depth, relevance_probability, band_probability,
                improvement_probability, conditional_gain, incumbent_loss, calibration):
    """Return per-query depths, updated scores and the reroutable budget band.

    Challenger arrays have eight slots. The three band probabilities respectively
    cover documents outside the hybrid prefix, in its first 28, and in ranks
    29 to 40. Loss columns predict exit and within-top-ten demotion for four
    incumbents. Padded challenger slots have zero relevance probability.
    All ranks used to produce these predictions must come from first-pass BM25.
    """
    ids = np.asarray(query_ids)
    if ids.ndim != 1 or ids.dtype.kind not in 'iU' or len(np.unique(ids)) != len(ids):
        raise ValueError("query_ids must be unique integers or Unicode strings")
    n = len(ids)
    prior = _finite_array(prior_score, (n,), 'prior_score')
    depths = np.asarray(prior_depth)
    if depths.shape != (n,) or depths.dtype.kind not in 'iu' or not np.isin(depths, [0, 28, 40]).all():
        raise ValueError("prior_depth must contain integer depths 0, 28 or 40")
    rel = _finite_array(relevance_probability, (n, 8), 'relevance_probability')
    bands = _finite_array(band_probability, (n, 8, 3), 'band_probability')
    event = _finite_array(improvement_probability, (n, 8, 3), 'improvement_probability')
    amount = _finite_array(conditional_gain, (n, 8), 'conditional_gain')
    loss = _finite_array(incumbent_loss, (n, 4, 2), 'incumbent_loss')
    for name, values in [('relevance_probability', rel), ('band_probability', bands),
                         ('improvement_probability', event), ('conditional_gain', amount),
                         ('incumbent_loss', loss)]:
        if np.any((values < 0) | (values > 1)):
            raise ValueError(f"{name} must be in [0, 1]")
    active = rel > 0
    if active.any() and not np.allclose(bands.sum(axis=2)[active], 1., atol=1e-6, rtol=0):
        raise ValueError("band probabilities must sum to one for active challenger slots")

    # Old score is mapped to net gain and the two expected incumbent losses.
    coefficients = _finite_array(calibration['net_and_losses'], (2, 3), 'net_and_losses')
    gross_coefficients = _finite_array(calibration['challenger_gain'], (2,), 'challenger_gain')
    predictors = np.column_stack([np.ones(n), prior])
    anchors = predictors @ coefficients
    new_gross = (rel * (bands * event).sum(axis=2) * amount).sum(axis=1)
    scores = anchors.sum(axis=1) - loss.sum(axis=(1, 2)) + new_gross - predictors @ gross_coefficients

    actions = depths.copy()
    eligible = np.flatnonzero(depths != 0)
    old_order = eligible[np.lexsort((ids[eligible], -prior[eligible]))]
    slots = int(np.count_nonzero(depths == 40))
    radius = min(max(1, slots // 4), slots, len(eligible) - slots)
    band = old_order[slots - radius:slots + radius]
    available = int(np.count_nonzero(depths[band] == 40))
    new_order = band[np.lexsort((ids[band], -scores[band]))]
    actions[band] = 28
    actions[new_order[:available]] = 40
    in_band = np.zeros(n, dtype=bool)
    in_band[band] = True
    return actions, scores, in_band


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--calibration', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
    with np.load(args.inputs, allow_pickle=False) as batch:
        fields = ['query_ids', 'prior_score', 'prior_depth', 'relevance_probability',
                  'band_probability', 'improvement_probability', 'conditional_gain', 'incumbent_loss']
        actions, scores, band = route_batch(*(batch[key] for key in fields), calibration=calibration)
        ids = batch['query_ids']
    with args.output.open('xb') as out:
        np.savez_compressed(out, query_ids=ids, depth=actions, score=scores, in_band=band)


if __name__ == '__main__':
    main()
