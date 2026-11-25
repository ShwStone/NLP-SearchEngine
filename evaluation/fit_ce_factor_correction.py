"""Fit the final gain and loss heads on frozen out-of-fold nominee features."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def tree(classifier=False):
    cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
    return cls(max_iter=55, max_leaf_nodes=15, min_samples_leaf=64,
               learning_rate=.07, l2_regularization=2, early_stopping=False,
               random_state=17)


def fit_heads(fit, calibration):
    # Rows are flattened in query-major, then stable nomination order.
    x = fit['challenger_x']
    valid = fit['challenger_valid'].ravel()
    known = fit['relevant'].ravel() & valid
    slot = fit['band'].ravel()
    gain = fit['gain'].ravel()
    positive = gain > 0
    exposed = known & (slot > 0)
    if not np.array_equal(np.unique(slot[known]), [0, 1, 2]):
        raise ValueError('FIT must contain known relevant nominees in all three bands')
    if not positive.any() or np.any(positive & ~exposed):
        raise ValueError('invalid positive gain supervision')
    rel = tree(True).fit(x[valid], known[valid])
    place = tree(True).fit(x[known], slot[known])
    event = tree(True).fit(np.column_stack([x[exposed], slot[exposed]]), positive[exposed])
    amount = tree().fit(x[positive], gain[positive])
    loss_x = fit['incumbent_x']
    keep = fit['incumbent_valid'].ravel()
    loss = fit['loss'].reshape(-1, 2)
    heads = [tree().fit(loss_x[keep], loss[keep, j]) for j in (0, 1)]

    s = calibration['prior_score']
    z = np.column_stack([np.ones(len(s)), s])
    moments = np.column_stack([calibration['net_gain'], calibration['loss'].sum(1)])
    coef = np.linalg.lstsq(z, moments, rcond=None)[0]
    extra = np.linalg.lstsq(z, calibration['gain'].sum(1), rcond=None)[0]
    return dict(factor=(rel, place, event), amount=amount, loss=heads, coef=coef, extra=extra)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fit', type=Path, required=True)
    cli.add_argument('--calibration', type=Path, required=True)
    cli.add_argument('--nomination-model', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    prior = joblib.load(args.nomination_model)
    model = {name: prior[name] for name in ['membership', 'platt', 'joint', 'router']}
    with np.load(args.fit, allow_pickle=False) as fit, np.load(args.calibration, allow_pickle=False) as cal:
        if not set(fit['groups']).isdisjoint(cal['groups']):
            raise ValueError('FIT and CAL groups overlap')
        model.update(fit_heads(fit, cal))
    with args.output.open('xb') as fp:
        joblib.dump(model, fp, compress=3)


if __name__ == '__main__':
    main()
