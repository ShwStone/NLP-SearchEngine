"""Audit full-corpus Router drift and CE-28/40 budget selectors.

The command reads frozen score captures and qrels, never changes the frozen
policy artifacts, and writes an exploratory JSON report. It is intended for
investigation before a Router change is promoted.
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from evaluation.dureader_policy_features import gate_features, query_groups, rerank, FrozenLinear
from evaluation.prepare_bm25_router import metrics
from evaluation.train_dureader_policies import read_scores, qrels


def rich_features(row, dfs):
    x = list(gate_features(row, dfs))
    ce = np.asarray(row['ce28']['scores'], dtype=float)
    for k in (1, 3, 5, 10, 20, 28):
        z = ce[:k]
        x.extend((float(z.mean()), float(z.std()), float(z[0] - z[-1])))
    lex = {h['doc_id'] for h in row['lexical'][:100]}
    dense = {h['doc_id'] for h in row['dense'][:100]}
    x.extend((len(row['text']), sum(c.isdigit() for c in row['text']),
              sum(c.isascii() for c in row['text']), len(lex & dense) / 100.0))
    return np.asarray(x, dtype=float)


def fixed_prefix(row, budget):
    copy = dict(row)
    copy[f'ce{budget}'] = {'scores': row['ce40']['scores'][:budget]}
    return rerank(copy, budget)


def main(args):
    train, _ = read_scores(args.train_scores)
    dev, _ = read_scores(args.dev_scores)
    train_labels = qrels(args.train_qrels)
    dev_labels = qrels(args.dev_qrels)
    dfs = json.loads(Path(args.statistics).read_text())['dfs']
    model_bundle = json.loads(Path(args.models).read_text())
    frozen_gate = FrozenLinear(model_bundle['gate']['model'])
    train_x = np.stack([rich_features(r, dfs) for r in train])
    dev_x = np.stack([rich_features(r, dfs) for r in dev])
    train_gain = np.asarray([
        metrics(rerank(r, 40), train_labels[r['query_id']])['ndcg'] -
        metrics(rerank(r, 28), train_labels[r['query_id']])['ndcg']
        for r in train
    ])
    train_groups = query_groups(train, train_labels)
    eligible = [i for i, r in enumerate(dev) if not r['explicit_constraints']]
    fixed = [metrics(fixed_prefix(r, 33), dev_labels[r['query_id']]) for r in dev]
    fixed_ndcg = float(np.mean([v['ndcg'] for v in fixed]))
    output = {
        'status': 'exploratory',
        'queries': {'train': len(train), 'dev': len(dev)},
        'feature_drift': {},
        'frozen_gate': {},
        'selectors': [],
    }
    gate_train = frozen_gate.predict([gate_features(r, dfs) for r in train])
    gate_dev = frozen_gate.predict([gate_features(r, dfs) for r in dev])
    for j, name in enumerate(('n_terms', 'vocab_coverage', 'relative_gap', 'candidate_hits')):
        output['feature_drift'][name] = {
            'train_unique': int(len(np.unique([gate_features(r, dfs)[j] for r in train]))),
            'dev_unique': int(len(np.unique([gate_features(r, dfs)[j] for r in dev]))),
            'train_quantiles': np.quantile(train_x[:, j], [0, .5, .9, 1]).tolist(),
            'dev_quantiles': np.quantile(dev_x[:, j], [0, .5, .9, 1]).tolist(),
        }
    output['frozen_gate'] = {
        'threshold': model_bundle['gate']['threshold'],
        'train_heavy_rate': float(np.mean(gate_train > model_bundle['gate']['threshold'])),
        'dev_heavy_rate': float(np.mean([(s > model_bundle['gate']['threshold']) and not r['explicit_constraints'] for s, r in zip(gate_dev, dev)])),
        'dev_gain_correlation': float(np.corrcoef(gate_dev, [
            metrics(rerank(r, 40), dev_labels[r['query_id']])['ndcg'] -
            metrics(r['pool'], dev_labels[r['query_id']])['ndcg'] for r in dev
        ])[0, 1]),
    }
    for rate in (0.4, 0.433, 0.5, 0.6):
        n = int(round(rate * len(eligible)))
        model = ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5,
                                    max_features=.7, random_state=20261113,
                                    n_jobs=-1)
        oof = np.zeros(len(train))
        for ti, vi in GroupKFold(4).split(train_x, train_gain, groups=train_groups):
            model.fit(train_x[ti], train_gain[ti])
            oof[vi] = model.predict(train_x[vi])
        model.fit(train_x, train_gain)
        pred = model.predict(dev_x)
        selected = set(sorted(eligible, key=lambda i: (-pred[i], dev[i]['query_id']))[:n])
        values = [metrics(rerank(r, 40) if i in selected else rerank(r, 28), dev_labels[r['query_id']]) for i, r in enumerate(dev)]
        output['selectors'].append({
            'kind': 'extra_trees_ce28_ce40', 'heavy_rate': float(len(selected) / len(dev)),
            'mean_ce_pairs': float(28 + 12 * len(selected) / len(dev)),
            'cv_gain_correlation': float(np.corrcoef(oof, train_gain)[0, 1]),
            'ndcg': float(np.mean([v['ndcg'] for v in values])),
            'recall': float(np.mean([v['recall'] for v in values])),
            'fixed_ce33_ndcg': fixed_ndcg,
            'ndcg_delta_vs_fixed_ce33': float(np.mean([v['ndcg'] for v in values]) - fixed_ndcg),
        })
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--train-scores', type=Path, required=True)
    p.add_argument('--dev-scores', type=Path, required=True)
    p.add_argument('--train-qrels', type=Path, required=True)
    p.add_argument('--dev-qrels', type=Path, required=True)
    p.add_argument('--statistics', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
