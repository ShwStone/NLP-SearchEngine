"""Refit both cascade actions and gain gate on all development groups.

Requires a passing independently audited HTTP development run. The final
serving manifest excludes training outcomes and labels; training diagnostics
are stored separately and are never quality acceptance evidence.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .prebge_nested import DevelopmentData, NestedTrainer, QUANTILES, safe_partition
from .prepare_bm25_router import jsonl_rows, sha
from .runtime_nested_prebge import gain_features
from .study_prebge_learned_gate_v4 import GATE_SPECS


def run(a):
    source = json.loads((a.study / 'study.json').read_text())
    http_audit = json.loads((a.http / 'independent-audit.json').read_text())
    http = json.loads((a.http / 'summary.json').read_text())
    if (not http_audit['development_accepted'] or
            http_audit['summary_sha256'] != sha(a.http / 'summary.json') or
            sha(a.study / 'study.json') not in http['protocol']['inputs'].values()):
        raise ValueError('final training requires this candidate to pass audited HTTP development')
    p = source['protocol']
    dataset = next(Path(s).parent for s in p['inputs'] if Path(s).name == 'dataset.json')
    capture = next(Path(s) for s in p['inputs'] if Path(s).name == 'captures.jsonl')
    canonical = next(Path(s) for s in p['inputs'] if Path(s).name == 'logits.jsonl')
    support = next(Path(s) for s in p['inputs'] if Path(s).name == 'vectors.npy')
    d = DevelopmentData(dataset, capture, canonical, support)
    captures = {r['query_id']: r for r in jsonl_rows(capture)}
    gx = np.array([gain_features(d.m['bm25_features'][i], d.m['bm25_feature_names'],
                                [(r['doc_id'], r['score']) for r in captures[q]['lexical']],
                                [(r['doc_id'], r['score']) for r in captures[q]['dense']])
                   for i, q in enumerate(d.ids)])
    if not np.array_equal(gx, np.load(a.study / 'gate-features.npy', allow_pickle=False)):
        raise ValueError('final gain feature reconstruction differs')
    a.out.mkdir(parents=True, exist_ok=False)
    trainer = NestedTrainer(d, a.out)
    all_ids = np.arange(d.n)
    quality, parts = trainer.gate_training_outcomes(all_ids, 'all-development')
    action, action_trials = trainer.select_action(all_ids, 'all-development/final-action')
    target = ((quality - d.cheap_quality) / d.baseline.mean(0)).mean(1)
    max_heavy = p.get('max_heavy_rate', .95)
    trials, roles = [], []
    for spec_index, spec in enumerate(GATE_SPECS):
        predictions = np.empty(d.n)
        for fold, (train, valid) in enumerate(GroupKFold(3).split(all_ids, groups=d.groups)):
            safe_partition(train, valid, all_ids, d.groups)
            gate = HistGradientBoostingRegressor(**spec, max_iter=80, learning_rate=.05,
                                                early_stopping=False, random_state=20261030)
            gate.fit(gx[train], target[train])
            path = a.out / 'models' / f'gate-spec-{spec_index}-inner-{fold}.joblib'
            joblib.dump(gate, path)
            predictions[valid] = joblib.load(path).predict(gx[valid])
            roles.append(dict(spec_index=spec_index, spec=spec, train=train.tolist(), valid=valid.tolist(),
                              model_file=str(path.relative_to(a.out)), model_sha256=sha(path)))
        for q in QUANTILES:
            threshold = float(np.quantile(predictions, q))
            mask = predictions > threshold
            if not mask.any() or not (~mask).any() or mask.mean() > max_heavy + 1e-12:
                continue
            actual = np.where(mask[:, None], quality, d.cheap_quality)
            lift = actual.mean(0) / d.baseline.mean(0) - 1
            trials.append(dict(spec_index=spec_index, spec=spec, threshold=threshold,
                               heavy_rate=float(mask.mean()), lifts=lift.tolist(),
                               key=[float(lift.min()), float(lift.mean()), -float(mask.mean())]))
    selected = max(trials, key=lambda t: tuple(t['key']))
    gate = HistGradientBoostingRegressor(**selected['spec'], max_iter=80, learning_rate=.05,
                                        early_stopping=False, random_state=20261030).fit(gx, target)
    ranker = HistGradientBoostingClassifier(**action['spec'], max_iter=100, learning_rate=.08,
                                           early_stopping=False, random_state=20261030)
    ranker.fit(d.x.reshape(-1, d.x.shape[-1]), d.y.reshape(-1))
    model_files = {}
    for name, model, xx in [('gate', gate, gx), ('action', ranker, d.x.reshape(-1, d.x.shape[-1]))]:
        path = a.out / 'models' / f'final-{name}.joblib'
        joblib.dump(model, path)
        prediction = model.predict(xx) if name == 'gate' else model.predict_proba(xx)
        loaded = joblib.load(path)
        reloaded = loaded.predict(xx) if name == 'gate' else loaded.predict_proba(xx)
        if not np.array_equal(prediction, reloaded):
            raise ValueError('saved final model predictions changed')
        model_files[name + '_file'] = str(path.relative_to(a.out))
        model_files[name + '_sha256'] = sha(path)
    protocol = dict(kind='audited-nested-preBGE-serving-v1',
                    source_study_sha256=sha(a.study / 'study.json'),
                    qualified_http_sha256=sha(a.http / 'summary.json'),
                    online_scorer=d.encoder, support_sha256=sha(support), features=d.feature_names,
                    bm25_names=d.m['bm25_feature_names'], gate_features=p['gate_features'],
                    label_assets_in_serving=False, query_prediction_cache=False,
                    all_development_groups_refitted=True, training_queries=d.n,
                    independent_confirmation_read=False, goal_complete=False)
    final = dict(fold='final', heldout_ids=[], threshold=selected['threshold'],
                 weight=action['weight'], **model_files)
    (a.out / 'study.json').write_text(json.dumps(dict(protocol=protocol, folds=[final]), indent=2) + '\n')
    training = dict(inputs=d.inputs, source_study=str(a.study.resolve()), qualified_http=str(a.http.resolve()),
                    training_ids=d.ids, groups=d.groups.tolist(), gate_outcome_parts=parts,
                    gate_oof_quality=quality.tolist(), action_selection=action, action_trials=action_trials,
                    gate_target=target.tolist(), gate_models=roles, gate_trials=trials,
                    gate_selection=selected, action_training_roles=trainer.records,
                    max_heavy_rate=max_heavy, final_models_train=list(range(d.n)),
                    final_training_quality_not_acceptance=True, goal_complete=False)
    (a.out / 'training.json').write_text(json.dumps(training, indent=2) + '\n')
    np.save(a.out / 'gate-features.npy', gx, allow_pickle=False)
    (a.out / 'source.py').write_bytes(Path(__file__).read_bytes())
    (a.out / 'source-sha256.json').write_text(json.dumps({str(Path(__file__).resolve()): sha(__file__)}) + '\n')
    print(json.dumps(dict(training_queries=d.n, action_selection=action, gate_selection=selected,
                          resubstitution_is_not_acceptance=True, confirmation_read=False), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('study', 'http', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
