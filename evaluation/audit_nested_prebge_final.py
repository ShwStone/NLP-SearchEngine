"""Replay final selection and independently retrain both full-development models."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .prebge_nested import DevelopmentData, SPECS, WEIGHTS, QUANTILES, safe_partition
from .prepare_bm25_router import metrics, sha
from .study_prebge_learned_gate_v4 import GATE_SPECS


def audit(root):
    s = json.loads((root / 'study.json').read_text())
    t = json.loads((root / 'training.json').read_text())
    source_hashes = json.loads((root / 'source-sha256.json').read_text())
    if len(source_hashes) != 1 or sha(root / 'source.py') != next(iter(source_hashes.values())):
        raise ValueError('executed final training source snapshot differs')
    source_root, http_root = Path(t['source_study']), Path(t['qualified_http'])
    source = json.loads((source_root / 'study.json').read_text())
    ha = json.loads((http_root / 'independent-audit.json').read_text())
    if (s['protocol']['source_study_sha256'] != sha(source_root / 'study.json') or
            s['protocol']['qualified_http_sha256'] != sha(http_root / 'summary.json') or
            ha['summary_sha256'] != sha(http_root / 'summary.json') or not ha['development_accepted']):
        raise ValueError('final preparation lacks matching qualified development evidence')
    if t['max_heavy_rate'] != source['protocol'].get('max_heavy_rate', .95):
        raise ValueError('final gate budget differs from qualified development')
    for path, digest in t['inputs'].items():
        if sha(path) != digest:
            raise ValueError('final input drift: ' + path)
    inputs = t['inputs']
    dataset = next(Path(p).parent for p in inputs if Path(p).name == 'dataset.json')
    capture = next(Path(p) for p in inputs if Path(p).name == 'captures.jsonl')
    canonical = next(Path(p) for p in inputs if Path(p).name == 'logits.jsonl')
    support = next(Path(p) for p in inputs if Path(p).name == 'vectors.npy')
    d = DevelopmentData(dataset, capture, canonical, support)
    if t['training_ids'] != d.ids or t['groups'] != d.groups.tolist() or t['final_models_train'] != list(range(d.n)):
        raise ValueError('final training does not cover all development groups')
    gx = np.load(root / 'gate-features.npy', allow_pickle=False)
    if not np.array_equal(gx, np.load(source_root / 'gate-features.npy', allow_pickle=False)):
        raise ValueError('final gate inputs differ from qualified development')
    roles, preds = {}, {}
    for r in t['action_training_roles']:
        safe_partition(r['train'], r['valid'], r['allowed'], d.groups)
        path = root / r['model_file']
        if sha(path) != r['model_sha256']:
            raise ValueError('final selection action model changed')
        key = r['role'], json.dumps(r['spec'], sort_keys=True)
        if key in roles:
            raise ValueError('duplicate final selection role')
        roles[key] = r
        xx = d.x[r['valid']]
        prob = np.clip(joblib.load(path).predict_proba(xx.reshape(-1, xx.shape[-1]))[:, 1], 1e-7, 1 - 1e-7)
        z = np.log(prob / (1 - prob)).reshape(xx.shape[:2])
        preds[r['model_file']] = (z - z.mean(1, keepdims=True)) / np.maximum(z.std(1, keepdims=True), 1e-8)

    def quality(ids, pred, weight):
        result = []
        score = weight * d.teacher[ids] + (1 - weight) * pred
        for j, i in enumerate(ids):
            order = sorted(range(40), key=lambda k: (-score[j, k], d.identities[d.heavy_docs[i][k]]))
            ranking = [d.heavy_docs[i][k] for k in order]
            v = metrics(ranking, d.m['relevance'][d.ids[i]])
            result.append([v['ndcg'], v['recall']])
        return np.array(result)

    used_roles = set()

    def replay_selection(train, role, selected, trials):
        train = np.array(train)
        positions = {int(i): j for j, i in enumerate(train)}
        replayed = []
        for spec in SPECS:
            out = np.empty((len(train), 40))
            seen = set()
            for split in range(3):
                key = role + f'/action-inner-{split}', json.dumps(spec, sort_keys=True)
                r = roles[key]; used_roles.add(key)
                if set(r['allowed']) != set(train.tolist()) or seen & set(r['valid']):
                    raise ValueError('final action selection escaped training scope')
                seen.update(r['valid'])
                out[[positions[i] for i in r['valid']]] = preds[r['model_file']]
            if seen != set(train.tolist()):
                raise ValueError('incomplete final action OOF')
            for weight in WEIGHTS:
                q = quality(train, out, weight)
                lift = q.mean(0) / d.baseline[train].mean(0) - 1
                replayed.append(dict(spec=spec, weight=weight, lifts=lift.tolist(),
                                     key=[float(lift.min()), float(lift.mean()), weight]))
        if replayed != trials or max(replayed, key=lambda r: tuple(r['key'])) != selected:
            raise ValueError('final action selection replay differs')

    all_ids = np.arange(d.n)
    oof = np.empty((d.n, 2)); seen = set()
    for split, part in enumerate(t['gate_outcome_parts']):
        safe_partition(part['train'], part['valid'], all_ids, d.groups)
        if seen & set(part['valid']):
            raise ValueError('final gate outcome folds overlap')
        seen.update(part['valid'])
        replay_selection(part['train'], f'all-development/gate-{split}', part['selection'], part['trials'])
        key = f'all-development/gate-outcome-{split}', json.dumps(part['selection']['spec'], sort_keys=True)
        r = roles[key]; used_roles.add(key)
        if r['train'] != part['train'] or r['valid'] != part['valid']:
            raise ValueError('final gate outcome model scope differs')
        oof[r['valid']] = quality(np.array(r['valid']), preds[r['model_file']], part['selection']['weight'])
    if seen != set(range(d.n)) or not np.array_equal(oof, t['gate_oof_quality']):
        raise ValueError('final gate outcomes not fully replayed')
    replay_selection(all_ids, 'all-development/final-action', t['action_selection'], t['action_trials'])
    if used_roles != set(roles):
        raise ValueError('unreplayed final action model roles')
    target = ((oof - d.cheap_quality) / d.baseline.mean(0)).mean(1)
    if not np.array_equal(target, t['gate_target']):
        raise ValueError('final gain targets differ')
    trials = []
    for spec_index, spec in enumerate(GATE_SPECS):
        pred = np.empty(d.n); seen = set()
        records = [r for r in t['gate_models'] if r['spec_index'] == spec_index]
        if len(records) != 3:
            raise ValueError('incomplete final gate inner models')
        for r in records:
            safe_partition(r['train'], r['valid'], all_ids, d.groups)
            path = root / r['model_file']
            if sha(path) != r['model_sha256'] or seen & set(r['valid']):
                raise ValueError('final gate model hash/partition differs')
            seen.update(r['valid']); pred[r['valid']] = joblib.load(path).predict(gx[r['valid']])
        if seen != set(range(d.n)):
            raise ValueError('incomplete gate OOF scope')
        for q in QUANTILES:
            threshold = float(np.quantile(pred, q)); mask = pred > threshold
            if not mask.any() or not (~mask).any() or mask.mean() > t['max_heavy_rate'] + 1e-12:
                continue
            actual = np.where(mask[:, None], oof, d.cheap_quality)
            lift = actual.mean(0) / d.baseline.mean(0) - 1
            trials.append(dict(spec_index=spec_index, spec=spec, threshold=threshold,
                               heavy_rate=float(mask.mean()), lifts=lift.tolist(),
                               key=[float(lift.min()), float(lift.mean()), -float(mask.mean())]))
    if trials != t['gate_trials'] or max(trials, key=lambda r: tuple(r['key'])) != t['gate_selection']:
        raise ValueError('final gate selection differs')
    # Independently refit both final models on all development rows. These
    # predictions verify implementation/training scope, never quality.
    row = s['folds'][0]
    if row['fold'] != 'final' or row['heldout_ids']:
        raise ValueError('invalid final serving fold')
    ranker = HistGradientBoostingClassifier(**t['action_selection']['spec'], max_iter=100,
                                           learning_rate=.08, early_stopping=False, random_state=20261030)
    ranker.fit(d.x.reshape(-1, d.x.shape[-1]), d.y.reshape(-1))
    gate = HistGradientBoostingRegressor(**t['gate_selection']['spec'], max_iter=80,
                                        learning_rate=.05, early_stopping=False, random_state=20261030).fit(gx, target)
    for kind, rebuilt, xx in [('action', ranker, d.x.reshape(-1, d.x.shape[-1])), ('gate', gate, gx)]:
        path = root / row[kind + '_file']
        if sha(path) != row[kind + '_sha256']:
            raise ValueError('final serving model changed')
        saved = joblib.load(path)
        actual = saved.predict_proba(xx) if kind == 'action' else saved.predict(xx)
        expected = rebuilt.predict_proba(xx) if kind == 'action' else rebuilt.predict(xx)
        if not np.array_equal(actual, expected):
            raise ValueError('independent all-development model refit differs: ' + kind)
    if row['weight'] != t['action_selection']['weight'] or row['threshold'] != t['gate_selection']['threshold']:
        raise ValueError('final serving configuration differs from selection')
    report = dict(kind='independent-full-development-nested-preBGE-final-audit-v1',
                  study_sha256=sha(root / 'study.json'), training_sha256=sha(root / 'training.json'),
                  queries=d.n, groups=len(set(d.groups)), action_models_replayed=len(roles),
                  gate_models_replayed=len(t['gate_models']), all_selections_and_training_scopes_replayed=True,
                  both_final_models_independently_refitted_on_all_development=True,
                  exact_refit_prediction_parity=True, quality_not_evaluated_on_training=True,
                  confirmation_read=False, goal_complete=False)
    (root / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--final', type=Path, required=True)
    audit(p.parse_args().final)
