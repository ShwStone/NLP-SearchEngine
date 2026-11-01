"""Grouped nested development study of BM25-first gains; never reads test qrels."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from online.router_features import FEATURE_NAMES, FEATURE_VERSION

THRESHOLDS = (-1., -.1, -.05, -.02, -.01, 0., .002, .005, .01, .02, .05, .1, .2, .5, 1.)
FAMILIES = ('calibrated-rules', 'linear-full', 'linear-query', 'linear-score', 'tree-full', 'combined')


def specifications():
    result = []
    for coverage in (.7, 1.):
        for gap in (.1, .25, .5, .75):
            for terms in (4, 12):
                result.append(dict(kind='rule', coverage=coverage, gap=gap, terms=terms, family='calibrated-rules'))
    for feature_set in ('full', 'query', 'score'):
        for alpha in (1., 10., 100.):
            result.append(dict(kind='ridge', alpha=alpha, features=feature_set, family='linear-'+feature_set))
    for depth, leaf in ((2, 30), (3, 30), (4, 50)):
        result.append(dict(kind='tree', depth=depth, leaf=leaf, features='full', family='tree-full'))
    return result


def columns(spec):
    subset = spec.get('features', 'full')
    return list(range(len(FEATURE_NAMES))) if subset=='full' else list(range(9)) if subset=='query' else list(range(9, len(FEATURE_NAMES)))


def fit(spec, x, y):
    if spec['kind'] in ('rule', 'constant'):
        return None
    xx = x[:, columns(spec)]
    if spec['kind']=='ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=spec['alpha'])).fit(xx, y)
    return DecisionTreeRegressor(max_depth=spec['depth'], min_samples_leaf=spec['leaf'], random_state=20261027).fit(xx, y)


def predict(spec, model, x):
    if spec['kind']=='constant':
        return np.full(len(x), spec['value'])
    if spec['kind']=='rule':
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        sufficient = ((x[:, index['vocab_coverage']]>=spec['coverage']) &
                      (x[:, index['relative_gap']]>=spec['gap']) &
                      (x[:, index['n_terms']]<=spec['terms']) & (x[:, index['candidate_hits']]>=2))
        return np.where(sufficient, -1., 1.)
    return model.predict(x[:, columns(spec)])


def outcome(gain, use):
    loss = gain*(~use)
    mean = float(np.mean(loss))
    upper = mean+1.645*float(np.std(loss, ddof=1))/math.sqrt(len(loss)) if len(loss)>1 else mean
    return dict(mean_loss=mean, loss_upper95=upper, vector_saving=float(np.mean(~use)))


def crossfit(specs, x, y, groups):
    splits = list(GroupKFold(4).split(x, y, groups))
    results = []
    for spec in specs:
        predicted = np.zeros(len(y))
        for train, valid in splits:
            predicted[valid] = predict(spec, fit(spec, x[train], y[train]), x[valid])
        results.append((spec, predicted))
    return results


def choose(candidates, gain, forced, family, budget):
    # Maximise saving under the predeclared empirical quality constraint.
    # 30% saving is assessed as an outer acceptance target, not forced by fallback.
    best, trials = None, []
    for spec, scores in candidates:
        if family!='combined' and spec['family']!=family:
            continue
        thresholds = (0.,) if spec['kind']=='rule' else THRESHOLDS
        for threshold in thresholds:
            use = (scores>threshold)|forced
            record = dict(spec=spec, threshold=threshold, **outcome(gain, use))
            trials.append(record)
            if record['loss_upper95']<=budget:
                key = (record['vector_saving'], -record['loss_upper95'], -record['mean_loss'])
                if best is None or key>best[0]:
                    best = key, record
    fallback = dict(spec=dict(kind='constant', value=1., family='hybrid'), threshold=0.,
                    **outcome(gain, np.ones(len(gain), dtype=bool)))
    return best[1] if best else fallback, trials


def group_bootstrap(values, groups, samples=2000):
    unique = np.unique(groups)
    blocks = [np.flatnonzero(groups==g) for g in unique]
    totals = np.array([values[b].sum() for b in blocks])
    sizes = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(20261027)
    means = []
    for _ in range(samples):
        selected = rng.integers(0, len(blocks), len(blocks))
        means.append(float(totals[selected].sum()/sizes[selected].sum()))
    return dict(mean=float(values.mean()), ci95=np.quantile(means, [.025,.975]).tolist(), unit='resampled query groups')


def summary(rows, use, groups):
    gain = np.array([r['hybrid']['ndcg']-r['lexical']['ndcg'] for r in rows])
    result = outcome(gain, use)
    result['metrics'] = {key:float(np.mean([r['hybrid' if u else 'lexical'][key] for r,u in zip(rows,use)]))
                         for key in ('ndcg','recall','mrr','map')}
    result['delta_hybrid'] = group_bootstrap(-gain*(~use), groups)
    result['vector_attempt_rate'] = float(use.mean())
    return result


def export(spec, model, threshold, provenance):
    result = dict(schema_version=1, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                  selected_features=[FEATURE_NAMES[i] for i in columns(spec)],
                  threshold=threshold, specification=spec, research_only=True, promoted_to_default=False,
                  provenance=provenance)
    if spec['kind'] in ('rule','constant'):
        result.update(kind=spec['kind'])
    elif spec['kind']=='ridge':
        scaler, reg = model.steps[0][1], model.steps[1][1]
        result.update(kind='linear', mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                      coef=reg.coef_.tolist(), intercept=float(reg.intercept_))
    else:
        tree = model.tree_
        result.update(kind='tree', feature=tree.feature.tolist(), left=tree.children_left.tolist(),
                      right=tree.children_right.tolist(), split=tree.threshold.tolist(), value=tree.value[:,0,0].tolist())
    return result


def train(table, out):
    table, out = Path(table), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in table.read_text().splitlines()]
    if len({r['query_id'] for r in rows})!=len(rows):
        raise ValueError('duplicate query id')
    x = np.array([[r['features'][f] for f in FEATURE_NAMES] for r in rows], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite features')
    y = np.array([r['hybrid']['ndcg']-r['lexical']['ndcg'] for r in rows])
    groups = np.array([r['group'] for r in rows])
    forced = np.array([r['forced_vector'] for r in rows], dtype=bool)
    specs = specifications()
    keys = [(family,budget) for family in FAMILIES for budget in (.01,.002)]
    uses = {key:np.ones(len(rows),dtype=bool) for key in keys}
    folds = {key:[] for key in keys}
    for fold,(training,validation) in enumerate(GroupKFold(5).split(x,y,groups)):
        assert not set(groups[training])&set(groups[validation])
        candidates = crossfit(specs,x[training],y[training],groups[training])
        for key in keys:
            family,budget = key
            choice,trials = choose(candidates,y[training],forced[training],family,budget)
            model = fit(choice['spec'],x[training],y[training])
            uses[key][validation] = (predict(choice['spec'],model,x[validation])>choice['threshold'])|forced[validation]
            folds[key].append(dict(fold=fold, choice=choice, inner_trials=trials,
                                   heldout_ids=[rows[i]['query_id'] for i in validation]))
        print('outer fold',fold,'complete',flush=True)
    results = {}
    predictions = []
    for key in keys:
        family,budget = key
        name = f'{family}-loss{budget}'
        result = summary(rows,uses[key],groups)
        result.update(budget=budget,meets_target=result['loss_upper95']<=budget and result['vector_saving']>=.3,folds=folds[key])
        results[name] = result
        predictions.extend(dict(protocol=name,query_id=r['query_id'],group=r['group'],use_vector=bool(u),
                                gain=float(y[i])) for i,(r,u) in enumerate(zip(rows,uses[key])))
        print(name,{k:v for k,v in result.items() if k not in ('folds',)},flush=True)
    baseline = {}
    for name in ('lexical','hybrid','dense'):
        baseline[name] = {k:float(np.mean([r[name][k] for r in rows])) for k in ('ndcg','recall','mrr','map')}
    baseline['rules-v1'] = summary(rows,np.array([r['rules_use_vector'] for r in rows]),groups)
    final_candidates = crossfit(specs,x,y,groups)
    choice, final_trials = choose(final_candidates,y,forced,'combined',.01)
    model = fit(choice['spec'],x,y)
    provenance = dict(table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),queries=len(rows),groups=len(set(groups)),
                      source=json.loads(table.with_name('provenance.json').read_text()),
                      selection=choice,outer_meets_target=results['combined-loss0.01']['meets_target'],
                      caveat='reused observed dev; empirical quality constraint, not conformal risk control or independent validation')
    artifact = export(choice['spec'],model,choice['threshold'],provenance)
    (out/'model.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    from online.gain_router import GainRouter
    loaded=GainRouter(out/'model.json')
    expected=predict(choice['spec'],model,x)
    actual=np.array([loaded.predict(r['features']) for r in rows])
    if not np.allclose(expected,actual,atol=1e-10,rtol=1e-10):
        raise ValueError('exported router inference differs from fitted model')
    (out/'export-audit.json').write_text(json.dumps(dict(model_sha256=loaded.sha256,
        max_abs_error=float(np.max(abs(expected-actual))),queries=len(rows)))+'\n')
    result = dict(baselines=baseline,results=results,provenance=provenance,final_trials=final_trials)
    (out/'study.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'outer-predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in predictions))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--table',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    train(args.table,args.out)
