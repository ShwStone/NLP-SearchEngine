"""Train grouped nested CE rankers and a pre-CE gain Router on train only.

Dev predictions are frozen before opening dev qrels. Router heavy rate is
selected from train OOF scores with a predeclared 60% ceiling. Random controls
match the realized dev budget by label-independent query hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from contracts.retrieval import RetrievalConfig
from online.router import QueryFeatures, RuleRouter

from contracts.index_format import sha256_of
from evaluation.dureader_policy_features import (gate_features, rank_features, rerank,
                                                 query_groups, GATE_FEATURES, RANK_FEATURES, FrozenLinear)
from evaluation.prepare_bm25_router import jsonl_rows, metrics
from evaluation.router_metrics import percentile
from evaluation.train_bm25_router import group_bootstrap

ALPHAS=(.1,1.,10.,100.)
HEAVY_BUDGET=.6
SEED=20261113


def qrels(path):
    result={}
    for line in Path(path).read_text().splitlines():
        q,_,d,g=line.split();result.setdefault(q,{})[d]=float(g)
    return result


def read_scores(directory):
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    if protocol['scores_sha256']!=sha256_of(directory/'scores.jsonl') or protocol['qrels_read']:
        raise ValueError('score hash mismatch or labels accessed during scoring')
    rows=list(jsonl_rows(directory/'scores.jsonl'))
    if len(rows)!=protocol['queries'] or len({r['query_id'] for r in rows})!=len(rows):
        raise ValueError('invalid score query coverage')
    return rows,protocol


def linear(alpha):
    return make_pipeline(StandardScaler(),Ridge(alpha=alpha))


def fit_ranker(rows, labels, budget, alpha):
    if alpha is None:return None
    x=np.vstack([rank_features(r,budget) for r in rows])
    y=np.array([labels[r['query_id']].get(d,0.) for r in rows for d in r['pool'][:budget]])
    return linear(alpha).fit(x,y)


def splits(groups, maximum):
    n=min(maximum,len(set(groups)))
    if n<2:raise ValueError('not enough independent training groups')
    return list(GroupKFold(n).split(np.zeros(len(groups)),groups=groups))


def select_ranker(rows, labels, groups, budget):
    trials=[{'alpha':None,'inner_oof_ndcg':float(np.mean([
        metrics(rerank(r,budget),labels[r['query_id']])['ndcg'] for r in rows]))}]
    for alpha in ALPHAS:
        values=[]
        for ti,vi in splits(groups,3):
            model=fit_ranker([rows[i] for i in ti],labels,budget,alpha)
            values.extend(metrics(rerank(rows[i],budget,model),labels[rows[i]['query_id']])['ndcg'] for i in vi)
        trials.append({'alpha':alpha,'inner_oof_ndcg':float(np.mean(values))})
    chosen=max(trials,key=lambda t:(t['inner_oof_ndcg'],t['alpha'] is None,
                                   -(t['alpha'] or 0.)))['alpha']
    return chosen,trials


def fit_gate(x,y,groups):
    trials=[];predictions={}
    for alpha in ALPHAS:
        pred=np.zeros(len(y))
        for ti,vi in splits(groups,3):
            pred[vi]=linear(alpha).fit(x[ti],y[ti]).predict(x[vi])
        trials.append({'alpha':alpha,'inner_oof_mse':float(np.mean((pred-y)**2))})
        predictions[alpha]=pred
    alpha=min(trials,key=lambda t:(t['inner_oof_mse'],t['alpha']))['alpha']
    pred=predictions[alpha]
    # Threshold/budget uses train predictions only; no dev calibration.
    limit=int(np.floor(len(y)*HEAVY_BUDGET))
    threshold=float(np.sort(pred)[::-1][limit]) if limit<len(pred) else float('-inf')
    threshold=max(0.,threshold)
    return linear(alpha).fit(x,y),threshold,trials,pred


def export_linear(model):
    if model is None:return None
    scaler,reg=model.steps[0][1],model.steps[1][1]
    return {'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'coef':reg.coef_.tolist(),'intercept':float(reg.intercept_)}


def train(train_scores,dev_scores,train_qrels,dev_qrels,statistics,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    tr,tp=read_scores(train_scores);dv,dp=read_scores(dev_scores)
    if set(r['query_id'] for r in tr)&set(r['query_id'] for r in dv):
        raise ValueError('train/dev IDs overlap')
    if tp['corpus_sha256']!=dp['corpus_sha256'] or tp['ce']!=dp['ce']:
        raise ValueError('train/dev corpus or CE mismatch')
    dfs=json.loads(Path(statistics).read_text())['dfs']
    labels=qrels(train_qrels)
    if not set(r['query_id'] for r in tr)<=set(labels):raise ValueError('train labels missing')
    groups=query_groups(tr,labels)
    x=np.stack([gate_features(r,dfs) for r in tr]);xd=np.stack([gate_features(r,dfs) for r in dv])
    gain=np.array([metrics(rerank(r,40),labels[r['query_id']])['ndcg']-
                   metrics(r['pool'],labels[r['query_id']])['ndcg'] for r in tr])
    oof=[];selections=[]
    for fold,(ti,vi) in enumerate(splits(groups,4)):
        if set(groups[ti])&set(groups[vi]):raise ValueError('group leakage')
        train_rows=[tr[i] for i in ti]
        gate,threshold,gate_trials,_=fit_gate(x[ti],gain[ti],groups[ti])
        gate_scores=gate.predict(x[vi])
        rankers={};rank_trials={}
        for budget in (28,40):
            alpha,trials=select_ranker(train_rows,labels,groups[ti],budget)
            rankers[budget]=fit_ranker(train_rows,labels,budget,alpha);rank_trials[budget]=trials
        selections.append({'fold':fold,'train_qids':[tr[i]['query_id'] for i in ti],
                           'validation_qids':[tr[i]['query_id'] for i in vi],
                           'gate_threshold':threshold,'gate_trials':gate_trials,
                           'ranker_trials':rank_trials})
        for j,i in enumerate(vi):
            row=tr[i];q=labels[row['query_id']]
            use=bool(gate_scores[j]>threshold and not row['explicit_constraints'])
            oof.append({'query_id':row['query_id'],'group':groups[i],'fold':fold,
                        'gate_gain_prediction':float(gate_scores[j]),'gate_heavy':use,
                        'metrics':{'ce0':metrics(row['pool'],q),'ce40':metrics(rerank(row,40),q),
                                   'learned-router':metrics(rerank(row,40) if use else row['pool'],q),
                                   **{f'ce{b}-ranker':metrics(rerank(row,b,rankers[b]),q) for b in (28,40)}}})
    if len(oof)!=len(tr) or len({r['query_id'] for r in oof})!=len(tr):
        raise ValueError('incomplete train OOF coverage')
    models={};specs={}
    for budget in (28,40):
        alpha,trials=select_ranker(tr,labels,groups,budget)
        models[budget]=fit_ranker(tr,labels,budget,alpha)
        specs[budget]={'alpha':alpha,'trials':trials,'model':export_linear(models[budget])}
    gate,threshold,trials,train_pred=fit_gate(x,gain,groups)
    if not np.allclose(gate.predict(xd),FrozenLinear(export_linear(gate)).predict(xd),atol=1e-12):
        raise ValueError('gate exported inference mismatch')
    for budget in (28,40):
        if models[budget] is None:continue
        for row in dv:
            if rerank(row,budget,models[budget])!=rerank(row,budget,FrozenLinear(export_linear(models[budget]))):
                raise ValueError('ranker exported inference mismatch')
    dev_scores_pred=gate.predict(xd)
    heavy=np.array([bool(s>threshold and not r['explicit_constraints']) for s,r in zip(dev_scores_pred,dv)])
    # Exact same realized cheap/heavy budget, selected without dev labels.
    controls={}
    for seed in (SEED,SEED+1):
        eligible=[i for i,r in enumerate(dv) if not r['explicit_constraints']]
        order=sorted(eligible,key=lambda i:hashlib.sha256(f'{seed}:{dv[i]["query_id"]}'.encode()).digest())
        controls[f'random-{seed}']=set(order[:int(heavy.sum())])
    predictions=[]
    rule_router=RuleRouter(RetrievalConfig())
    fixed_pairs=int(np.floor(40*float(heavy.mean())))
    for i,row in enumerate(dv):
        f=xd[i]
        rules=rule_router.decide(QueryFeatures(row['explicit_constraints'],int(f[0]),
                                              float(f[1]),int(f[3]),float(f[2]) if f[3]>1 else None))
        actions={'ce0':row['pool'],'bm25':[h['doc_id'] for h in row['lexical']],
                 'dense':[h['doc_id'] for h in row['dense']],
                 'old-rules-router':row['pool'] if rules.use_vector else [h['doc_id'] for h in row['lexical']],
                 'ce28':rerank(row,28),'ce40':rerank(row,40),
                 'learned-router':rerank(row,40) if heavy[i] else row['pool'],
                 **{f'ce{b}-ranker':rerank(row,b,models[b]) for b in (28,40)},
                 **{name:rerank(row,40) if i in selected else row['pool'] for name,selected in controls.items()}}
        # Scores from CE-40 can be reused for quality only. Timing requires
        # separately executing this fixed budget in the HTTP harness.
        fixed_row=dict(row)
        fixed_row[f'ce{fixed_pairs}']={'scores':row['ce40']['scores'][:fixed_pairs]}
        actions['fixed-pair-budget']=rerank(fixed_row,fixed_pairs) if fixed_pairs else row['pool']
        predictions.append({'query_id':row['query_id'],'heavy':bool(heavy[i]),
                            'gain_prediction':float(dev_scores_pred[i]),
                            'rules_vector':rules.use_vector,'actions':actions})
    output.mkdir(parents=True)
    def write_jsonl(name,rows):
        with (output/name).open('x',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_jsonl('dev-predictions.jsonl',predictions)
    write_jsonl('train-oof.jsonl',oof)
    (output/'nested-selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    bundle={'rankers':specs,'gate':{'model':export_linear(gate),'threshold':threshold,'trials':trials},
            'gate_features':list(GATE_FEATURES),'rank_features':list(RANK_FEATURES),
            'train_scores_sha256':tp['scores_sha256'],'dev_scores_sha256':dp['scores_sha256'],
            'train_qrels_sha256':sha256_of(train_qrels),'statistics_sha256':sha256_of(statistics),
            'train_only':True,'heavy_budget_train':HEAVY_BUDGET,'research_only':True}
    (output/'models.json').write_text(json.dumps(bundle,indent=2)+'\n')
    prediction_hash=sha256_of(output/'dev-predictions.jsonl')
    # Only now open development labels. No model/threshold changes below.
    dev_labels=qrels(dev_qrels);summary={}
    dev_groups=query_groups(dv,dev_labels)
    evaluated={}
    for name in predictions[0]['actions']:
        values=[];included=[]
        for i,(row,pred) in enumerate(zip(dv,predictions)):
            if name=='dense' and row['explicit_constraints']:continue
            values.append(metrics(pred['actions'][name],dev_labels[row['query_id']]));included.append(i)
        evaluated[name]=(values,included)
        summary[name]={'queries':len(values),'metrics':{k:float(np.mean([v[k] for v in values])) for k in values[0]}}
        if name!='dense':
            if name.startswith('ce28'): budgets=[28]*len(dv)
            elif name.startswith('ce40'):budgets=[40]*len(dv)
            elif name=='learned-router':budgets=[40 if b else 0 for b in heavy]
            elif name in controls:budgets=[40 if i in controls[name] else 0 for i in range(len(dv))]
            else:budgets=[0]*len(dv)
            times=[r[f'ce{b}']['component_ms'] if b else 0 for r,b in zip(dv,budgets)]
            pairs=[r[f'ce{b}']['pairs'] if b else 0 for r,b in zip(dv,budgets)]
            summary[name].update(ce_pairs_mean=float(np.mean(pairs)),heavy_rate=float(np.mean(np.array(budgets)>0)),
                                 ce_component_ms={'p50':percentile(times,.5),'p95':percentile(times,.95)},
                                 e2e_http_ms=None)
            if name=='fixed-pair-budget':
                summary[name].update(ce_pairs_mean=float(np.mean([min(fixed_pairs,len(r['pool'])) for r in dv])),
                                     heavy_rate=float(fixed_pairs>0),ce_component_ms=None)
    base=evaluated['ce0'][0]
    for name,(values,indices) in evaluated.items():
        summary[name]['paired_vs_ce0']={k:group_bootstrap(
            np.array([v[k]-base[i][k] for v,i in zip(values,indices)]),dev_groups[indices])
            for k in ('ndcg','recall')}
    for name in controls:
        summary['learned-router'].setdefault('paired_vs_same_budget_random',{})[name]={
            k:group_bootstrap(np.array([a[k]-b[k] for a,b in
                                        zip(evaluated['learned-router'][0],evaluated[name][0])]),dev_groups)
            for k in ('ndcg','recall')}
    summary['old-rules-router']['vector_attempt_rate']=float(np.mean([p['rules_vector'] for p in predictions]))
    for budget in (28,40):
        learned=evaluated[f'ce{budget}-ranker'][0];raw=evaluated[f'ce{budget}'][0]
        summary[f'ce{budget}-ranker']['paired_vs_raw']={
            k:group_bootstrap(np.array([a[k]-b[k] for a,b in zip(learned,raw)]),dev_groups)
            for k in ('ndcg','recall')}
    summary['train_nested_oof']={
        name:{k:float(np.mean([r['metrics'][name][k] for r in oof]))
              for k in ('ndcg','recall','mrr','map')} for name in oof[0]['metrics']}
    summary['protocol']={'train_queries':len(tr),'dev_queries':len(dv),'train_groups':len(set(groups)),
                         'nested_outer_folds':min(4,len(set(groups))),'inner_folds':3,
                         'dev_ci_groups':len(set(dev_groups)),
                         'dev_prediction_sha256_before_labels':prediction_hash,
                         'dev_qrels_sha256':sha256_of(dev_qrels),'models_sha256':sha256_of(output/'models.json'),
                         'train_oof_sha256':sha256_of(output/'train-oof.jsonl'),
                         'same_budget_random':'exact realized Router heavy count; query hash ordering, no labels',
                         'fixed_pair_budget':fixed_pairs,
                         'all_dev_metrics_exploratory':True,
                         'latency_scope':'CE components under current load; HTTP latency pending'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('train-scores','dev-scores','train-qrels','dev-qrels','statistics','output'):
        p.add_argument('--'+name,required=True)
    print(json.dumps(train(**vars(p.parse_args())),indent=2))
