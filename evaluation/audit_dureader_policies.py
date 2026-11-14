"""Independent readback of policy exports, group exclusion and predictions.

No fitting or threshold selection. Recomputes deployed gate/ranker decisions
and metrics from frozen inputs, then checks the published aggregates.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
from contracts.index_format import sha256_of
from evaluation.dureader_policy_features import FrozenLinear,gate_features,query_groups,rerank
from evaluation.prepare_bm25_router import jsonl_rows,metrics
from evaluation.train_dureader_policies import read_scores,qrels


def audit(train_scores,dev_scores,train_qrels,dev_qrels,statistics,policy_dir,output):
    output=Path(output);root=Path(policy_dir)
    if output.exists():raise FileExistsError(output)
    train,tp=read_scores(train_scores);dev,dp=read_scores(dev_scores)
    bundle=json.loads((root/'models.json').read_text());summary=json.loads((root/'summary.json').read_text())
    for name,key in [('models.json','models_sha256'),('dev-predictions.jsonl','dev_prediction_sha256_before_labels'),
                     ('train-oof.jsonl','train_oof_sha256')]:
        if sha256_of(root/name)!=summary['protocol'][key]:raise ValueError('policy artifact hash mismatch: '+name)
    if (bundle['train_qrels_sha256']!=sha256_of(train_qrels)or
        summary['protocol']['dev_qrels_sha256']!=sha256_of(dev_qrels)or
        bundle['statistics_sha256']!=sha256_of(statistics)or
        bundle['train_scores_sha256']!=tp['scores_sha256']or bundle['dev_scores_sha256']!=dp['scores_sha256']):
        raise ValueError('training/evaluation dependency mismatch')
    labels=qrels(train_qrels);groups=query_groups(train,labels)
    group_by_id={r['query_id']:g for r,g in zip(train,groups)}
    train_ids=set(group_by_id)
    folds=json.loads((root/'nested-selection.json').read_text())
    covered=set()
    for fold in folds:
        tr,va=set(fold['train_qids']),set(fold['validation_qids'])
        if tr&va or tr|va!=train_ids or covered&va:raise ValueError('invalid outer fold partition')
        if {group_by_id[q]for q in tr}&{group_by_id[q]for q in va}:raise ValueError('outer group leakage')
        covered.update(va)
    if covered!=train_ids:raise ValueError('missing OOF coverage')
    oof=list(jsonl_rows(root/'train-oof.jsonl'))
    if len(oof)!=len(train)or {r['query_id']for r in oof}!=train_ids:raise ValueError('OOF rows incomplete')
    for row in oof:
        if row['group']!=group_by_id[row['query_id']] or row['query_id']not in folds[row['fold']]['validation_qids']:
            raise ValueError('OOF record fold/group mismatch')
    predictions=list(jsonl_rows(root/'dev-predictions.jsonl'))
    by_id={p['query_id']:p for p in predictions}
    if len(predictions)!=len(dev)or set(by_id)!={r['query_id']for r in dev}:raise ValueError('dev coverage mismatch')
    dfs=json.loads(Path(statistics).read_text())['dfs'];gate=FrozenLinear(bundle['gate']['model'])
    rankers={b:FrozenLinear(bundle['rankers'][str(b)]['model'])if bundle['rankers'][str(b)]['model']else None for b in (28,40)}
    for row in dev:
        prediction=by_id[row['query_id']]
        score=float(gate.predict([gate_features(row,dfs)])[0])
        heavy=score>bundle['gate']['threshold']and not row['explicit_constraints']
        if not math.isclose(score,prediction['gain_prediction'],abs_tol=1e-12)or heavy!=prediction['heavy']:
            raise ValueError('exported gate prediction mismatch')
        expected={'ce0':row['pool'],'ce28':rerank(row,28),'ce40':rerank(row,40),
                  'learned-router':rerank(row,40)if heavy else row['pool'],
                  **{f'ce{b}-ranker':rerank(row,b,rankers[b])for b in (28,40)}}
        for name,ids in expected.items():
            if prediction['actions'][name]!=ids:raise ValueError('exported ranking mismatch: '+name)
    dev_labels=qrels(dev_qrels)
    for name in summary:
        if name in ('protocol','train_nested_oof'):continue
        values=[]
        for row in dev:
            if name=='dense'and row['explicit_constraints']:continue
            values.append(metrics(by_id[row['query_id']]['actions'][name],dev_labels[row['query_id']]))
        if len(values)!=summary[name]['queries']:raise ValueError('metric denominator mismatch')
        for metric,reported in summary[name]['metrics'].items():
            actual=float(np.mean([v[metric]for v in values]))
            if not math.isclose(actual,reported,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('metric mismatch')
    result={'status':'verified','train_queries':len(train),'dev_queries':len(dev),'train_groups':len(set(groups)),
            'outer_group_exclusion_verified':True,'oof_coverage_verified':True,
            'all_exported_gate_and_ranker_predictions_verified':True,'all_aggregate_quality_metrics_recomputed':True,
            'summary_sha256':sha256_of(root/'summary.json'),'models_sha256':sha256_of(root/'models.json'),
            'scope':'artifact integrity, outer group exclusion, exported inference and quality; no latency or inner-fold refitting audit'}
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('train-scores','dev-scores','train-qrels','dev-qrels','statistics','policy-dir','output'):
        p.add_argument('--'+n,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
