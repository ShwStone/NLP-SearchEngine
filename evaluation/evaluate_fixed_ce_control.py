"""Evaluate separately executed fixed CE depth against frozen Router outputs.

Never adjusts models, budget or threshold. Previous reused-score evidence is
retained, and numerical differences are counted explicitly.
"""
import argparse,json
from pathlib import Path
import numpy as np
from contracts.index_format import sha256_of
from evaluation.train_dureader_policies import read_scores,qrels
from evaluation.dureader_policy_features import rerank,query_groups
from evaluation.prepare_bm25_router import jsonl_rows,metrics
from evaluation.train_bm25_router import group_bootstrap
from evaluation.router_metrics import percentile


def evaluate(scores,policy_dir,dev_qrels,output):
    output=Path(output);policy_dir=Path(policy_dir)
    if output.exists():raise FileExistsError(output)
    rows,protocol=read_scores(scores)
    previous=json.loads((policy_dir/'summary.json').read_text())
    budget=previous['protocol']['fixed_pair_budget']
    if protocol['budgets']!=[budget]:raise ValueError('fixed budget differs from frozen Router budget')
    predictions={r['query_id']:r for r in jsonl_rows(policy_dir/'dev-predictions.jsonl')}
    if set(predictions)!={r['query_id']for r in rows}:raise ValueError('query universe differs')
    fixed=[];changed=0
    for row in rows:
        ranked=rerank(row,budget)
        changed+=ranked!=predictions[row['query_id']]['actions']['fixed-pair-budget']
        fixed.append({'query_id':row['query_id'],'results':ranked,
                      'ce_pairs':row[f'ce{budget}']['pairs'],'ce_ms':row[f'ce{budget}']['component_ms']})
    output.mkdir(parents=True)
    with(output/'predictions.jsonl').open('x')as handle:
        for row in fixed:handle.write(json.dumps(row)+'\n')
    ph=sha256_of(output/'predictions.jsonl')
    labels=qrels(dev_qrels);groups=query_groups(rows,labels)
    vals=[metrics(r['results'],labels[r['query_id']])for r in fixed]
    router=[metrics(predictions[r['query_id']]['actions']['learned-router'],labels[r['query_id']])for r in rows]
    summary={'queries':len(rows),'fixed_ce_pairs':budget,
             'metrics':{m:float(np.mean([v[m]for v in vals]))for m in vals[0]},
             'router_minus_fixed':{m:group_bootstrap(np.array([a[m]-b[m]for a,b in zip(router,vals)]),groups)
                                   for m in ['ndcg','recall']},
             'ce_component_ms':{k:percentile([r['ce_ms']for r in fixed],p)for k,p in [('p50',.5),('p95',.95)]},
             'changed_rankings_vs_reused_ce40_prefix':changed,
             'protocol':{'separately_scored':True,'model_or_threshold_changed':False,
                         'predictions_sha256_before_qrels':ph,'scores_protocol_sha256':sha256_of(Path(scores)/'protocol.json'),
                         'policy_summary_sha256':sha256_of(policy_dir/'summary.json'),
                         'qrels_sha256':sha256_of(dev_qrels),'e2e_http_latency_pending':True}}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('scores','policy-dir','dev-qrels','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(evaluate(**vars(p.parse_args())),indent=2))
