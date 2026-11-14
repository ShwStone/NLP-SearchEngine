"""Independently refit every recorded inner selection on train labels only."""
import argparse,json,math
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from contracts.index_format import sha256_of
from evaluation.train_dureader_policies import read_scores,qrels
from evaluation.dureader_policy_features import gate_features,rank_features,rerank,query_groups
from evaluation.prepare_bm25_router import metrics


def audit(train_scores,train_qrels,statistics,policy_dir,output):
    output=Path(output);policy_dir=Path(policy_dir)
    if output.exists():raise FileExistsError(output)
    rows,protocol=read_scores(train_scores);labels=qrels(train_qrels)
    bundle=json.loads((policy_dir/'models.json').read_text())
    if (bundle['train_scores_sha256']!=protocol['scores_sha256']or
        bundle['train_qrels_sha256']!=sha256_of(train_qrels)or
        bundle['statistics_sha256']!=sha256_of(statistics)):
        raise ValueError('training dependency identity mismatch')
    dfs=json.loads(Path(statistics).read_text())['dfs'];groups=query_groups(rows,labels)
    byid={r['query_id']:i for i,r in enumerate(rows)}
    x=np.stack([gate_features(r,dfs)for r in rows])
    gain=np.array([metrics(rerank(r,40),labels[r['query_id']])['ndcg']-
                   metrics(r['pool'],labels[r['query_id']])['ndcg']for r in rows])
    def model(alpha):return make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    checks=0;fold_records=[]
    specs=json.loads((policy_dir/'nested-selection.json').read_text())
    specs.append({'fold':'final-train','train_qids':[r['query_id']for r in rows],
                  'ranker_trials':{b:bundle['rankers'][b]['trials']for b in ['28','40']},
                  'gate_trials':bundle['gate']['trials'],'gate_threshold':bundle['gate']['threshold']})
    for spec in specs:
        indices=np.array([byid[q]for q in spec['train_qids']]);rr=[rows[i]for i in indices]
        gg=groups[indices];xx=x[indices];yy=gain[indices]
        folds=list(GroupKFold(min(3,len(set(gg)))).split(xx,groups=gg))
        for ti,vi in folds:
            if set(gg[ti])&set(gg[vi]):raise ValueError('inner group leakage')
        selected_scores={}
        for trial in spec['gate_trials']:
            pred=np.zeros(len(rr))
            for ti,vi in folds:pred[vi]=model(trial['alpha']).fit(xx[ti],yy[ti]).predict(xx[vi])
            mse=float(np.mean((pred-yy)**2))
            if not math.isclose(mse,trial['inner_oof_mse'],abs_tol=1e-12):raise ValueError('inner gate MSE mismatch')
            selected_scores[trial['alpha']]=pred;checks+=1
        chosen=min(spec['gate_trials'],key=lambda t:(t['inner_oof_mse'],t['alpha']))['alpha']
        pred=selected_scores[chosen];limit=int(np.floor(len(rr)*bundle['heavy_budget_train']))
        threshold=max(0.,float(np.sort(pred)[::-1][limit]))
        if not math.isclose(threshold,spec['gate_threshold'],abs_tol=1e-12):raise ValueError('inner gate threshold mismatch')
        for budget,trials in spec['ranker_trials'].items():
            budget=int(budget)
            for trial in trials:
                if trial['alpha']is None:
                    values=[metrics(rerank(r,budget),labels[r['query_id']])['ndcg']for r in rr]
                else:
                    values=[]
                    for ti,vi in folds:
                        features=np.vstack([rank_features(rr[i],budget)for i in ti])
                        targets=np.array([labels[rr[i]['query_id']].get(d,0.)for i in ti for d in rr[i]['pool'][:budget]])
                        ranker=model(trial['alpha']).fit(features,targets)
                        values.extend(metrics(rerank(rr[i],budget,ranker),labels[rr[i]['query_id']])['ndcg']for i in vi)
                if not math.isclose(float(np.mean(values)),trial['inner_oof_ndcg'],abs_tol=1e-12):
                    raise ValueError('inner ranker selection metric mismatch')
                checks+=1
        fold_records.append({'fold':spec['fold'],'queries':len(rr),'inner_folds':len(folds),
                             'group_disjoint':True,'gate_threshold_verified':True})
    result={'status':'verified','train_queries':len(rows),'dev_labels_read':False,
            'recorded_trials_independently_recomputed':checks,'folds':fold_records,
            'selection_sha256':sha256_of(policy_dir/'nested-selection.json'),
            'models_sha256':sha256_of(policy_dir/'models.json'),
            'scope':'independent train-only inner ranker/gate refits and threshold reconstruction'}
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('train-scores','train-qrels','statistics','policy-dir','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
