"""Outer-isolated gain regression using verified v2 action-training outcomes."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .prebge_nested import DevelopmentData,QUANTILES,safe_partition
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .study_bge_score_calibration import summary

GATE_SPECS=[dict(max_leaf_nodes=leaf,min_samples_leaf=child,l2_regularization=reg)
            for leaf,child,reg in ((3,80,10.),(7,80,10.),(7,40,10.),(15,40,10.))]


def run(a):
    parent=json.loads((a.parent/'study.json').read_text());audit=json.loads((a.parent/'independent-audit.json').read_text())
    if audit['study_sha256']!=sha(a.parent/'study.json') or not audit['outer_groups_isolated'] or not audit['all_model_and_gate_selections_replayed']:
        raise ValueError('learned gate requires independently replayed outer-isolated action outcomes')
    p=parent['protocol'];dataset=next(Path(x).parent for x in p['inputs'] if Path(x).name=='dataset.json')
    capture=next(Path(x) for x in p['inputs'] if Path(x).name=='captures.jsonl')
    canonical=next(Path(x) for x in p['inputs'] if Path(x).name=='logits.jsonl')
    support=next(Path(x) for x in p['inputs'] if Path(x).name=='vectors.npy')
    d=DevelopmentData(dataset,capture,canonical,support)
    captures={r['query_id']:r for r in jsonl_rows(capture)}
    names=list(p['gate_features'])+[f'BM25_{n}' for n in d.m['bm25_feature_names']]
    names+=['dense_mean10','dense_std10','dense_score10','dense_score20','dense_score40',
            'lexical_mean10','lexical_std10','intersection_top10_fraction','intersection_top20_fraction']
    x=[]
    for i,q in enumerate(d.ids):
        c=captures[q];ds=[r['score'] for r in c['dense']];ls=[r['score'] for r in c['lexical']]
        overlap=[len({r['doc_id'] for r in c['dense'][:k]}&{r['doc_id'] for r in c['lexical'][:k]})/k for k in (10,20)]
        x.append([*d.gate_x[i],*[d.m['bm25_features'][i][n] for n in d.m['bm25_feature_names']],
                  float(np.mean(ds[:10])),float(np.std(ds[:10])),ds[9],ds[19],ds[39],
                  float(np.mean(ls[:10])) if ls else 0.,float(np.std(ls[:10])) if ls else 0.,*overlap])
    x=np.array(x);a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir()
    protocol=dict(kind='outer-isolated-preBGE-gain-regression-v4',parent_study_sha256=sha(a.parent/'study.json'),
        inputs=dict(p['inputs'],**{str((a.parent/n).resolve()):sha(a.parent/n) for n in ('study.json','independent-audit.json','training-roles.json')}),
        gate_features=names,gate_specs=GATE_SPECS,gate_inner_folds=3,gate_target='mean normalized nDCG/Recall action gain on outer-training-only nested outcomes',
        source_sha256=sha(Path(__file__)),global_OOF_used_for_gate=False,confirmation_read=False,http_evaluated=False,goal_complete=False)
    (a.out/'source.py').write_bytes(Path(__file__).read_bytes());(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    np.save(a.out/'gate-features.npy',x,allow_pickle=False);outer=np.empty_like(d.baseline);folds=[];outputs={};counts={'cheap':0,'heavy':0}
    for fold in parent['folds']:
        f=fold['fold'];train=np.array(fold['train']);valid=np.array(fold['valid']);safe_partition(train,valid,np.arange(d.n),d.groups)
        hq=np.array(fold['gate_oof_quality']);cq=d.cheap_quality[train]
        target=((hq-cq)/d.baseline[train].mean(0)).mean(1);trials=[];models=[]
        for spec_index,spec in enumerate(GATE_SPECS):
            prediction=np.empty(len(train));roles=[]
            for split,(t,v) in enumerate(GroupKFold(3).split(train,groups=d.groups[train])):
                safe_partition(train[t],train[v],train,d.groups)
                model=HistGradientBoostingRegressor(**spec,max_iter=80,learning_rate=.05,early_stopping=False,random_state=20261030)
                model.fit(x[train[t]],target[t]);path=a.out/'models'/f'outer-{f}-spec-{spec_index}-inner-{split}.joblib';joblib.dump(model,path)
                prediction[v]=joblib.load(path).predict(x[train[v]])
                roles.append(dict(train=train[t].tolist(),valid=train[v].tolist(),model_file=str(path.relative_to(a.out)),model_sha256=sha(path)))
            models.append(dict(spec=spec,roles=roles))
            for quantile in QUANTILES:
                threshold=float(np.quantile(prediction,quantile));mask=prediction>threshold
                if not mask.any() or not (~mask).any() or mask.mean()>.95+1e-12:continue
                actual=np.where(mask[:,None],hq,cq);lift=actual.mean(0)/d.baseline[train].mean(0)-1
                trials.append(dict(spec_index=spec_index,spec=spec,threshold=threshold,heavy_rate=float(mask.mean()),
                    lifts=lift.tolist(),key=[float(lift.min()),float(lift.mean()),-float(mask.mean())]))
        selected=max(trials,key=lambda t:tuple(t['key']))
        model=HistGradientBoostingRegressor(**selected['spec'],max_iter=80,learning_rate=.05,early_stopping=False,random_state=20261030)
        model.fit(x[train],target);path=a.out/'models'/f'outer-{f}-gate.joblib';joblib.dump(model,path)
        mask=joblib.load(path).predict(x[valid])>selected['threshold']
        action=fold['action'];heavy=joblib.load(a.parent/fold['outer_model']['model_file'])
        features=d.x[valid];prob=np.clip(heavy.predict_proba(features.reshape(-1,features.shape[-1]))[:,1],1e-7,1-1e-7)
        z=np.log(prob/(1-prob)).reshape(len(valid),40);z=(z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),1e-8)
        hvalid,rankings=d.rank_quality(valid,z,action['weight']);outer[valid]=np.where(mask[:,None],hvalid,d.cheap_quality[valid])
        for j,i in enumerate(valid):
            name='heavy' if mask[j] else 'cheap';counts[name]+=1
            outputs[int(i)]=dict(query_id=d.ids[i],group=str(d.groups[i]),fold=f,action=name,
                ranked=rankings[j] if mask[j] else d.cheap_rankings[i],metrics=dict(zip(('ndcg','recall'),outer[i].tolist())))
        folds.append(dict(fold=f,train=train.tolist(),valid=valid.tolist(),gate_models=models,selection=selected,trials=trials,
            final_gate_file=str(path.relative_to(a.out)),final_gate_sha256=sha(path),target=target.tolist()))
    path=a.out/'outer-predictions.jsonl';path.write_text(''.join(json.dumps(outputs[i])+'\n' for i in range(d.n)))
    quality=summary(outer,d.baseline,d.groups);result=dict(protocol=protocol,folds=folds,outer=quality,action_counts=counts,
        quality_point_gate_passed=all(v['passes_20_percent'] and v['delta_ci']['ci95'][0]>0 for v in quality.values()) and all(counts.values()),
        predictions_sha256=sha(path),http_evaluated=False,confirmation_evaluated=False,goal_complete=False)
    (a.out/'study.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(outer=quality,actions=counts,quality_passed=result['quality_point_gate_passed']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--parent',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
