"""Dev-only nested group selection for query/vector-first lexical gain models."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .router_model import FEATURES,CascadeRouter
from .router_metrics import metrics,bootstrap

THRESHOLDS=(-.1,-.05,-.02,-.01,-.005,0,.005,.01,.02,.03,.05,.075,.1,.15,.2,.3,1.)


def descriptions():
    return ([dict(kind='constant',value=0.)]+[dict(kind='ridge',alpha=a) for a in (1,10,100)]
        +[dict(kind='tree',depth=d,leaf=l) for d in (2,3,4) for l in (20,50)]
        +[dict(kind='forest',depth=d,leaf=l,trees=50) for d in (3,5) for l in (20,40)])


def fit(spec,x,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    if spec['kind']=='constant':return None
    if spec['kind']=='ridge':model=make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha']))
    elif spec['kind']=='tree':model=DecisionTreeRegressor(max_depth=spec['depth'],min_samples_leaf=spec['leaf'],random_state=20261026)
    else:model=RandomForestRegressor(n_estimators=spec['trees'],max_depth=spec['depth'],min_samples_leaf=spec['leaf'],random_state=20261026,n_jobs=1)
    return model.fit(x,y)


def predictions(model,x):
    import numpy as np
    return np.zeros(len(x)) if model is None else model.predict(x)


def loss_record(gain,use):
    import numpy as np
    loss=gain*(~use);mean=float(np.mean(loss));upper=mean+1.645*float(np.std(loss,ddof=1))/math.sqrt(len(loss))
    return dict(mean_loss=mean,upper95=upper,saving=float(np.mean(~use)))


def select(x,gain,groups,forced,folds=4):
    import numpy as np
    from sklearn.model_selection import GroupKFold
    trials=[];best=None
    for spec in descriptions():
        predicted=np.zeros(len(gain))
        if spec['kind']!='constant':
            for train,valid in GroupKFold(folds).split(x,gain,groups):predicted[valid]=predictions(fit(spec,x[train],gain[train]),x[valid])
        thresholds=(1.,) if spec['kind']=='constant' else THRESHOLDS
        for threshold in thresholds:
            use=(predicted>threshold)|forced;record=loss_record(gain,use)
            record.update(model=spec,threshold=threshold);trials.append(record)
            if record['saving']>=.3 and record['upper95']<=.01:
                key=(-record['mean_loss'],record['saving'],-record['upper95'])
                if best is None or key>best[0]:best=(key,spec,threshold,record)
    return best,trials


def export(model,spec,threshold,info,provenance):
    result=dict(feature_version='vector-first-v1',feature_names=list(FEATURES),threshold=threshold,
        retrieval={k:info['retrieval']['config'][k] for k in ('candidate_k','rrf_c','lexical_weight','dense_weight')},
        corpus_sha256=info['retrieval']['vector_manifest']['corpus_sha256'],embedding=info['retrieval']['vector_manifest']['embedding'],
        vector_search_params=info['retrieval']['vector_search_params'],provenance=provenance)
    if model is None:
        result.update(kind='linear',mean=[0.]*len(FEATURES),scale=[1.]*len(FEATURES),coef=[0.]*len(FEATURES),intercept=0.)
    elif spec['kind']=='ridge':
        scaler=model.steps[0][1];reg=model.steps[1][1]
        result.update(kind='linear',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=reg.coef_.tolist(),intercept=float(reg.intercept_))
    else:
        trees=model.estimators_ if spec['kind']=='forest' else [model];exported=[]
        for estimator in trees:
            tree=estimator.tree_;exported.append(dict(feature=tree.feature.tolist(),left=tree.children_left.tolist(),right=tree.children_right.tolist(),threshold=tree.threshold.tolist(),value=tree.value[:,0,0].tolist()))
        result.update(kind='forest',trees=exported)
    return result


def main():
    import numpy as np
    from sklearn.model_selection import GroupKFold
    p=argparse.ArgumentParser();p.add_argument('--capture',required=True,type=Path);p.add_argument('--qrels',required=True,type=Path)
    p.add_argument('--audit',required=True,type=Path);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(line) for line in args.capture.read_text().splitlines()];qids={r['query_id'] for r in rows};qrels={}
    for line in args.qrels.read_text().splitlines():
        q,_,d,g=line.split()
        if q in qids:qrels.setdefault(q,{})[d]=int(g)
    audit=json.loads(args.audit.read_text());groups=np.array([audit['groups'][r['query_id']] for r in rows])
    x=np.array([[r['features'][f] for f in FEATURES] for r in rows]);forced=np.array([r['features']['hits']==0 for r in rows])
    dense=np.array([metrics([h['doc_id'] for h in r['dense']],qrels[r['query_id']])['ndcg@10'] for r in rows])
    fixed=np.array([metrics([h['doc_id'] for h in r['hybrid']],qrels[r['query_id']])['ndcg@10'] for r in rows]);gain=fixed-dense
    if not np.isfinite(x).all():raise ValueError('nonfinite training features')
    uses=np.ones(len(rows),dtype=bool);fold_records=[]
    for fold,(train,valid) in enumerate(GroupKFold(5).split(x,gain,groups)):
        choice,trials=select(x[train],gain[train],groups[train],forced[train])
        if choice is None:spec=None;threshold=None
        else:
            _,spec,threshold,_=choice;model=fit(spec,x[train],gain[train]);uses[valid]=(predictions(model,x[valid])>threshold)|forced[valid]
        record=dict(fold=fold,selection=spec,threshold=threshold,valid_query_ids=[rows[i]['query_id'] for i in valid],
                    outer=loss_record(gain[valid],uses[valid]),inner_trials=trials);fold_records.append(record)
        print('outer',fold,spec,record['outer'],flush=True)
    outer=loss_record(gain,uses);outer.update(adaptive_ndcg10=float(np.mean(np.where(uses,fixed,dense))),fixed_ndcg10=float(np.mean(fixed)),
        dense_ndcg10=float(np.mean(dense)),adaptive_minus_fixed=bootstrap((-gain*(~uses)).tolist()),
        adaptive_minus_dense=bootstrap((gain*uses).tolist()),budget_met=outer['upper95']<=.01 and outer['saving']>=.3)
    choice,trials=select(x,gain,groups,forced,folds=5)
    if choice is None:spec=dict(kind='constant',value=0);threshold=-1.;model=None;selection=None
    else:_,spec,threshold,selection=choice;model=fit(spec,x,gain)
    info=json.loads(args.capture.with_suffix('.provenance.json').read_text())['index_info']
    provenance=dict(capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),queries=len(rows),selected_model=spec,
        threshold=threshold,selection=selection,reads_test_or_external_labels=False,
        caveat='nested group CV conditional on fusion selected in a previous round; empirical intervals, no population guarantee')
    artifact=export(model,spec,threshold,info,provenance);pending=args.out/'model.pending.json';pending.write_text(json.dumps(artifact,indent=2)+'\n')
    deployed=CascadeRouter(pending);expected=predictions(model,x)
    actual=[deployed.predict(r['features']) for r in rows]
    if not np.allclose(expected,actual,atol=1e-10,rtol=1e-10):raise ValueError('export inference mismatch')
    (args.out/'nested-cv.json').write_text(json.dumps(dict(outer=outer,folds=fold_records,final_trials=trials,provenance=provenance),indent=2)+'\n')
    (args.out/'outer-predictions.jsonl').write_text(''.join(json.dumps(dict(query_id=r['query_id'],use_lexical=bool(uses[i]),gain=float(gain[i]),dense_ndcg10=float(dense[i]),fixed_ndcg10=float(fixed[i])))+'\n' for i,r in enumerate(rows)))
    pending.replace(args.out/'frozen-model.json');print(json.dumps(dict(outer=outer,selected=provenance),indent=2),flush=True)


if __name__=='__main__':main()
