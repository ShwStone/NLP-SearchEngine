"""Dev-only nested study of corpus-aware, cost-weighted dense-first routing.

No test/external relevance is used. Work defaults to a sum(df) proxy; an optional
997-query trace substitutes measured lexical-stage ms, never full HTTP savings.
All groups are inherited from the frozen audit.
No deployable model is emitted: a new independent validation set is required.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from contracts import analyzer
from offline.disk_index import DiskIndex
from .router_model import FEATURES
from .evaluate import _metric_for_query
from .router_metrics import bootstrap

THRESHOLDS=(-.1,-.02,-.01,-.005,0.,.002,.005,.01,.02,.05,.1,1.)
LEXICAL_FEATURES=('oov_fraction','mean_idf','min_idf','max_idf','std_idf','sum_df_fraction','max_df_fraction')


def stats(gain,use,work,budget):
    import numpy as np
    loss=gain*(~use);mean=float(np.mean(loss));upper=mean+1.645*float(np.std(loss,ddof=1))/math.sqrt(len(loss))
    saving=float(np.mean(~use));cost=float(np.sum(work*(~use))/np.sum(work))
    return dict(mean_loss=mean,loss_upper95=upper,bm25_call_saving=saving,posting_work_saving=cost,
                meets_budget=upper<=budget and saving>=.3)


def select(x,gain,groups,forced,work,budget,objective,inner_folds=4,feature_sets=('vector','lexical-only','combined')):
    import numpy as np
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    trials=[];best=None
    for features in feature_sets:
        subset=(slice(None,len(FEATURES)) if features=='vector' else slice(len(FEATURES),None) if features=='lexical-only' else slice(None))
        xx=x[:,subset]
        for alpha in (1.,10.,100.):
            scores=np.empty(len(gain))
            for train,valid in GroupKFold(inner_folds).split(xx,gain,groups):
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(xx[train],gain[train])
                scores[valid]=model.predict(xx[valid])
            for threshold in THRESHOLDS:
                use=(scores>threshold)|forced;record=stats(gain,use,work,budget)
                record.update(features=features,alpha=alpha,threshold=threshold)
                trials.append(record)
    record=stats(gain,forced,work,budget);record.update(features='always-dense',alpha=None,threshold=None);trials.append(record)
    for record in trials:
        if record['meets_budget']:
            key=((-record['mean_loss'],record['bm25_call_saving']) if objective=='quality' else
                 (record['posting_work_saving'],-record['mean_loss']))
            if best is None or key>best[0]:best=(key,record)
    return best[1] if best else None,trials


def fit_predict(spec,xtrain,ytrain,xvalid):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if spec is None:return np.full(len(xvalid),float('inf'))
    if spec['features']=='always-dense':return np.full(len(xvalid),float('-inf'))
    feature_slice=(slice(None,len(FEATURES)) if spec['features']=='vector' else
                   slice(len(FEATURES),None) if spec['features']=='lexical-only' else slice(None))
    model=make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha'])).fit(xtrain[:,feature_slice],ytrain)
    return model.predict(xvalid[:,feature_slice])


def main():
    import numpy as np
    from sklearn.model_selection import GroupKFold
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('artifacts/round4/router'))
    parser.add_argument('--cost-file',type=Path,help='optional 997-query packed-64 HTTP trace; lexical stage milliseconds only')
    for name in ('capture', 'audit', 'qrels', 'index'):
        parser.add_argument('--'+name, type=Path, required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    capture=args.capture;audit_path=args.audit
    qrels_path=args.qrels;audit=json.loads(audit_path.read_text())
    qrels={}
    for line in qrels_path.read_text().splitlines():
        qid,_,docid,grade=line.split()
        if qid in audit['groups']:qrels.setdefault(qid,{})[docid]=int(grade)
    features=[];work=[];gains=[];dense=[];fixed=[];qids=[]
    reader=DiskIndex(args.index)
    with capture.open() as source:
        for line in source:
            row=json.loads(line);qid=row['query_id'];terms=sorted({t.term for t in analyzer.analyze(row['text'])})
            dfs=[reader.df(term) for term in terms];idfs=[math.log1p((reader.n_docs-df+.5)/(df+.5)) for df in dfs if df]
            mean=statistics.mean(idfs) if idfs else 0.
            extra=[sum(df==0 for df in dfs)/max(len(dfs),1),mean,min(idfs,default=0.),max(idfs,default=0.),
                   statistics.pstdev(idfs) if idfs else 0.,sum(dfs)/reader.n_docs,max(dfs,default=0)/reader.n_docs]
            features.append([row['features'][name] for name in FEATURES]+extra)
            work.append(sum(reader.df(t.term) for t in analyzer.analyze(row['text'])))  # duplicates match scorer traversal
            scores=[_metric_for_query(qrels[qid],[r['doc_id'] for r in row[mode]],10,100)[0] for mode in ('dense','hybrid')]
            dense.append(scores[0]);fixed.append(scores[1]);gains.append(scores[1]-scores[0]);qids.append(qid)
    x=np.array(features);gain=np.array(gains);work=np.array(work);groups=np.array([audit['groups'][qid] for qid in qids])
    if args.cost_file:
        with args.cost_file.open() as source:
            costs={r['query_id']:r['stages']['lexical'] for line in source for r in [json.loads(line)] if r['variant']=='packed-64'}
        if set(costs)!=set(qids) or any(not math.isfinite(v) or v<=0 for v in costs.values()):
            raise ValueError('need positive measured lexical costs for all 997 dev queries')
        work=np.array([costs[qid] for qid in qids])
    forced=x[:,FEATURES.index('hits')]==0;dense=np.array(dense);fixed=np.array(fixed)
    all_results={};all_predictions=[]
    # Nested CV evaluates selection over feature sets/alpha/threshold and two budgets/objectives.
    for feature_search,feature_sets in (('vector-control',('vector',)),('corpus-aware',('vector','lexical-only','combined'))):
      for budget in (.01,.002):
        for objective in ('quality','work'):
            uses=np.ones(len(qids),dtype=bool);folds=[]
            for fold,(train,valid) in enumerate(GroupKFold(5).split(x,gain,groups)):
                chosen,_=select(x[train],gain[train],groups[train],forced[train],work[train],budget,objective,feature_sets=feature_sets)
                scores=fit_predict(chosen,x[train],gain[train],x[valid])
                threshold=chosen['threshold'] if chosen and chosen['threshold'] is not None else 0.
                uses[valid]=(scores>threshold)|forced[valid]
                folds.append(dict(fold=fold,selected=chosen,heldout=stats(gain[valid],uses[valid],work[valid],budget)))
            final,trials=select(x,gain,groups,forced,work,budget,objective,inner_folds=5,feature_sets=feature_sets)
            record=stats(gain,uses,work,budget)
            record.update(ndcg10=float(np.mean(np.where(uses,fixed,dense))),dense_ndcg10=float(dense.mean()),fixed_ndcg10=float(fixed.mean()),
                delta_fixed=bootstrap((-gain*(~uses)).tolist()),delta_dense=bootstrap((gain*uses).tolist()),folds=folds,final_oof_choice=final,
                final_oof_frontier=[r for r in trials if r['meets_budget']],always_dense=stats(gain,forced,work,budget))
            name=f'{feature_search}-{objective}-loss{budget}'
            all_results[name]=record
            all_predictions.extend(dict(protocol=name,query_id=qid,use_bm25=bool(uses[i]),work=float(work[i]),gain=float(gain[i])) for i,qid in enumerate(qids))
            print(name,json.dumps({k:v for k,v in record.items() if k not in ('folds','final_oof_frontier','final_oof_choice')}),flush=True)
    result=dict(queries=len(qids),feature_names=list(FEATURES)+list(LEXICAL_FEATURES),results=all_results,
        provenance=dict(capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            reads_only_dev_relevance=True,new_independent_external_validation=False,model_exported=False,
            cost_file_sha256=hashlib.sha256(args.cost_file.read_bytes()).hexdigest() if args.cost_file else None,
            work_unit='one measured lexical-stage ms observation per dev query' if args.cost_file else 'sum(df(term)) posting visits',
            proxy='partial lexical-stage savings; excludes route inference, encode/vector, fusion, rendering, HTTP and cache-state changes' if args.cost_file else 'sum(df(term)) estimates scorer posting visits, not measured milliseconds or complete HTTP cost',
            limitations=['Du dev and old fusion already observed; adaptive research CV, not an unbiased whole-project estimate',
                         'Empirical intervals and learned selection have no population guarantee',
                         'Lexical statistics extraction and rendering/encoding/vector costs excluded from work proxy']))
    if args.cost_file:
        def measured_names(value):
            if isinstance(value,list):return [measured_names(v) for v in value]
            if isinstance(value,dict):return {('lexical_stage_ms_saving' if k=='posting_work_saving' else k):measured_names(v) for k,v in value.items()}
            return value
        result=measured_names(result)
    (args.out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.out/'outer-predictions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in all_predictions))


if __name__=='__main__':main()
