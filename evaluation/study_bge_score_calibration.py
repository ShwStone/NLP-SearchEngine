"""Grouped nested calibration using one verified BGE cross-encoder pass.

All selection uses outer-training groups. Persist predictions, scoped labels,
models and source bindings so failures and improvements can both be replayed.
This is an action feasibility study, not an HTTP or Router acceptance result.
"""
import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .prepare_bm25_router import jsonl_rows, metrics, sha
from .ranking_dataset import load_dataset
from .train_bm25_router import group_bootstrap
from online.router_features import FAST_FEATURE_NAMES,BASE_FEATURE_NAMES


SPECS = [dict(max_leaf_nodes=leaves, min_samples_leaf=leaf, l2_regularization=reg)
         for leaves, leaf, reg in ((7, 50, 10.), (15, 40, 10.),
                                   (31, 40, 10.), (31, 20, 1.))]
WEIGHTS = (0., .25, .5, .75, 1.)


def standardized(values):
    values = np.asarray(values, dtype=float)
    return (values-values.mean()) / max(values.std(), 1e-8)


def summary(actual, baseline, groups):
    if actual.shape != baseline.shape or actual.shape != (len(groups), 2):
        raise ValueError('expected separate nDCG and Recall per query')
    result = {}
    for j, name in enumerate(('ndcg', 'recall')):
        delta = actual[:, j]-baseline[:, j]
        if delta.ndim != 1:
            raise ValueError('bootstrap requires one metric at a time')
        result[name] = dict(value=float(actual[:, j].mean()),
            baseline=float(baseline[:, j].mean()),
            relative_lift=float(actual[:, j].mean()/baseline[:, j].mean()-1),
            delta_ci=group_bootstrap(delta, groups),
            passes_20_percent=bool(actual[:, j].mean()>1.2*baseline[:, j].mean()))
    return result


def checked_scores(rows, captures, ids, pool_size=40):
    by_id = {r['query_id']:r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError('BGE query identity/uniqueness mismatch')
    output = {}
    for q in ids:
        logits = by_id[q]['logits']
        expected = captures[q]['scores']['40']['doc_ids'][:pool_size]
        if len(expected) != pool_size or set(logits) != set(expected):
            raise ValueError('BGE scores must cover the exact declared pool')
        if not np.isfinite(list(logits.values())).all():
            raise ValueError('nonfinite BGE scores')
        # Never read GTE logits here: captures provide only candidate identities.
        output[q] = logits
    return output


def run(args):
    started = time.monotonic()
    rank_calibration=getattr(args,'rank_calibration',False)
    no_bm25_stats=getattr(args,'no_bm25_stats',False)
    fast_bm25_stats=getattr(args,'fast_bm25_stats',False)
    if no_bm25_stats and fast_bm25_stats:raise ValueError('choose one BM25 feature mode')
    if rank_calibration:
        from .bge_rank_calibrator import RANK_SPECS,RankCalibrator
    specs=RANK_SPECS if rank_calibration else SPECS
    parent = load_dataset(args.dataset)
    meta = parent['metadata']; ids = meta['query_ids']; groups = parent['groups']
    source = json.loads((args.dataset/'protocol.json').read_text())
    teacher_report = json.loads(args.logits.with_name('summary.json').read_text())
    if teacher_report['logits_sha256'] != sha(args.logits):
        raise ValueError('BGE logits checksum mismatch')
    encoder = teacher_report['protocol']['encoder']
    if encoder['model_id'] != 'BAAI/bge-reranker-v2-m3':
        raise ValueError('expected frozen BGE-M3 scorer')
    if (encoder['max_length'], encoder['batch_size']) not in ((256,20),(224,20),(192,20)):
        raise ValueError('unexpected BGE input recipe')
    if teacher_report['protocol']['inputs'].get(str(args.capture)) != sha(args.capture):
        raise ValueError('BGE scores/candidate capture binding differs')
    if source.get('confirmation_labels_read') is not False:
        raise ValueError('development input must exclude confirmation labels')
    captures = {r['query_id']:r for r in jsonl_rows(args.capture)}
    if set(captures) != set(ids):
        raise ValueError('capture/dataset query coverage differs')
    if args.pool_size not in (20,32,40):raise ValueError('unsupported BGE pool size')
    scores = checked_scores(list(jsonl_rows(args.logits)), captures, ids, args.pool_size)
    support_path=getattr(args,'support_vectors',None);support_inputs=[];support_names=[]
    if support_path is not None:
        from .bge_candidate_support import bge_support,FEATURE_NAMES as support_names
        support_protocol=json.loads(support_path.with_name('protocol.json').read_text())
        if sha(support_path)!=support_protocol['vector_sha256']:raise ValueError('BGE support vectors changed')
        documents=next(Path(p) for p in support_protocol['inputs'] if Path(p).name=='docs.jsonl')
        if sha(documents)!=support_protocol['inputs'][str(documents)] or sha(documents) not in source['inputs'].values():
            raise ValueError('BGE support corpus identity mismatch')
        support_ids={r['doc_id']:r['docid'] for r in jsonl_rows(documents)}
        if any(support_ids[d]!=v for d,v in meta['identities'].items()):raise ValueError('BGE support document order differs')
        support_vectors=np.load(support_path,mmap_mode='r',allow_pickle=False)
        if support_vectors.shape!=(len(support_ids),512):raise ValueError('BGE support vector shape differs')
        support_inputs=[support_path,support_path.with_name('protocol.json'),documents]
    # For reduced pools, checked_scores already selected the first declared
    # members of the frozen BGE pool. Do not silently substitute dense order.
    args.out.mkdir(parents=True, exist_ok=False)
    names = meta['feature_names']
    # Explicit whitelist: support anchors in the parent depend on GTE rankings.
    from .study_candidate_ranking import FEATURES, TEXT_FEATURES
    raw_names = list(FEATURES)+list(TEXT_FEATURES)
    raw_columns = [names.index(name) for name in raw_names]
    blocks=[]; labels=[]; offsets=[0]; pool_docs=[]; teacher_z=[]
    for i, q in enumerate(ids):
        docs = sorted(scores[q], key=meta['identities'].__getitem__)
        parent_slot = {d:j for j,d in enumerate(meta['docs'][i])}
        xx = np.asarray(parent['x'][parent['offsets'][i]:parent['offsets'][i+1]])
        raw = xx[[parent_slot[d] for d in docs]][:, raw_columns]
        bge = np.array([scores[q][d] for d in docs], dtype=float)
        z = standardized(bge)
        order = sorted(range(len(docs)), key=lambda j:(-bge[j], meta['identities'][docs[j]]))
        ranks=np.empty(len(docs)); ranks[order]=np.arange(1,len(docs)+1)
        query_names=list(BASE_FEATURE_NAMES) if fast_bm25_stats else meta['bm25_feature_names']
        query_values=np.array([meta['bm25_features'][i][n] for n in query_names]) if not no_bm25_stats else np.empty(0)
        extras=np.column_stack((bge,z,1./ranks,bge.max()-bge))
        parts=[raw,extras]
        if support_path is not None:
            parts.append(bge_support([support_ids[d] for d in docs],bge,
                [support_ids[r['doc_id']] for r in captures[q]['lexical']],
                [support_ids[r['doc_id']] for r in captures[q]['dense']],support_vectors))
        if not no_bm25_stats:parts.append(np.tile(query_values,(len(docs),1)))
        blocks.append(np.column_stack(parts))
        labels.extend(meta['relevance'][q].get(d,0)>0 for d in docs)
        teacher_z.append(z); pool_docs.append(docs); offsets.append(offsets[-1]+len(docs))
    x=np.concatenate(blocks).astype('float32'); y=np.array(labels,dtype=int)
    offsets=np.array(offsets); sizes=np.diff(offsets)
    def row_indices(selected):
        return np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in selected])
    def fit(spec, selected):
        indices=row_indices(selected)
        weights=np.concatenate([np.full(sizes[i],1./sizes[i]) for i in selected])
        weights*=len(weights)/weights.sum()
        if rank_calibration:
            return RankCalibrator(spec).fit(x[indices],y[indices],sizes[selected],weights)
        model=HistGradientBoostingClassifier(**spec,max_iter=100,learning_rate=.08,
            early_stopping=False,random_state=20261030)
        return model.fit(x[indices],y[indices],sample_weight=weights)
    def predict(model, selected):
        result={}
        for i in selected:
            probability=np.clip(model.predict_proba(x[offsets[i]:offsets[i+1]])[:,1],1e-7,1-1e-7)
            result[int(i)]=standardized(np.log(probability/(1-probability)))
        return result
    def quality(selected, predictions, weight):
        result=[]; rankings={}
        for i in selected:
            blended=weight*teacher_z[i]+(1-weight)*predictions[int(i)]
            docs=pool_docs[i]
            order=sorted(range(len(docs)),key=lambda j:(-blended[j],meta['identities'][docs[j]]))
            ranking=[docs[j] for j in order]
            actual=metrics(ranking,meta['relevance'][ids[i]])
            result.append([actual['ndcg'],actual['recall']]);rankings[int(i)]=ranking
        return np.array(result),rankings
    baseline=np.array([[r[m] for m in ('ndcg','recall')] for r in meta['baselines']['hybrid']])
    protocol=dict(kind='nested-single-BGE-score-calibration-v1',queries=len(ids),
        groups=len(set(groups)),outer_folds=5,inner_folds=3,specs=specs,teacher_weights=WEIGHTS,
        calibration_loss='query-group LambdaRank/XE-NDCG' if rank_calibration else 'binary log loss',
        inputs={str(p):sha(p) for p in [args.dataset/'dataset.json',args.capture,args.logits,args.logits.with_name('summary.json')]+support_inputs},
        code_sha256={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('ranking_dataset.py'),
            Path(__file__).with_name('train_bm25_router.py'),Path(__file__).with_name('study_candidate_ranking.py'))},
        online_scorer=encoder,features=raw_names+['bge_logit','bge_z','bge_rank_reciprocal','bge_gap']+list(support_names)+([] if no_bm25_stats else query_names),
        bm25_query_statistics=not no_bm25_stats,bm25_statistics_kind='fast' if fast_bm25_stats else ('none' if no_bm25_stats else 'full'),
        bge_support_features=list(support_names),bge_support_anchors='current BGE top5, native dense top1 and lexical top1' if support_names else None,
        candidate_selection=f'frozen native dense top{args.pool_size}',
        scored_subset_proxy=False,
        subset_proxy_limit=None,gte_logits_in_features=False,
        gte_support_features_in_inputs=False,query_weight_mean=1.,
        confirmation_labels_read=False,http_evaluated=False,router_acceptance=False,goal_complete=False,
        route_calibration=args.route_calibration,max_cheap_rate=args.max_cheap_rate,
        actions=dict(cheap='one BGE40 forward plus raw ranking',heavy='same BGE40 forward plus calibrated ranking'),
        gate_scope='post-BGE confidence or BM25; no second cross-encoder')
    if rank_calibration:
        protocol['code_sha256'][str(Path(__file__).with_name('bge_rank_calibrator.py'))]=sha(Path(__file__).with_name('bge_rank_calibrator.py'))
    if support_path is not None:
        for name in ('bge_candidate_support.py','prepare_candidate_support.py'):
            p=Path(__file__).with_name(name);protocol['code_sha256'][str(p)]=sha(p)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'source.py').write_bytes(Path(__file__).read_bytes())
    np.save(args.out/'features.npy',x,allow_pickle=False)
    np.save(args.out/'labels.npy',y,allow_pickle=False)
    (args.out/'metadata.json').write_text(json.dumps(dict(query_ids=ids,groups=groups.tolist(),
        docs=pool_docs,offsets=offsets.tolist(),relevance=meta['relevance'],identities=meta['identities'],
        baseline=baseline.tolist(),teacher_z=[z.tolist() for z in teacher_z]))+'\n')
    # Optional quality/cost routing after the single BGE scorer. A cheap action
    # returns raw BGE scores; heavy action applies the learned calibration.
    # Rules use only BM25 or BGE confidence and are fitted inside outer training.
    routing_features=[]
    for i,q in enumerate(ids):
        z=np.sort(teacher_z[i])[::-1]
        routing_features.append([z[0]-z[1],z[9]-z[10],
            meta['bm25_features'][i]['query_chars'],
            meta['bm25_features'][i]['relative_gap']])
    routing_features=np.array(routing_features,dtype=float)
    route_names=['BGE_top_gap_z','BGE_cutoff_gap_z'] if no_bm25_stats else ['BGE_top_gap_z','BGE_cutoff_gap_z','query_chars','BM25_relative_gap']
    zero={i:np.zeros(len(teacher_z[i])) for i in range(len(ids))}
    raw_quality,raw_rankings=quality(np.arange(len(ids)),zero,1.)
    def policies(selected):
        if not args.route_calibration:
            return [(dict(kind='constant',heavy=True),np.ones(len(selected),dtype=bool))]
        result=[]
        for col,name in enumerate(route_names):
            values=routing_features[selected,col]
            for quantile in (.01,.02,.05,.1,.25,.5,.75,.9,.95,.98,.99):
                threshold=float(np.quantile(values,quantile))
                for direction in (-1,1):
                    mask=direction*values>direction*threshold
                    if mask.any() and not mask.all() and (1-mask.mean())<=args.max_cheap_rate+1e-12:
                        result.append((dict(kind='threshold',feature=col,feature_name=name,
                            direction=direction,threshold=threshold),mask))
        return result
    folds=[]; outer=np.empty_like(baseline); outputs={};counts={'cheap':0,'heavy':0}
    for fold,(training,valid) in enumerate(GroupKFold(5).split(ids,groups=groups)):
        if set(groups[training])&set(groups[valid]):raise ValueError('outer group leak')
        trials=[]
        for spec in specs:
            predictions={}
            for t,v in GroupKFold(3).split(training,groups=groups[training]):
                tr,va=training[t],training[v]
                if set(groups[tr])&set(groups[va]):raise ValueError('inner group leak')
                predictions.update(predict(fit(spec,tr),va))
            for weight in WEIGHTS:
                values,_=quality(training,predictions,weight)
                lifts=values.mean(axis=0)/baseline[training].mean(axis=0)-1
                for policy,uses_calibration in policies(training):
                    actual=np.where(uses_calibration[:,None],values,raw_quality[training])
                    lifts=actual.mean(axis=0)/baseline[training].mean(axis=0)-1
                    trials.append(dict(spec=spec,weight=weight,policy=policy,lifts=lifts.tolist(),
                        key=[float(lifts.min()),float(lifts.mean()),-float(uses_calibration.mean()),weight]))
        selected=max(trials,key=lambda t:tuple(t['key']))
        model=fit(selected['spec'],training); model_path=args.out/f'fold-{fold}.joblib'
        joblib.dump(model,model_path)
        predicted=predict(model,valid);replayed=predict(joblib.load(model_path),valid)
        if any(not np.array_equal(predicted[i],replayed[i]) for i in predicted):
            raise ValueError('persisted calibration model replay differs')
        observed,rankings=quality(valid,predicted,selected['weight'])
        policy=selected['policy']
        if policy['kind']=='constant':heavy=np.ones(len(valid),dtype=bool)
        else:
            heavy=policy['direction']*routing_features[valid,policy['feature']]>policy['direction']*policy['threshold']
        outer[valid]=np.where(heavy[:,None],observed,raw_quality[valid])
        for j,i in enumerate(valid):
            action='heavy' if heavy[j] else 'cheap';counts[action]+=1
            outputs[int(i)]=dict(query_id=ids[i],group=str(groups[i]),fold=fold,action=action,
                metrics=dict(zip(('ndcg','recall'),outer[i].tolist())),
                ranked=rankings[int(i)] if heavy[j] else raw_rankings[int(i)])
        record=dict(fold=fold,training_ids=[ids[i] for i in training],heldout_ids=[ids[i] for i in valid],
            selection=selected,trials=trials,model_file=model_path.name,model_sha256=sha(model_path))
        folds.append(record)
        (args.out/'fold-progress.json').write_text(json.dumps(folds,indent=2)+'\n')
        print('BGE calibration fold',fold,'selected',selected,flush=True)
    predictions_path=args.out/'outer-predictions.jsonl'
    predictions_path.write_text(''.join(json.dumps(outputs[i])+'\n' for i in range(len(ids))))
    result=summary(outer,baseline,groups)
    zero={i:np.zeros(len(teacher_z[i])) for i in range(len(ids))}
    teacher_quality,_=quality(np.arange(len(ids)),zero,1.)
    study=dict(protocol=protocol,folds=folds,outer=result,
        fixed_BGE_control=summary(teacher_quality,baseline,groups),
        fixed_dense_control={m:float(np.mean([r[m] for r in meta['baselines']['dense']])) for m in ('ndcg','recall')},
        quality_point_gate_passed=all(v['passes_20_percent'] and v['delta_ci']['ci95'][0]>0 for v in result.values()),
        artifact_sha256={p.name:sha(p) for p in args.out.iterdir() if p.is_file()},
        http_evaluated=False,independent_confirmation_evaluated=False,
        action_counts=counts,nonconstant_router_verified=bool(args.route_calibration and all(counts.values())),
        goal_complete=False,elapsed_seconds=time.monotonic()-started)
    (args.out/'study.json').write_text(json.dumps(study,indent=2)+'\n')
    print(json.dumps(dict(outer=result,quality_point_gate_passed=study['quality_point_gate_passed']),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('dataset','capture','logits','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--route-calibration',action='store_true')
    p.add_argument('--rank-calibration',action='store_true')
    p.add_argument('--support-vectors',type=Path)
    p.add_argument('--no-bm25-stats',action='store_true')
    p.add_argument('--fast-bm25-stats',action='store_true')
    p.add_argument('--max-cheap-rate',type=float,default=1.)
    p.add_argument('--pool-size',type=int,choices=(20,32,40),default=40)
    run(p.parse_args())
