"""Leakage-free pre-BGE routing primitives, with persisted training roles."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .bge_candidate_support import bge_support,FEATURE_NAMES as SUPPORT_NAMES
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .ranking_dataset import load_dataset
from .study_candidate_ranking import FEATURES,TEXT_FEATURES

SPECS=[dict(max_leaf_nodes=l,min_samples_leaf=n,l2_regularization=r)
       for l,n,r in ((7,50,10.),(15,40,10.),(31,40,10.),(31,20,1.))]
WEIGHTS=(0.,.25,.5,.75,1.)
GATE_NAMES=['query_chars','n_terms','BM25_relative_gap','dense_top_gap',
            'lexical_top_gap','candidate_hits','vocab_coverage']
QUANTILES=(.01,.02,.03,.04,.05,.1,.15,.2,.25,.3,.5,.7,.8,.9,.95,.98,.99)


def normalized(v):
    v=np.asarray(v,dtype=float)
    return (v-v.mean(axis=-1,keepdims=True))/np.maximum(v.std(axis=-1,keepdims=True),1e-8)


def safe_partition(train,valid,allowed,groups):
    if any(len(list(v))!=len(set(map(int,v))) for v in (train,valid,allowed)):
        raise ValueError('duplicate query index in training roles')
    train,valid,allowed=set(map(int,train)),set(map(int,valid)),set(map(int,allowed))
    if not train or not valid or train&valid or train|valid!=allowed:
        raise ValueError('training/validation partition differs from its allowed scope')
    if {groups[i] for i in train}&{groups[i] for i in valid}:
        raise ValueError('query group crosses a training boundary')


def gate_features(stats,lexical_scores,dense_scores):
    return [stats['query_chars'],stats['n_terms'],stats['relative_gap'],
            dense_scores[0]-dense_scores[1] if len(dense_scores)>1 else 0.,
            lexical_scores[0]-lexical_scores[1] if len(lexical_scores)>1 else 0.,
            stats['candidate_hits'],stats['vocab_coverage']]


class DevelopmentData:
    def __init__(self,dataset,capture,canonical,support,score_pool=40):
        if score_pool not in (28,40):raise ValueError('unsupported calibrated candidate pool')
        self.score_pool=score_pool
        self.paths=dict(dataset=Path(dataset),capture=Path(capture),canonical=Path(canonical),support=Path(support))
        parent=load_dataset(dataset);m=parent['metadata'];self.m=m
        self.ids=m['query_ids'];self.groups=np.array(m['groups']);self.n=len(self.ids)
        self.baseline=np.array([[r[k] for k in ('ndcg','recall')] for r in m['baselines']['hybrid']])
        source=json.loads((Path(dataset)/'protocol.json').read_text())
        if source.get('confirmation_labels_read') is not False:raise ValueError('development must exclude confirmation labels')
        report=json.loads(Path(canonical).with_name('summary.json').read_text())
        if report['protocol']['kind']!='canonical-preBGE-development-logits-v2' or report['logits_sha256']!=sha(canonical):
            raise ValueError('canonical actual logits identity differs')
        for path,digest in report['protocol']['inputs'].items():
            if sha(path)!=digest:raise ValueError('canonical scorer source changed: '+path)
        if report['protocol']['inputs'].get(str(Path(capture).resolve()))!=sha(capture):raise ValueError('candidate capture differs')
        support_report=json.loads(Path(support).with_name('protocol.json').read_text())
        if sha(support)!=support_report['vector_sha256']:raise ValueError('support vectors changed')
        docs_path=next(Path(p) for p in support_report['inputs'] if Path(p).name=='docs.jsonl')
        if support_report['inputs'][str(docs_path)]!=sha(docs_path):raise ValueError('support corpus differs')
        self.identities={r['doc_id']:r['docid'] for r in jsonl_rows(docs_path)}
        vectors=np.load(support,mmap_mode='r',allow_pickle=False)
        if vectors.shape!=(len(self.identities),512):raise ValueError('support matrix shape differs')
        captures={r['query_id']:r for r in jsonl_rows(capture)}
        logits={r['query_id']:r['scores'] for r in jsonl_rows(canonical)}
        if set(captures)!=set(self.ids) or set(logits)!=set(self.ids):raise ValueError('development coverage differs')
        raw_names=list(FEATURES)+list(TEXT_FEATURES);cols=[m['feature_names'].index(n) for n in raw_names]
        self.feature_names=raw_names+['bge_logit','bge_z','bge_rank_reciprocal','bge_gap']+list(SUPPORT_NAMES)+m['bm25_feature_names']
        x=[];labels=[];teacher=[];gates=[];self.heavy_docs=[];self.cheap_rankings=[];cheap_quality=[]
        for i,q in enumerate(self.ids):
            cap=captures[q];pool=sorted((r['doc_id'] for r in cap['dense'][:score_pool]),key=self.identities.__getitem__)
            cheap_pool=sorted((r['doc_id'] for r in cap['dense'][:28]),key=self.identities.__getitem__)
            if list(logits[q][str(score_pool)])!=pool or list(logits[q]['28'])!=cheap_pool:
                raise ValueError('actual scoring order differs from live canonical pools')
            slots={d:j for j,d in enumerate(m['docs'][i])};raw=parent['x'][parent['offsets'][i]:parent['offsets'][i+1]][[slots[d] for d in pool]][:,cols]
            values=np.array([logits[q][str(score_pool)][d] for d in pool]);z=normalized(values)
            ranks=np.empty(score_pool);ranks[np.argsort(-values,kind='stable')]=np.arange(1,score_pool+1)
            support_values=bge_support([self.identities[d] for d in pool],values,
                [self.identities[r['doc_id']] for r in cap['lexical']],
                [self.identities[r['doc_id']] for r in cap['dense']],vectors)
            bm25=np.tile([m['bm25_features'][i][n] for n in m['bm25_feature_names']],(score_pool,1))
            x.append(np.column_stack((raw,values,z,1./ranks,values.max()-values,support_values,bm25)).astype(np.float32))
            labels.append([m['relevance'][q].get(d,0)>0 for d in pool]);teacher.append(z);self.heavy_docs.append(pool)
            ranked=sorted(cheap_pool,key=lambda d:(-logits[q]['28'][d],self.identities[d]))
            self.cheap_rankings.append(ranked);v=metrics(ranked,m['relevance'][q]);cheap_quality.append([v['ndcg'],v['recall']])
            gates.append(gate_features(m['bm25_features'][i],[r['score'] for r in cap['lexical'][:10]],
                                       [r['score'] for r in cap['dense'][:10]]))
        self.x=np.array(x);self.y=np.array(labels,dtype=int);self.teacher=np.array(teacher)
        self.gate_x=np.array(gates);self.cheap_quality=np.array(cheap_quality)
        self.encoder=report['protocol']['encoder']
        self.inputs={str(p.resolve()):sha(p) for p in (Path(dataset)/'dataset.json',Path(capture),Path(canonical),
            Path(canonical).with_name('summary.json'),Path(support),Path(support).with_name('protocol.json'),docs_path)}
        isolation_path=Path(capture).with_name('isolation.json')
        if isolation_path.exists():
            isolation=json.loads(isolation_path.read_text())
            if (isolation.get('queries')!=self.n or isolation.get('consumed_ID_overlap')!=0 or
                    isolation.get('consumed_near_duplicate_overlap')!=0 or isolation.get('new_confirmation_read') is not False or
                    isolation.get('dataset_manifest_sha256')!=sha(Path(dataset)/'dataset.json')):
                raise ValueError('expanded development isolation evidence differs')
            for section in ('historical_sources','consumed_confirmation_query_files'):
                for path,digest in isolation[section].items():
                    if sha(path)!=digest:raise ValueError('isolation history/input changed: '+path)
                    self.inputs[str(Path(path).resolve())]=digest
            self.inputs[str(isolation_path.resolve())]=sha(isolation_path)

    def rank_quality(self,selected,prediction,weight):
        selected=np.asarray(selected,dtype=int);score=weight*self.teacher[selected]+(1-weight)*prediction
        quality=[];rankings=[]
        for j,i in enumerate(selected):
            order=np.argsort(-score[j],kind='stable');ranked=[self.heavy_docs[i][k] for k in order]
            value=metrics(ranked,self.m['relevance'][self.ids[i]])
            rankings.append(ranked);quality.append([value['ndcg'],value['recall']])
        return np.array(quality),rankings


def predict(model,x):
    shape=x.shape[:2];prob=np.clip(model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1],1e-7,1-1e-7)
    return normalized(np.log(prob/(1-prob)).reshape(shape))


class NestedTrainer:
    def __init__(self,data,out):
        self.data=data;self.out=Path(out);self.records=[];self.counter=0
        (self.out/'models').mkdir(parents=True,exist_ok=True)

    def fit(self,spec,train,valid,allowed,role):
        d=self.data;safe_partition(train,valid,allowed,d.groups)
        model=HistGradientBoostingClassifier(**spec,max_iter=100,learning_rate=.08,early_stopping=False,random_state=20261030)
        model.fit(d.x[train].reshape(-1,d.x.shape[-1]),d.y[train].reshape(-1))
        path=self.out/'models'/f'{self.counter:04d}.joblib';self.counter+=1;joblib.dump(model,path)
        record=dict(role=role,spec=spec,train=list(map(int,train)),valid=list(map(int,valid)),
            allowed=list(map(int,allowed)),model_file=str(path.relative_to(self.out)),model_sha256=sha(path))
        self.records.append(record)
        return joblib.load(path),predict(model,d.x[valid])

    def select_action(self,train,role):
        d=self.data;trials=[]
        for spec in SPECS:
            oof=np.empty((len(train),d.x.shape[1]))
            for split,(t,v) in enumerate(GroupKFold(3).split(train,groups=d.groups[train])):
                _,pred=self.fit(spec,train[t],train[v],train,role+f'/action-inner-{split}')
                oof[v]=pred
            for weight in WEIGHTS:
                quality,_=d.rank_quality(train,oof,weight);lifts=quality.mean(0)/d.baseline[train].mean(0)-1
                trials.append(dict(spec=spec,weight=weight,lifts=lifts.tolist(),key=[float(lifts.min()),float(lifts.mean()),weight]))
        return max(trials,key=lambda r:tuple(r['key'])),trials

    def gate_training_outcomes(self,train,role):
        d=self.data;oof=np.empty((len(train),2));parts=[]
        for split,(t,v) in enumerate(GroupKFold(3).split(train,groups=d.groups[train])):
            fit_ids,hold_ids=train[t],train[v]
            selection,trials=self.select_action(fit_ids,role+f'/gate-{split}')
            _,pred=self.fit(selection['spec'],fit_ids,hold_ids,train,role+f'/gate-outcome-{split}')
            oof[v],_=d.rank_quality(hold_ids,pred,selection['weight'])
            parts.append(dict(train=fit_ids.tolist(),valid=hold_ids.tolist(),selection=selection,trials=trials))
        return oof,parts


def select_gate(data,train,heavy_quality):
    trials=[]
    for col,name in enumerate(GATE_NAMES):
        values=data.gate_x[train,col]
        for quantile in QUANTILES:
            threshold=float(np.quantile(values,quantile))
            for direction in (-1,1):
                mask=direction*values>direction*threshold
                if not mask.any() or not (~mask).any() or mask.mean()>.95+1e-12:continue
                actual=np.where(mask[:,None],heavy_quality,data.cheap_quality[train])
                lifts=actual.mean(0)/data.baseline[train].mean(0)-1
                trials.append(dict(feature=col,name=name,threshold=threshold,direction=direction,
                    heavy_rate=float(mask.mean()),lifts=lifts.tolist(),key=[float(lifts.min()),float(lifts.mean()),-float(mask.mean())]))
    if not trials:raise ValueError('no nonconstant gate in training scope')
    return max(trials,key=lambda r:tuple(r['key'])),trials


def gate_mask(gate,x):
    return gate['direction']*x[:,gate['feature']]>gate['direction']*gate['threshold']
