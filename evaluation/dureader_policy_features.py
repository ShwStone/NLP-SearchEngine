"""Identical train/serve features for the frozen CE policy experiment."""
import hashlib
import json
import unicodedata

import numpy as np

from contracts import analyzer
from online.parser import parse

GATE_FEATURES = ('n_terms','vocab_coverage','relative_gap','candidate_hits')
RANK_FEATURES = ('ce_score','reciprocal_hybrid_rank','reciprocal_lexical_rank',
                 'reciprocal_dense_rank','lexical_hit','dense_hit')


class FrozenLinear:
    """Portable scaler plus ridge inference; no pickle or sklearn required."""
    def __init__(self, spec):
        self.mean=np.asarray(spec['mean'],dtype=float)
        self.scale=np.asarray(spec['scale'],dtype=float)
        self.coef=np.asarray(spec['coef'],dtype=float)
        self.intercept=float(spec['intercept'])
        if (self.mean.ndim!=1 or self.mean.shape!=self.scale.shape or
                self.mean.shape!=self.coef.shape or not np.all(self.scale>0) or
                not all(np.isfinite(a).all() for a in (self.mean,self.scale,self.coef)) or
                not np.isfinite(self.intercept)):
            raise ValueError('invalid exported linear model')
    def predict(self, values):
        x=np.asarray(values,dtype=float)
        if x.ndim!=2 or x.shape[1]!=len(self.mean) or not np.isfinite(x).all():
            raise ValueError('invalid inference features')
        return ((x-self.mean)/self.scale)@self.coef+self.intercept


def gate_features(row, dfs):
    terms = {t.term for raw in parse(row['text'])['terms'] for t in analyzer.analyze(raw)}
    if any(t not in dfs for t in terms):
        raise ValueError('missing frozen query-term df')
    lexical = row['lexical']
    gap = ((lexical[0]['score']-lexical[1]['score'])/max(abs(lexical[0]['score']),1e-12)
           if len(lexical)>1 else 0.)
    return np.array([len(terms),sum(dfs[t]>0 for t in terms)/max(1,len(terms)),
                     gap,len(lexical)], dtype=float)


def rank_features(row, budget):
    scores = row[f'ce{budget}']['scores']
    lexical = {h['doc_id']:i+1 for i,h in enumerate(row['lexical'])}
    dense = {h['doc_id']:i+1 for i,h in enumerate(row['dense'])}
    if len(scores) != min(budget,len(row['pool'])):
        raise ValueError('CE score cardinality mismatch')
    return np.array([[s,1/(i+1),1/lexical[d] if d in lexical else 0.,
                      1/dense[d] if d in dense else 0.,float(d in lexical),float(d in dense)]
                     for i,(d,s) in enumerate(zip(row['pool'],scores))], dtype=float)


def rerank(row, budget, model=None):
    pool = row['pool']
    scores = (model.predict(rank_features(row,budget)) if model is not None
              else row[f'ce{budget}']['scores'])
    chosen = sorted(range(len(scores)),key=lambda i:(-float(scores[i]),row['global_docids'][i]))
    return [pool[i] for i in chosen]+pool[len(scores):]


def query_groups(rows, qrels):
    """Connected components of normalized duplicate queries/shared positives.

    Training groups must use training labels. Evaluation groups may use judged
    documents for inference after policies and rankings have been frozen.
    """
    parent = list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen = {}
    for i,row in enumerate(rows):
        normalized=''.join(unicodedata.normalize('NFKC',row['text']).casefold().split())
        keys=[('query',normalized)]+[('positive',d) for d,g in qrels[row['query_id']].items() if g>0]
        for key in keys:
            if key in seen:
                a,b=root(i),root(seen[key]);parent[max(a,b)]=min(a,b)
            else:seen[key]=i
    members={}
    for i,row in enumerate(rows):members.setdefault(root(i),[]).append(row['query_id'])
    names={i:hashlib.sha256(json.dumps(sorted(v)).encode()).hexdigest() for i,v in members.items()}
    return np.array([names[root(i)] for i in range(len(rows))])
