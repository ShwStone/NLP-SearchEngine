"""Pre-CE cost router features and portable tree inference."""
import json
from pathlib import Path
import numpy as np
from contracts import analyzer
from online.parser import parse

FEATURE_NAMES = ('n_terms','query_chars','digit_count','ascii_count',
                 'lex_mean','lex_std','lex_max','lex_min','lex_gap12','lex_top20_mass',
                 'dense_mean','dense_std','dense_max','dense_min','dense_gap12','dense_top20_mass',
                 'lex_dense_overlap','dense_lex_overlap','legacy_zero','pool_lex10','pool_dense10','pool_lexmean10','pool_densemean10',
                 'pool_lex20','pool_dense20','pool_lexmean20','pool_densemean20',
                 'pool_lex50','pool_dense50','pool_lexmean50','pool_densemean50',
                 'pool_lex100','pool_dense100','pool_lexmean100','pool_densemean100')

def extract_features(row):
    lex=row.get('lexical',[]); dense=row.get('dense',[]); pool=row.get('pool',[]); text=row['text']
    lr={h['doc_id']:i for i,h in enumerate(lex)}; dr={h['doc_id']:i for i,h in enumerate(dense)}
    terms={t.term for raw in parse(text)['terms'] for t in analyzer.analyze(raw)}
    x=[len(terms),len(text),sum(c.isdigit() for c in text),sum(c.isascii() for c in text)]
    for hs in (lex,dense):
        s=np.asarray([h['score'] for h in hs[:50]],dtype=float)
        if not len(s): s=np.zeros(1,dtype=float)
        x.extend((float(s.mean()),float(s.std()),float(s.max()),float(s.min()),
                  float(s[0]-s[1] if len(s)>1 else 0),float(np.mean(s>=np.quantile(s,.8)))))
    x.extend((sum(h['doc_id'] in dr for h in lex[:50]),sum(h['doc_id'] in lr for h in dense[:50]),0.0))
    for k in (10,20,50,100):
        ids=pool[:k]
        x.extend((sum(d in lr for d in ids),sum(d in dr for d in ids),
                  float(np.mean([lr.get(d,200) for d in ids])) if ids else 200.0,
                  float(np.mean([dr.get(d,200) for d in ids])) if ids else 200.0))
    return np.asarray(x,dtype=float)

class FrozenTreeEnsemble:
    def __init__(self, spec):
        self.threshold=float(spec['threshold']); self.feature_names=tuple(spec['feature_names']); self.trees=spec['trees']
        if self.feature_names!=FEATURE_NAMES: raise ValueError('pre-ce feature schema mismatch')
    def predict_one(self, values):
        x=np.asarray(values,dtype=float)
        if x.shape!=(len(self.feature_names),) or not np.isfinite(x).all(): raise ValueError('invalid pre-ce features')
        total=0.0
        for t in self.trees:
            i=0
            while t['feature'][i]>=0:
                i=t['left'][i] if x[t['feature'][i]]<=t['threshold'][i] else t['right'][i]
            total+=t['value'][i]
        return total/len(self.trees)
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))

class FrozenScalarTree:
    def __init__(self,spec):
        self.threshold=float(spec['threshold']); self.trees=spec['trees']
    def predict_one(self,value):
        x=float(value); total=0.0
        for t in self.trees:
            i=0
            while t['feature'][i]>=0:
                i=t['left'][i] if x<=t['threshold'][i] else t['right'][i]
            total+=t['value'][i]
        return total/len(self.trees)
    @classmethod
    def load(cls,path): return cls(json.loads(Path(path).read_text()))
