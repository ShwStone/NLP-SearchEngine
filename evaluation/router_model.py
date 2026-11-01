"""Historical dense-first research inference. Not an online Router policy."""

import hashlib
import json
import math
from pathlib import Path
from contracts import analyzer

FEATURES=('query_chars','n_terms','ascii_fraction','digit_fraction','hits','top1','mean5','mean10',
          'std10','gap2','gap5','gap10','gap50','score200')


def dense_features(query,ranked):
    scores=[s for _,s in ranked];head=scores[:10];mean=sum(head)/max(len(head),1)
    at=lambda i:scores[min(i,len(scores)-1)] if scores else 0.
    return dict(zip(FEATURES,[len(query),len({t.term for t in analyzer.analyze(query)}),
        sum(ord(c)<128 for c in query)/max(len(query),1),sum(c.isdigit() for c in query)/max(len(query),1),
        len(scores),at(0),sum(scores[:5])/max(len(scores[:5]),1),mean,
        math.sqrt(sum((s-mean)**2 for s in head)/max(len(head),1)),
        at(0)-at(1),at(0)-at(4),at(0)-at(9),at(0)-at(49),at(199)]))


class CascadeRouter:
    def __init__(self,path):
        data=Path(path).read_bytes();self.sha256=hashlib.sha256(data).hexdigest();self.model=json.loads(data)
        model=self.model
        if model.get('feature_version')!='vector-first-v1' or model.get('feature_names')!=list(FEATURES):raise ValueError('cascade feature mismatch')
        if not isinstance(model.get('threshold'),(int,float)) or not math.isfinite(model['threshold']):raise ValueError('invalid cascade threshold')
        if model['kind']=='linear':
            for key in ('mean','scale','coef'):
                if len(model[key])!=len(FEATURES) or any(not math.isfinite(v) for v in model[key]):raise ValueError('invalid linear cascade')
            if any(v<=0 for v in model['scale']) or not math.isfinite(model['intercept']):raise ValueError('invalid scaling/intercept')
        elif model['kind']=='forest':
            if not 0<len(model['trees'])<=100:raise ValueError('invalid tree count')
            for tree in model['trees']:
                n=len(tree['feature'])
                if not 0<n<=1024 or any(len(tree[k])!=n for k in ('left','right','threshold','value')):raise ValueError('invalid tree shape')
                for i,f in enumerate(tree['feature']):
                    if f==-2:
                        if tree['left'][i]!=-1 or tree['right'][i]!=-1:raise ValueError('invalid tree leaf')
                    elif type(f) is not int or not 0<=f<len(FEATURES) or not i<tree['left'][i]<n or not i<tree['right'][i]<n:
                        raise ValueError('invalid tree split/cycle')
                    if not math.isfinite(tree['value'][i]) or not math.isfinite(tree['threshold'][i]):raise ValueError('invalid tree numeric value')
        else:raise ValueError('invalid cascade kind')

    def predict(self,features):
        x=[features[name] for name in FEATURES];m=self.model
        if any(not math.isfinite(v) for v in x):raise ValueError('nonfinite cascade input')
        if m['kind']=='linear':return m['intercept']+sum(c*(v-a)/b for v,a,b,c in zip(x,m['mean'],m['scale'],m['coef']))
        predictions=[]
        for tree in m['trees']:
            i=0
            while tree['feature'][i]!=-2:i=tree['left'][i] if x[tree['feature'][i]]<=tree['threshold'][i] else tree['right'][i]
            predictions.append(tree['value'][i])
        return sum(predictions)/len(predictions)

    def use_lexical(self,features):
        return features['hits']==0 or self.predict(features)>self.model['threshold']
