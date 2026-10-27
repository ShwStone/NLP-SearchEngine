"""Opt-in, safely serialised BM25-first research policy. No training dependencies."""

import hashlib
import json
import math
from pathlib import Path

from .router import RouteDecision
from .router_features import FEATURE_NAMES, FEATURE_VERSION


class GainRouter:
    requires_statistics = True

    def __init__(self, path, *, allow_transfer=False):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.policy_version = 'bm25-gain-v1-'+self.sha256[:12]
        self.allow_transfer = allow_transfer
        self.model = m = json.loads(raw)
        if (m.get('schema_version')!=1 or m.get('feature_version')!=FEATURE_VERSION
                or m.get('feature_names')!=list(FEATURE_NAMES)):
            raise ValueError('gain router feature/schema mismatch')
        selected = m['selected_features']
        if (not selected or len(set(selected))!=len(selected) or any(name not in FEATURE_NAMES for name in selected)
                or not math.isfinite(m['threshold'])):
            raise ValueError('invalid features or threshold')
        if m['kind']=='linear':
            for key in ('mean','scale','coef'):
                if len(m[key])!=len(selected) or any(not math.isfinite(v) for v in m[key]):
                    raise ValueError('invalid linear router')
            if any(v<=0 for v in m['scale']) or not math.isfinite(m['intercept']):
                raise ValueError('invalid router scaling')
        elif m['kind']=='tree':
            n = len(m['feature'])
            if not 0<n<=1024 or any(len(m[k])!=n for k in ('left','right','split','value')):
                raise ValueError('invalid tree shape')
            for i, f in enumerate(m['feature']):
                if f==-2:
                    if m['left'][i]!=-1 or m['right'][i]!=-1:
                        raise ValueError('invalid tree leaf')
                elif type(f) is not int or not 0<=f<len(selected) or not i<m['left'][i]<n or not i<m['right'][i]<n:
                    raise ValueError('invalid tree split/cycle')
                if not math.isfinite(m['value'][i]) or not math.isfinite(m['split'][i]):
                    raise ValueError('invalid tree value')
        elif m['kind']=='constant':
            if not math.isfinite(m['specification']['value']):
                raise ValueError('invalid constant router')
        elif m['kind']=='rule':
            spec = m['specification']
            if (not 0<=spec['coverage']<=1 or not 0<=spec['gap']<=1
                    or type(spec['terms']) is not int or spec['terms']<=0):
                raise ValueError('invalid rule router')
        else:
            raise ValueError('unknown router model')

    def check_binding(self, reader, config, vectors):
        source = self.model['provenance']['source']
        expected = source['source_manifest']
        if config.candidate_k!=source['candidate_k'] or config.rrf_c!=source['rrf_c']:
            raise ValueError('gain router retrieval configuration mismatch')
        if reader.manifest().get('bm25')!=expected.get('bm25'):
            raise ValueError('gain router BM25 configuration mismatch')
        if not self.allow_transfer and reader.manifest()!=expected:
            raise ValueError('gain router index mismatch; explicit transfer required')
        if vectors is None or vectors.manifest().to_dict()['embedding']!=source['vector_manifest']['embedding']:
            raise ValueError('gain router embedding mismatch')
        if vectors.manifest().index_params!=source['vector_manifest']['index_params']:
            raise ValueError('gain router HNSW build parameters mismatch')
        if getattr(vectors,'search_params',None)!=source['vector_search_params']:
            raise ValueError('gain router ANN configuration mismatch')

    def predict(self, features):
        m = self.model
        x = [features[name] for name in m['selected_features']]
        if any(not math.isfinite(v) for v in x):
            raise ValueError('nonfinite gain router input')
        if m['kind']=='linear':
            return m['intercept']+sum(c*(v-a)/b for v,a,b,c in zip(x,m['mean'],m['scale'],m['coef']))
        if m['kind']=='tree':
            i = 0
            while m['feature'][i]!=-2:
                i = m['left'][i] if x[m['feature'][i]]<=m['split'][i] else m['right'][i]
            return m['value'][i]
        spec = m['specification']
        if m['kind']=='constant':
            return spec['value']
        sufficient = (features['vocab_coverage']>=spec['coverage'] and features['relative_gap']>=spec['gap']
                      and features['n_terms']<=spec['terms'] and features['candidate_hits']>=2)
        return -1. if sufficient else 1.

    def decide(self, features):
        if features.explicit_constraints:
            return RouteDecision(False,'explicit_constraints')
        if not features.lexical_hits:
            return RouteDecision(True,'no_lexical_hits')
        use = self.predict(features.statistics)>self.model['threshold']
        return RouteDecision(use,'predicted_hybrid_gain' if use else 'predicted_lexical_sufficient')
