"""Opt-in, safely serialised BM25-first research policy. No training dependencies."""

import hashlib
import hmac
import json
import math
import os
import re
from pathlib import Path

from contracts.index_format import BM25Spec

from .router import RouteDecision
from .router_features import (
    FEATURE_NAMES, FEATURE_NAMES_V2, FEATURE_VERSION, FEATURE_VERSION_V2,
    extract_statistics, extract_statistics_v2,
)


class GainRouter:
    requires_statistics = True

    def __init__(self, path, *, allow_transfer=False, expected_sha256=None):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and not hmac.compare_digest(self.sha256, expected_sha256):
            raise ValueError('gain router model SHA256 mismatch')
        self.policy_version = 'bm25-gain-v1-'+self.sha256[:12]
        self.allow_transfer = allow_transfer
        self.model = m = json.loads(raw)
        if m.get('kind')=='action-linear':
            self.policy_version = 'bm25-actions-v1-'+self.sha256[:12]
        self.feature_version = m.get('feature_version')
        allowed = {FEATURE_VERSION:list(FEATURE_NAMES), FEATURE_VERSION_V2:list(FEATURE_NAMES_V2)}
        if (m.get('schema_version')!=1 or m.get('feature_version') not in allowed
                or m.get('feature_names')!=allowed[m.get('feature_version')]):
            raise ValueError('gain router feature/schema mismatch')
        source = m.get('provenance', {}).get('source', {})
        if not {'source_manifest','candidate_k','rrf_c','vector_manifest','vector_search_params'} <= source.keys():
            raise ValueError('incomplete gain router retrieval provenance')
        selected = m['selected_features']
        if (not selected or len(set(selected))!=len(selected) or any(name not in allowed[m['feature_version']] for name in selected)
                or not math.isfinite(m['threshold'])):
            raise ValueError('invalid features or threshold')
        if m['kind'] in ('linear','action-linear'):
            for key in ('mean','scale'):
                if len(m[key])!=len(selected) or any(not math.isfinite(v) for v in m[key]):
                    raise ValueError('invalid linear router')
            heads = [m] if m['kind']=='linear' else list(m.get('heads',{}).values())
            if m['kind']=='action-linear' and (set(m.get('heads',{}))!={'dense','hybrid'} or not math.isfinite(m['margin'])):
                raise ValueError('invalid action heads')
            for head in heads:
                if len(head['coef'])!=len(selected) or any(not math.isfinite(v) for v in head['coef']) or not math.isfinite(head['intercept']):
                    raise ValueError('invalid linear coefficients')
            if any(v<=0 for v in m['scale']):
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

    def statistics_for_query(self, query, reader, terms, ranked):
        extractor = (extract_statistics_v2 if self.feature_version == FEATURE_VERSION_V2
                     else extract_statistics)
        return extractor(query, reader, terms, ranked)

    def check_retrieval_binding(self, reader, config, vectors):
        source = self.model['provenance']['source']
        expected = source['source_manifest']
        if config.candidate_k!=source['candidate_k'] or config.rrf_c!=source['rrf_c']:
            raise ValueError('gain router retrieval configuration mismatch')
        manifest = reader.manifest()
        # Legacy readers and artifacts may omit BM25 defaults. Compare the
        # settings used by the scorer, then bind the rest of the index identity.
        try:
            actual_bm25 = BM25Spec(**manifest.get('bm25', {}))
            expected_bm25 = BM25Spec(**expected.get('bm25', {}))
        except (TypeError, ValueError) as error:
            raise ValueError('gain router invalid BM25 configuration') from error
        if actual_bm25 != expected_bm25:
            raise ValueError('gain router BM25 configuration mismatch')
        if not self.allow_transfer and (
            {**manifest, 'bm25': actual_bm25.to_dict()}
            != {**expected, 'bm25': expected_bm25.to_dict()}
        ):
            raise ValueError('gain router index mismatch; explicit transfer required')
        if vectors is None:
            raise ValueError('gain router requires a vector index')
        actual_vector = vectors.manifest().to_dict()
        expected_vector = source['vector_manifest']
        if actual_vector['embedding'] != expected_vector['embedding']:
            raise ValueError('gain router embedding mismatch')
        if actual_vector['index_params'] != expected_vector['index_params']:
            raise ValueError('gain router HNSW build parameters mismatch')
        if actual_vector['backend'] != expected_vector['backend']:
            raise ValueError('gain router vector backend mismatch')
        if not self.allow_transfer and actual_vector != expected_vector:
            raise ValueError('gain router vector manifest mismatch; explicit transfer required')

        expected_search = source['vector_search_params']
        required = {'hnsw_ef', 'exact'}
        if (not isinstance(expected_search, dict) or not required <= expected_search.keys()
                or expected_search.keys() - (required | {'timeout_s'})):
            raise ValueError('gain router requires explicit ANN configuration')
        expected_ef = expected_search['hnsw_ef']
        exact = expected_search['exact']
        if type(exact) is not bool:
            raise ValueError('gain router requires a boolean exact search setting')
        if not (exact and expected_ef is None) and (
            type(expected_ef) is not int or expected_ef <= 0
        ):
            raise ValueError('gain router requires an explicit positive hnsw_ef for ANN')
        actual_search = getattr(vectors, 'search_config', None)
        if actual_search is None:
            actual_search = getattr(vectors, 'search_params', None)
        if (not isinstance(actual_search, dict)
                or any(key not in actual_search or type(actual_search[key]) is not type(value)
                       or actual_search[key] != value
                       for key, value in expected_search.items() if key != 'timeout_s')
                or ('timeout_s' in expected_search
                    and ('timeout_s' not in actual_search
                         or actual_search['timeout_s'] != expected_search['timeout_s']))):
            raise ValueError('gain router ANN configuration mismatch')

    check_vector_binding = check_retrieval_binding

    def predict(self, features):
        m = self.model
        if m['kind']=='action-linear':
            raise ValueError('action router requires predict_actions')
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

    def predict_actions(self, features):
        m=self.model
        x=[features[name] for name in m['selected_features']]
        if any(not math.isfinite(v) for v in x):
            raise ValueError('nonfinite action router input')
        return {name:head['intercept']+sum(c*(v-a)/b for v,a,b,c in zip(x,m['mean'],m['scale'],head['coef']))
                for name,head in m['heads'].items()}

    def decide(self, features):
        if features.explicit_constraints:
            return RouteDecision(False,'explicit_constraints')
        if self.model['kind']=='action-linear':
            if not features.lexical_hits:
                return RouteDecision(True,'no_lexical_hits','dense')
            values=self.predict_actions(features.statistics)
            if max(values.values())<=self.model['threshold']:
                return RouteDecision(False,'predicted_lexical_sufficient')
            mode='dense' if values['dense']>=values['hybrid']+self.model['margin'] else 'hybrid'
            return RouteDecision(True,'predicted_'+mode+'_gain',mode)
        if not features.lexical_hits:
            return RouteDecision(True,'no_lexical_hits')
        use = self.predict(features.statistics)>self.model['threshold']
        return RouteDecision(use,'predicted_hybrid_gain' if use else 'predicted_lexical_sufficient')


def load_gain_router(config):
    """Load a digest-pinned local model for the engine's retrieval binding check."""
    path = os.environ.get('NLP_PEER_GAIN_MODEL')
    digest = os.environ.get('NLP_PEER_GAIN_SHA256', '')
    if not path:
        raise ValueError('NLP_PEER_GAIN_MODEL must specify a local model path')
    if not re.fullmatch(r'[0-9a-fA-F]{64}', digest):
        raise ValueError('NLP_PEER_GAIN_SHA256 must specify a SHA256 hex digest')
    return GainRouter(path, expected_sha256=digest.lower())
