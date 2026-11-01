"""Policy integration, fallback and frozen research model checks."""

import json
import math
from pathlib import Path
import tempfile

from contracts.api import ApiError, SearchRequest
from contracts.retrieval import RetrievalConfig, RetrievalUnavailable
from evaluation.router_model import CascadeRouter, FEATURES
from online.router import QueryFeatures, RuleRouter
from tests.test_hybrid import _raises, _setup


def test_policy_controls_change_single_hit_decision_and_report_actual_policy():
    # A single lexical hit gives no score-gap evidence. It is not proof of recall.
    for policy, attempted in [('rules-v1', False), ('recall-v1', True), ('hybrid-v1', True)]:
        engine, encoder, _ = _setup(RetrievalConfig(adaptive_policy=policy))
        response = engine.search(SearchRequest('phone', mode='adaptive'))
        assert response.routing.vector_attempted is attempted
        assert encoder.query_calls == int(attempted)
        assert response.routing.policy_version == policy
        if policy == 'recall-v1':
            assert response.routing.reason == 'insufficient_lexical_evidence'
        # Explicit constraints take priority over both optional controls.
        before = encoder.query_calls
        constrained = engine.search(SearchRequest('phone AND NOT camera', mode='adaptive'))
        assert constrained.routing.reason == 'explicit_constraints'
        assert not constrained.routing.vector_attempted and encoder.query_calls == before
        assert [r.doc_id for r in constrained.results] == ['z']


def test_policy_controls_preserve_fallback_forced_modes_and_candidate_limit():
    for policy in ('recall-v1', 'hybrid-v1'):
        engine, _, vectors = _setup(RetrievalConfig(adaptive_policy=policy, candidate_k=2))
        lexical = engine.search(SearchRequest('phone', mode='lexical', k=3))
        assert not lexical.routing.vector_attempted
        def fail(*_):
            raise RetrievalUnavailable('backend unavailable')
        vectors.search = fail
        response = engine.search(SearchRequest('phone', mode='adaptive', k=2))
        assert response.routing.fallback and response.routing.reason == 'vector_unavailable'
        assert response.routing.executed_mode == 'lexical'
        _raises(RetrievalUnavailable, lambda: engine.search(SearchRequest('phone', mode='hybrid', k=2)))
        _raises(ApiError, lambda: engine.search(SearchRequest('phone', mode='adaptive', k=2, offset=1)))
    _raises(ValueError, lambda: RetrievalConfig(adaptive_policy='typo'))


def test_recall_control_preserves_sufficient_multi_hit_rules():
    router = RuleRouter(RetrievalConfig(adaptive_policy='recall-v1'))
    assert not router.decide(QueryFeatures(False, 2, 1., 3, .5)).use_vector
    assert router.decide(QueryFeatures(False, 2, .5, 3, .5)).reason == 'low_vocab_coverage'
    assert router.decide(QueryFeatures(False, 2, 1., 3, .01)).reason == 'small_relative_gap'


def test_archived_model_is_evaluation_only_and_matches_frozen_linear_formula():
    path = Path(__file__).resolve().parents[1]/'evaluation/router_evidence/vector-first-du-v1.json'
    model = CascadeRouter(path)
    features = dict(zip(FEATURES, model.model['mean']))
    assert math.isclose(model.predict(features), model.model['intercept'], abs_tol=1e-14)
    features['query_chars'] += model.model['scale'][0]
    assert math.isclose(model.predict(features), model.model['intercept']+model.model['coef'][0], abs_tol=1e-14)
    features['hits'] = 0
    assert model.use_lexical(features)
    features['top1'] = float('nan')
    _raises(ValueError, lambda: model.predict(features))
    with tempfile.TemporaryDirectory() as tmp:
        malformed = dict(model.model, scale=[0.]*len(FEATURES))
        invalid = Path(tmp)/'invalid.json'
        invalid.write_text(json.dumps(malformed))
        _raises(ValueError, lambda: CascadeRouter(invalid))


ALL_TESTS = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
