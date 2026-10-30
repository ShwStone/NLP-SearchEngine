"""Online-visible statistics, opt-in policy isolation and safe JSON model tests."""

import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

from contracts.api import ApiError, SearchRequest
from contracts.retrieval import RetrievalUnavailable
from online.engine import SearchEngine
from online.gain_router import GainRouter
from online.router import RouteDecision
from online.router_features import extract_statistics, FEATURE_NAMES, FEATURE_VERSION
from tests.test_hybrid import _setup, _raises


def test_jsonl_preserves_unicode_separators_inside_documents():
    from evaluation.prepare_bm25_router import jsonl_rows
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'corpus.jsonl'
        rows=[{'doc_id':'a','text':'心肌\u2028梗死\u0085急救'}, {'doc_id':'b','text':'正常正文'}]
        path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
        assert list(jsonl_rows(path))==rows


def _artifact(engine):
    return dict(schema_version=1, feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES),
                selected_features=list(FEATURE_NAMES), threshold=0., kind='linear',
                mean=[0.]*len(FEATURE_NAMES), scale=[1.]*len(FEATURE_NAMES),
                coef=[0.]*len(FEATURE_NAMES), intercept=.1,
                provenance={'source':dict(source_manifest=engine.reader.manifest(), candidate_k=200, rrf_c=60.,
                                          vector_manifest=engine.vectors.manifest().to_dict(),
                                          vector_search_params={'hnsw_ef':128,'exact':False})})


def test_statistics_are_finite_and_do_not_need_dense_or_labels():
    engine, encoder, _ = _setup()
    before = encoder.query_calls
    values = extract_statistics('phone phone',engine.reader,['phone','phone'],[(0,2.)])
    assert tuple(values)==FEATURE_NAMES
    assert values['n_terms']==1 and values['vocab_coverage']==1
    assert values['relative_gap']==0 and values['top1_share10']==1
    assert all(math.isfinite(v) for v in values.values())
    assert encoder.query_calls==before
    empty = extract_statistics('',engine.reader,[],[])
    assert empty['candidate_hits']==0 and empty['entropy10']==0
    _raises(ValueError,lambda:extract_statistics('a',engine.reader,['a'],[(0,float('nan'))]))


def test_injected_router_preserves_constraints_request_validation_and_fallback():
    engine, encoder, vectors = _setup()
    class Probe:
        policy_version='test-only-statistics'
        requires_statistics=True
        calls=0
        def decide(self,features):
            self.calls+=1
            assert tuple(features.statistics)==FEATURE_NAMES
            return RouteDecision(True,'probe')
    router=Probe()
    experiment=SearchEngine(engine.reader,encoder,vectors,router=router)
    response=experiment.search(SearchRequest('phone',mode='adaptive'))
    assert response.routing.vector_used and response.routing.policy_version==router.policy_version
    assert router.calls==1
    experiment.search(SearchRequest('phone AND NOT camera',mode='adaptive'))
    _raises(ApiError,lambda:SearchRequest('phone',mode='adaptive',scorer='tfidf'))
    _raises(ApiError,lambda:SearchRequest('phone',mode='adaptive',expander='rm3'))
    assert router.calls==1
    def fail(*_):raise RetrievalUnavailable('backend timeout')
    vectors.search=fail
    result=experiment.search(SearchRequest('phone',mode='adaptive'))
    assert result.routing.fallback and result.routing.reason=='vector_unavailable'


def test_gain_router_binding_and_malformed_model_rejection():
    engine, encoder, vectors = _setup()
    vectors.search_params={'hnsw_ef':128,'exact':False}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'router.json'
        path.write_text(json.dumps(_artifact(engine)))
        router=GainRouter(path)
        experiment=SearchEngine(engine.reader,encoder,vectors,router=router)
        assert experiment.search(SearchRequest('phone',mode='adaptive')).routing.vector_used
        with patch.object(vectors,'search_params',{'hnsw_ef':32,'exact':False}):
            _raises(ValueError,lambda:SearchEngine(engine.reader,encoder,vectors,router=router))
        for change in ({'feature_version':'vector-first-v1'},{'scale':[0.]*len(FEATURE_NAMES)},
                       {'selected_features':['dense_score']},{'threshold':float('nan')}):
            artifact=_artifact(engine);artifact.update(change);path.write_text(json.dumps(artifact))
            _raises(ValueError,lambda:GainRouter(path))
        artifact=_artifact(engine)
        artifact.update(kind='tree',feature=[0],left=[0],right=[0],split=[.5],value=[0.])
        path.write_text(json.dumps(artifact))
        _raises(ValueError,lambda:GainRouter(path))


def test_v2_coverage_and_online_feature_dispatch_are_consistent():
    from online.router_features import extract_statistics_v2, FEATURE_NAMES_V2, FEATURE_VERSION_V2
    engine, encoder, vectors = _setup()
    values=extract_statistics_v2('phone water missing',engine.reader,
                                 ['phone','water','missing'],[(0,2.),(1,1.)])
    assert tuple(values)==FEATURE_NAMES_V2
    assert math.isclose(values['top1_term_coverage'],2/3)
    assert math.isclose(values['top10_term_coverage'],2/3)
    assert math.isclose(values['top10_mean_term_coverage'],1/3)
    assert math.isclose(values['top10_rare_term_coverage'],2/3)
    assert values['top10_all_terms']==0
    vectors.search_params={'hnsw_ef':128,'exact':False}
    artifact=_artifact(engine)
    artifact.update(feature_version=FEATURE_VERSION_V2,feature_names=list(FEATURE_NAMES_V2),
                    selected_features=['top1_term_coverage'],mean=[0.],scale=[1.],coef=[1.],intercept=0.,threshold=.9)
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'v2.json';path.write_text(json.dumps(artifact))
        router=GainRouter(path)
        experiment=SearchEngine(engine.reader,encoder,vectors,router=router)
        assert experiment.search(SearchRequest('phone',mode='adaptive')).routing.vector_attempted
        assert not experiment.search(SearchRequest('phone absent',mode='adaptive')).routing.vector_attempted
        constrained=experiment.search(SearchRequest('phone AND NOT camera',mode='adaptive'))
        assert not constrained.routing.vector_attempted
        broken=dict(artifact,provenance={'source':{'index_manifest':engine.reader.manifest()}})
        path.write_text(json.dumps(broken))
        _raises(ValueError,lambda:GainRouter(path))


def test_v2_coverage_does_not_scan_whole_postings_blocks():
    from online.router_features import extract_statistics_v2
    class LookupOnly(list):
        def __iter__(self):
            raise AssertionError('coverage scanned every posting')
    class Reader:
        n_docs=5000
        values=LookupOnly((i,1,[0]) for i in range(5000))
        def df(self,term):return 5000
        def postings(self,term):return self.values
    values=extract_statistics_v2('term',Reader(),['term'],[(999,1.),(2500,.5)])
    assert values['top1_term_coverage']==values['top10_term_coverage']==1.


ALL_TESTS=[value for name,value in sorted(globals().items()) if name.startswith('test_')]
