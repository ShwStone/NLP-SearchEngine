import json
import sys
from types import SimpleNamespace

import pytest

from contracts.retrieval import EmbeddingSpec, VectorManifest
from evaluation import capture_full_dense as mod


@pytest.mark.parametrize('duplicate', [False, True])
def test_explicit_parameters_unicode_and_duplicate_point_rejection(tmp_path, monkeypatch, duplicate):
    spec = EmbeddingSpec('fixture', 'pinned', 2)
    manifest = VectorManifest('a'*64, 3, spec, 'fixture')
    mp, qp = tmp_path/'manifest', tmp_path/'queries'
    mp.write_text(json.dumps(manifest.to_dict()))
    model = tmp_path/'model'; model.mkdir()
    (model/'model_inventory.json').write_text('{}')
    rows = [{'query_id':'q1','text':'苹果\u2028香蕉'},
            {'query_id':'q2','text':'苹果 AND 香蕉'}]
    qp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    calls = []
    class Encoder:
        def __init__(self, *args, **kwargs):
            pass
        def encode_query(self, text):
            assert text == rows[0]['text']
            return [1., 0.]
    class Client:
        def query_points(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(points=[
                SimpleNamespace(id=0, payload={'doc_id':'a'}, score=.9),
                SimpleNamespace(id=0 if duplicate else 1, payload={'doc_id':'b'}, score=.8)])
    monkeypatch.setitem(sys.modules, 'adapters.embedding', SimpleNamespace(BGEEncoder=Encoder))
    monkeypatch.setitem(sys.modules, 'qdrant_client',
                        SimpleNamespace(models=SimpleNamespace(SearchParams=SimpleNamespace)))
    monkeypatch.setattr(mod, 'QdrantVectorIndex', lambda *args: SimpleNamespace(_client=Client()))
    output = tmp_path/'capture'
    if duplicate:
        with pytest.raises(ValueError, match='invalid dense hit'):
            mod.capture(qp, mp, model, 'fixture', output, k=2)
        assert not (output/'protocol.json').exists()
    else:
        result = mod.capture(qp, mp, model, 'fixture', output, k=2)
        assert result['queries'] == 2
        assert result['explicit_dense_skipped'] == 1
        assert result['qrels_read'] is False
        assert len(calls) == 1
        assert vars(calls[0]['search_params']) == {'hnsw_ef':128,'exact':False}
        saved = list(mod.jsonl_rows(output/'capture.jsonl'))
        assert saved[0]['text'] == rows[0]['text']
        assert saved[1]['dense'] == []
