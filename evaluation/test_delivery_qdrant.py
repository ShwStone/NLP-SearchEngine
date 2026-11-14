import json
from types import SimpleNamespace

import pytest

from contracts.retrieval import EmbeddingSpec, VectorManifest, corpus_fingerprint
from evaluation import audit_delivery_qdrant as mod


@pytest.mark.parametrize('fault', [None, 'payload', 'order', 'missing', 'extra', 'notindexed'])
def test_full_scroll_identity_fail_closed(tmp_path, monkeypatch, fault):
    docs = [SimpleNamespace(doc_id=f'id-{i}', title='', text=f'文档{i}') for i in range(3)]
    reader = SimpleNamespace(n_docs=3, doc=lambda i: SimpleNamespace(docid=i, **vars(docs[i])))
    manifest = VectorManifest(corpus_fingerprint(reader), 3,
                              EmbeddingSpec('fixture', 'pinned', 2), 'fixture')
    cp, mp, out = (tmp_path / n for n in ('corpus', 'manifest', 'report'))
    cp.write_text(''.join(json.dumps(vars(d)) + '\n' for d in docs))
    mp.write_text(json.dumps(manifest.to_dict()))
    points = [SimpleNamespace(id=i, payload={'doc_id': d.doc_id}) for i, d in enumerate(docs)]
    if fault == 'payload':
        points[1].payload['doc_id'] = 'wrong'
    elif fault == 'order':
        points[1].id = 7
    elif fault == 'missing':
        points.pop()
    elif fault == 'extra':
        points.append(SimpleNamespace(id=3, payload={'doc_id': 'extra'}))
    class Client:
        def get_collection(self, collection):
            return SimpleNamespace(status='green', indexed_vectors_count=2 if fault=='notindexed' else 3)
        def scroll(self, *, offset, limit, **kwargs):
            start = offset or 0
            end = min(start+limit, len(points))
            return points[start:end], (end if end < len(points) else None)
    monkeypatch.setattr(mod, 'QdrantVectorIndex',
                        lambda *args: SimpleNamespace(_client=Client()))
    if fault:
        with pytest.raises(ValueError):
            mod.audit(cp, mp, 'fixture', out, batch_size=2)
        assert not out.exists()
    else:
        result = mod.audit(cp, mp, 'fixture', out, batch_size=2)
        assert result['all_server_point_ids_and_payload_ids_verified']
        assert result['documents'] == 3
        assert not result['vectors_numerically_verified']
