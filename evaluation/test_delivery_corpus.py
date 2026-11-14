import hashlib
import json
from types import SimpleNamespace

import pytest

from contracts.retrieval import EmbeddingSpec, VectorManifest, corpus_fingerprint
from evaluation.audit_delivery_corpus import audit


def test_contract_parity_order_and_mutation_fail_closed(tmp_path):
    docs = [SimpleNamespace(doc_id='甲', title='标题', text='第一篇'),
            SimpleNamespace(doc_id='乙', title='', text='第二篇')]
    reader = SimpleNamespace(n_docs=len(docs), doc=lambda i: SimpleNamespace(docid=i, **vars(docs[i])))
    spec = EmbeddingSpec('fixture', 'pinned', 2)
    manifest = VectorManifest(corpus_fingerprint(reader), 2, spec, 'fixture')
    mp, cp, out = (tmp_path / x for x in ('manifest', 'corpus', 'report'))
    mp.write_text(json.dumps(manifest.to_dict()))
    def write(rows):
        cp.write_text(''.join(json.dumps(vars(d), ensure_ascii=False) + '\n' for d in rows), encoding='utf-8')
    write(docs)
    result = audit(cp, mp, out)
    assert result['corpus_contract_fingerprint'] == corpus_fingerprint(reader)
    assert result['corpus_file_sha256'] == hashlib.sha256(cp.read_bytes()).hexdigest()
    assert not result['server_payloads_verified']
    with pytest.raises(FileExistsError):
        audit(cp, mp, out)
    write(docs[::-1])
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        audit(cp, mp, tmp_path / 'reorder')
    write(docs[:1])
    with pytest.raises(ValueError, match='count mismatch'):
        audit(cp, mp, tmp_path / 'missing')
    cp.write_text(json.dumps({'doc_id': '甲', 'title': '', 'text': None}) + '\n')
    with pytest.raises(ValueError, match='invalid document'):
        audit(cp, mp, tmp_path / 'invalid')
    assert not (tmp_path / 'reorder').exists()
    assert not (tmp_path / 'missing').exists()
    assert not (tmp_path / 'invalid').exists()
