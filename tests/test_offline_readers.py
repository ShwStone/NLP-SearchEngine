"""Independent reader parity and streaming writer regressions on the base engine."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
from unittest.mock import patch

from contracts.api import SearchRequest
from contracts.conformance import check_reader
from contracts.retrieval import VectorRecord
from offline.build import build
from offline.compact import CompactIndex, compile_index
from offline.disk_index import DiskIndex
from offline.packed_index import PackedIndex
from offline.vector_build import QdrantSnapshotWriter
from online.engine import SearchEngine
from tests.test_qdrant import _Client, _manifest, _raises, _sdk


def test_compact_and_packed_conformance_and_base_engine_parity():
    corpus = [dict(doc_id=str(i), title='', text=text) for i, text in enumerate([
        'alpha beta beta gamma', 'alpha gamma', '😀心肌梗死急救。',
        'delta beta', 'alpha alpha beta', '肺炎喘息。', 'zero negative'])]
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / 'corpus.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in corpus))
        build(path, root/'json', 'reader-parity')
        compile_index(root/'json', root/'compact')
        reference = DiskIndex(root/'json')
        with CompactIndex(root/'compact') as compact, PackedIndex(root/'compact', cache_bytes=2048) as packed:
            for reader in (compact, packed):
                check_reader(reader, corpus)
                assert reader.manifest() == reference.manifest()
                engine, baseline = SearchEngine(reader), SearchEngine(reference)
                requests = [SearchRequest(q, k=3) for q in (
                    'alpha beta', '心肌梗死', 'NOT alpha', 'alpha AND NOT gamma',
                    '"alpha beta"', 'absent')]
                requests += [SearchRequest('alpha', expander='rm3', k=3),
                             SearchRequest('alpha beta', scorer='tfidf', k=2, offset=1)]
                for req in requests:
                    actual, expected = engine.search(req), baseline.search(req)
                    assert [r.to_dict() for r in actual.results] == [r.to_dict() for r in expected.results]
                    assert actual.expanded == expected.expanded
                with ThreadPoolExecutor(max_workers=4) as pool:
                    responses = list(pool.map(engine.search, requests*2))
                assert len(responses) == len(requests)*2
            assert packed.cache_info()['cached_bytes'] <= 2048
        _raises(ValueError, lambda: compile_index(root/'json', root/'compact'))
        with (root/'compact/postings.bin').open('ab') as file:
            file.write(b'corruption')
        _raises(ValueError, lambda: PackedIndex(root/'compact'))


def test_packed_byte_budget_eviction_and_concurrent_single_load():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        corpus = root/'corpus'
        corpus.write_text(json.dumps(dict(doc_id='a', text='alpha beta gamma delta'))+'\n')
        build(corpus, root/'json', 'cache')
        compile_index(root/'json', root/'compact')
        with PackedIndex(root/'compact', cache_bytes=0) as reader:
            entered, release, joined = threading.Event(), threading.Event(), threading.Event()
            original = reader._load_block
            def controlled(term):
                entered.set()
                assert release.wait(5)
                return original(term)
            with patch.object(reader, '_load_block', side_effect=controlled) as load:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    owner = pool.submit(reader.postings, 'alpha')
                    assert entered.wait(5)
                    future = reader._pending['alpha']
                    result = future.result
                    def waiting(*args, **kwargs):
                        joined.set()
                        return result(*args, **kwargs)
                    with patch.object(future, 'result', side_effect=waiting):
                        follower = pool.submit(reader.postings, 'alpha')
                        assert joined.wait(5)
                        release.set()
                        assert owner.result() == follower.result() == [(0, 1, [0])]
                assert load.call_count == 1
            assert reader.cache_info()['cached_bytes'] == 0
            with patch.object(reader, '_load_block', side_effect=OSError('disk failed')):
                _raises(OSError, lambda: reader.postings('beta'))
            assert reader.cache_info()['pending'] == 0
            assert reader.postings('beta') == [(0, 1, [1])]
        with PackedIndex(root/'compact') as reader:
            reader._budget = reader._load_block('alpha').charge('alpha') + 16
            for term in ('alpha', 'beta', 'gamma', 'delta'):
                reader.postings(term)
            info = reader.cache_info()
            assert info['evictions'] >= 2 and info['terms'] == 1
            assert info['cached_bytes'] <= info['budget_bytes']
            reader.clear_cache()
            assert reader.cache_info()['cached_bytes'] == 0


def test_qdrant_records_are_consumed_incrementally_and_failures_clean_only_new_collection():
    for failure in (None, 'duplicate', 'generator', 'snapshot'):
        client = _Client()
        manifest = replace(_manifest(), collection='new-build', snapshot_files={}, index_params={'m': 16})
        # An unrelated existing collection must survive every failure.
        client.create_collection('old', None, {})
        def records():
            yield VectorRecord(0, 'd0', [1., 0.])
            assert len(client.records['new-build']) == 1  # disproves list(records)
            if failure == 'generator':
                raise RuntimeError('encoder failed')
            yield VectorRecord(0 if failure == 'duplicate' else 1, 'd1', [1., 0.])
        with tempfile.TemporaryDirectory() as tmp, _sdk(client):
            writer = QdrantSnapshotWriter('http://fixture', Path(tmp), batch_size=1)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'snapshot')):
                if failure == 'snapshot':
                    with patch.object(client, 'create_snapshot', side_effect=RuntimeError('snapshot failed')):
                        _raises(RuntimeError, lambda: writer.replace(manifest, records()))
                elif failure:
                    _raises(ValueError if failure == 'duplicate' else RuntimeError,
                            lambda: writer.replace(manifest, records()))
                else:
                    assert writer.replace(manifest, records()).snapshot_files
            assert 'old' in client.collections
            assert ('new-build' in client.collections) == (failure is None)


ALL_TESTS = [value for name, value in sorted(globals().items()) if name.startswith('test_')]
