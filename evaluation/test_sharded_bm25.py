import json
import pytest
from contracts import analyzer
from offline.build import build
from offline.disk_index import DiskIndex
from online.scorers import bm25
from evaluation.sharded_bm25 import shard_topk, merge_topk


def test_global_shards_match_monolithic_with_unequal_df_and_lengths(tmp_path):
    corpus = [dict(doc_id=str(i), title='', text=text) for i, text in enumerate([
        '苹果 苹果 香蕉', '香蕉', '苹果', '汽车 汽车 汽车 汽车', '香蕉 汽车', '苹果 香蕉 汽车'])]
    def index(name, rows):
        source = tmp_path / (name + '.jsonl')
        source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        build(source, tmp_path / name, name)
        return DiskIndex(tmp_path / name)
    whole = index('whole', corpus)
    shards = [index('left', corpus[:2]), index('right', corpus[2:])]
    for query in ['苹果', '香蕉 汽车', '苹果 苹果 汽车', '不存在']:
        terms = [t.term for t in analyzer.analyze(query)]
        dfs = {t: sum(s.df(t) for s in shards) for t in terms}
        avgdl = sum(s.n_docs * s.avgdl for s in shards) / whole.n_docs
        for k in (1, 3, 10):
            left = shard_topk(shards[0], terms, n_docs=whole.n_docs, avgdl=avgdl,
                             dfs=dfs, offset=0, k=k)
            right = shard_topk(shards[1], terms, n_docs=whole.n_docs, avgdl=avgdl,
                              dfs=dfs, offset=2, k=k)
            got = merge_topk([left, right], k)
            expected = sorted(bm25(whole, terms).items(), key=lambda x: (-x[1], x[0]))[:k]
            assert [d for d, _ in got] == [d for d, _ in expected]
            assert [s for _, s in got] == pytest.approx([s for _, s in expected], abs=1e-12)
    with pytest.raises(KeyError):
        shard_topk(shards[0], ['missing'], n_docs=6, avgdl=1, dfs={}, offset=0, k=1)


def test_resume_rejects_corrupt_completed_shard(tmp_path):
    from evaluation.build_full_sharded_bm25 import verified
    corpus = tmp_path / 'source.jsonl'
    corpus.write_text(json.dumps(dict(doc_id='a', title='', text='苹果')) + '\n')
    output = tmp_path / 'index'
    build(corpus, output, 'dureader-full-shard-000')
    assert verified(output, 'dureader-full-shard-000').n_docs == 1
    with (output / 'postings.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='corrupt'):
        verified(output, 'dureader-full-shard-000')


def test_streaming_global_statistics_and_incomplete_build_rejection(tmp_path):
    from evaluation.prepare_sharded_bm25_statistics import prepare
    source = tmp_path / 'sources'
    indices = tmp_path / 'indices'
    source.mkdir()
    all_rows = []
    for i, texts in enumerate([['苹果 苹果', '香蕉'], ['汽车', '苹果 香蕉 汽车']]):
        rows = [dict(doc_id=f'{i}-{j}', title='', text=text) for j, text in enumerate(texts)]
        all_rows.extend(rows)
        path = source / f'part-{i:03d}'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        build(path, indices / path.name, f'dureader-full-shard-{i:03d}')
    queries = tmp_path / 'queries.jsonl'
    queries.write_text(json.dumps(dict(query_id='q', text='苹果 汽车 不存在')) + '\n')
    full = tmp_path / 'all.jsonl'
    full.write_text(''.join(json.dumps(r) + '\n' for r in all_rows))
    build(full, tmp_path / 'whole', 'whole')
    whole = DiskIndex(tmp_path / 'whole')
    stats = prepare(source, indices, queries, tmp_path / 'global.json')
    assert stats['n_docs'] == whole.n_docs
    assert stats['avgdl'] == whole.avgdl
    assert stats['dfs'] == {t.term: whole.df(t.term) for t in analyzer.analyze('苹果 汽车 不存在')}
    assert [s['offset'] for s in stats['shards']] == [0, 2]
    (source / 'part-002').write_text('{}\n')
    with pytest.raises(FileNotFoundError):
        prepare(source, indices, queries, tmp_path / 'incomplete.json')
    assert not (tmp_path / 'incomplete.json').exists()


def test_source_audit_rejects_wrong_corpus_even_when_index_hashes_pass(tmp_path):
    from evaluation.audit_sharded_bm25_sources import audit
    source = tmp_path / 'sources'
    source.mkdir()
    part = source / 'part-000'
    part.write_text(json.dumps(dict(doc_id='a', title='', text='苹果')) + '\n')
    full = tmp_path / 'full.jsonl'
    full.write_bytes(part.read_bytes())
    indices = tmp_path / 'indices'
    build(part, indices / part.name, 'dureader-full-shard-000')
    result = audit(source, indices, full, 1, tmp_path / 'ok.json')
    assert result['source_record_parity_verified']
    # Corrupt the source, leaving an internally valid index and manifest.
    part.write_text(json.dumps(dict(doc_id='a', title='', text='香蕉')) + '\n')
    with pytest.raises(ValueError, match='text mismatch'):
        audit(source, indices, full, 1, tmp_path / 'wrong.json')
    assert not (tmp_path / 'wrong.json').exists()
    part.write_bytes(full.read_bytes())
    full.write_bytes(full.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='concatenation'):
        audit(source, indices, full, 1, tmp_path / 'different.json')


def test_capture_matches_monolithic_engine_including_explicit_constraints(tmp_path):
    from evaluation.prepare_sharded_bm25_statistics import prepare
    from evaluation.capture_sharded_bm25 import capture
    from evaluation.prepare_bm25_router import jsonl_rows
    from online.engine import SearchEngine
    from contracts.api import SearchRequest
    source = tmp_path / 'sources'
    source.mkdir()
    indices = tmp_path / 'indices'
    all_rows = []
    for i, texts in enumerate([['苹果 香蕉', '香蕉'], ['汽车', '苹果 苹果', '香蕉 汽车']]):
        rows = [dict(doc_id=f'{i}-{j}', title='', text=text) for j, text in enumerate(texts)]
        all_rows.extend(rows)
        part = source / f'part-{i:03d}'
        part.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        build(part, indices / part.name, f'dureader-full-shard-{i:03d}')
    full = tmp_path / 'full.jsonl'
    full.write_text(''.join(json.dumps(r) + '\n' for r in all_rows))
    build(full, tmp_path / 'whole', 'whole')
    engine = SearchEngine(DiskIndex(tmp_path / 'whole'))
    queries = tmp_path / 'queries.jsonl'
    queries.write_text(''.join(json.dumps(dict(query_id=str(i), text=q)) + '\n'
                               for i, q in enumerate(['苹果', '苹果 AND 香蕉', 'NOT 苹果', '"苹果 香蕉"'])))
    stats = tmp_path / 'global.json'
    prepare(source, indices, queries, stats)
    capture(indices, queries, stats, tmp_path / 'capture', k=100)
    for row in jsonl_rows(tmp_path / 'capture/capture.jsonl'):
        expected = engine.search(SearchRequest(query=row['text'], mode='lexical', k=100))
        assert [x['doc_id'] for x in row['lexical']] == [x.doc_id for x in expected.results]
        assert [x['score'] for x in row['lexical']] == pytest.approx([x.score for x in expected.results])
    queries.write_text(queries.read_text() + '\n')
    with pytest.raises(ValueError, match='query bytes'):
        capture(indices, queries, stats, tmp_path / 'wrong-capture')
