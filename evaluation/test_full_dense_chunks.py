from dataclasses import dataclass
import json
import subprocess
import sys
import pytest
from evaluation.artifact_lock import exclusive_artifact_lock
from evaluation.encode_full_dense_chunks import encode_shard


@dataclass
class Spec:
    dimension: int = 2
    query_prefix: str = '查询：'


class Encoder:
    spec = Spec()
    def __init__(self):
        self.calls = []
    def encode_documents(self, texts):
        self.calls.append(texts)
        return [[1., 0.]] * len(texts)


def test_chunk_resume_binds_source_ids_and_vector_bytes(tmp_path):
    source = tmp_path / 'corpus'
    rows = [dict(doc_id=f'编号{i}', title='标题', text='正文') for i in range(5)]
    source.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                      encoding='utf-8', newline='\n')
    encoder = Encoder()
    out = tmp_path / 'chunks'
    result = encode_shard(source, out, encoder, 2)
    assert result['documents'] == 5
    assert len(encoder.calls) == 3
    assert encoder.calls[0] == ['标题\n正文'] * 2
    assert json.loads((out / 'chunk-00000.ids.json').read_text(encoding='utf-8')) == ['编号0', '编号1']
    encoder.calls.clear()
    encode_shard(source, out, encoder, 2)
    assert not encoder.calls
    (out / 'chunk-00000.npy').write_bytes(b'broken')
    with pytest.raises(ValueError, match='corruption'):
        encode_shard(source, out, encoder, 2)


def test_source_change_is_rejected_before_reusing_chunk(tmp_path):
    source = tmp_path / 'corpus'
    source.write_text(json.dumps(dict(doc_id='a', title='', text='one')) + '\n')
    out = tmp_path / 'out'
    encode_shard(source, out, Encoder())
    source.write_text(json.dumps(dict(doc_id='b', title='', text='two')) + '\n')
    with pytest.raises(ValueError, match='identity'):
        encode_shard(source, out, Encoder())


def test_stream_records_use_global_ids_and_reject_truncated_summary(tmp_path):
    from evaluation.stream_dense_chunks import iter_shard_records
    source = tmp_path / 'corpus'
    source.write_text(''.join(json.dumps(dict(doc_id=str(i), title='', text='text')) + '\n' for i in range(5)))
    out = tmp_path / 'encoded'
    encode_shard(source, out, Encoder(), chunk_size=2)
    records = list(iter_shard_records(source, out, Spec(), global_offset=100))
    assert [r.docid for r in records] == list(range(100, 105))
    assert [r.doc_id for r in records] == [str(i) for i in range(5)]
    summary = json.loads((out / 'encoding-summary.json').read_text(encoding='utf-8'))
    summary['chunks'].pop()
    (out / 'encoding-summary.json').write_text(json.dumps(summary), encoding='utf-8')
    with pytest.raises(ValueError, match='coverage'):
        list(iter_shard_records(source, out, Spec(), global_offset=100))


@pytest.mark.parametrize('name', ('.build.lock', '.encode.lock'))
def test_writer_lock_excludes_a_second_process_and_releases(tmp_path, name):
    path = tmp_path / name
    probe = """import sys
from evaluation.artifact_lock import exclusive_artifact_lock
try:
    with exclusive_artifact_lock(sys.argv[1]):
        pass
except OSError:
    raise SystemExit(5)
"""
    command = [sys.executable, '-B', '-c', probe, str(path)]
    with exclusive_artifact_lock(path):
        assert subprocess.run(command, timeout=10).returncode == 5
    assert subprocess.run(command, timeout=10).returncode == 0
