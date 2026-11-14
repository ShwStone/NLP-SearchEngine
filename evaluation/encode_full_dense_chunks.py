"""Bounded-memory, resumable document encoding; no Qdrant index is claimed."""
import argparse
from dataclasses import asdict
from itertools import islice
import json
from pathlib import Path
import sys
import time
import numpy as np
from contracts.index_format import sha256_of
from evaluation.artifact_lock import exclusive_artifact_lock
from evaluation.prepare_bm25_router import jsonl_rows


def encode_shard(source, out, encoder, chunk_size=1024):
    if chunk_size < 1:
        raise ValueError('invalid chunk size')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'source_sha256': sha256_of(source), 'embedding': asdict(encoder.spec),
                'chunk_size': chunk_size, 'dtype': 'float32'}
    chunks = []
    count = 0
    rows = iter(jsonl_rows(source))
    while batch := list(islice(rows, chunk_size)):
        number = len(chunks)
        stem = f'chunk-{number:05d}'
        vector_path = out / (stem + '.npy')
        ids_path = out / (stem + '.ids.json')
        metadata = out / (stem + '.json')
        ids = [r['doc_id'] for r in batch]
        expected = dict(identity, offset=count, documents=len(batch))
        if metadata.exists():
            record = json.loads(metadata.read_text(encoding='utf-8'))
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError('chunk identity mismatch')
            if (sha256_of(vector_path) != record['vectors_sha256'] or
                    sha256_of(ids_path) != record['ids_sha256']
                    or json.loads(ids_path.read_text(encoding='utf-8')) != ids):
                raise ValueError('chunk corruption or ID mismatch')
            vectors = np.load(vector_path, mmap_mode='r', allow_pickle=False)
        else:
            values = encoder.encode_documents([r['title'] + '\n' + r['text'] for r in batch])
            vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (len(batch), encoder.spec.dimension) or vectors.dtype != np.float32:
            raise ValueError('vector shape or dtype mismatch')
        if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            raise ValueError('nonfinite or nonunit embeddings')
        if not metadata.exists():
            pending = vector_path.with_suffix('.npy.pending')
            with pending.open('wb') as handle:
                np.save(handle, vectors, allow_pickle=False)
            pending.replace(vector_path)
            ids_pending = ids_path.with_suffix('.pending')
            ids_pending.write_text(json.dumps(ids, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')
            ids_pending.replace(ids_path)
            record = dict(expected, vectors_sha256=sha256_of(vector_path), ids_sha256=sha256_of(ids_path))
            pending = metadata.with_suffix('.pending')
            pending.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n',
                               encoding='utf-8', newline='\n')
            pending.replace(metadata)
        chunks.append(record)
        count += len(batch)
        print(Path(source).name, 'encoded or verified', count, flush=True)
    report = dict(identity, documents=count, chunks=chunks, qdrant_built=False)
    pending = out / 'encoding-summary.pending'
    pending.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    pending.replace(out / 'encoding-summary.json')
    return report


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('corpus-dir', 'out', 'model-path', 'vector-manifest', 'adapter-root'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--chunk-size', type=int, default=1024)
    p.add_argument('--limit-shards', type=int)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with exclusive_artifact_lock(a.out / '.encode.lock'):
        sys.path.append(str(a.adapter_root.resolve()))
        from adapters.embedding import BGEEncoder
        from contracts.retrieval import VectorManifest
        spec = VectorManifest.from_dict(json.loads(a.vector_manifest.read_text(encoding='utf-8'))).embedding
        encoder = BGEEncoder(spec, str(a.model_path), a.device, batch_size=32)
        parts = sorted(a.corpus_dir.glob('part-*'))
        if a.limit_shards is not None:
            parts = parts[:a.limit_shards]
        started = time.monotonic()
        for part in parts:
            encode_shard(part, a.out / part.name, encoder, a.chunk_size)
        print('encoded selected shards', len(parts), 'seconds', time.monotonic() - started, flush=True)


if __name__ == '__main__':
    main()
