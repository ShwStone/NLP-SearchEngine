"""Verify source/index document parity and exact full-corpus shard concatenation."""
import argparse
import hashlib
from itertools import zip_longest
import json
from pathlib import Path

from contracts.index_format import sha256_of
from evaluation.build_full_sharded_bm25 import verified
from evaluation.prepare_bm25_router import jsonl_rows


def audit(corpus_dir, index_dir, full_corpus, expected_documents, output):
    parts = sorted(Path(corpus_dir).glob('part-*'))
    if not parts or expected_documents < 1:
        raise ValueError('missing shards or invalid expected document count')
    concatenated = hashlib.sha256()
    result = []
    total = 0
    for number, source in enumerate(parts):
        directory = Path(index_dir) / source.name
        manifest = verified(directory, f'dureader-full-shard-{number:03d}')
        count = 0
        for local_id, pair in enumerate(zip_longest(jsonl_rows(source),
                                                   jsonl_rows(directory / 'docs.jsonl'))):
            original, indexed = pair
            if original is None or indexed is None:
                raise ValueError(f'{source.name}: source/index length mismatch')
            if indexed.get('docid') != local_id or type(indexed.get('docid')) is not int:
                raise ValueError(f'{source.name}: internal document order mismatch')
            for key in ('doc_id', 'title', 'text'):
                if original.get(key) != indexed.get(key):
                    raise ValueError(f'{source.name}: source/index {key} mismatch at {local_id}')
            count += 1
        if count != manifest.n_docs:
            raise ValueError(f'{source.name}: manifest document count mismatch')
        with source.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                concatenated.update(chunk)
        result.append({'shard': source.name, 'offset': total, 'documents': count,
                       'source_sha256': sha256_of(source),
                       'manifest_sha256': sha256_of(directory / 'manifest.json')})
        total += count
        print('source parity', number + 1, '/', len(parts), flush=True)
    full_digest = sha256_of(full_corpus)
    if concatenated.hexdigest() != full_digest:
        raise ValueError('shard concatenation differs from full corpus')
    if total != expected_documents:
        raise ValueError('unexpected full-corpus document count')
    report = {'status': 'verified', 'documents': total, 'shards': result,
              'full_corpus_sha256': full_digest,
              'source_record_parity_verified': True,
              'shard_concatenation_verified': True,
              'scope': 'source/document parity and artifact hashes; not postings semantic conformance'}
    with Path(output).open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('corpus-dir', 'index-dir', 'full-corpus', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--expected-documents', type=int, required=True)
    audit(**vars(p.parse_args()))
