"""Prepare exact corpus-wide statistics for a declared set of query terms.

All declared shards must be complete. One JSONL posting row at a time is read;
this avoids loading all shard vocabularies or readers simultaneously.
"""
import argparse
import json
from pathlib import Path

from contracts import analyzer
from online.parser import parse
from contracts.index_format import sha256_of
from evaluation.build_full_sharded_bm25 import verified
from evaluation.prepare_bm25_router import jsonl_rows


def prepare(corpus_dir, index_dir, queries, output):
    parts = sorted(Path(corpus_dir).glob('part-*'))
    if not parts:
        raise ValueError('no declared shards')
    terms = {t.term for row in jsonl_rows(queries)
             for raw in parse(row.get('query', row.get('text', '')))['terms']
             for t in analyzer.analyze(raw)}
    dfs = dict.fromkeys(terms, 0)
    count = total_length = 0
    provenance = []
    for number, part in enumerate(parts):
        directory = Path(index_dir) / part.name
        manifest = verified(directory, f'dureader-full-shard-{number:03d}')
        stats = json.loads((directory / 'stats.json').read_text())
        lengths = stats['doc_lengths']
        if set(lengths) != {str(i) for i in range(manifest.n_docs)}:
            raise ValueError('document length coverage mismatch')
        if any(type(v) is not int or v < 0 for v in lengths.values()):
            raise ValueError('invalid document length')
        shard_length = sum(lengths.values())
        if abs(shard_length / manifest.n_docs - manifest.avgdl) > 1e-9:
            raise ValueError('average document length mismatch')
        for row in jsonl_rows(directory / 'postings.jsonl'):
            term = row['term']
            if term in terms:
                if row['df'] != len(row['docs']):
                    raise ValueError('posting df mismatch')
                dfs[term] += row['df']
        provenance.append({'shard': part.name, 'offset': count,
                           'documents': manifest.n_docs,
                           'manifest_sha256': sha256_of(directory / 'manifest.json'),
                           'source_sha256': sha256_of(part)})
        count += manifest.n_docs
        total_length += shard_length
        print('statistics', number + 1, '/', len(parts), flush=True)
    result = {'schema_version': 1, 'n_docs': count, 'total_document_length': total_length,
              'avgdl': total_length / count, 'dfs': dfs, 'shards': provenance,
              'queries_sha256': sha256_of(queries), 'analyzer_hash': analyzer.config_hash(),
              'scope': 'query-term global statistics; not a full vocabulary',
              'source_record_parity_verified': False}
    # No output is published when any declared shard fails verification.
    output = Path(output)
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('corpus-dir', 'index-dir', 'queries', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a.corpus_dir, a.index_dir, a.queries, a.output)
