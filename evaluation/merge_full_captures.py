"""Join independently captured lexical/dense rankings by frozen query ID.

Explicit constraints remain lexical-only. Equal RRF uses global internal IDs
for ties. No labels are accessed. Inputs must carry verified capture hashes.
"""
import argparse
import json
from pathlib import Path

from contracts.index_format import sha256_of
from evaluation.prepare_bm25_router import jsonl_rows
from online.fusion import rrf


def read_capture(directory):
    directory = Path(directory)
    protocol = json.loads((directory / 'protocol.json').read_text())
    if sha256_of(directory / 'capture.jsonl') != protocol['capture_sha256']:
        raise ValueError('capture SHA256 mismatch')
    rows = list(jsonl_rows(directory / 'capture.jsonl'))
    ids = [r['query_id'] for r in rows]
    if len(rows) != protocol['queries'] or len(ids) != len(set(ids)):
        raise ValueError('capture query count/uniqueness mismatch')
    return rows, protocol


def merge(lexical_dir, dense_dir, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    lexical, lp = read_capture(lexical_dir)
    dense, dp = read_capture(dense_dir)
    if (lp['queries_sha256'] != dp['queries_sha256'] or lp['documents'] != dp['documents'] or
            lp['k'] != dp['k'] or lp['qrels_read'] or dp['qrels_read']):
        raise ValueError('incompatible frozen capture protocols')
    by_query = {r['query_id']: r for r in dense}
    if set(by_query) != {r['query_id'] for r in lexical}:
        raise ValueError('capture query universes differ')
    merged = []
    for lrow in lexical:
        drow = by_query[lrow['query_id']]
        if (lrow['text'] != drow['text'] or
                lrow['explicit_constraints'] != drow['explicit_constraints']):
            raise ValueError('query text/constraint status mismatch')
        external, internal = {}, {}
        for row, mode in [(lrow, 'lexical'), (drow, 'dense')]:
            seen = set()
            for hit in row[mode]:
                i, d = hit['docid'], hit['doc_id']
                if (type(i) is not int or not 0 <= i < lp['documents'] or
                        not isinstance(d, str) or not d or i in seen or
                        (i in external and external[i] != d) or
                        (d in internal and internal[d] != i)):
                    raise ValueError('inconsistent/duplicate global document identity')
                seen.add(i); external[i] = d; internal[d] = i
        if lrow['explicit_constraints']:
            if drow['dense']:
                raise ValueError('explicit constraints cannot use dense')
            hybrid = list(lrow['lexical'])
        else:
            fused = rrf([[h['docid'] for h in lrow['lexical']],
                         [h['docid'] for h in drow['dense']]], c=60.)[:lp['k']]
            hybrid = [{'doc_id': external[i], 'docid': i, 'score': score} for i, score in fused]
        merged.append(dict(lrow, dense=drow['dense'], hybrid=hybrid))
    output.mkdir(parents=True)
    with (output / 'capture.jsonl').open('x', encoding='utf-8') as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    protocol = {'queries': len(merged), 'documents': lp['documents'], 'k': lp['k'],
                'queries_sha256': lp['queries_sha256'],
                'lexical_capture_sha256': lp['capture_sha256'],
                'dense_capture_sha256': dp['capture_sha256'],
                'lexical_protocol_sha256': sha256_of(Path(lexical_dir) / 'protocol.json'),
                'dense_protocol_sha256': sha256_of(Path(dense_dir) / 'protocol.json'),
                'fusion': {'method': 'equal-RRF', 'c': 60., 'tie_break': 'global_internal_docid'},
                'explicit_constraints': 'lexical_only', 'qrels_read': False,
                'capture_sha256': sha256_of(output / 'capture.jsonl'),
                'latency_scope': 'offline merge; no end-to-end latency claim'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('lexical-dir', 'dense-dir', 'output'):
        parser.add_argument('--'+name, required=True)
    print(json.dumps(merge(**vars(parser.parse_args())), indent=2))
