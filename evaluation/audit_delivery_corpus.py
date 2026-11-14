"""Bind normalized full-corpus bytes to the vector contract fingerprint.

Streams records without loading the full corpus. This verifies input identity,
not the actual server payloads or vectors; those require a separate readback.
"""
import argparse
import hashlib
import json
from pathlib import Path

from contracts.retrieval import VectorManifest


def audit(corpus, manifest_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = VectorManifest.from_dict(json.loads(manifest_bytes))
    raw, fingerprint = hashlib.sha256(), hashlib.sha256()
    count = 0
    with Path(corpus).open('rb') as handle:
        for line in handle:
            raw.update(line)
            if not line.strip():
                continue
            record = json.loads(line)
            if (not isinstance(record, dict) or
                    any(not isinstance(record.get(k), str)
                        for k in ('doc_id', 'title', 'text')) or not record['doc_id']):
                raise ValueError(f'invalid document at ordinal {count}')
            row = [count, record['doc_id'], record['title'], record['text']]
            fingerprint.update((json.dumps(row, ensure_ascii=False,
                                           separators=(',', ':')) + '\n').encode('utf-8'))
            count += 1
            if count % 1000000 == 0:
                print('fingerprinted', count, flush=True)
    if count != manifest.n_docs:
        raise ValueError(f'document count mismatch: {count} != {manifest.n_docs}')
    if fingerprint.hexdigest() != manifest.corpus_sha256:
        raise ValueError('vector manifest corpus fingerprint mismatch')
    result = {
        'status': 'verified', 'documents': count,
        'corpus_file_sha256': raw.hexdigest(),
        'corpus_contract_fingerprint': fingerprint.hexdigest(),
        'vector_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
        'server_payloads_verified': False, 'server_vectors_verified': False,
        'scope': 'ordered normalized corpus identity against vector manifest only',
    }
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('corpus', 'manifest-path', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
