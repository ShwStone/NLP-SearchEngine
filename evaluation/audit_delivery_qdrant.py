"""Read back every Qdrant point ID and external ID against ordered corpus.

Does not certify vector values. Uses bounded scroll batches and never reads
query text or relevance labels. Report is published only after full coverage.
"""
import argparse
import hashlib
import json
from pathlib import Path

from contracts.retrieval import VectorManifest
from online.vector_index import QdrantVectorIndex


def audit(corpus, manifest_path, url, output, batch_size=4096):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('invalid batch size')
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = VectorManifest.from_dict(json.loads(manifest_bytes))
    vector = QdrantVectorIndex(manifest, url)
    client = vector._client
    before = client.get_collection(manifest.collection)
    if (str(before.status).lower().split('.')[-1] != 'green' or
            before.indexed_vectors_count != manifest.n_docs):
        raise ValueError('collection not fully indexed or not green')
    offset, count = None, 0
    raw, fingerprint = hashlib.sha256(), hashlib.sha256()
    with Path(corpus).open('rb') as handle:
        while True:
            points, next_offset = client.scroll(
                collection_name=manifest.collection, offset=offset, limit=batch_size,
                with_payload=['doc_id'], with_vectors=False)
            for point in points:
                line = handle.readline()
                raw.update(line)
                while line and not line.strip():
                    line = handle.readline()
                    raw.update(line)
                if not line:
                    raise ValueError('server has more points than corpus')
                doc = json.loads(line)
                if (type(point.id) is not int or point.id != count or
                        (point.payload or {}).get('doc_id') != doc['doc_id']):
                    raise ValueError(f'server ID/payload mismatch at {count}')
                row = [count, doc['doc_id'], doc['title'], doc['text']]
                fingerprint.update((json.dumps(row, ensure_ascii=False,
                                               separators=(',', ':')) + '\n').encode('utf-8'))
                count += 1
            if next_offset is None:
                break
            if not points or next_offset == offset or count > manifest.n_docs:
                raise ValueError('invalid/nonprogressing scroll')
            offset = next_offset
            if count % (batch_size * 100) == 0:
                print('read back aligned payloads', count, flush=True)
        for line in handle:
            raw.update(line)
            if line.strip():
                raise ValueError('corpus has more records than server')
    if count != manifest.n_docs or fingerprint.hexdigest() != manifest.corpus_sha256:
        raise ValueError('incomplete server coverage or corpus fingerprint mismatch')
    # Recheck build metadata and exact count after the scan.
    QdrantVectorIndex(manifest, url)
    report = {'status': 'verified', 'documents': count,
              'corpus_file_sha256': raw.hexdigest(),
              'corpus_contract_fingerprint': fingerprint.hexdigest(),
              'vector_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'all_server_point_ids_and_payload_ids_verified': True,
              'indexed_vectors_count': before.indexed_vectors_count,
              'vectors_numerically_verified': False,
              'scope': 'all ordered corpus fields fingerprinted; every server ID/payload read back'}
    with output.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('corpus', 'manifest-path', 'url', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--batch-size', type=int, default=4096)
    print(json.dumps(audit(**vars(parser.parse_args())), indent=2))
