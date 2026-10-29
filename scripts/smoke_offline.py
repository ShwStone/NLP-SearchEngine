"""Optional real BGE/Qdrant build-and-snapshot-import smoke using tiny fixture.

Starts an isolated local Qdrant process and removes its temporary data on exit.
Never points at an existing server. The import is into the same isolated
process after deleting the collection; it is not a clean-instance restore or
an HNSW performance experiment. Requires the optional offline dependencies.
"""

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import ProxyHandler, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def smoke(binary, model, device):
    import requests
    from qdrant_client import QdrantClient
    from adapters.embedding import BGEEncoder
    from contracts.retrieval import VectorManifest, corpus_fingerprint
    from offline.build_hybrid import build_hybrid
    from offline.disk_documents import DiskDocuments
    from offline.vector_build import QdrantSnapshotWriter
    from online.vector_index import QdrantVectorIndex

    start = time.perf_counter()
    binary, model = Path(binary).resolve(), Path(model).resolve()
    fixture = Path(__file__).resolve().parents[1]/'fixtures/tiny_corpus.jsonl'
    opener = build_opener(ProxyHandler({}))
    with tempfile.TemporaryDirectory(prefix='offline-smoke-') as tmp:
        root = Path(tmp)
        port, grpc_port = free_port(), free_port()
        while grpc_port == port:
            grpc_port = free_port()
        url = f'http://127.0.0.1:{port}'
        config = root/'qdrant.yaml'
        config.write_text(f'service:\n  host: 127.0.0.1\n  http_port: {port}\n  grpc_port: {grpc_port}\n')
        with (root/'qdrant.log').open('w+') as log:
            process = subprocess.Popen([str(binary), '--config-path', str(config)], cwd=root,
                                       stdout=log, stderr=log, stdin=subprocess.DEVNULL)
            try:
                deadline = time.monotonic()+30
                while time.monotonic() < deadline:
                    try:
                        with opener.open(url+'/healthz', timeout=1):
                            break
                    except OSError:
                        if process.poll() is not None:
                            log.seek(0)
                            raise RuntimeError(log.read())
                        time.sleep(.1)
                else:
                    raise RuntimeError('isolated Qdrant did not start')
                encoder = BGEEncoder(model_path=model, device=device, batch_size=2)
                bundle = root/'bundle'
                result = build_hybrid(fixture, bundle, 'tiny-real-smoke', encoder,
                                      QdrantSnapshotWriter(url, bundle, batch_size=2),
                                      'offline_smoke', batch_size=2,
                                      index_params={'m': 16, 'ef_construct': 100})
                manifest = VectorManifest.from_dict(result['vector'])
                client = QdrantClient(url=url)
                try:
                    before = QdrantVectorIndex(manifest, url).search(encoder.encode_query('手机进水'), 5)
                    with DiskDocuments(bundle/'bm25') as docs:
                        assert corpus_fingerprint(docs) == manifest.corpus_sha256
                        assert docs.n_docs == manifest.n_docs
                    client.delete_collection(manifest.collection)
                    snapshot = bundle/next(iter(manifest.snapshot_files))
                    session = requests.Session()
                    session.trust_env = False
                    try:
                        with snapshot.open('rb') as file:
                            response = session.post(url+'/collections/'+manifest.collection+'/snapshots/upload',
                                                    params={'priority': 'snapshot', 'wait': 'true'},
                                                    files={'snapshot': file}, timeout=120)
                            response.raise_for_status()
                    finally:
                        session.close()
                    after = QdrantVectorIndex(manifest, url).search(encoder.encode_query('手机进水'), 5)
                    assert [h.doc_id for h in before] == [h.doc_id for h in after]
                    assert client.count(manifest.collection, exact=True).count == manifest.n_docs
                    return dict(n_docs=manifest.n_docs, snapshot_imported=True,
                                same_process_restore=True,
                                corpus_sha256=manifest.corpus_sha256,
                                seconds=time.perf_counter()-start)
                finally:
                    client.close()
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--qdrant-binary', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(smoke(args.qdrant_binary, args.model_path, args.device), indent=2))
