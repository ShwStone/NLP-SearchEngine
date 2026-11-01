"""Canonical BGE development scoring with a declared TensorRT batch size."""
import argparse
import json
from pathlib import Path

from .prepare_bm25_router import jsonl_rows, sha
from .ranking_dataset import load_dataset


def run(a):
    from .tensorrt_cross_encoder import TensorRTCrossEncoder
    metadata = load_dataset(a.dataset)['metadata']
    captures = {r['query_id']: r for r in jsonl_rows(a.capture)}
    ids = metadata['query_ids']
    if set(captures) != set(ids):
        raise ValueError('canonical scorer query scope differs')
    documents = {r['doc_id']: (r['docid'], r['title'] + '\n' + r['text'])
                 for r in jsonl_rows(a.documents)}
    scorer = TensorRTCrossEncoder(a.engine, a.model, max_length=a.max_length,
                                  batch_size=a.batch_size)
    a.out.mkdir(parents=True, exist_ok=False)
    files = [a.dataset / 'dataset.json', a.capture, a.documents, a.engine / 'summary.json']
    code = [Path(__file__), Path(__file__).with_name('tensorrt_cross_encoder.py'),
            Path(__file__).with_name('export_bge_tensorrt_input.py'),
            Path(__file__).with_name('study_rerank_actions.py')]
    protocol = dict(kind='canonical-preBGE-development-logits-v2', query_ids=ids,
                    pools=[28, 40], candidate_order='internal numeric docid ascending',
                    encoder=scorer.info, requested_batch_size=a.batch_size,
                    inputs={str(p.resolve()): sha(p) for p in files},
                    code_sha256={str(p.resolve()): sha(p) for p in code},
                    qrels_read=False, scoped_development_metadata_loaded=True,
                    confirmation_read=False, quality_evaluated=False, goal_complete=False)
    (a.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (a.out / 'code').mkdir()
    for p in code:
        (a.out / 'code' / p.name).write_bytes(p.read_bytes())
    scorer.score('预热', ['文档内容 ' * 40] * min(a.batch_size, 40))
    with (a.out / 'logits.jsonl').open('w') as output:
        for index, q in enumerate(ids):
            scores, costs = {}, {}
            for k in (28, 40):
                pool = sorted((r['doc_id'] for r in captures[q]['dense'][:k]),
                              key=lambda d: documents[d][0])
                if len(pool) != k or len(set(pool)) != k:
                    raise ValueError('invalid native dense pool')
                values, elapsed = scorer.score(captures[q]['text'],
                                               [documents[d][1] for d in pool])
                scores[str(k)] = dict(zip(pool, values)); costs[str(k)] = elapsed
            output.write(json.dumps(dict(query_id=q, scores=scores,
                                         component_ms=costs)) + '\n')
            output.flush()
            if (index + 1) % 100 == 0:
                print('canonical batch', a.batch_size, index + 1, '/', len(ids), flush=True)
    summary = dict(protocol=protocol, logits_sha256=sha(a.out / 'logits.jsonl'),
                   queries=len(ids), goal_complete=False)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(dict(queries=len(ids), logits_sha256=summary['logits_sha256']), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('dataset', 'capture', 'documents', 'engine', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=40)
    p.add_argument('--max-length', type=int, default=256)
    run(p.parse_args())
