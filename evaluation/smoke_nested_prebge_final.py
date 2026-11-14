"""Verify final all-development models over HTTP using observed queries only."""
import argparse
import functools
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from . import benchmark_native_router_http as harness
from .prebge_nested import DevelopmentData, predict
from .prepare_bm25_router import jsonl_rows, sha
from .runtime_nested_prebge import NestedGainPreBGERouter


def run(a):
    final = a.study
    manifest = json.loads((final / 'study.json').read_text())
    training = json.loads((final / 'training.json').read_text())
    fa = json.loads((final / 'independent-audit.json').read_text())
    if (fa['study_sha256'] != sha(final / 'study.json') or
            not fa['both_final_models_independently_refitted_on_all_development']):
        raise ValueError('final HTTP smoke requires an independent full-development refit audit')
    qualified = json.loads((Path(training['qualified_http']) / 'summary.json').read_text())
    for path, digest in qualified['protocol']['code_sha256'].items():
        if sha(path) != digest:
            raise ValueError('qualified serving source changed before final HTTP smoke: ' + path)
    ip = training['inputs']
    dataset = next(Path(p).parent for p in ip if Path(p).name == 'dataset.json')
    capture = next(Path(p) for p in ip if Path(p).name == 'captures.jsonl')
    canonical = next(Path(p) for p in ip if Path(p).name == 'logits.jsonl')
    support = next(Path(p) for p in ip if Path(p).name == 'vectors.npy')
    if sha(a.capture) != sha(capture):
        raise ValueError('final smoke development query scope differs')
    d = DevelopmentData(dataset, capture, canonical, support)
    fold = manifest['folds'][0]
    gate = joblib.load(final / fold['gate_file'])
    action = joblib.load(final / fold['action_file'])
    gx = np.load(final / 'gate-features.npy', allow_pickle=False)
    selected = sorted(range(d.n), key=lambda i: hashlib.sha256(
        ('final-nested-preBGE-smoke-v1|' + d.ids[i]).encode()).hexdigest())[:128]
    chosen = np.array(selected)
    mask = gate.predict(gx[chosen]) > fold['threshold']
    _, rankings = d.rank_quality(chosen, predict(action, d.x[chosen]), fold['weight'])
    client = a.out.with_name(a.out.name + '-client')
    client.mkdir(parents=True, exist_ok=False)
    (client / 'study.json').write_text(json.dumps(dict(
        protocol=dict(kind='observed-final-model-smoke-client-only', model_sha256=sha(final / 'study.json')),
        folds=[dict(fold='final', heldout_ids=[d.ids[i] for i in chosen])])) + '\n')
    expected = [dict(query_id=d.ids[i], fold='final', action='heavy' if mask[j] else 'cheap',
                     ranked=rankings[j] if mask[j] else d.cheap_rankings[i]) for j, i in enumerate(chosen)]
    (client / 'outer-predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in expected))
    rows = [dict(query_id=d.ids[i], group=str(d.groups[i]),
                 **{name: d.m['baselines'][name][i] for name in ('hybrid', 'lexical', 'dense')}) for i in chosen]
    (client / 'table.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (client / 'qrels.tsv').write_text(''.join(f'{d.ids[i]}\t0\t{doc}\t{grade}\n'
                                            for i in chosen for doc, grade in d.m['relevance'][d.ids[i]].items()))
    factory = functools.partial(NestedGainPreBGERouter, export_path=a.export,
                                model_path=a.reranker_path, scorer_batch_size=20,
                                document_token_parity=a.document_token_parity)

    def serve(base, backend, unused_client_study, fold):
        return factory(base, backend, final, fold)

    harness.NativeLengthRouter = serve
    a.study, a.table, a.qrels = client, client / 'table.jsonl', client / 'qrels.tsv'
    a.scorer_batch_size, a.query_graph, a.scorer_graph = 20, True, False
    a.fast_support, a.freeze_index_gc = True, True
    a.limit, a.repeats, a.quality_replay_all = 128, 1, False
    paths = [Path(__file__), Path(__file__).with_name('runtime_nested_prebge.py'),
             Path(__file__).with_name('tensorrt_cross_encoder.py'), Path(__file__).with_name('stable_dense_topk.py'),
             Path(__file__).with_name('bge_candidate_support.py'), Path(__file__).with_name('gte_document_tokens.py')]
    snapshots = {str(p.resolve()): p.read_bytes() for p in paths}
    harness.run(a)
    summary = json.loads((a.out / 'summary.json').read_text())
    lookup = {r['query_id']: r for r in expected}
    live = [r for r in jsonl_rows(a.out / 'requests.jsonl') if r['variant'] == 'native-length-router']
    if len(live) != 128 or len({r['query_id'] for r in live}) != 128:
        raise ValueError('final HTTP smoke request coverage differs')
    mismatches = []
    for row in live:
        expected_row = lookup[row['query_id']]
        pool = len(expected_row['ranked'])
        if (row['ranked'][:pool] != expected_row['ranked'] or
                row['research_route']['action'] != expected_row['action']):
            mismatches.append(row['query_id'])
    p = summary['protocol']
    p['kind'] = 'final-nested-preBGE-observed-HTTP-smoke-v1'
    for path in (final / 'study.json', final / 'independent-audit.json', final / fold['gate_file'], final / fold['action_file']):
        p['inputs'][str(path.resolve())] = sha(path)
    for path, contents in snapshots.items():
        target = a.out / 'code' / Path(path).name
        target.write_bytes(contents); p['code_sha256'][path] = sha(target)
    summary.update(model_sha256=sha(final / 'study.json'), top_scored_pool_and_actions_exact=not mismatches,
                   mismatches=mismatches, observed_queries_only=True,
                   quality_on_training_is_not_acceptance=True, goal_complete=False)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.out / 'protocol.json').write_text(json.dumps(p, indent=2) + '\n')
    print(json.dumps(dict(smoke_queries=len(live), top_scored_pool_and_actions_exact=not mismatches,
                          model_sha256=sha(final / 'study.json'), quality_is_not_acceptance=True), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('index', 'manifest', 'storage', 'qdrant-binary', 'small-model-path', 'native-model-path',
                 'reranker-path', 'vectors-dir', 'support-vectors', 'study', 'previous-router',
                 'capture', 'adapter-root', 'out', 'export'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--feature-extension', type=Path)
    p.add_argument('--document-token-parity', type=Path)
    run(p.parse_args())
