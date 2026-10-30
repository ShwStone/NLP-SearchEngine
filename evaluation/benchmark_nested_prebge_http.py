"""Export label-free outer serving models and replay real HTTP controls."""
import argparse
import functools
import json
from pathlib import Path
import shutil

from . import benchmark_native_router_http as harness
from .prepare_bm25_router import jsonl_rows, sha
from .ranking_dataset import load_dataset
from .runtime_nested_prebge import NestedGainPreBGERouter


def export_serving(study_root, out):
    study = json.loads((study_root / 'study.json').read_text())
    audit = json.loads((study_root / 'independent-audit.json').read_text())
    if (audit['study_sha256'] != sha(study_root / 'study.json') or
            not audit['quality_gate_passed'] or not audit['outer_groups_isolated']):
        raise ValueError('HTTP requires an audited quality-passing nested gate')
    inputs = study['protocol']['inputs']
    for path, digest in inputs.items():
        if sha(path) != digest:
            raise ValueError('source input changed: ' + path)
    parent_path = next(Path(p) for p in inputs if Path(p).name == 'study.json')
    parent = json.loads(parent_path.read_text())
    dataset = next(Path(p).parent for p in inputs if Path(p).name == 'dataset.json')
    m = load_dataset(dataset)['metadata']
    out.mkdir(parents=True, exist_ok=False)
    (out / 'models').mkdir()
    folds = []
    for gate, action in zip(study['folds'], parent['folds']):
        if gate['valid'] != action['valid'] or gate['fold'] != action['fold']:
            raise ValueError('gate and action outer folds differ')
        f = gate['fold']
        row = dict(fold=f, heldout_ids=[m['query_ids'][i] for i in gate['valid']],
                   threshold=gate['selection']['threshold'], weight=action['action']['weight'])
        for kind, source, digest in (
                ('gate', study_root / gate['final_gate_file'], gate['final_gate_sha256']),
                ('action', parent_path.parent / action['outer_model']['model_file'],
                 action['outer_model']['model_sha256'])):
            if sha(source) != digest:
                raise ValueError('outer model changed before serving export')
            name = f'models/fold-{f}-{kind}.joblib'
            shutil.copyfile(source, out / name)
            row[kind + '_file'], row[kind + '_sha256'] = name, digest
        folds.append(row)
    support = next(Path(p) for p in inputs if Path(p).name == 'vectors.npy')
    protocol = dict(kind='audited-nested-preBGE-serving-v1',
                    source_study_sha256=sha(study_root / 'study.json'),
                    source_audit_sha256=sha(study_root / 'independent-audit.json'),
                    online_scorer=parent['protocol']['online_scorer'],
                    support_sha256=sha(support), features=parent['protocol']['features'],
                    bm25_names=m['bm25_feature_names'], gate_features=study['protocol']['gate_features'],
                    label_assets_in_serving=False, query_prediction_cache=False,
                    parent_action_study_sha256=sha(parent_path),
                    goal_complete=False)
    (out / 'study.json').write_text(json.dumps(dict(protocol=protocol, folds=folds), indent=2) + '\n')
    # These are client-side evaluation inputs. Runtime opens only serving
    # study.json and models; it never opens the following three files.
    shutil.copyfile(study_root / 'outer-predictions.jsonl', out / 'outer-predictions.jsonl')
    with (out / 'qrels.tsv').open('w') as handle:
        for q in m['query_ids']:
            for d, grade in m['relevance'][q].items():
                handle.write(f'{q}\t0\t{d}\t{grade}\n')
    rows = [dict(query_id=q, group=m['groups'][i],
                 **{name: m['baselines'][name][i] for name in ('hybrid', 'lexical', 'dense')})
            for i, q in enumerate(m['query_ids'])]
    (out / 'table.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return dataset


def run(a):
    source_study = a.study
    bundle = a.out.with_name(a.out.name + '-serving')
    dataset = export_serving(source_study, bundle)
    capture = next(Path(p) for p in json.loads((source_study / 'study.json').read_text())['protocol']['inputs']
                   if Path(p).name == 'captures.jsonl')
    if sha(a.capture) != sha(capture):
        raise ValueError('live capture differs from audited action captures')
    if a.document_token_parity is not None:
        parity = json.loads(a.document_token_parity.read_text())
        source_inputs = json.loads((source_study / 'study.json').read_text())['protocol']['inputs']
        canonical = next(Path(p) for p in source_inputs if Path(p).name == 'logits.jsonl')
        for path in (capture, canonical):
            if parity['inputs'].get(str(path.resolve())) != sha(path):
                raise ValueError('document-token parity was not checked on this development scope')
    paths = [Path(__file__), Path(__file__).with_name('runtime_nested_prebge.py'),
             Path(__file__).with_name('tensorrt_cross_encoder.py'),
             Path(__file__).with_name('bge_candidate_support.py'),
             Path(__file__).with_name('export_bge_tensorrt_input.py'),
             Path(__file__).with_name('study_rerank_actions.py'),
             Path(__file__).with_name('stable_dense_topk.py'),
             Path(__file__).with_name('gte_document_tokens.py'),
             Path(__file__).resolve().parents[1] / 'online' / 'router_features.py']
    snapshots = {str(p.resolve()): p.read_bytes() for p in paths}
    harness.NativeLengthRouter = functools.partial(NestedGainPreBGERouter,
                                                   export_path=a.export, model_path=a.reranker_path,
                                                   scorer_batch_size=a.scorer_batch_size,
                                                   document_token_parity=a.document_token_parity)
    a.study, a.qrels, a.table = bundle, bundle / 'qrels.tsv', bundle / 'table.jsonl'
    a.scorer_batch_size, a.query_graph, a.scorer_graph = a.scorer_batch_size, True, False
    a.fast_support, a.freeze_index_gc = True, True
    harness.run(a)
    summary = json.loads((a.out / 'summary.json').read_text())
    protocol = summary['protocol']
    protocol['kind'] = 'real-http-audited-nested-preBGE-development-v1'
    bindings = [source_study / 'study.json', source_study / 'independent-audit.json',
                dataset / 'dataset.json', a.export / 'summary.json']
    bindings += list((bundle / 'models').glob('*.joblib'))
    if a.document_token_parity is not None:
        bindings.append(a.document_token_parity)
    for path in bindings:
        protocol['inputs'][str(path.resolve())] = sha(path)
    for path, contents in snapshots.items():
        target = a.out / 'code' / Path(path).name
        target.write_bytes(contents)
        protocol['code_sha256'][path] = sha(target)
    captures = {r['query_id']: r for r in jsonl_rows(a.capture)}
    mismatch = [dict(query_id=r['query_id'], repeat=r['repeat'])
                for r in jsonl_rows(a.out / 'requests.jsonl')
                if r['variant'] == 'hybrid' and r['ranked'] != captures[r['query_id']]['baselines']['hybrid']]
    summary['original_hybrid_capture_mismatches'] = mismatch
    summary['development_acceptance'] = bool(
        not mismatch and not summary['outer_replay_mismatches'] and
        summary['compiled_hybrid_ranking_mismatches'] == 0 and summary['latency_gate_passed'] and
        a.quality_replay_all and all(summary['full_quality_replay']['action_counts'].values()) and
        all(v['passes_20_percent'] and v['delta_ci']['ci95'][0] > 0
            for v in summary['full_quality_replay']['metrics'].values()))
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    print(json.dumps(dict(development_acceptance=summary['development_acceptance'],
                          original_hybrid_mismatches=len(mismatch),
                          outer_replay_mismatches=len(summary['outer_replay_mismatches']),
                          p95_ratio=summary['paired']['native-length-router']['p95_ratio']), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('index', 'manifest', 'storage', 'qdrant-binary', 'small-model-path', 'native-model-path',
                 'reranker-path', 'vectors-dir', 'support-vectors', 'study', 'previous-router',
                 'capture', 'adapter-root', 'out', 'export'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--feature-extension', type=Path)
    p.add_argument('--limit', type=int, default=48)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--quality-replay-all', action='store_true')
    p.add_argument('--scorer-batch-size', type=int, default=20)
    p.add_argument('--document-token-parity', type=Path)
    run(p.parse_args())
