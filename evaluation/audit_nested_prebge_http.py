"""Independently reconstruct paired HTTP quality, model parity and latency."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .prepare_bm25_router import jsonl_rows, metrics, sha
from .ranking_dataset import load_dataset


def audit(root):
    summary = json.loads((root / 'summary.json').read_text())
    p = summary['protocol']
    if p['kind'] != 'real-http-audited-nested-preBGE-development-v1':
        raise ValueError('unsupported HTTP protocol')
    for path, digest in p['inputs'].items():
        if sha(path) != digest:
            raise ValueError('HTTP input binding changed: ' + path)
    for path, digest in p['code_sha256'].items():
        if sha(root / 'code' / Path(path).name) != digest:
            raise ValueError('executed HTTP source snapshot changed: ' + path)
    bundle = next(Path(path).parent for path in p['inputs']
                  if Path(path).name == 'table.jsonl')
    serving = json.loads((bundle / 'study.json').read_text())
    source_path = next(Path(path) for path, digest in p['inputs'].items()
                       if digest == serving['protocol']['source_study_sha256'])
    source = json.loads(source_path.read_text())
    source_audit = json.loads(source_path.with_name('independent-audit.json').read_text())
    if (source_audit['study_sha256'] != sha(source_path) or
            not source_audit['outer_groups_isolated'] or not source_audit['quality_gate_passed']):
        raise ValueError('HTTP source lacks an isolated passing audit')
    sp = source['protocol']
    for path, digest in sp['inputs'].items():
        if sha(path) != digest:
            raise ValueError('nested source input changed: ' + path)
    dataset = next(Path(path).parent for path in sp['inputs'] if Path(path).name == 'dataset.json')
    m = load_dataset(dataset)['metadata']
    ids = m['query_ids']
    slots = {q: i for i, q in enumerate(ids)}
    saved_gx = np.load(source_path.with_name('gate-features.npy'), allow_pickle=False)
    capture_path = next(Path(path) for path in sp['inputs'] if Path(path).name == 'captures.jsonl')
    captures = {r['query_id']: r for r in jsonl_rows(capture_path)}
    canonical_path = next(Path(path) for path in sp['inputs'] if Path(path).name == 'logits.jsonl')
    canonical = {r['query_id']: r['scores'] for r in jsonl_rows(canonical_path)}
    reference = {r['query_id']: r for r in jsonl_rows(source_path.with_name('outer-predictions.jsonl'))}
    if sha(root / 'requests.jsonl') != summary['requests_sha256']:
        raise ValueError('HTTP request log changed')
    if sha(root / 'quality-replay.jsonl') != summary['full_quality_replay']['file_sha256']:
        raise ValueError('HTTP full-quality log changed')
    rows = list(jsonl_rows(root / 'requests.jsonl'))
    live = list(jsonl_rows(root / 'quality-replay.jsonl'))
    expected = {(q, r, v) for q in p['queries'] for r in range(p['repeats']) for v in p['variants']}
    keys = [(r['query_id'], r['repeat'], r['variant']) for r in rows]
    if len(set(keys)) != len(rows) or set(keys) != expected:
        raise ValueError('unbalanced or duplicate paired HTTP requests')
    if len(live) != len(ids) or {r['query_id'] for r in live} != set(ids):
        raise ValueError('incomplete or duplicated full-quality requests')
    lookup = dict(zip(keys, rows))
    for q in p['queries']:
        for repeat in range(p['repeats']):
            h = lookup[q, repeat, 'hybrid']['ranked']
            if h != captures[q]['baselines']['hybrid'] or h != lookup[q, repeat, 'hybrid-compiled']['ranked']:
                raise ValueError('original/compiled hybrid capture parity failed')
    max_logit_delta = 0.
    for row in live + [r for r in rows if r['variant'] == 'native-length-router']:
        q, route = row['query_id'], row['research_route']
        expected_row = reference[q]
        pool = '40' if route['action'] == 'heavy' else '28'
        if row['ranked'][:int(pool)] != expected_row['ranked']:
            raise ValueError('full scored-pool ranking differs from isolated outer prediction')
        if (route['action'] != expected_row['action'] or route['fold'] != expected_row['fold'] or
                route['ce_max_length'] != serving['protocol']['online_scorer']['max_length'] or
                not route['one_cross_encoder'] or route['ce_pairs'] != int(pool) or
                row['routing']['fallback'] or route.get('delegated')):
            raise ValueError('outer model/action or executed CE count differs')
        if not np.array_equal(row['audit']['gate_features'], saved_gx[slots[q]]):
            raise ValueError('live pre-CE gate features differ from audited development')
        logits = row['audit']['teacher_logits']
        if list(logits) != list(canonical[q][pool]):
            raise ValueError('canonical candidate identities/order differ from HTTP')
        max_logit_delta = max(max_logit_delta, max(abs(v - canonical[q][pool][d]) for d, v in logits.items()))
    # Rebuild every per-query baseline/metric, independently resample group
    # totals and sizes; never substitute the aggregate baseline mean.
    live_by_id = {r['query_id']: r for r in live}
    actual = np.array([[metrics(live_by_id[q]['ranked'], m['relevance'][q])[name]
                        for name in ('ndcg', 'recall')] for q in ids])
    baseline = np.array([[r[name] for name in ('ndcg', 'recall')] for r in m['baselines']['hybrid']])
    groups = np.array(m['groups'])
    blocks = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    sizes = np.array([len(b) for b in blocks])
    totals = np.array([(actual - baseline)[b].sum(0) for b in blocks])
    rng = np.random.default_rng(20261027)
    draws = rng.integers(0, len(blocks), (2000, len(blocks)))
    boot = totals[draws].sum(1) / sizes[draws].sum(1)[:, None]
    quality = {}
    for j, name in enumerate(('ndcg', 'recall')):
        value, base = float(actual[:, j].mean()), float(baseline[:, j].mean())
        ci = np.quantile(boot[:, j], [.025, .975]).tolist()
        stated = summary['full_quality_replay']['metrics'][name]
        if (abs(value - stated['value']) > 1e-12 or
                not np.allclose(ci, stated['delta_ci']['ci95'], rtol=0, atol=1e-12)):
            raise ValueError('paired HTTP quality or group CI differs')
        quality[name] = dict(value=value, baseline=base, relative_lift=value / base - 1,
                             paired_delta_ci95=ci)
    latency = {}
    for variant in p['variants']:
        obs = [r['latency_ms'] for r in rows if r['variant'] == variant]
        val = float(np.quantile(obs, .95))
        if abs(val - summary['variants'][variant]['latency_ms']['p95']) > 1e-9:
            raise ValueError('HTTP p95 differs')
        latency[variant] = val
    actions = dict(Counter(r['research_route']['action'] for r in live))
    ratio = latency['native-length-router'] / latency['hybrid']
    qp = all(v['relative_lift'] > .2 and v['paired_delta_ci95'][0] > 0 for v in quality.values())
    accepted = bool(qp and ratio <= 1.05 and set(actions) == {'cheap', 'heavy'})
    if bool(summary['development_acceptance']) != accepted:
        raise ValueError('HTTP acceptance flag differs')
    stopped = json.loads((root / 'services-stopped.json').read_text())
    if not stopped['exited']:
        raise ValueError('owned HTTP database has not exited')
    report = dict(kind='independent-nested-preBGE-HTTP-audit-v1', summary_sha256=sha(root / 'summary.json'),
                  requests=len(rows), quality_queries=len(live),
                  paired_request_coverage=True, original_compiled_hybrid_parity=True,
                  full_scored_pool_rankings_actions_features_replayed=True,
                  maximum_live_canonical_logit_delta=max_logit_delta,
                  per_query_paired_group_CIs=True, quality=quality, action_counts=actions,
                  p95_ms=latency, p95_ratio=ratio, quality_passed=qp,
                  latency_passed=ratio <= 1.05, development_accepted=accepted,
                  confirmation_evaluated=False, goal_complete=False, owned_services_stopped=True)
    (root / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--http', type=Path, required=True)
    audit(p.parse_args().http)
