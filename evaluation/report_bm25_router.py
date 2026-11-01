"""Audit and summarise frozen research evidence, including compressed archives."""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    path=Path(path)
    if path.exists():return path.read_bytes()
    return gzip.decompress(path.with_name(path.name+'.gz').read_bytes())


def data(path):
    return json.loads(read(path))


def rows(path):
    # Evidence records contain no raw document text; split only LF anyway.
    return [json.loads(line) for line in read(path).decode().split('\n') if line.strip()]


def same(actual,expected):
    if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'evidence arithmetic mismatch: {actual} != {expected}')


def report(root):
    root=Path(root)
    inventory=root/'inventory.json'
    if inventory.exists():
        for name,expected in data(inventory)['files'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError('archive checksum mismatch: '+name)
    table=rows(root/'dev/table.jsonl')
    study=data(root/'study/study.json')
    by_id={r['query_id']:r for r in table}
    if len(by_id)!=len(table):raise ValueError('duplicate dev queries')
    if hashlib.sha256(read(root/'dev/table.jsonl')).hexdigest()!=study['provenance']['table_sha256']:
        raise ValueError('dev table checksum mismatch')
    frozen=data(root/'FREEZE.json')
    model_hash=hashlib.sha256(read(root/'study/model.json')).hexdigest()
    if frozen['model_sha256']!=model_hash:raise ValueError('frozen model changed')
    predictions=defaultdict(list)
    for row in rows(root/'study/outer-predictions.jsonl'):
        predictions[row['protocol']].append(row)
    errors={}
    for name,result in study['results'].items():
        seen=set()
        for fold in result['folds']:
            selected=set(fold['heldout_ids'])
            if seen&selected:raise ValueError('repeated outer validation query')
            training=set(by_id)-selected
            if {by_id[q]['group'] for q in selected}&{by_id[q]['group'] for q in training}:
                raise ValueError('outer group leakage')
            seen|=selected
        if seen!=set(by_id):raise ValueError('outer coverage incomplete')
        observed=predictions[name]
        if len(observed)!=len(table) or {r['query_id'] for r in observed}!=set(by_id):
            raise ValueError('prediction identities mismatch')
        losses=[];quality=[];missed=called_without_ndcg_gain=skipped_with_better_lexical=0
        for row in observed:
            original=by_id[row['query_id']]
            gain=original['hybrid']['ndcg']-original['lexical']['ndcg']
            same(row['gain'],gain)
            use=row['use_vector']
            losses.append(gain*int(not use))
            quality.append(original['hybrid' if use else 'lexical']['ndcg'])
            missed+=int(not use and gain>.01)
            called_without_ndcg_gain+=int(use and gain<=0)
            skipped_with_better_lexical+=int(not use and gain<0)
        same(statistics.mean(losses),result['mean_loss'])
        same(statistics.mean(quality),result['metrics']['ndcg'])
        same(statistics.mean(not r['use_vector'] for r in observed),result['vector_saving'])
        same(statistics.mean(losses)+1.645*statistics.stdev(losses)/math.sqrt(len(losses)),result['loss_upper95'])
        errors[name]=dict(missed_gain_above_001=missed,called_without_ndcg_gain=called_without_ndcg_gain,
                          skipped_with_better_lexical=skipped_with_better_lexical)
    runtime={}
    transfer_loss_upper=None
    for folder in ('http-dev','http-transfer'):
        summary=data(root/folder/'summary.json')
        if summary['protocol']['model_sha256']!=model_hash:raise ValueError('HTTP model changed')
        raw=rows(root/folder/'requests.jsonl')
        if len(raw)!=len(summary['protocol']['queries'])*5*summary['protocol']['repeats']:
            raise ValueError('missing HTTP requests')
        if any(r['routing']['fallback'] for r in raw):raise ValueError('HTTP fallback contaminated evaluation')
        for name,expected in summary['variants'].items():
            subset=[r for r in raw if r['variant']==name]
            identities=Counter(r['query_id'] for r in subset)
            if set(identities)!=set(summary['protocol']['queries']) or set(identities.values())!={summary['protocol']['repeats']}:
                raise ValueError('unbalanced HTTP repetition')
            same(statistics.mean(r['latency_ms'] for r in subset),expected['latency_ms']['mean'])
            same(statistics.mean(r['routing']['vector_attempted'] for r in subset),expected['vector_attempt_rate'])
            same(statistics.mean(r['metrics']['ndcg'] for r in subset),expected['metrics']['ndcg'])
        if not data(root/folder/'services-stopped.json')['exited']:
            raise ValueError('owned process did not stop')
        runtime[folder]=dict(variants=summary['variants'],paired=summary['paired'])
        if folder=='http-transfer':
            decisions={r['query_id']:r for r in raw if r['variant']=='gain-router'}
            losses=[r['metrics']['ndcg']-decisions[r['query_id']]['metrics']['ndcg']
                    for r in raw if r['variant']=='hybrid']
            transfer_loss_upper=statistics.mean(losses)+1.645*statistics.stdev(losses)/math.sqrt(len(losses))
    checks=dict(dev_target_met=study['results']['combined-loss0.01']['meets_target'],
                transfer_target_met=(transfer_loss_upper<=.01 and
                    runtime['http-transfer']['variants']['gain-router']['vector_attempt_rate']<=.7),
                dev_http_speedup_supported=runtime['http-dev']['paired']['gain-router']['delta_latency_ms']['ci95'][1]<0,
                transfer_http_speedup_supported=runtime['http-transfer']['paired']['gain-router']['delta_latency_ms']['ci95'][1]<0)
    result=dict(model_sha256=model_hash,queries=len(table),groups=len({r['group'] for r in table}),
                dev_baselines=study['baselines'],
                dev_candidates={name:{k:v for k,v in r.items() if k!='folds'} for name,r in study['results'].items()},
                error_counts=errors,
                current_actions_oracle=dict(ndcg=statistics.mean(max(r['lexical']['ndcg'],r['hybrid']['ndcg']) for r in table),
                    zero_loss_skip_fraction=statistics.mean(r['lexical']['ndcg']>=r['hybrid']['ndcg'] for r in table),
                    scope='post-hoc qrels choice only between these frozen BM25/hybrid results; not an online policy or an upper bound for all retrieval systems'),
                runtime=runtime,
                checks=checks,transfer_loss_upper95=transfer_loss_upper,
                accepted_for_default=False,
                decision=('Research targets passed; team review still required before changing defaults' if all(checks.values()) else
                          'Do not promote; failed checks: '+', '.join(k for k,v in checks.items() if not v)),
                audit='table/model hashes, outer group coverage, prediction arithmetic, balanced HTTP requests, no fallbacks and child-process exit verified')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('artifacts/rfc01'))
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    result=report(args.root)
    text=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.out:args.out.write_text(text)
    print(text)
