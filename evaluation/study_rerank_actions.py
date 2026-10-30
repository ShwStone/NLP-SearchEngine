"""Frozen cross-encoder feasibility for new quality-oriented Router actions.

Consumes development candidates only. Cached logits permit later grouped
selection; they are not HTTP timings or independent validation. Pair scoring
times are recorded on-device, with synchronization, including tokenization.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

from .audit_router_ceiling import weighted_rrf
from .prepare_bm25_router import jsonl_rows, metrics, sha
from .router_metrics import percentile
from .train_bm25_router import group_bootstrap


MODEL_ID = 'BAAI/bge-reranker-base'
REVISION = '2cfc18c9415c912f9d8155881c133215df768a70'
MODEL_SHA = 'ced967c45fd1902eb92716c9ceeca7c95a936770ea9db611f5a841b926e33fbd'
ALLOWED_MODELS={
    (MODEL_ID,REVISION):MODEL_SHA,
    ('BAAI/bge-reranker-v2-m3','modelscope:e099d4b9cdbd291b1569d416f19aaa6523570bc3'):
        'd9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286',
}
POOLS = ('dense-10', 'dense-20', 'dense-50', 'dense-100',
         'hybrid-20', 'hybrid-50', 'union-10', 'union-25', 'union-50')
EXTENDED_POOLS = ('dense-200', 'hybrid-100', 'hybrid-200', 'union-100', 'union-200')


def length_batches(lengths,batch_size):
    if type(batch_size) is not int or batch_size<1 or any(type(n) is not int or n<1 for n in lengths):
        raise ValueError('invalid pair lengths or batch size')
    order=sorted(range(len(lengths)),key=lambda i:(lengths[i],i))
    return [order[offset:offset+batch_size] for offset in range(0,len(order),batch_size)]


class FrozenCrossEncoder:
    def __init__(self, path, *, device='cuda', batch_size=16, max_length=512,batch_strategy='input-order'):
        path = Path(path)
        inventory = json.loads((path/'model_inventory.json').read_text())
        identity=(inventory['model_id'],inventory['revision'])
        if identity not in ALLOWED_MODELS:
            raise ValueError('frozen cross-encoder identity mismatch')
        if inventory['files'].get('model.safetensors') != ALLOWED_MODELS[identity]:
            raise ValueError('unexpected model weights')
        for name, digest in inventory['files'].items():
            if sha(path/name) != digest:
                raise ValueError('cross-encoder file checksum mismatch: '+name)
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if batch_size < 1 or not 32 <= max_length <= 512:
            raise ValueError('invalid cross-encoder scoring configuration')
        if batch_strategy not in ('input-order','length'):
            raise ValueError('invalid scoring batch strategy')
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        dtype = torch.float16 if device.startswith('cuda') else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True, use_safetensors=True, torch_dtype=dtype).to(device).eval()
        self.batch_size, self.max_length = batch_size, max_length
        self.batch_strategy=batch_strategy
        self.info = dict(model_id=identity[0], revision=identity[1], inventory_sha256=sha(path/'model_inventory.json'),
                         max_length=max_length, batch_size=batch_size, device=device, dtype=str(dtype))
        if batch_strategy!='input-order':
            self.info['batch_strategy']=batch_strategy

    def score(self, query, texts):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        values = []
        with self.torch.inference_mode():
            if self.batch_strategy=='length' and texts:
                encoded=self.tokenizer([query]*len(texts),texts,padding=False,truncation=True,max_length=self.max_length)
                batches=length_batches([len(row) for row in encoded['input_ids']],self.batch_size)
                values=[None]*len(texts)
                for indices in batches:
                    batch={key:[rows[i] for i in indices] for key,rows in encoded.items()}
                    inputs=self.tokenizer.pad(batch,padding=True,return_tensors='pt').to(self.device)
                    scored=self.model(**inputs).logits.view(-1).float().cpu().tolist()
                    if len(scored)!=len(indices):
                        raise ValueError('bucketed scorer returned incorrect cardinality')
                    for index,value in zip(indices,scored):
                        values[index]=value
            else:
                for offset in range(0, len(texts), self.batch_size):
                    batch = texts[offset:offset+self.batch_size]
                    inputs = self.tokenizer([query]*len(batch), batch, padding=True,
                        truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)
                    values.extend(self.model(**inputs).logits.view(-1).float().cpu().tolist())
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()
        elapsed = (time.perf_counter()-started)*1000
        if len(values) != len(texts) or any(not math.isfinite(v) for v in values):
            raise ValueError('invalid cross-encoder logits')
        return values, elapsed


def pools_for(lexical, dense, identities, *, extended=False,dense_limits=None):
    if dense_limits is not None:
        if (extended or not dense_limits or len(set(dense_limits))!=len(dense_limits) or
                any(type(n) is not int or not 1<=n<=200 for n in dense_limits)):
            raise ValueError('invalid dense-only pool limits')
        return {f'dense-{size}':dense[:size] for size in dense_limits}
    hybrid = weighted_rrf(lexical, dense, .5, 60., identities)
    result = {}
    for name in POOLS + (EXTENDED_POOLS if extended else ()):
        kind, size = name.split('-')
        size = int(size)
        if kind == 'union':
            result[name] = sorted(set(lexical[:size]) | set(dense[:size]), key=identities.__getitem__)
        else:
            result[name] = (dense if kind == 'dense' else hybrid)[:size]
    return result


def run(args):
    if args.limit < 0:
        raise ValueError('pilot limit must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=False)
    table = {row['query_id']: row for row in jsonl_rows(args.table)}
    source = json.loads(args.table.with_name('provenance.json').read_text())
    if sha(args.documents) != source['source_manifest']['files']['docs.jsonl']:
        raise ValueError('frozen candidate/document identity mismatch')
    import hashlib
    ordered = sorted(table, key=lambda q: hashlib.sha256(('rfc01-ce-feasibility-v1|'+q).encode()).hexdigest())
    selected = set(ordered[:args.limit] if args.limit else ordered)
    captured = {row['query_id']: row for row in jsonl_rows(args.capture) if row['query_id'] in selected}
    if set(captured) != selected:
        raise ValueError('incomplete development capture')
    texts, identities = {}, {}
    for row in jsonl_rows(args.documents):
        texts[row['doc_id']] = row['title']+'\n'+row['text']
        identities[row['doc_id']] = row['docid']
    protocol = dict(kind='frozen-reranking-development-feasibility', pools=POOLS,
        queries=[q for q in ordered if q in selected], selection='hash of query ID before scoring; development only',
        limit=args.limit, max_length=args.max_length, batch_size=args.batch_size,
        inputs={str(p): sha(p) for p in (args.table, args.capture, args.qrels, args.documents)},
        reference='frozen equal-RRF c=60; engine internal docid tie-break',
        confirmation_read=False, includes_http_latency=False,
        timing_scope='scoring largest union of all declared pools once per query; GPU synchronization; no caching across queries; excludes BM25, encoding, ANN and HTTP')
    pool_names = tuple(pools_for([],[],{},extended=args.extended,dense_limits=args.dense_pool_limits))
    protocol['pools'] = pool_names
    protocol['dense_pool_limits']=args.dense_pool_limits
    protocol['code_sha256']={str(p):sha(p) for p in (Path(__file__),Path(__file__).with_name('prepare_bm25_router.py'))}
    (args.out/'code').mkdir()
    for path in protocol['code_sha256']:(args.out/'code'/Path(path).name).write_bytes(Path(path).read_bytes())
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    encoder = FrozenCrossEncoder(args.model_path, batch_size=args.batch_size, max_length=args.max_length, device=args.device,
                                 batch_strategy=args.batch_strategy)
    encoder.score('模型预热', ['文本匹配测试']*args.batch_size)
    protocol['encoder'] = encoder.info
    prior = None
    if args.reuse_logits:
        report = json.loads(args.reuse_logits.with_name('summary.json').read_text())
        if sha(args.reuse_logits) != report['logits_sha256']:
            raise ValueError('prior logits checksum mismatch')
        if report['protocol']['encoder'] != encoder.info:
            raise ValueError('cannot merge logits from a different scorer')
        if report['protocol']['inputs'] != protocol['inputs']:
            raise ValueError('prior logits use different development inputs')
        prior = {row['query_id']:row for row in jsonl_rows(args.reuse_logits)}
        if not selected <= set(prior):
            raise ValueError('prior scores do not cover selected queries')
        protocol['reused_logits_sha256'] = sha(args.reuse_logits)
        protocol['timing_scope'] = 'sum of previous scoring pass plus missing-pair scoring; offline reconstruction, not online latency'
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    with (args.out/'logits.jsonl').open('w', encoding='utf-8') as handle:
        for index, query_id in enumerate(protocol['queries']):
            row = captured[query_id]
            pools = pools_for([h['doc_id'] for h in row['lexical']], [h['doc_id'] for h in row['dense']], identities,
                              extended=args.extended,dense_limits=args.dense_pool_limits)
            candidates = sorted({doc_id for pool in pools.values() for doc_id in pool}, key=identities.__getitem__)
            scores = dict(prior[query_id]['logits']) if prior is not None else {}
            missing = [d for d in candidates if d not in scores]
            logits, elapsed = encoder.score(row['text'], [texts[d] for d in missing])
            scores.update(zip(missing, logits))
            if prior is not None:
                elapsed += prior[query_id]['scoring_ms']
            ranked = {name: sorted(pool, key=lambda d: (-scores[d], identities[d])) for name, pool in pools.items()}
            handle.write(json.dumps(dict(query_id=query_id, group=table[query_id]['group'], logits=scores,
                ranked=ranked, scoring_ms=elapsed, candidates=len(candidates)), ensure_ascii=False)+'\n')
            handle.flush()
            if (index+1)%8 == 0:
                print('scored', index+1, '/', len(selected), 'latest_ms', round(elapsed, 1), flush=True)
    # Labels are consumed only after all frozen-model scoring has finished.
    relevance = {}
    with args.qrels.open() as handle:
        for line in handle:
            query_id, _, doc_id, grade = line.split()
            if query_id in selected:
                relevance.setdefault(query_id, {})[doc_id] = float(grade)
    if set(relevance) != selected:
        raise ValueError('missing development qrels')
    rows = list(jsonl_rows(args.out/'logits.jsonl'))
    groups = np.array([row['group'] for row in rows])
    baselines = {name: {metric: float(np.mean([table[row['query_id']][name][metric] for row in rows]))
                       for metric in ('ndcg', 'recall')} for name in ('lexical', 'dense', 'hybrid')}
    results = {}
    with (args.out/'action-metrics.jsonl').open('w') as handle:
        for row in rows:
            q = row['query_id']
            observed = {name: metrics(ranking, relevance[q]) for name, ranking in row['ranked'].items()}
            handle.write(json.dumps(dict(query_id=q, group=row['group'], metrics=observed))+'\n')
    observed = list(jsonl_rows(args.out/'action-metrics.jsonl'))
    for pool in pool_names:
        report = {}
        for metric in ('ndcg', 'recall'):
            quality = np.array([row['metrics'][pool][metric] for row in observed])
            baseline = np.array([table[row['query_id']]['hybrid'][metric] for row in rows])
            report[metric] = dict(value=float(quality.mean()), relative_lift=float(quality.mean()/baseline.mean()-1),
                                 delta_ci=group_bootstrap(quality-baseline, groups))
        results[pool] = report
    times = [row['scoring_ms'] for row in rows]
    result = dict(protocol=protocol, baselines=baselines, actions=results,
        full_pool_scoring_ms=dict(mean=float(np.mean(times)), p50=percentile(times,.5), p95=percentile(times,.95)),
        paired_candidates=float(np.mean([row['candidates'] for row in rows])),
        logits_sha256=sha(args.out/'logits.jsonl'), action_metrics_sha256=sha(args.out/'action-metrics.jsonl'),
        goal_complete=False, limitation='fixed-action pilot; no learned routing, no HTTP or independent validation')
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='protocol'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('table', 'capture', 'qrels', 'documents', 'model-path', 'out'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--limit', type=int, default=96, help='hash pilot size; 0 = all development queries')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--extended',action='store_true')
    parser.add_argument('--reuse-logits',type=Path)
    parser.add_argument('--batch-strategy',choices=('input-order','length'),default='input-order')
    parser.add_argument('--dense-pool-limits',type=int,nargs='+',help='optional fixed dense-only pools, e.g. 20 40 80')
    run(parser.parse_args())
