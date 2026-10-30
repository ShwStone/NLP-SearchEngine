"""Independent confirmation audit covering successful and failed outcomes.

Mutable Qdrant files are fingerprinted before startup by the frozen runner;
their preflight manifest is checked here. Immutable model/code/data hashes,
selection bindings, isolation, request balance and every metric are rebuilt.
"""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .router_metrics import percentile,bootstrap
from .train_bm25_router import group_bootstrap
from .confirm_nested_prebge_http import verify_frozen
from .prepare_router_transfer import grams
from online.parser import parse,QuerySyntaxError

VARIANTS=['hybrid','hybrid-compiled','dense','previous-router','bge-calibration-router']

def audit_history(freeze):
    manifest=json.loads(Path(freeze['history_source_manifest']).read_text())
    observed=set();pairs=set()
    for path,digest in manifest['inputs'].items():
        if freeze['files'].get(str(Path(path).resolve()))!=digest or sha(path)!=digest:
            raise ValueError('query exposure source is not frozen')
        for row in jsonl_rows(path):
            q=row.get('query_id',row.get('qid'))
            if q is None:continue
            q=str(q);observed.add(q);text=row.get('text',row.get('query'))
            if isinstance(text,str) and text:pairs.add((q,text))
    catalog=Path(manifest['catalog'])
    if sha(catalog)!=manifest['catalog_sha256']:raise ValueError('exposure text catalog changed')
    for row in jsonl_rows(catalog):
        if row['query_id'] in observed:pairs.add((row['query_id'],row['text']))
    covered={q for q,_ in pairs};pairs.update((q,'') for q in observed-covered)
    actual={(str(r['query_id']),r['text']) for path in freeze['history_sha256'] for r in jsonl_rows(path)}
    if pairs!=actual or len(observed)!=manifest['observed_ids']:raise ValueError('consolidated exposure history omits/changes observed queries')
    return dict(exposure_source_files_replayed=len(manifest['inputs']),observed_ids=len(observed),history_rows=len(actual),exposure_history_exact=True)

def audit_selection(freeze,root,selection,queries):
    """Replay the frozen sampling rule and derive labels from pinned raw sources."""
    import pyarrow.parquet as pq
    exposure=audit_history(freeze)
    policy=freeze['confirmation'];prior=set();history=[];index={}
    def remember(tokens):
        i=len(history);history.append(tokens)
        for token in tokens:index.setdefault(token,set()).add(i)
    for path in freeze['history_sha256']:
        for row in jsonl_rows(path):
            prior.add(row.get('query_id',row.get('qid')))
            remember(grams(row.get('text',row.get('query',''))))
    docs={r['doc_id'] for r in jsonl_rows(Path(freeze['corpus']['index'])/'docs.jsonl')}
    labels=pq.read_table(policy['raw_qrels']).to_pylist()
    answerable={str(r['qid']) for r in labels if r['score']>0 and str(r['pid']) in docs}
    raw=pq.read_table(policy['raw_queries']).to_pylist()
    ordered=sorted(raw,key=lambda r:hashlib.sha256((policy['hash_salt']+str(r['id'])).encode()).hexdigest())
    expected=[];excluded=Counter()
    for row in ordered:
        q=str(row['id'])
        if q in prior:excluded['prior_id']+=1;continue
        if q not in answerable:excluded['no_positive_in_fixed_corpus']+=1;continue
        try:
            if parse(row['text'])['explicit_constraints']:excluded['explicit_constraints']+=1;continue
        except QuerySyntaxError:excluded['invalid_syntax']+=1;continue
        tokens=grams(row['text']);shared=Counter(i for token in tokens for i in index.get(token,()))
        if any(n/(len(tokens)+len(history[i])-n)>=policy['near_duplicate_threshold'] for i,n in shared.items()):
            excluded['near_duplicate']+=1;continue
        expected.append(dict(query_id=q,text=row['text']));remember(tokens)
        if len(expected)==policy['limit']:break
    if expected!=queries or dict(excluded)!=selection['exclusions']:raise ValueError('selected queries differ from predeclared raw-source hash-order replay')
    selected={r['query_id'] for r in expected}
    expected_labels=''.join(f"{r['qid']}\t0\t{r['pid']}\t{r['score']}\n" for r in labels if str(r['qid']) in selected and str(r['pid']) in docs)
    if (root/'qrels.tsv').read_bytes()!=expected_labels.encode():raise ValueError('selected qrels differ from pinned raw labels/corpus')
    return dict(raw_query_selection_replayed=True,raw_qrels_exactly_reconstructed=True,queries=len(expected),documents=len(docs),**exposure)

def recompute(rows,queries,repeats,relevance,acceptance):
    if len(set(queries))!=len(queries) or set(relevance)!=set(queries):raise ValueError('query/relevance coverage differs')
    expected=Counter((q,r,v) for q in queries for r in range(repeats) for v in VARIANTS)
    if Counter((r['query_id'],r['repeat'],r['variant']) for r in rows)!=expected:raise ValueError('duplicate/missing/unbalanced HTTP observations')
    lookup={(r['query_id'],r['repeat'],r['variant']):r for r in rows}
    for row in rows:
        if (row['routing'].get('fallback') is not False or not np.isfinite(row['latency_ms']) or row['latency_ms']<=0 or
                len(row['ranked'])>100 or len(set(row['ranked']))!=len(row['ranked'])):raise ValueError('invalid HTTP result/fallback/timing')
        if row['variant']=='bge-calibration-router':
            route=row.get('research_route') or {};action=route.get('action')
            if (route.get('delegated') is not False or action not in ('cheap','heavy') or
                    (route.get('ce_pairs')!=(40 if action=='heavy' else 28) or not route.get('one_cross_encoder')) or route.get('fold')!='final' or route.get('ce_max_length')!=acceptance['ce_max_length']):
                raise ValueError('selected final BGE recipe was not executed')
    quality={};variants={}
    for name in VARIANTS:
        subset=[r for r in rows if r['variant']==name];times=[r['latency_ms'] for r in subset]
        quality[name]={m:np.array([np.mean([metrics(lookup[q,r,name]['ranked'],relevance[q])[m] for r in range(repeats)]) for q in queries]) for m in ('ndcg','recall')}
        variants[name]=dict(metrics={m:float(v.mean()) for m,v in quality[name].items()},latency_ms=dict(mean=float(np.mean(times)),p95=percentile(times,.95)))
    gains={};groups=np.array(queries)
    for m in ('ndcg','recall'):
        actual,reference=quality['bge-calibration-router'][m],quality['hybrid'][m]
        if reference.mean()<=0:raise ValueError('zero baseline cannot define relative improvement')
        gains[m]=dict(value=float(actual.mean()),baseline=float(reference.mean()),relative_lift=float(actual.mean()/reference.mean()-1),
            delta_ci=group_bootstrap(actual-reference,groups),passes_20_percent=bool(actual.mean()>1.2*reference.mean()))
    actions={a:sum(r['variant']=='bge-calibration-router' and r['research_route']['action']==a for r in rows) for a in ('cheap','heavy')}
    for q in queries:
        if len({lookup[q,r,'bge-calibration-router']['research_route']['action'] for r in range(repeats)})!=1:raise ValueError('query action changes across repeats')
    query_actions={a:sum(lookup[q,0,'bge-calibration-router']['research_route']['action']==a for q in queries) for a in actions}
    compiled=sum(lookup[q,r,'hybrid']['ranked']!=lookup[q,r,'hybrid-compiled']['ranked'] for q in queries for r in range(repeats))
    ratio=variants['bge-calibration-router']['latency_ms']['p95']/variants['hybrid']['latency_ms']['p95']
    paired=[float(np.mean([lookup[q,r,'bge-calibration-router']['latency_ms']-lookup[q,r,'hybrid']['latency_ms'] for r in range(repeats)])) for q in queries]
    quality_pass=all(v['passes_20_percent'] and v['delta_ci']['ci95'][0]>0 for v in gains.values())
    latency_pass=ratio<=acceptance['http_p95_ratio_max'];nonconstant=all(query_actions.values())
    return dict(variants=variants,gains=gains,action_counts=actions,query_action_counts=query_actions,
        http_p95_ratio=ratio,paired_latency_delta=bootstrap(paired),nonconstant=bool(nonconstant),
        compiled_hybrid_ranking_mismatches=compiled,quality_gate_passed=bool(quality_pass),latency_gate_passed=bool(latency_pass),
        confirmation_gate_passed=bool(quality_pass and latency_pass and nonconstant and compiled==0))

def close(actual,expected):
    if isinstance(expected,dict):
        if set(actual)!=set(expected):raise ValueError('reported metric schema differs')
        for k in expected:close(actual[k],expected[k])
    elif isinstance(expected,(list,tuple)):
        if len(actual)!=len(expected):raise ValueError('reported metric list differs')
        for a,b in zip(actual,expected):close(a,b)
    elif isinstance(expected,(float,int)) and not isinstance(expected,bool):
        if not np.isclose(actual,expected,rtol=0,atol=1e-12):raise ValueError('reported metric arithmetic differs')
    elif actual!=expected:raise ValueError('reported metric/flag differs')

def audit(folder,freeze_path):
    folder=Path(folder);f=json.loads(Path(freeze_path).read_text());s=json.loads((folder/'summary.json').read_text());p=s['protocol']
    if f.get('protocol_version')!=2:raise ValueError('v2 audit requires a fully bound v2 freeze')
    if p!=json.loads((folder/'protocol.json').read_text()) or not s['independent_confirmation_evaluated'] or not s['http_evaluated']:raise ValueError('confirmation protocol differs')
    root=Path(f['confirmation']['directory']);selection=json.loads((root/'manifest.json').read_text());queries=list(jsonl_rows(root/'queries.jsonl'))
    if p['freeze_sha256']!=sha(freeze_path) or p['model_sha256']!=f['model_sha256']:raise ValueError('frozen model binding differs')
    if p['selection_manifest_sha256']!=sha(root/'manifest.json') or p['query_sha256']!=sha(root/'queries.jsonl') or p['qrels_sha256']!=sha(root/'qrels.tsv'):raise ValueError('selection/query/qrels binding differs')
    if p['code_sha256']!=sha(folder/'source.py'):raise ValueError('executed confirmation source snapshot differs')
    if p['queries']!=[r['query_id'] for r in queries] or p['variants']!=VARIANTS:raise ValueError('query/variant protocol differs')
    isolation=verify_frozen(f,freeze_path,selection,queries,root,check_storage=False)
    selection_audit=audit_selection(f,root,selection,queries)
    for key,value in isolation.items():
        if key!='storage_preflight_verified' and p['isolation'].get(key)!=value:raise ValueError('isolation/preflight manifest differs')
    if p['isolation'].get('storage_preflight_verified') is not True:raise ValueError('pre-start storage binding was not checked')
    consumed=json.loads((root/'CONSUMED.json').read_text())
    if consumed.get('phase')!='completed' or consumed.get('freeze_sha256')!=sha(freeze_path) or consumed.get('summary_sha256')!=sha(folder/'summary.json'):raise ValueError('single-use completion marker differs')
    if s['requests_sha256']!=sha(folder/'requests.jsonl'):raise ValueError('HTTP observations changed')
    relevance={}
    for line in (root/'qrels.tsv').read_text().splitlines():
        q,_,d,g=line.split();relevance.setdefault(q,{})[d]=float(g)
    result=recompute(list(jsonl_rows(folder/'requests.jsonl')),p['queries'],p['repeats'],relevance,f['acceptance'])
    for name in ('variants','gains','action_counts','query_action_counts','http_p95_ratio','paired_latency_delta','nonconstant','quality_gate_passed','latency_gate_passed','confirmation_gate_passed'):
        close(result[name],s[name])
    if p['runtime_identity']['serving_sha256']!=f['model_sha256'] or p['runtime_identity']['backend']['reranker']!=f['online_scorer']:raise ValueError('loaded model/runtime policy differs')
    stopped=json.loads((folder/'services-stopped.json').read_text())
    if not stopped['exited']:raise ValueError('owned confirmation Qdrant still running')
    result.update(kind='independent-nested-preBGE-confirmation-audit-v1',queries=len(queries),request_count=len(p['queries'])*p['repeats']*len(VARIANTS),
        immutable_files_checked=len(f['files']),isolation_recomputed=True,selection_manifest_sha256=sha(root/'manifest.json'),
        storage_preflight_manifest_checked=True,confirmation_consumed=True,http_evaluated=True,independent_confirmation_evaluated=True,goal_complete=False)
    result['raw_selection_audit']=selection_audit
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=audit(a.folder,a.freeze)
    with a.out.open('x') as handle:handle.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
