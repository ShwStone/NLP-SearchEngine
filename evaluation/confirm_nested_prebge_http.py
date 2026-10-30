"""One frozen fully nested pre-BGE Router acceptance run on a newly selected independent confirmation.

All model/config/asset hashes are checked before starting. Relevance is opened
only after all timed HTTP observations. No fitting, policy selection, filtering
by quality, or threshold adjustment occurs in this program.
"""

import argparse
from collections import Counter
import functools
import hashlib
import importlib.metadata
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler,Request,build_opener

import numpy as np

from contracts.retrieval import VectorManifest
from offline.disk_index import DiskIndex
from online.engine import SearchEngine
from online.gain_router import GainRouter
from online.server import make_handler
from online.vector_index import QdrantVectorIndex
from .benchmark_native_router_http import free_port,research_handler
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .prepare_router_transfer import grams
from .runtime_native_router import NativeRouterBackend
from .runtime_nested_prebge import NestedGainPreBGERouter
from .router_metrics import percentile,bootstrap
from .train_bm25_router import group_bootstrap


def verify_frozen(freeze,freeze_path,selection,queries,root,*,check_storage=True):
    if freeze['kind']!='bge-calibration-router-independent-confirmation-freeze-v1':raise ValueError('invalid confirmation freeze')
    for path,digest in freeze['files'].items():
        if sha(path)!=digest:raise ValueError('frozen confirmation file changed: '+path)
    storage_files=freeze.get('storage_files_before_start',{})
    if freeze.get('protocol_version')==2 and (not freeze['files'] or not storage_files or 'environment_path' not in freeze):
        raise ValueError('version2 confirmation freeze lacks code/assets/storage/environment binding')
    if check_storage:
        for path,digest in storage_files.items():
            if sha(path)!=digest:raise ValueError('frozen pre-start Qdrant storage changed: '+path)
    if 'environment_path' in freeze:
        environment=json.loads(Path(freeze['environment_path']).read_text())
        if environment['python']!=sys.version or environment['executable']!=sys.executable:
            raise ValueError('frozen Python runtime changed')
        for package,version in environment['dependencies'].items():
            if importlib.metadata.version(package)!=version:raise ValueError('frozen dependency changed: '+package)
        for name,value in environment.get('environment',{}).items():
            if os.environ.get(name)!=value:raise ValueError('frozen process environment changed: '+name)
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],
            check=True,capture_output=True,text=True).stdout.strip()
        if hardware!=environment['hardware']:raise ValueError('frozen confirmation GPU changed')
        for name,value in environment['cgroup_limits'].items():
            if Path('/sys/fs/cgroup',name).read_text().strip()!=value:raise ValueError('frozen cgroup limits changed')
    if 'history_sha256' in freeze and selection['history_sha256']!=freeze['history_sha256']:
        raise ValueError('frozen exclusion history changed')
    if sha(freeze['model_path'])!=freeze['model_sha256']:raise ValueError('frozen final model changed')
    if selection.get('freeze_sha256')!=sha(freeze_path) or selection.get('model_sha256')!=freeze['model_sha256']:
        raise ValueError('confirmation was not selected under this frozen model')
    policy=freeze['confirmation']
    if selection['hash_salt']!=policy['hash_salt'] or selection['limit']!=policy['limit']:
        raise ValueError('confirmation sampling policy changed')
    if selection['preparation_started_epoch']<freeze['created_epoch']:raise ValueError('confirmation selection predates freeze')
    if not policy['minimum_queries']<=len(queries)<=policy['limit'] or len({r['query_id'] for r in queries})!=len(queries):
        raise ValueError('confirmation query count/identity mismatch')
    if selection['queries']!=len(queries) or queries!=list(jsonl_rows(root/'queries.jsonl')):
        raise ValueError('confirmation query list differs from selected file')
    for name,digest in selection['output_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('fresh confirmation output changed')
    prior_ids=set();history=[];index={}
    def remember(tokens):
        i=len(history);history.append(tokens)
        for token in tokens:index.setdefault(token,set()).add(i)
    for path,digest in selection['history_sha256'].items():
        if sha(path)!=digest:raise ValueError('confirmation exclusion history changed')
        for row in jsonl_rows(path):
            prior_ids.add(row.get('query_id',row.get('qid')))
            remember(grams(row.get('text',row.get('query',''))))
    for row in queries:
        if row['query_id'] in prior_ids:raise ValueError('confirmation reused an observed query ID')
        tokens=grams(row['text']);shared=Counter(i for token in tokens for i in index.get(token,()))
        if any(n/(len(tokens)+len(history[i])-n)>=policy['near_duplicate_threshold'] for i,n in shared.items()):
            raise ValueError('confirmation contains a near duplicate of prior or selected queries')
        remember(tokens)
    result=dict(queries=len(queries),prior_ids=len(prior_ids),prior_or_selected_texts=len(history),
        history_files=len(selection['history_sha256']),id_disjoint=True,near_duplicate_check_passed=True,
        labels_used_for_isolation=False)
    if storage_files:
        result.update(storage_preflight_verified=check_storage,
            storage_preflight_manifest_sha256=hashlib.sha256(json.dumps(storage_files,sort_keys=True,separators=(',',':')).encode()).hexdigest())
    return result


def run(args):
    freeze=json.loads(args.freeze.read_text());root=Path(freeze['confirmation']['directory'])
    if freeze.get('recipe_kind')!='fully-nested-preBGE-final-v1':raise ValueError('wrong final nested recipe')
    if (root/'CONSUMED.json').exists():raise ValueError('BGE confirmation has already been consumed')
    selection=json.loads((root/'manifest.json').read_text());queries=list(jsonl_rows(root/'queries.jsonl'))
    isolation=verify_frozen(freeze,args.freeze,selection,queries,root)
    args.out.mkdir(parents=True,exist_ok=False)
    runtime=freeze['runtime'];corpus=freeze['corpus'];repeats=freeze['confirmation']['repeats']
    sys.path.append(runtime['adapter_root'])
    from adapters.embedding import BGEEncoder
    from qdrant_client import models
    variants=['hybrid','hybrid-compiled','dense','previous-router','bge-calibration-router']
    protocol=dict(kind='fresh-independent-nested-preBGE-confirmation-http-v1',freeze_sha256=sha(args.freeze),
        model_sha256=freeze['model_sha256'],queries=[q['query_id'] for q in queries],repeats=repeats,k=100,
        selection_manifest_sha256=sha(root/'manifest.json'),source=freeze['confirmation']['source'],
        query_sha256=sha(root/'queries.jsonl'),qrels_sha256=sha(root/'qrels.tsv'),variants=variants,
        isolation=isolation,order='deterministic shuffled variants per query/repeat; sequential, warm resident models/indexes',
        no_training_or_tuning=True,confirmation_consumed=False,goal_complete=False,
        code_sha256=sha(Path(__file__)),limitation=freeze['limitation'])
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'source.py').write_bytes(Path(__file__).read_bytes())
    http,grpc=free_port(),free_port()
    while grpc==http:grpc=free_port()
    url=f'http://127.0.0.1:{http}';opener=build_opener(ProxyHandler({}))
    env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':corpus['storage'],
        'QDRANT__STORAGE__SNAPSHOTS_PATH':str((args.out/'snapshots').resolve()),'QDRANT__SERVICE__HOST':'127.0.0.1',
        'QDRANT__SERVICE__HTTP_PORT':str(http),'QDRANT__SERVICE__GRPC_PORT':str(grpc)}
    log=(args.out/'qdrant.log').open('w')
    database=subprocess.Popen([runtime['qdrant_binary'],'--disable-telemetry'],env=env,cwd=args.out,stdout=log,stderr=subprocess.STDOUT)
    servers=[];vectors=None;consumed=False
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if database.poll() is not None:raise RuntimeError('confirmation Qdrant startup failed')
            try:
                with opener.open(url+'/collections',timeout=1):break
            except OSError:time.sleep(.1)
        else:raise TimeoutError('confirmation Qdrant readiness')
        print('confirmation loading immutable corpus and original controls',flush=True)
        reader=DiskIndex(corpus['index'])
        import gc
        gc.collect();gc.freeze()
        manifest=VectorManifest.from_dict(json.loads(Path(corpus['vector_manifest']).read_text()))
        encoder=BGEEncoder(manifest.embedding,runtime['small_model'],'cuda')
        vectors=QdrantVectorIndex(manifest,url);vectors.search_params={'hnsw_ef':runtime['hnsw_ef'],'exact':False}
        vectors._client.query_points=functools.partial(vectors._client.query_points,
            search_params=models.SearchParams(hnsw_ef=runtime['hnsw_ef'],exact=False))
        baseline=SearchEngine(reader,encoder,vectors)
        backend=NativeRouterBackend(reader,runtime['native_model'],runtime['reranker_model'],corpus['native_vectors'],corpus['support_vectors'],
            scorer_batch_size=runtime['scorer_batch_size'],query_graph=runtime.get('query_graph',True),scorer_graph=runtime.get('scorer_graph',False),
            feature_extension=runtime['feature_extension'],fast_support=runtime['fast_support'])
        optimized=SearchEngine(reader,encoder,vectors,bm25_accelerator=backend.compiled)
        previous=SearchEngine(reader,encoder,vectors,router=GainRouter(runtime['previous_router'],allow_transfer=True))
        native=NestedGainPreBGERouter(baseline,backend,Path(freeze['model_path']).parent,'final',
            export_path=runtime['tensorrt_engine'],model_path=runtime['reranker_model'])
        if native.identity['serving_sha256']!=freeze['model_sha256'] or backend.reranker.info!=freeze['online_scorer']:
            raise ValueError('loaded BGE calibration model/engine differs from frozen recipe')
        engines={'baseline':baseline,'compiled':optimized,'previous':previous,'native':native};endpoints={}
        for name,engine in engines.items():
            handler=research_handler(engine) if name=='native' else make_handler(engine)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler);thread=threading.Thread(target=server.serve_forever,daemon=True)
            thread.start();servers.append((server,thread))
            endpoints[name]=f'http://127.0.0.1:{server.server_port}'+('/research/search' if name=='native' else '/search')
        protocol['runtime_identity']=native.identity
        protocol['previous_router_transfer']=True
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        def request(name,row):
            target={'hybrid':'baseline','hybrid-compiled':'compiled','dense':'baseline','previous-router':'previous','bge-calibration-router':'native'}[name]
            mode={'hybrid':'hybrid','hybrid-compiled':'hybrid','dense':'dense','previous-router':'adaptive','bge-calibration-router':'adaptive'}[name]
            body=json.dumps(dict(query=row['text'],mode=mode,k=100)).encode();tick=time.perf_counter()
            with opener.open(Request(endpoints[target],body,{'Content-Type':'application/json'}),timeout=120) as response:result=json.load(response)
            elapsed=(time.perf_counter()-tick)*1000
            if result['routing']['fallback']:raise ValueError('confirmation fallback invalidates comparison')
            if name=='bge-calibration-router' and result.get('research_route',{}).get('delegated',True):raise ValueError('confirmation research query unexpectedly delegated')
            if name=='bge-calibration-router':
                route=result['research_route']
                if (route.get('delegated',True) or route.get('ce_pairs')!=(40 if route.get('action')=='heavy' else 28) or
                        route.get('ce_max_length')!=freeze['online_scorer']['max_length'] or not route.get('one_cross_encoder')):
                    raise ValueError('confirmation shared-prefix execution mismatch')
            return result,elapsed
        with (root/'CONSUMED.json').open('x') as handle:
            json.dump(dict(freeze_sha256=sha(args.freeze),started_epoch=time.time(),phase='HTTP requests starting',output=str(args.out.resolve())),handle)
        consumed=True;protocol['confirmation_consumed']=True
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        print('BGE confirmation consumed under frozen recipe; warming fixed variants',flush=True)
        for name in variants:request(name,queries[0])
        records=[]
        with (args.out/'requests.jsonl').open('w') as handle:
            for repeat in range(repeats):
                for index,row in enumerate(queries):
                    order=list(variants);random.Random(f"fresh-confirmation-BGE-TensorRT-v1|{row['query_id']}|{repeat}").shuffle(order)
                    for name in order:
                        result,elapsed=request(name,row)
                        record=dict(query_id=row['query_id'],repeat=repeat,variant=name,latency_ms=elapsed,
                            ranked=[r['doc_id'] for r in result['results']],routing=result['routing'],
                            research_route=result.get('research_route'),timings_ms=result['timings_ms'],audit=result.get('audit'))
                        records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                    if (index+1)%16==0:print('confirmation HTTP queries',repeat+1,index+1,'/',len(queries),flush=True)
        # All fresh BGE predictions/timings are finalized before labels are read.
        relevance={};selected={q['query_id'] for q in queries}
        with (root/'qrels.tsv').open() as handle:
            for line in handle:
                q,_,d,g=line.split()
                if q in selected:relevance.setdefault(q,{})[d]=float(g)
        if set(relevance)!=selected:raise ValueError('fresh confirmation relevance coverage mismatch')
        lookup={(r['query_id'],r['repeat'],r['variant']):r for r in records}
        if len(lookup)!=len(queries)*repeats*len(variants):raise ValueError('fresh confirmation HTTP observations unbalanced')
        metrics_by_variant={};quality={};groups=np.array([q['query_id'] for q in queries])
        for name in variants:
            rows=[r for r in records if r['variant']==name];times=[r['latency_ms'] for r in rows]
            quality[name]={m:np.array([np.mean([metrics(lookup[q['query_id'],r,name]['ranked'],relevance[q['query_id']])[m]
                for r in range(repeats)]) for q in queries]) for m in ('ndcg','recall')}
            metrics_by_variant[name]=dict(metrics={m:float(v.mean()) for m,v in quality[name].items()},
                latency_ms=dict(mean=float(np.mean(times)),p95=percentile(times,.95)))
        gains={}
        for m in ('ndcg','recall'):
            actual=quality['bge-calibration-router'][m];reference=quality['hybrid'][m]
            gains[m]=dict(value=float(actual.mean()),baseline=float(reference.mean()),
                relative_lift=float(actual.mean()/reference.mean()-1),delta_ci=group_bootstrap(actual-reference,groups),
                passes_20_percent=bool(actual.mean()>reference.mean()*1.2))
        native_rows=[r for r in records if r['variant']=='bge-calibration-router']
        actions={name:sum(r['research_route']['action']==name for r in native_rows) for name in ('cheap','heavy')}
        query_actions={name:sum(lookup[q['query_id'],0,'bge-calibration-router']['research_route']['action']==name for q in queries) for name in actions}
        ratio=metrics_by_variant['bge-calibration-router']['latency_ms']['p95']/metrics_by_variant['hybrid']['latency_ms']['p95']
        quality_pass=all(v['passes_20_percent'] and v['delta_ci']['ci95'][0]>0 for v in gains.values())
        nonconstant=all(query_actions.values());latency_pass=ratio<=freeze['acceptance']['http_p95_ratio_max']
        compiled=sum(lookup[q['query_id'],r,'hybrid']['ranked']!=lookup[q['query_id'],r,'hybrid-compiled']['ranked'] for q in queries for r in range(repeats))
        paired_latency=[]
        for q in queries:
            paired_latency.append(float(np.mean([lookup[q['query_id'],r,'bge-calibration-router']['latency_ms']-lookup[q['query_id'],r,'hybrid']['latency_ms'] for r in range(repeats)])))
        result=dict(protocol=protocol,variants=metrics_by_variant,gains=gains,action_counts=actions,query_action_counts=query_actions,
            nonconstant=bool(nonconstant),http_p95_ratio=ratio,paired_latency_delta=bootstrap(paired_latency),
            quality_gate_passed=bool(quality_pass),latency_gate_passed=bool(latency_pass),
            confirmation_gate_passed=bool(quality_pass and nonconstant and latency_pass and compiled==0),
            compiled_hybrid_ranking_mismatches=compiled,
            requests_sha256=sha(args.out/'requests.jsonl'),http_evaluated=True,independent_confirmation_evaluated=True,
            no_confirmation_training_or_tuning=True,goal_complete=False)
        (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (root/'CONSUMED.json').write_text(json.dumps(dict(freeze_sha256=sha(args.freeze),phase='completed',
            summary_sha256=sha(args.out/'summary.json'),output=str(args.out.resolve())),indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='protocol'},indent=2),flush=True)
    finally:
        for server,thread in servers:server.shutdown();server.server_close();thread.join(timeout=5)
        if vectors is not None:vectors._client.close()
        database.terminate()
        try:database.wait(timeout=15)
        except subprocess.TimeoutExpired:database.kill();database.wait(timeout=5)
        log.close();(args.out/'services-stopped.json').write_text(json.dumps(dict(qdrant_pid=database.pid,exited=database.poll() is not None,confirmation_consumed=consumed))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);run(parser.parse_args())
