"""Real HTTP paired replay of outer-heldout native length Router and controls.

Research response schema is explicit. Runtime receives query text only; fold
endpoints hold outer-trained models. No captured features/predictions are used
in requests. Saved OOF rankings are opened only after all HTTP observations.
"""

import argparse
import functools
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler,Request,build_opener

import numpy as np

from contracts.api import ApiError,SearchRequest
from contracts.retrieval import RetrievalUnavailable,VectorManifest
from offline.disk_index import DiskIndex
from online.engine import SearchEngine
from online.gain_router import GainRouter
from online.parser import QuerySyntaxError
from online.server import make_handler
from online.vector_index import QdrantVectorIndex
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .router_metrics import percentile,bootstrap
from .runtime_native_router import NativeRouterBackend,NativeLengthRouter


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return sock.getsockname()[1]


def research_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass

        def do_POST(self):
            if self.path!='/research/search':return self.send_payload(404,dict(error='unknown research path'))
            try:
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<1024*1024:raise ApiError('invalid request length')
                request=SearchRequest.from_dict(json.loads(self.rfile.read(length)))
                result=engine.search(request)
            except (ApiError,QuerySyntaxError,json.JSONDecodeError) as error:
                return self.send_payload(400,dict(error=str(error)))
            except RetrievalUnavailable as error:
                return self.send_payload(503,dict(error=str(error),fallback=False))
            self.send_payload(200,result)

        def send_payload(self,code,payload):
            body=json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(code);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    return Handler


def run(args):
    if args.limit<0 or args.repeats<1:raise ValueError('invalid HTTP query/repeat count')
    args.out.mkdir(parents=True,exist_ok=False)
    sys.path.append(str(args.adapter_root.resolve()))
    from adapters.embedding import BGEEncoder
    from qdrant_client import models
    study=json.loads((args.study/'study.json').read_text())
    table={r['query_id']:r for r in jsonl_rows(args.table)}
    text={r['query_id']:r['text'] for r in jsonl_rows(args.capture) if r['query_id'] in table}
    if set(text)!=set(table):raise ValueError('HTTP development text coverage mismatch')
    fold_for={q:f['fold'] for f in study['folds'] for q in f['heldout_ids']}
    if set(fold_for)!=set(table):raise ValueError('HTTP fold query coverage mismatch')
    import hashlib
    queries=sorted(table,key=lambda q:hashlib.sha256(('rfc01-native-http|'+q).encode()).hexdigest())
    if args.limit:queries=queries[:args.limit]
    source={str(p):sha(p) for p in (args.study/'study.json',args.table,args.capture,args.manifest,args.previous_router,args.qrels)}
    protocol=dict(kind='real-http-outer-native-length-router-v1',queries=queries,repeats=args.repeats,
        inputs=source,k=100,fold_endpoints={q:fold_for[q] for q in queries},
        variants=['hybrid','hybrid-compiled','dense','previous-router','native-length-router'],
        timing='client wall time, actual HTTP including encode/retrieval/gate/selected CE/calibration/highlights/JSON',
        order='deterministic shuffled per query/repeat; sequential warm resident models and indexes',
        acceptance_reference='original fixed equal-RRF H with original scorer; compiled H also reported as same-quality stronger cost control',
        live_query_features=True,cached_query_vectors=False,cached_query_predictions=False,
        quality_replay_all=args.quality_replay_all,
        freeze_immutable_index_gc=args.freeze_index_gc,
        confirmation_read=False,goal_complete=False,code_sha256={str(p):sha(p) for p in (
            Path(__file__),Path(__file__).with_name('runtime_native_router.py'),Path(__file__).with_name('study_candidate_ranking.py'),
            Path(__file__).resolve().parents[1]/'online'/'engine.py',Path(__file__).resolve().parents[1]/'online'/'compiled_bm25.py',
            Path(__file__).resolve().parents[1]/'online'/'highlight.py',
            Path(__file__).with_name('cuda_graph_native_query.py'),Path(__file__).with_name('cuda_graph_cross_encoder.py'),
            Path(__file__).with_name('prepare_candidate_support.py'))})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (args.out/'code').mkdir()
    for path in protocol['code_sha256']:(args.out/'code'/Path(path).name).write_bytes(Path(path).read_bytes())
    http,grpc=free_port(),free_port()
    while grpc==http:grpc=free_port()
    url=f'http://127.0.0.1:{http}'
    env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(args.storage.resolve()),
        'QDRANT__STORAGE__SNAPSHOTS_PATH':str((args.out/'snapshots').resolve()),'QDRANT__SERVICE__HOST':'127.0.0.1',
        'QDRANT__SERVICE__HTTP_PORT':str(http),'QDRANT__SERVICE__GRPC_PORT':str(grpc)}
    log=(args.out/'qdrant.log').open('w')
    database=subprocess.Popen([str(args.qdrant_binary.resolve()),'--disable-telemetry'],env=env,cwd=args.out,stdout=log,stderr=subprocess.STDOUT)
    opener=build_opener(ProxyHandler({}));servers=[];vectors=None
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if database.poll() is not None:raise RuntimeError('Qdrant failed: '+(args.out/'qdrant.log').read_text()[-2000:])
            try:
                with opener.open(url+'/collections',timeout=1):break
            except OSError:time.sleep(.1)
        else:raise TimeoutError('Qdrant readiness')
        startup={};tick=time.perf_counter()
        print('HTTP loading checked lexical index',flush=True);reader=DiskIndex(args.index)
        startup['checked_lexical_reader_seconds']=time.perf_counter()-tick
        if args.freeze_index_gc:
            import gc
            tick=time.perf_counter();gc.collect();gc.freeze()
            startup['freeze_immutable_reader_gc_seconds']=time.perf_counter()-tick
        tick=time.perf_counter()
        manifest=VectorManifest.from_dict(json.loads(args.manifest.read_text()))
        encoder=BGEEncoder(manifest.embedding,str(args.small_model_path),'cuda')
        vectors=QdrantVectorIndex(manifest,url)
        vectors.search_params={'hnsw_ef':128,'exact':False}
        vectors._client.query_points=functools.partial(vectors._client.query_points,search_params=models.SearchParams(hnsw_ef=128,exact=False))
        baseline=SearchEngine(reader,encoder,vectors)
        startup['original_encoder_vector_binding_seconds']=time.perf_counter()-tick
        print('HTTP loading frozen native backend',flush=True)
        tick=time.perf_counter()
        backend=NativeRouterBackend(reader,args.native_model_path,args.reranker_path,args.vectors_dir,args.support_vectors,
            scorer_batch_size=args.scorer_batch_size,query_graph=args.query_graph,scorer_graph=args.scorer_graph,
            feature_extension=args.feature_extension,fast_support=args.fast_support)
        startup['native_backend_seconds']=time.perf_counter()-tick
        optimized=SearchEngine(reader,encoder,vectors,bm25_accelerator=backend.compiled)
        # Historical cross-corpus controls may carry the same learned gate but
        # a different corpus binding. Keep the control visible while requiring
        # the router to use the current index's explicit transfer path.
        previous=SearchEngine(reader,encoder,vectors,router=GainRouter(args.previous_router,allow_transfer=True))
        engines={'baseline':baseline,'compiled':optimized,'previous':previous}
        for fold in sorted(set(fold_for.values()) if args.quality_replay_all else {fold_for[q] for q in queries}):
            engines[f'fold-{fold}']=NativeLengthRouter(baseline,backend,args.study,fold)
        endpoints={}
        for name,engine in engines.items():
            handler=research_handler(engine) if name.startswith('fold-') else make_handler(engine)
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();servers.append((server,thread))
            endpoints[name]=f'http://127.0.0.1:{server.server_port}'+('/research/search' if name.startswith('fold-') else '/search')
        protocol['runtime_identity']=backend.identity
        protocol['startup_seconds']=startup
        protocol['router_identities']={name:engine.identity for name,engine in engines.items() if name.startswith('fold-')}
        (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        def request(name,q):
            if name=='native-length-router':target=f'fold-{fold_for[q]}';mode='adaptive'
            else:
                target={'hybrid':'baseline','hybrid-compiled':'compiled','dense':'baseline','previous-router':'previous'}[name]
                mode={'hybrid':'hybrid','hybrid-compiled':'hybrid','dense':'dense','previous-router':'adaptive'}[name]
            body=json.dumps(dict(query=text[q],mode=mode,k=100)).encode()
            tick=time.perf_counter()
            with opener.open(Request(endpoints[target],body,{'Content-Type':'application/json'}),timeout=120) as response:
                result=json.load(response)
            elapsed=(time.perf_counter()-tick)*1000
            if result['routing']['fallback']:raise ValueError('HTTP fallback invalidates comparison')
            if name=='native-length-router' and result.get('research_route',{}).get('delegated',True):
                raise ValueError('native HTTP request delegated unexpectedly')
            return result,elapsed
        print('HTTP warming all variants/fold endpoints',flush=True)
        for name in protocol['variants']:request(name,queries[0])
        for fold in {fold_for[q] for q in queries}:
            request('native-length-router',next(q for q in queries if fold_for[q]==fold))
        records=[]
        with (args.out/'requests.jsonl').open('w') as handle:
            for repeat in range(args.repeats):
                for index,q in enumerate(queries):
                    order=list(protocol['variants']);random.Random(f'native-router-http|{q}|{repeat}').shuffle(order)
                    for name in order:
                        result,elapsed=request(name,q)
                        record=dict(query_id=q,repeat=repeat,variant=name,latency_ms=elapsed,
                            ranked=[r['doc_id'] for r in result['results']],routing=result['routing'],
                            research_route=result.get('research_route'),timings_ms=result['timings_ms'],audit=result.get('audit'))
                        records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                    if (index+1)%8==0:print('HTTP queries',repeat+1,index+1,'/',len(queries),flush=True)
        quality_records=[]
        if args.quality_replay_all:
            print('HTTP full development quality replay',flush=True)
            with (args.out/'quality-replay.jsonl').open('w') as handle:
                for index,q in enumerate(fold_for):
                    result,elapsed=request('native-length-router',q)
                    record=dict(query_id=q,variant='native-length-router',repeat='full-quality',latency_ms=elapsed,
                        ranked=[r['doc_id'] for r in result['results']],routing=result['routing'],
                        research_route=result['research_route'],timings_ms=result['timings_ms'],audit=result.get('audit'))
                    quality_records.append(record);handle.write(json.dumps(record)+'\n');handle.flush()
                    if (index+1)%100==0:print('HTTP full quality',index+1,'/',len(fold_for),flush=True)
        # Check labels and historical OOF predictions only after HTTP replay.
        reference={r['query_id']:r for r in jsonl_rows(args.study/'outer-predictions.jsonl')}
        relevance={};selected=set(fold_for) if args.quality_replay_all else set(queries)
        with args.qrels.open() as handle:
            for line in handle:
                q,_,d,g=line.split()
                if q in selected:relevance.setdefault(q,{})[d]=float(g)
        if set(relevance)!=selected:raise ValueError('HTTP qrels coverage mismatch')
        variants={};lookup={(r['query_id'],r['repeat'],r['variant']):r for r in records}
        for name in protocol['variants']:
            rows=[r for r in records if r['variant']==name];times=[r['latency_ms'] for r in rows]
            stages=sorted({k for r in rows for k in r['timings_ms']})
            variants[name]=dict(queries=len(queries),requests=len(rows),latency_ms=dict(mean=float(np.mean(times)),p95=percentile(times,.95)),
                metrics={m:float(np.mean([metrics(r['ranked'],relevance[r['query_id']])[m] for r in rows])) for m in ('ndcg','recall')},
                stage_mean_ms={k:float(np.mean([r['timings_ms'].get(k,0.) for r in rows])) for k in stages})
        native=[r for r in records if r['variant']=='native-length-router']
        mismatch=[dict(query_id=r['query_id'],repeat=r['repeat'],action=r['research_route']['action'],
            expected_action=reference[r['query_id']]['action'],top10_changed=r['ranked'][:10]!=reference[r['query_id']]['ranked'][:10])
            for r in native+quality_records if r['ranked'][:10]!=reference[r['query_id']]['ranked'][:10] or r['research_route']['action']!=reference[r['query_id']]['action']]
        compiled_mismatch=sum(lookup[q,r,'hybrid']['ranked']!=lookup[q,r,'hybrid-compiled']['ranked'] for q in queries for r in range(args.repeats))
        paired={}
        for name in protocol['variants'][1:]:
            differences=[float(np.mean([lookup[q,r,name]['latency_ms']-lookup[q,r,'hybrid']['latency_ms'] for r in range(args.repeats)])) for q in queries]
            paired[name]=dict(latency_delta=bootstrap(differences),p95_ratio=variants[name]['latency_ms']['p95']/variants['hybrid']['latency_ms']['p95'])
        counts={action:sum(r['research_route']['action']==action for r in native) for action in ('cheap','heavy')}
        result=dict(protocol=protocol,variants=variants,paired=paired,action_counts=counts,
            outer_replay_mismatches=mismatch,compiled_hybrid_ranking_mismatches=compiled_mismatch,
            requests_sha256=sha(args.out/'requests.jsonl'),http_evaluated=True,confirmation_evaluated=False,
            latency_gate_passed=bool(paired['native-length-router']['p95_ratio']<=1.05),goal_complete=False)
        if quality_records:
            from .train_bm25_router import group_bootstrap
            groups=np.array([table[r['query_id']]['group'] for r in quality_records]);full={}
            for m in ('ndcg','recall'):
                values=np.array([metrics(r['ranked'],relevance[r['query_id']])[m] for r in quality_records])
                baseline_values=np.array([table[r['query_id']]['hybrid'][m] for r in quality_records])
                full[m]=dict(value=float(values.mean()),relative_lift=float(values.mean()/baseline_values.mean()-1),
                    delta_ci=group_bootstrap(values-baseline_values,groups),passes_20_percent=bool(values.mean()>baseline_values.mean()*1.2))
            result['full_quality_replay']=dict(queries=len(quality_records),metrics=full,
                action_counts={action:sum(r['research_route']['action']==action for r in quality_records) for action in ('cheap','heavy')},
                file_sha256=sha(args.out/'quality-replay.jsonl'))
        (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='protocol'},indent=2),flush=True)
    finally:
        for server,thread in servers:server.shutdown();server.server_close();thread.join(timeout=5)
        if vectors is not None:vectors._client.close()
        database.terminate()
        try:database.wait(timeout=15)
        except subprocess.TimeoutExpired:database.kill();database.wait(timeout=5)
        log.close();(args.out/'services-stopped.json').write_text(json.dumps(dict(qdrant_pid=database.pid,exited=database.poll() is not None))+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('index','manifest','storage','qdrant-binary','small-model-path','native-model-path','reranker-path',
                 'vectors-dir','support-vectors','study','previous-router','table','capture','qrels','adapter-root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--limit',type=int,default=48);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--quality-replay-all',action='store_true')
    parser.add_argument('--scorer-batch-size',type=int,default=16)
    parser.add_argument('--query-graph',action='store_true');parser.add_argument('--scorer-graph',action='store_true')
    parser.add_argument('--feature-extension',type=Path);parser.add_argument('--fast-support',action='store_true')
    parser.add_argument('--freeze-index-gc',action='store_true',help='freeze immutable reader objects before graph/model construction; applies to all variants')
    run(parser.parse_args())
