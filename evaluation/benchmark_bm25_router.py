"""Real local HTTP cost/effectiveness for frozen BM25-first policies.

Owns the started Qdrant/HTTP processes only. Reuses existing immutable collections;
does not rebuild indexes, tune models or modify the production defaults.
"""

import argparse
import functools
import hashlib
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

from contracts.retrieval import VectorManifest
from offline.disk_index import DiskIndex
from online.engine import SearchEngine
from online.gain_router import GainRouter
from online.router import RouteDecision
from online.server import make_handler
from online.vector_index import QdrantVectorIndex
from .prepare_bm25_router import sha, metrics, jsonl_rows
from .router_metrics import percentile, bootstrap
from .evaluate import QueryRouting


class BM25ThenDense:
    policy_version='bm25-actions-v1-dense-control'
    requires_statistics=False

    def decide(self,features):
        return RouteDecision(not features.explicit_constraints,'fixed_bm25_then_dense','dense')


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        return sock.getsockname()[1]


def run(args):
    if args.adapter_root:
        # Optional RFC-0004 shared encoder; current repo's contracts/online stay first.
        sys.path.append(str(args.adapter_root.resolve()))
    from adapters.embedding import BGEEncoder
    from qdrant_client import models

    args.out.mkdir(parents=True,exist_ok=False)
    root=args.out
    model=GainRouter(args.model,allow_transfer=args.transfer)
    frozen=json.loads(args.freeze.read_text())
    if model.sha256!=frozen['model_sha256']:
        raise ValueError('model differs from frozen protocol')
    queries=list(jsonl_rows(args.queries))
    if not args.transfer:
        queries=sorted(queries,key=lambda q:hashlib.sha256(('rfc01-http|'+q['query_id']).encode()).hexdigest())[:48]
    qrels={}
    allowed_ids={q['query_id'] for q in queries}
    for line in args.qrels.read_text().splitlines():
        qid,_,docid,grade=line.split()
        if qid in allowed_ids:qrels.setdefault(qid,{})[docid]=int(grade)
    variants={'lexical':('baseline','lexical'),'dense':('baseline','dense'),
              'hybrid':('baseline','hybrid'),'rules-v1':('baseline','adaptive'),
              'gain-router':('learned','adaptive')}
    if args.action_study:
        variants['bm25-dense']=('dense-control','adaptive')
    repeats=1 if args.transfer else 3
    protocol=dict(model_sha256=model.sha256,queries=[q['query_id'] for q in queries],k=100,repeats=repeats,
                  variants=variants,transfer=args.transfer,source_sha256={str(p):sha(p) for p in (args.queries,args.qrels,args.manifest)},
                  cost_caveat='same-process reader/encoder/shared backend; deterministic per-query shuffled order; warm resident indexes; sequential load only')
    (root/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    http,grpc=free_port(),free_port()
    while grpc==http:grpc=free_port()
    url=f'http://127.0.0.1:{http}'
    env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(args.storage.resolve()),
         'QDRANT__STORAGE__SNAPSHOTS_PATH':str((root/'snapshots').resolve()),
         'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(http),
         'QDRANT__SERVICE__GRPC_PORT':str(grpc)}
    log=(root/'qdrant.log').open('w')
    database=subprocess.Popen([str(args.qdrant_binary.resolve()),'--disable-telemetry'],env=env,
                              cwd=root,stdout=log,stderr=subprocess.STDOUT)
    servers=[];vectors=None
    opener=build_opener(ProxyHandler({}))
    try:
        deadline=time.monotonic()+60
        while time.monotonic()<deadline:
            if database.poll() is not None:raise RuntimeError('Qdrant failed: '+(root/'qdrant.log').read_text()[-2000:])
            try:
                with opener.open(url+'/collections',timeout=1):break
            except OSError:time.sleep(.1)
        else:raise TimeoutError('Qdrant readiness')
        print('loading lexical reader',flush=True)
        reader=DiskIndex(args.index)
        print('loading frozen encoder and vector reader',flush=True)
        manifest=VectorManifest.from_dict(json.loads(args.manifest.read_text()))
        encoder=BGEEncoder(manifest.embedding,str(args.model_path),args.device)
        vectors=QdrantVectorIndex(manifest,url)
        vectors.search_params={'hnsw_ef':128,'exact':False}
        # Pin the experimental ANN call while reusing the upstream response validator.
        vectors._client.query_points=functools.partial(vectors._client.query_points,
            search_params=models.SearchParams(**vectors.search_params))
        baseline=SearchEngine(reader,encoder,vectors)
        learned=SearchEngine(reader,encoder,vectors,router=model)
        (root/'engine-info.json').write_text(json.dumps({'baseline':baseline.info(),'learned':learned.info()},ensure_ascii=False,indent=2)+'\n')
        endpoints={}
        engines=[('baseline',baseline),('learned',learned)]
        if args.action_study:
            engines.append(('dense-control',SearchEngine(reader,encoder,vectors,router=BM25ThenDense())))
        for name,engine in engines:
            handler=make_handler(engine)
            handler.log_message=lambda *args:None
            server=ThreadingHTTPServer(('127.0.0.1',0),handler)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            servers.append((server,thread));endpoints[name]=f'http://127.0.0.1:{server.server_port}/search'
        def request(name,query):
            target,mode=variants[name]
            body=json.dumps(dict(query=query,mode=mode,k=100)).encode()
            started=time.perf_counter()
            with opener.open(Request(endpoints[target],body,{'Content-Type':'application/json'}),timeout=120) as response:
                payload=json.load(response)
            elapsed=(time.perf_counter()-started)*1000
            if payload['routing']['fallback']:raise ValueError('fallback invalidates comparison')
            QueryRouting.from_response(payload['routing'],mode,allow_adaptive_dense=args.action_study)
            return payload,elapsed
        for name in variants:
            for query in queries[:3]:request(name,query['text'])
        rows=[]
        with (root/'requests.jsonl').open('w') as output:
            for repeat in range(repeats):
                for i,query in enumerate(queries):
                    order=list(variants)
                    random.Random('rfc01-order|'+str(repeat)+'|'+query['query_id']).shuffle(order)
                    for name in order:
                        response,elapsed=request(name,query['text'])
                        ids=[hit['doc_id'] for hit in response['results']]
                        row=dict(query_id=query['query_id'],variant=name,repeat=repeat,latency_ms=elapsed,
                                 metrics=metrics(ids,qrels[query['query_id']]),routing=response['routing'],
                                 timings_ms=response['timings_ms'],ids=ids)
                        rows.append(row);output.write(json.dumps(row,ensure_ascii=False)+'\n');output.flush()
                    if (i+1)%12==0:print('evaluated',repeat,i+1,'/',len(queries),flush=True)
        summary={}
        for name in variants:
            subset=[r for r in rows if r['variant']==name]
            times=[r['latency_ms'] for r in subset]
            summary[name]=dict(queries=len(queries),requests=len(subset),
                latency_ms=dict(mean=statistics.mean(times),p50=percentile(times,.5),p95=percentile(times,.95)),
                router_mean_ms=statistics.mean(r['timings_ms']['router'] for r in subset),
                vector_attempt_rate=statistics.mean(r['routing']['vector_attempted'] for r in subset),
                metrics={key:statistics.mean(r['metrics'][key] for r in subset) for key in ('ndcg','recall','mrr','map')})
        paired={}
        for name in ('gain-router','rules-v1'):
            latency=[];loss=[]
            for query in queries:
                a=[r for r in rows if r['query_id']==query['query_id'] and r['variant']==name]
                b=[r for r in rows if r['query_id']==query['query_id'] and r['variant']=='hybrid']
                latency.append(statistics.mean(r['latency_ms'] for r in a)-statistics.mean(r['latency_ms'] for r in b))
                loss.append(a[0]['metrics']['ndcg']-b[0]['metrics']['ndcg'])
            paired[name]=dict(delta_latency_ms=bootstrap(latency),delta_ndcg=bootstrap(loss))
        result=dict(protocol=protocol,variants=summary,paired=paired)
        (root/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    finally:
        for server,thread in servers:server.shutdown();server.server_close();thread.join(timeout=5)
        if vectors is not None:vectors._client.close()
        database.terminate()
        try:database.wait(timeout=15)
        except subprocess.TimeoutExpired:database.kill();database.wait(timeout=5)
        log.close()
        (root/'services-stopped.json').write_text(json.dumps(dict(qdrant_pid=database.pid,exited=database.poll() is not None))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('index','manifest','storage','model','freeze','model-path','qdrant-binary','queries','qrels','out'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--adapter-root',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--transfer',action='store_true')
    p.add_argument('--action-study',action='store_true',help='explicit experimental adaptive dense action and BM25-then-dense control')
    run(p.parse_args())
