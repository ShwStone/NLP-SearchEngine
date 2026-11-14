import argparse,json,os,subprocess,time,urllib.request,statistics,threading
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from contracts.api import SearchRequest
from contracts.retrieval import VectorManifest,RetrievalConfig
from online.engine import SearchEngine
from offline.disk_index import DiskIndex
from online.vector_index import QdrantVectorIndex
from adapters.embedding import BGEEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import percentile,bootstrap
p=argparse.ArgumentParser();p.add_argument('--index-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--qdrant',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.index_root;data=a.data_root;port=6397
queries=[json.loads(x) for x in (data/'queries.dev.jsonl').read_text().splitlines()];qrels={}
for line in (data/'qrels.dev.tsv').read_text().splitlines():
 q,_,d,g=line.split();qrels.setdefault(q,{})[d]=float(g)
log=(root/'http-router-qdrant.log').open('w');proc=subprocess.Popen([str(a.qdrant),'--disable-telemetry'],env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(root/'qdrant-storage'),'QDRANT__STORAGE__SNAPSHOTS_PATH':str(root/'qdrant-snapshots'),'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(port),'QDRANT__SERVICE__GRPC_PORT':str(port+1)},stdout=log,stderr=subprocess.STDOUT,cwd=root)
try:
 for _ in range(200):
  if proc.poll() is not None:raise RuntimeError(log.read_text()[-2000:])
  try:urllib.request.urlopen(f'http://127.0.0.1:{port}/collections',timeout=1);break
  except:time.sleep(.2)
 reader=DiskIndex(root/'bm25');manifest=VectorManifest.from_dict(json.loads((root/'vector_manifest.json').read_text()));enc=BGEEncoder(manifest.embedding,str(a.model),'cuda',batch_size=32);vec=QdrantVectorIndex(manifest,f'http://127.0.0.1:{port}');vec.search_params={'hnsw_ef':128,'exact':False}
 engines={name:SearchEngine(reader,enc,vec,config=RetrievalConfig(adaptive_policy=name)) for name in ('rules-v1','hybrid-v1')}
 class H(BaseHTTPRequestHandler):
  def do_POST(self):
   try:
    b=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))));policy=b.get('policy','rules-v1');t=time.perf_counter();r=engines[policy].search(SearchRequest(query=b['query'],mode='adaptive',k=100));e=(time.perf_counter()-t)*1000;payload={'results':[x.doc_id for x in r.results],'routing':r.routing.__dict__,'timings_ms':r.timings_ms,'e2e_ms':e};raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
   except Exception as ex:
    raw=json.dumps({'error':str(ex)}).encode();self.send_response(500);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  def log_message(self,*x):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}';op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 for policy in engines:op.open(urllib.request.Request(url,json.dumps({'policy':policy,'query':queries[0]['query']}).encode()),timeout=120).read()
 rows={p:[] for p in engines}
 for i,q in enumerate(queries):
  qid=q.get('qid',q.get('query_id'))
  for policy in engines:
   b=json.dumps({'policy':policy,'query':q.get('query',q.get('text'))}).encode();resp=json.load(op.open(urllib.request.Request(url,b),timeout=120));rows[policy].append({'qid':qid,'metrics':metrics(resp['results'],qrels[qid]),'e2e':resp['e2e_ms'],'routing':resp['routing']})
  if (i+1)%100==0:print('router http',i+1,'/',len(queries),flush=True)
 summary={}
 for policy,rs in rows.items():
  summary[policy]={'queries':len(rs),'metrics':{m:statistics.mean(r['metrics'][m] for r in rs) for m in ('ndcg','recall','mrr','map')},'e2e_ms':{'p50':percentile([r['e2e'] for r in rs],.5),'p95':percentile([r['e2e'] for r in rs],.95)},'vector_attempt_rate':statistics.mean(r['routing']['vector_attempted'] for r in rs),'executed_modes':{m:sum(r['routing']['executed_mode']==m for r in rs) for m in ('lexical','hybrid','dense')}}
 summary['paired_ndcg_hybrid_minus_rules']=bootstrap([a['metrics']['ndcg']-b['metrics']['ndcg'] for a,b in zip(rows['hybrid-v1'],rows['rules-v1'])]);a.out.mkdir();(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
finally:
 try:server.shutdown();server.server_close()
 except:pass
 try:proc.terminate();proc.wait(timeout=20)
 except:proc.kill()
 log.close()
