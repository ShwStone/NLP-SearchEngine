import argparse,json,os,subprocess,time,urllib.request,urllib.error,statistics,threading,hashlib,random
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from contracts.api import SearchRequest
from contracts.retrieval import VectorManifest
from online.engine import SearchEngine
from offline.disk_index import DiskIndex
from online.vector_index import QdrantVectorIndex
from adapters.embedding import BGEEncoder
from evaluation.study_rerank_actions import FrozenCrossEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import percentile,bootstrap
p=argparse.ArgumentParser();p.add_argument('--index-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--small-model',type=Path,required=True);p.add_argument('--ce-model',type=Path,required=True);p.add_argument('--qdrant',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.index_root;data=a.data_root;port=6389
queries=[json.loads(x) for x in (data/'queries.dev.jsonl').read_text().splitlines()];qrels={}
for line in (data/'qrels.dev.tsv').read_text().splitlines():
 q,_,d,g=line.split();qrels.setdefault(q,{})[d]=float(g)
docs={}
for line in (root/'bm25/docs.jsonl').open():
 x=json.loads(line);docs[x['doc_id']]=x['title']+'\n'+x['text']
log=(root/'http-ce-qdrant.log').open('w');proc=subprocess.Popen([str(a.qdrant),'--disable-telemetry'],env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(root/'qdrant-storage'),'QDRANT__STORAGE__SNAPSHOTS_PATH':str(root/'qdrant-snapshots'),'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(port),'QDRANT__SERVICE__GRPC_PORT':str(port+1)},stdout=log,stderr=subprocess.STDOUT,cwd=root)
try:
 for _ in range(200):
  if proc.poll() is not None:raise RuntimeError(log.read_text()[-2000:])
  try:urllib.request.urlopen(f'http://127.0.0.1:{port}/collections',timeout=1);break
  except:time.sleep(.2)
 reader=DiskIndex(root/'bm25');manifest=VectorManifest.from_dict(json.loads((root/'vector_manifest.json').read_text()));enc=BGEEncoder(manifest.embedding,str(a.small_model),'cuda',batch_size=32);vec=QdrantVectorIndex(manifest,f'http://127.0.0.1:{port}');vec.search_params={'hnsw_ef':128,'exact':False};engine=SearchEngine(reader,enc,vec);ce=FrozenCrossEncoder(a.ce_model,device='cuda',batch_size=20,max_length=256)
 class H(BaseHTTPRequestHandler):
  def do_POST(self):
   try:
    body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))));q=body['query'];mode=body.get('mode','ce40');t=time.perf_counter();base=engine.search(SearchRequest(query=q,mode='hybrid',k=100));pool=[x.doc_id for x in base.results];chosen=pool[:({'ce0':0,'ce28':28,'ce40':40}[mode])];ce_ms=0
    if chosen:
     vals,ce_ms=ce.score(q,[docs[d] for d in chosen]);ranked=[d for _,d in sorted(zip(vals,chosen),key=lambda z:(-z[0],z[1]))]+pool[len(chosen):]
    else: ranked=pool
    elapsed=(time.perf_counter()-t)*1000;payload={'results':ranked[:100],'timings_ms':{'e2e':elapsed,'ce':ce_ms,'retrieval':max(0,elapsed-ce_ms)}};raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
   except Exception as e:
    raw=json.dumps({'error':str(e)}).encode();self.send_response(500);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  def log_message(self,*x):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}/search';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 # warm each action
 for mode in ('ce0','ce28','ce40'): opener.open(urllib.request.Request(url,json.dumps({'query':queries[0]['query'],'mode':mode}).encode(),{'Content-Type':'application/json'})).read()
 rows={m:[] for m in ('ce0','ce28','ce40')}
 for i,q in enumerate(queries):
  for mode in rows:
   started=time.perf_counter();resp=json.load(opener.open(urllib.request.Request(url,json.dumps({'query':q['query'],'mode':mode}).encode(),{'Content-Type':'application/json'}),timeout=120));wall=(time.perf_counter()-started)*1000;ids=resp['results'];rows[mode].append({'qid':q.get('qid',q.get('query_id')),'metrics':metrics(ids,qrels[q.get('qid',q.get('query_id'))]),'latency_ms':wall,'ce_ms':resp['timings_ms']['ce'],'retrieval_ms':resp['timings_ms']['retrieval']})
  if (i+1)%100==0:print('http evaluated',i+1,'/',len(queries),flush=True)
 summary={}
 for mode,rs in rows.items():
  summary[mode]={'queries':len(rs),'metrics':{m:statistics.mean(r['metrics'][m] for r in rs) for m in ('ndcg','recall','mrr','map')},'e2e_ms':{'p50':percentile([r['latency_ms'] for r in rs],.5),'p95':percentile([r['latency_ms'] for r in rs],.95),'mean':statistics.mean(r['latency_ms'] for r in rs)},'ce_ms':{'p50':percentile([r['ce_ms'] for r in rs],.5),'p95':percentile([r['ce_ms'] for r in rs],.95)},'retrieval_ms':{'p50':percentile([r['retrieval_ms'] for r in rs],.5),'p95':percentile([r['retrieval_ms'] for r in rs],.95)}}
 for mode in ('ce28','ce40'):
  summary[mode]['paired_ndcg_vs_ce0']=bootstrap([a['metrics']['ndcg']-b['metrics']['ndcg'] for a,b in zip(rows[mode],rows['ce0'])])
 a.out.mkdir(parents=True);(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
finally:
 try:server.shutdown();server.server_close()
 except:pass
 try:proc.terminate();proc.wait(timeout=20)
 except:proc.kill()
 log.close()
