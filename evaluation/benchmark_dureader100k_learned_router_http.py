import argparse,json,os,subprocess,time,urllib.request,statistics,threading,random
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from contracts.api import SearchRequest
from contracts.retrieval import VectorManifest
from online.engine import SearchEngine
from offline.disk_index import DiskIndex
from online.vector_index import QdrantVectorIndex
from adapters.embedding import BGEEncoder
from contracts import analyzer
from online.scorers import bm25
from online.router_features import extract_statistics
from evaluation.study_rerank_actions import FrozenCrossEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import percentile,bootstrap
p=argparse.ArgumentParser();p.add_argument('--index-root',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--train-capture',type=Path,required=True);p.add_argument('--dev-capture',type=Path,required=True);p.add_argument('--small-model',type=Path,required=True);p.add_argument('--ce-model',type=Path,required=True);p.add_argument('--qdrant',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.index_root;data=a.data_root;port=6411

def qrels(path):
 d={}
 for line in path.read_text().splitlines():
  q,_,doc,g=line.split();d.setdefault(q,{})[doc]=float(g)
 return d
trq=qrels(data/'qrels.train.tsv');dvq=qrels(data/'qrels.dev.tsv');train=[json.loads(x) for x in a.train_capture.read_text().splitlines()];dev=[json.loads(x) for x in a.dev_capture.read_text().splitlines()]
docs={}
for line in (root/'bm25/docs.jsonl').open():
 x=json.loads(line);docs[x['doc_id']]=x['title']+'\n'+x['text']
reader=DiskIndex(root/'bm25');ce=FrozenCrossEncoder(a.ce_model,device='cuda',batch_size=20,max_length=256)
def fvec(q):
 terms=[t.term for t in analyzer.analyze(q)];ranked=sorted(bm25(reader,terms).items(),key=lambda z:(-z[1],z[0]))[:200];s=extract_statistics(q,reader,terms,ranked);return [s['n_terms'],s['vocab_coverage'],s['relative_gap'] or 0.,s['candidate_hits']]
# Fit labels only on train: whether CE40 improves over hybrid for that query.
X=[];y=[]
for i,row in enumerate(train):
 pool=[x['doc_id'] for x in row['hybrid'][:100]];chosen=pool[:40];scores,_=ce.score(row['text'],[docs[d] for d in chosen]);reranked=[d for _,d in sorted(zip(scores,chosen),key=lambda z:(-z[0],z[1]))]+pool[40:];gain=metrics(reranked,trq[row['query_id']])['ndcg']-metrics(pool,trq[row['query_id']])['ndcg'];X.append(fvec(row['text']));y.append(int(gain>0))
router=make_pipeline(StandardScaler(),LogisticRegression(C=.5,max_iter=1000,class_weight='balanced')).fit(np.array(X),np.array(y))
log=(root/'http-learned-qdrant.log').open('w');proc=subprocess.Popen([str(a.qdrant),'--disable-telemetry'],env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(root/'qdrant-storage'),'QDRANT__STORAGE__SNAPSHOTS_PATH':str(root/'qdrant-snapshots'),'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(port),'QDRANT__SERVICE__GRPC_PORT':str(port+1)},stdout=log,stderr=subprocess.STDOUT,cwd=root)
try:
 for _ in range(200):
  if proc.poll() is not None:raise RuntimeError(log.read_text()[-2000:])
  try:urllib.request.urlopen(f'http://127.0.0.1:{port}/collections',timeout=1);break
  except:time.sleep(.2)
 manifest=VectorManifest.from_dict(json.loads((root/'vector_manifest.json').read_text()));enc=BGEEncoder(manifest.embedding,str(a.small_model),'cuda',batch_size=32);vec=QdrantVectorIndex(manifest,f'http://127.0.0.1:{port}');vec.search_params={'hnsw_ef':128,'exact':False};engine=SearchEngine(reader,enc,vec)
 class H(BaseHTTPRequestHandler):
  def do_POST(self):
   try:
    b=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))));q=b['query'];t=time.perf_counter();base=engine.search(SearchRequest(query=q,mode='hybrid',k=100));pool=[x.doc_id for x in base.results];heavy=bool(router.predict(np.array([fvec(q)]))[0]);ce_ms=0
    if heavy:
     chosen=pool[:40];scores,ce_ms=ce.score(q,[docs[d] for d in chosen]);ids=[d for _,d in sorted(zip(scores,chosen),key=lambda z:(-z[0],z[1]))]+pool[40:]
    else:ids=pool
    payload={'results':ids[:100],'heavy':heavy,'ce_ms':ce_ms,'e2e_ms':(time.perf_counter()-t)*1000};raw=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
   except Exception as e:
    raw=json.dumps({'error':str(e)}).encode();self.send_response(500);self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
  def log_message(self,*x):pass
 server=ThreadingHTTPServer(('127.0.0.1',0),H);threading.Thread(target=server.serve_forever,daemon=True).start();url=f'http://127.0.0.1:{server.server_port}';op=urllib.request.build_opener(urllib.request.ProxyHandler({}));op.open(urllib.request.Request(url,json.dumps({'query':dev[0]['text']}).encode()),timeout=120).read()
 rows=[]
 for i,row in enumerate(dev):
  qid=row['query_id'];resp=json.load(op.open(urllib.request.Request(url,json.dumps({'query':row['text']}).encode()),timeout=120));rows.append({'qid':qid,'metrics':metrics(resp['results'],dvq[qid]),'heavy':resp['heavy'],'e2e':resp['e2e_ms'],'ce':resp['ce_ms']})
  if (i+1)%100==0:print('learned http',i+1,'/',len(dev),flush=True)
 summary={'queries':len(rows),'metrics':{m:statistics.mean(r['metrics'][m] for r in rows) for m in ('ndcg','recall','mrr','map')},'heavy_rate':statistics.mean(r['heavy'] for r in rows),'e2e_ms':{'p50':percentile([r['e2e'] for r in rows],.5),'p95':percentile([r['e2e'] for r in rows],.95)},'ce_ms':{'p50':percentile([r['ce'] for r in rows if r['heavy']],.5),'p95':percentile([r['ce'] for r in rows if r['heavy']],.95)},'train_positive_rate':statistics.mean(y),'paired_delta_vs_hybrid':bootstrap([r['metrics']['ndcg']-metrics([z['doc_id'] for z in next(x for x in dev if x['query_id']==r['qid'])['hybrid'][:100]],dvq[r['qid']])['ndcg'] for r in rows]),'protocol':{'train_queries':len(train),'dev_queries':len(dev),'labels_train_only':True,'features':['n_terms','vocab_coverage','relative_gap','candidate_hits']}}
 a.out.mkdir();(a.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
finally:
 try:server.shutdown();server.server_close()
 except:pass
 try:proc.terminate();proc.wait(timeout=20)
 except:proc.kill()
 log.close()
