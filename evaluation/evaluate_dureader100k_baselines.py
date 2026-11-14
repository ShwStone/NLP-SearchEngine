import json,os,subprocess,time,urllib.request,statistics,hashlib
from pathlib import Path
import sys
sys.path[:0]=['/root/autodl-tmp/NLP-SearchEngine-router','/root/autodl-tmp/NLP-SearchEngine']
from offline.disk_index import DiskIndex
from online.engine import SearchEngine
from online.parser import parse
from contracts.api import SearchRequest
from online.vector_index import QdrantVectorIndex
from contracts.retrieval import VectorManifest
from adapters.embedding import BGEEncoder
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import bootstrap,percentile
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--index-root',type=Path,required=True)
parser.add_argument('--data-root',type=Path,required=True)
parser.add_argument('--model-path',type=Path,required=True)
parser.add_argument('--qdrant-binary',type=Path,required=True)
args=parser.parse_args()
root=args.index_root; data=args.data_root
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
qrels={}
for line in (data/'qrels.dev.tsv').read_text().splitlines():
 q,_,d,g=line.split();qrels.setdefault(q,{})[d]=float(g)
queries=[]
for line in (data/'queries.dev.jsonl').read_text().splitlines():
 x=json.loads(line); queries.append((x.get('qid',x.get('query_id')),x.get('query',x.get('text'))))
port=6359; storage=root/'qdrant-storage'; log=(root/'baseline-qdrant.log').open('w'); proc=subprocess.Popen([str(args.qdrant_binary),'--disable-telemetry'],env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(storage),'QDRANT__STORAGE__SNAPSHOTS_PATH':str(root/'qdrant-snapshots'),'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(port),'QDRANT__SERVICE__GRPC_PORT':str(port+1)},stdout=log,stderr=subprocess.STDOUT,cwd=root)
try:
 for _ in range(200):
  if proc.poll() is not None: raise RuntimeError(log.read_text()[-2000:])
  try: urllib.request.urlopen(f'http://127.0.0.1:{port}/collections',timeout=1);break
  except:time.sleep(.2)
 reader=DiskIndex(root/'bm25'); manifest=VectorManifest.from_dict(json.loads((root/'vector_manifest.json').read_text())); enc=BGEEncoder(manifest.embedding,str(args.model_path),'cuda',batch_size=32); vec=QdrantVectorIndex(manifest,f'http://127.0.0.1:{port}'); vec.search_params={'hnsw_ef':128,'exact':False}; base=SearchEngine(reader,enc,vec)
 vals={m:[] for m in ('lexical','dense','hybrid','adaptive')}; times={m:[] for m in vals}; groups=[]; skipped_dense=0
 for i,(qid,q) in enumerate(queries):
  groups.append(qid)
  for mode in vals:
   if mode == 'dense' and parse(q).get('explicit_constraints', False):
    skipped_dense += 1; continue
   t=time.perf_counter(); resp=base.search(SearchRequest(query=q,mode=mode,k=100)); times[mode].append((time.perf_counter()-t)*1000); ids=[x.doc_id for x in resp.results]; vals[mode].append(metrics(ids,qrels[qid]))
  if (i+1)%100==0: print('evaluated',i+1,'/',len(queries),flush=True)
 summary={}
 for mode in vals:
  summary[mode]={'queries':len(vals[mode]),'metrics':{k:statistics.mean(x[k] for x in vals[mode]) for k in ('ndcg','recall','mrr','map')},'latency_ms':{'p50':percentile(times[mode],.5),'p95':percentile(times[mode],.95),'mean':statistics.mean(times[mode])}}
 summary['protocol']={'dataset_manifest_sha256':sha(data/'dataset_manifest.json'),'corpus_sha256':sha(data/'corpus.jsonl'),'queries_sha256':sha(data/'queries.dev.jsonl'),'qrels_sha256':sha(data/'qrels.dev.tsv'),'vector_manifest_sha256':sha(root/'vector_manifest.json'),'index_manifest_sha256':sha(root/'bm25/manifest.json'),'queries':len(queries),'labels_read_after_frozen_index':True,'per_query_records_written':False,'cutoffs':{'ndcg':10,'recall':10,'map':100},'dense_explicit_skipped':skipped_dense}
 (root/'baseline-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2))
finally:
 proc.terminate();proc.wait(timeout=20);log.close()
