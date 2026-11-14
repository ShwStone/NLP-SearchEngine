import argparse,json,os,subprocess,time,urllib.request,hashlib
from pathlib import Path
from contracts.api import SearchRequest
from contracts.retrieval import VectorManifest
from online.engine import SearchEngine
from online.parser import parse
from offline.disk_index import DiskIndex
from adapters.embedding import BGEEncoder
from online.vector_index import QdrantVectorIndex

a=argparse.ArgumentParser();a.add_argument('--index-root',type=Path,required=True);a.add_argument('--data-root',type=Path,required=True);a.add_argument('--model-path',type=Path,required=True);a.add_argument('--qdrant-binary',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
root=args.index_root; data=args.data_root; out=args.out
if out.exists(): raise SystemExit('output exists')
queries=[json.loads(x) for x in (data/'queries.dev.jsonl').read_text().splitlines()]
port=6367; log=(root/'capture-qdrant.log').open('w'); proc=subprocess.Popen([str(args.qdrant_binary),'--disable-telemetry'],env={**os.environ,'QDRANT__STORAGE__STORAGE_PATH':str(root/'qdrant-storage'),'QDRANT__STORAGE__SNAPSHOTS_PATH':str(root/'qdrant-snapshots'),'QDRANT__SERVICE__HOST':'127.0.0.1','QDRANT__SERVICE__HTTP_PORT':str(port),'QDRANT__SERVICE__GRPC_PORT':str(port+1)},stdout=log,stderr=subprocess.STDOUT,cwd=root)
try:
 for _ in range(200):
  if proc.poll() is not None: raise RuntimeError(log.read_text()[-2000:])
  try: urllib.request.urlopen(f'http://127.0.0.1:{port}/collections',timeout=1);break
  except: time.sleep(.2)
 reader=DiskIndex(root/'bm25'); manifest=VectorManifest.from_dict(json.loads((root/'vector_manifest.json').read_text())); enc=BGEEncoder(manifest.embedding,str(args.model_path),'cuda',batch_size=32); vec=QdrantVectorIndex(manifest,f'http://127.0.0.1:{port}'); vec.search_params={'hnsw_ef':128,'exact':False}; engine=SearchEngine(reader,enc,vec)
 out.mkdir(parents=True)
 with (out/'capture.jsonl').open('w') as f:
  for i,row in enumerate(queries):
   qid=row.get('qid',row.get('query_id')); q=row.get('query',row.get('text')); explicit=parse(q).get('explicit_constraints',False); result={'query_id':qid,'text':q,'explicit_constraints':explicit}
   for mode in ('lexical','dense','hybrid'):
    if mode=='dense' and explicit: result[mode]=[]; continue
    resp=engine.search(SearchRequest(query=q,mode=mode,k=200)); result[mode]=[{'doc_id':x.doc_id,'score':x.score} for x in resp.results]
   f.write(json.dumps(result,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%100==0: print('captured',i+1,'/',len(queries),flush=True)
 meta={'queries':len(queries),'k':200,'dataset_manifest_sha256':hashlib.sha256((data/'dataset_manifest.json').read_bytes()).hexdigest(),'corpus_sha256':hashlib.sha256((data/'corpus.jsonl').read_bytes()).hexdigest(),'queries_sha256':hashlib.sha256((data/'queries.dev.jsonl').read_bytes()).hexdigest(),'qrels_sha256':hashlib.sha256((data/'qrels.dev.tsv').read_bytes()).hexdigest(),'vector_manifest_sha256':hashlib.sha256((root/'vector_manifest.json').read_bytes()).hexdigest(),'explicit_dense_skipped':sum(parse(x.get('query',x.get('text'))).get('explicit_constraints',False) for x in queries),'qrels_read_after_capture':False}
 (out/'protocol.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n');print(json.dumps(meta,indent=2))
finally:
 proc.terminate();proc.wait(timeout=20);log.close()
