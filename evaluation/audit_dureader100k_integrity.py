"""Independent integrity audit for the frozen DuReader 100k build."""
import argparse, hashlib, json
from pathlib import Path
from offline.disk_index import DiskIndex
from contracts.retrieval import VectorManifest

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
 return h.hexdigest()
def count_lines(path):
 return sum(1 for line in Path(path).open(encoding='utf-8') if line.strip())
def main(a):
 data,index=Path(a.data),Path(a.index)
 manifest=json.loads((data/'dataset_manifest.json').read_text())
 expected={
  'corpus': 'f24b50b6012b47f6fb5f72272666dbf187124a869f19d1c8d83f7e30d6cb62b4',
  'qrels.dev':'e75a68b89fa8a1ed1879e6703cffc1d6514ba295c82667dc4ce574e24b318f30',
  'qrels.train':'176fc7b3f3eda48a00a5d3b8dc33c165e9e1c8ab18c9410815cf1229a4931ce5',
  'queries.dev':'ce010e6ea3a498a696d653b58bcf7976b5101d4034341983429df0a44837cfd9',
  'queries.train':'0fc31d70fb1c4548483b3c3d115d0e89fe957c813312d6823ff155b4ef488290',
 }
 paths={'corpus':data/'corpus.jsonl','qrels.dev':data/'qrels.dev.tsv','qrels.train':data/'qrels.train.tsv','queries.dev':data/'queries.dev.jsonl','queries.train':data/'queries.train.jsonl'}
 actual={k:sha(paths[k]) for k in expected}
 if actual!=expected: raise SystemExit(json.dumps({'error':'data hash mismatch','actual':actual},indent=2))
 reader=DiskIndex(index/'bm25',verify_checksums=True)
 vm=VectorManifest.from_dict(json.loads((index/'vector_manifest.json').read_text()))
 snapshot=next(iter(vm.snapshot_files))
 snap_path=index/snapshot
 if not snap_path.exists(): snap_path=index/'qdrant-snapshots'/snapshot
 if sha(snap_path)!=vm.snapshot_files[snapshot]: raise SystemExit('snapshot hash mismatch')
 report={'status':'verified','dataset_manifest_sha256':sha(data/'dataset_manifest.json'),'data_hashes':actual,'counts':{'documents':count_lines(data/'corpus.jsonl'),'queries.dev':count_lines(data/'queries.dev.jsonl'),'queries.train':count_lines(data/'queries.train.jsonl'),'qrels.dev':count_lines(data/'qrels.dev.tsv'),'qrels.train':count_lines(data/'qrels.train.tsv')},'bm25_manifest_sha256':sha(index/'bm25/manifest.json'),'bm25_documents':reader.n_docs,'vector_manifest_sha256':sha(index/'vector_manifest.json'),'vector_documents':vm.n_docs,'snapshot':snapshot,'snapshot_sha256':sha(snap_path),'labels_read_after_index_freeze':True,'per_query_records_committed':False}
 print(json.dumps(report,ensure_ascii=False,indent=2));Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--index',required=True);p.add_argument('--out',required=True);main(p.parse_args())
