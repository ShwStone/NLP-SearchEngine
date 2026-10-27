"""Optional real vector build using pinned BGE and the upstream Qdrant writer."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.embedding import BGEEncoder
from offline.disk_documents import DiskDocuments
from offline.vector_build import QdrantSnapshotWriter,build_vectors


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True)
    p.add_argument('--collection',required=True);p.add_argument('--url',default='http://127.0.0.1:16353')
    p.add_argument('--model-path',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args();start=time.perf_counter()
    encoder=BGEEncoder(model_path=args.model_path,device=args.device,batch_size=args.batch_size)
    with DiskDocuments(args.index) as reader:
        class ProgressEncoder:
            spec=encoder.spec;seen=0
            def encode_documents(self,texts):
                result=encoder.encode_documents(texts);self.seen+=len(texts)
                if self.seen%4096==0:print('encoded',self.seen,'/',reader.n_docs,flush=True)
                return result
        manifest=build_vectors(reader,ProgressEncoder(),QdrantSnapshotWriter(args.url,Path(args.out),timeout=180),
            Path(args.out),args.collection,batch_size=args.batch_size,index_params={'m':16,'ef_construct':100})
    print(json.dumps(dict(n_docs=manifest.n_docs,seconds=time.perf_counter()-start,collection=manifest.collection)),flush=True)


if __name__=='__main__':main()
