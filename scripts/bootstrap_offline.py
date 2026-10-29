"""Download optional pinned BGE files and portable Qdrant1.16, with SHA checks."""

import argparse
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adapters.model_files import download,fetch
from contracts.index_format import sha256_of

QDRANT_URL='https://github.com/qdrant/qdrant/releases/download/v1.16.0/qdrant-x86_64-unknown-linux-musl.tar.gz'
QDRANT_SHA256='d4752e9bf673c7f98153264cb8f416398928c7787e1269350c179fb856de3ef7'


def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',default='.runtime/models/bge-small-zh-v1.5')
    p.add_argument('--qdrant-dir',type=Path,default=Path('.runtime/round3'));p.add_argument('--proxy')
    args=p.parse_args()
    download(args.model_dir,args.proxy);args.qdrant_dir.mkdir(parents=True,exist_ok=True)
    archive=args.qdrant_dir/'qdrant-v1.16.0-musl.tar.gz'
    if not archive.exists():
        pending=archive.with_suffix('.part');fetch(QDRANT_URL,pending,args.proxy)
        if sha256_of(pending)!=QDRANT_SHA256:raise ValueError('Qdrant download SHA mismatch')
        pending.replace(archive)
    if sha256_of(archive)!=QDRANT_SHA256:raise ValueError('Qdrant archive SHA mismatch')
    with tarfile.open(archive) as tar:
        member=next(m for m in tar.getmembers() if Path(m.name).name=='qdrant' and m.isfile())
        pending=args.qdrant_dir/'qdrant.part';pending.write_bytes(tar.extractfile(member).read())
    pending.chmod(0o755);pending.replace(args.qdrant_dir/'qdrant')
    print('Pinned BGE and Qdrant1.16 musl ready')


if __name__=='__main__':main()
