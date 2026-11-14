"""Verify official archive bytes, extracted sources and every normalized row.

The ID map is the only full input materialized. Passage text and normalized
corpus are streamed in lockstep; no index, query candidate or qrel is read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from contracts.index_format import sha256_of

ARCHIVE_SHA256='b051d3d966f7b0288aed95a923fc3b706176767d8bdcff8aa8fe5fff84b20662'


def audit(archive,official_root,corpus,output,expected_documents=8096668):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if sha256_of(archive)!=ARCHIVE_SHA256:raise ValueError('official archive SHA256 mismatch')
    root=Path(official_root)
    required={f'passage-collection/part-{i:02d}'for i in range(4)}
    required.add('passage-collection/passage2id.map.json')
    required.update({'dev/dev.json','dev/q2qid.dev.json'})
    verified={}
    with tarfile.open(archive,'r|gz')as tar:
        for member in tar:
            path=Path(member.name)
            tail='/'.join(path.parts[1:])
            if tail not in required:continue
            if tail in verified or not member.isfile():raise ValueError('duplicate/non-file official member')
            digest=hashlib.sha256()
            with tar.extractfile(member)as handle:
                for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
            if sha256_of(root/tail)!=digest.hexdigest():raise ValueError('extracted official member mismatch: '+tail)
            verified[tail]=digest.hexdigest();print('official archive member verified',tail,flush=True)
    if set(verified)!=required:raise ValueError('official archive source members missing')
    mapping=json.loads((root/'passage-collection/passage2id.map.json').read_text())
    if not isinstance(mapping,dict)or len(mapping)!=expected_documents:raise ValueError('official ID map count mismatch')
    count=0;raw=hashlib.sha256();parts={}
    with Path(corpus).open('rb')as normalized:
        for i in range(4):
            part=root/f'passage-collection/part-{i:02d}';part_count=0
            with part.open(encoding='utf-8',newline='')as handle:
                for line in handle:
                    fields=line.rstrip('\r\n').split('\t')
                    if len(fields)<3:raise ValueError('invalid official passage fields')
                    actual=normalized.readline();raw.update(actual)
                    expected={'doc_id':mapping.get(str(count)),'title':'','text':fields[2]}
                    if not expected['doc_id']or json.loads(actual)!=expected:
                        raise ValueError(f'official normalized row mismatch: {count}')
                    count+=1;part_count+=1
                    if count%1000000==0:print('official source rows verified',count,flush=True)
            parts[part.name]=part_count
        if normalized.read(1):raise ValueError('extra normalized corpus records')
    if count!=expected_documents:raise ValueError('official record count mismatch')
    result={'status':'verified','documents':count,'parts':parts,
            'official_archive_sha256':ARCHIVE_SHA256,'verified_archive_members':verified,
            'corpus_file_sha256':raw.hexdigest(),'all_official_normalized_rows_verified':True,
            'scope':'official archive identity, extracted source hashes and every normalized document ID/text/order'}
    with output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('archive','official-root','corpus','output'):p.add_argument('--'+n,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
