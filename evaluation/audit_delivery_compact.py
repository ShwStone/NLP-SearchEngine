"""Verify every compact file checksum and all corpus document fields/order.

Uses the frozen delivered reader source as an external backend dependency.
This audit does not certify every posting's semantic correspondence to text.
"""
import argparse
import hashlib
import json
from pathlib import Path

from contracts import analyzer
from contracts.index_format import sha256_of


def audit(compact, corpus, frozen_source, vector_manifest, output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    compact=Path(compact);source=Path(frozen_source)
    metadata=json.loads((compact/'compact.json').read_text())
    required={'docs.jsonl','docs.idx','offsets.bin','offsets.idx','lengths.bin','terms.sqlite','postings.bin'}
    if metadata['compact_version']!=1 or set(metadata['files'])!=required:
        raise ValueError('compact file set/version mismatch')
    if metadata['source_manifest']['analyzer']['config_hash']!=analyzer.config_hash():
        raise ValueError('analyzer mismatch')
    for name,digest in metadata['files'].items():
        if sha256_of(compact/name)!=digest:raise ValueError('compact hash mismatch: '+name)
        print('compact checksum verified',name,flush=True)
    raw,fingerprint=hashlib.sha256(),hashlib.sha256();count=0
    with Path(corpus).open('rb') as original,(compact/'docs.jsonl').open('rb') as indexed:
        for line in original:
            raw.update(line)
            if not line.strip():continue
            doc=json.loads(line);actual=json.loads(indexed.readline())
            if actual.get('docid')!=count or any(actual.get(k)!=doc[k] for k in ('doc_id','title','text')):
                raise ValueError('compact source document mismatch')
            fingerprint.update((json.dumps([count,doc['doc_id'],doc['title'],doc['text']],
                               ensure_ascii=False,separators=(',',':'))+'\n').encode())
            count+=1
        if indexed.read(1):raise ValueError('extra compact documents')
    vm=json.loads(Path(vector_manifest).read_text())
    if count!=vm['n_docs'] or count!=metadata['source_manifest']['n_docs'] or fingerprint.hexdigest()!=vm['corpus_sha256']:
        raise ValueError('compact vector/corpus identity mismatch')
    report={'status':'verified','documents':count,'all_compact_files_verified':True,
            'all_document_fields_and_order_verified':True,
            'all_posting_semantics_verified':False,
            'corpus_file_sha256':raw.hexdigest(),'corpus_contract_fingerprint':fingerprint.hexdigest(),
            'compact_manifest_sha256':sha256_of(compact/'compact.json'),
            'vector_manifest_sha256':sha256_of(vector_manifest),
            'reader_sources':{n:sha256_of(source/n) for n in ['offline/compact.py','offline/packed_index.py']},
            'scope':'all compact file integrity and all source document fields/order'}
    with output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('compact','corpus','frozen-source','vector-manifest','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
