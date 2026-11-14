"""Independent full BM25 capture through the audited compact reader.

Sharded capture remains the independent ranking/score reference. This backend
does not use query candidate replay or relevance labels.
"""
import argparse,json,time
from pathlib import Path

from contracts import analyzer
from contracts.index_format import sha256_of
from evaluation.delivery_backend import load_reader,BoundedPackedBM25,READER_HASHES
from evaluation.prepare_bm25_router import jsonl_rows
from online.parser import parse
from online.scorers import boolean_docs


def capture(reader_source,compact,queries,output,k=200):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if type(k)is not int or not 1<=k<=200:raise ValueError('invalid k')
    reader=load_reader(reader_source,compact)
    scorer=BoundedPackedBM25(reader)
    rows=list(jsonl_rows(queries));qids=[r.get('query_id',r.get('qid')) for r in rows]
    if any(not isinstance(q,str) or not q for q in qids) or len(qids)!=len(set(qids)):
        raise ValueError('invalid query IDs')
    qhash=sha256_of(queries);output.mkdir(parents=True)
    start=time.monotonic()
    try:
        with (output/'capture.pending.jsonl').open('x',encoding='utf-8')as handle:
            for i,(row,qid)in enumerate(zip(rows,qids)):
                text=row.get('query',row.get('text'));parsed=parse(text)
                terms=[t.term for raw in parsed['terms'] for t in analyzer.analyze(raw)]
                candidates=boolean_docs(reader,parsed['ast'],analyzer.analyze) if parsed.get('explicit_constraints',False) else None
                hits=scorer.topk(terms,k,candidate_docs=candidates)
                if candidates is not None:
                    scores=dict(hits)
                    hits=sorted(((i,scores.get(i,0.))for i in candidates),key=lambda h:(-h[1],h[0]))[:k]
                result={'query_id':qid,'text':text,'explicit_constraints':parsed.get('explicit_constraints',False),
                        'lexical':[{'docid':d,'doc_id':reader.external_doc_id(d),'score':s}for d,s in hits]}
                handle.write(json.dumps(result,ensure_ascii=False)+'\n');handle.flush()
                if(i+1)%50==0:print('compact BM25 captured',i+1,'/',len(rows),flush=True)
        if sha256_of(queries)!=qhash:raise ValueError('queries changed')
        (output/'capture.pending.jsonl').rename(output/'capture.jsonl')
        protocol={'queries':len(rows),'queries_sha256':qhash,'documents':reader.n_docs,'k':k,
                  'capture_sha256':sha256_of(output/'capture.jsonl'),'qrels_read':False,
                  'compact_manifest_sha256':sha256_of(Path(compact)/'compact.json'),
                  'frozen_reader_sources':READER_HASHES,'analyzer_hash':analyzer.config_hash(),
                  'batch_wall_seconds':time.monotonic()-start,
                  'latency_scope':'offline compact capture; not HTTP latency',
                  'independent_sharded_parity_verified':False}
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        return protocol
    finally:reader.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for n in ('reader-source','compact','queries','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--k',type=int,default=200)
    print(json.dumps(capture(**vars(p.parse_args())),indent=2))
