"""Score canonical dense28/dense40 batches on observed development queries only."""
import argparse
import json
from pathlib import Path

from .prepare_bm25_router import jsonl_rows,sha


def run(a):
    from .tensorrt_cross_encoder import TensorRTCrossEncoder
    from .ranking_dataset import load_dataset
    metadata=load_dataset(a.dataset)['metadata']
    captures={row['query_id']:row for row in jsonl_rows(a.capture)}
    ids=metadata['query_ids']
    if set(captures)!=set(ids):raise ValueError('canonical scorer query scope differs')
    documents={row['doc_id']:(row['docid'],row['title']+'\n'+row['text']) for row in jsonl_rows(a.documents)}
    scorer=TensorRTCrossEncoder(a.engine,a.model,max_length=256,batch_size=20)
    a.out.mkdir(parents=True,exist_ok=False)
    files=[a.dataset/'dataset.json',a.capture,a.documents,a.engine/'summary.json']
    code=[Path(__file__),Path(__file__).with_name('tensorrt_cross_encoder.py'),
          Path(__file__).with_name('export_bge_tensorrt_input.py'),Path(__file__).with_name('study_rerank_actions.py')]
    protocol=dict(kind='canonical-preBGE-development-logits-v2',query_ids=ids,pools=[28,40],
        candidate_order='internal numeric docid ascending, identical to live sorted(ids[:k])',
        encoder=scorer.info,inputs={str(p.resolve()):sha(p) for p in files},
        code_sha256={str(p.resolve()):sha(p) for p in code},qrels_read=False,
        scoped_development_metadata_loaded=True,confirmation_read=False,quality_evaluated=False,goal_complete=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.out/'code').mkdir()
    for p in code:(a.out/'code'/p.name).write_bytes(p.read_bytes())
    scorer.score('预热',['文档内容 '*40]*40)
    with (a.out/'logits.jsonl').open('w') as output:
        for index,q in enumerate(ids):
            scores={};costs={}
            for k in (28,40):
                pool=sorted((r['doc_id'] for r in captures[q]['dense'][:k]),key=lambda d:documents[d][0])
                if len(pool)!=k or len(set(pool))!=k:raise ValueError('invalid native dense pool')
                values,elapsed=scorer.score(captures[q]['text'],[documents[d][1] for d in pool])
                scores[str(k)]=dict(zip(pool,values));costs[str(k)]=elapsed
            output.write(json.dumps(dict(query_id=q,scores=scores,component_ms=costs))+'\n');output.flush()
            if (index+1)%100==0:print('canonical preBGE pairs',index+1,'/',len(ids),flush=True)
    summary=dict(protocol=protocol,logits_sha256=sha(a.out/'logits.jsonl'),queries=len(ids),goal_complete=False)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(queries=len(ids),logits_sha256=summary['logits_sha256']),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('dataset','capture','documents','engine','model','out'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
