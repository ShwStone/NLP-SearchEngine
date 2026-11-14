"""Score frozen CE-28/40 actions without reading query relevance labels.

Only selected candidate documents are retained from a streaming corpus scan.
Each budget is scored separately; CE-28 does not inherit a CE-40 timing.
All measurements are component timings, not end-to-end HTTP timings.
"""
import argparse
import hashlib
import json
from pathlib import Path

from contracts.index_format import sha256_of
from evaluation.merge_full_captures import read_capture
from evaluation.study_rerank_actions import FrozenCrossEncoder


def score(capture_dir, corpus, model, output, expected_corpus_sha256, budgets='28,40'):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    budgets=[int(b) for b in budgets.split(',')]
    if not budgets or len(set(budgets))!=len(budgets)or any(not 1<=b<=100 for b in budgets):
        raise ValueError('invalid CE budgets')
    rows, source = read_capture(capture_dir)
    wanted = {h['doc_id'] for r in rows for h in r['hybrid'][:max(budgets)]}
    documents = {}
    digest = hashlib.sha256()
    count = 0
    with Path(corpus).open('rb') as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            doc = json.loads(line)
            count += 1
            if doc['doc_id'] in wanted:
                if doc['doc_id'] in documents:
                    raise ValueError('duplicate selected document ID')
                documents[doc['doc_id']] = doc['title']+'\n'+doc['text']
    if digest.hexdigest() != expected_corpus_sha256 or count != source['documents']:
        raise ValueError('corpus identity/count mismatch')
    if set(documents) != wanted:
        raise ValueError('candidate documents missing from corpus')
    ce = FrozenCrossEncoder(model, device='cuda', batch_size=20, max_length=256)
    # Warm both budgets without labels; retain no warmup times.
    for budget in budgets:
        for row in rows[:1]:
            ce.score(row['text'],[documents[h['doc_id']] for h in row['hybrid'][:budget]])
    output.mkdir(parents=True)
    with (output/'scores.pending.jsonl').open('x',encoding='utf-8') as handle:
        for i,row in enumerate(rows):
            result = {'query_id':row['query_id'],'text':row['text'],
                      'explicit_constraints':row['explicit_constraints'],
                      'pool':[h['doc_id'] for h in row['hybrid'][:100]],
                      'global_docids':[h['docid'] for h in row['hybrid'][:100]],
                      'lexical':row['lexical'], 'dense':row['dense']}
            for budget in budgets:
                selected = result['pool'][:budget]
                scores,elapsed = ce.score(row['text'],[documents[d] for d in selected])
                result[f'ce{budget}']={'scores':scores,'pairs':len(selected),'component_ms':elapsed}
            handle.write(json.dumps(result,ensure_ascii=False)+'\n'); handle.flush()
            if (i+1)%100==0:
                print('CE actions scored',i+1,'/',len(rows),flush=True)
    (output/'scores.pending.jsonl').rename(output/'scores.jsonl')
    protocol = {'queries':len(rows),'documents':count,
                'queries_sha256':source['queries_sha256'],
                'candidate_capture_sha256':source['capture_sha256'],
                'candidate_protocol_sha256':sha256_of(Path(capture_dir)/'protocol.json'),
                'corpus_sha256':digest.hexdigest(),'ce':ce.info,
                'scores_sha256':sha256_of(output/'scores.jsonl'),
                'budgets':budgets,'qrels_read':False,
                'latency_scope':'separately measured CE components; not HTTP latency'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('capture-dir','corpus','model','output','expected-corpus-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--budgets',default='28,40')
    print(json.dumps(score(**vars(p.parse_args())),indent=2))
