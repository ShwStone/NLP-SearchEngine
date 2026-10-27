"""Freeze unused CMedQA query evaluation, with no retrieval/model score access."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from online.parser import parse, QuerySyntaxError
from .prepare_bm25_router import sha, jsonl_rows


def grams(text):
    text = ''.join(c for c in unicodedata.normalize('NFKC',text).lower() if c.isalnum())
    return {text[i:i+2] for i in range(len(text)-1)} if len(text)>1 else {text}


def prepare(history_root, model, freeze, out):
    import pyarrow.parquet as pq
    root, out = Path(history_root), Path(out)
    frozen = json.loads(Path(freeze).read_text())
    if sha(model)!=frozen['model_sha256']:
        raise ValueError('frozen model changed')
    out.mkdir(parents=True,exist_ok=False)
    data = root/'artifacts/round5/cmedqa-data'
    manifest = json.loads((data/'manifest.json').read_text())
    raw = root/'.runtime/round5/cmedqa'
    for name, digest in manifest['source_sha256'].items():
        if sha(raw/name)!=digest:
            raise ValueError('CMedQA source checksum mismatch')
    for name, digest in manifest['outputs_sha256'].items():
        if sha(data/name)!=digest:
            raise ValueError('historical CMedQA corpus/query checksum mismatch')
    ids = {row['doc_id'] for row in jsonl_rows(data/'corpus.jsonl')}
    prior_ids = {row['qid'] for row in jsonl_rows(data/'queries.jsonl')}
    paths = sorted((root/'artifacts').rglob('queries*.jsonl'))
    history, index = [], defaultdict(set)
    def remember(tokens):
        i=len(history);history.append(tokens)
        for token in tokens:index[token].add(i)
    for path in paths:
        for row in jsonl_rows(path):
            remember(grams(row.get('text',row.get('query',''))))
    def near(tokens):
        shared=Counter(i for token in tokens for i in index.get(token,()))
        return any(n/(len(tokens)+len(history[i])-n)>=.8 for i,n in shared.items())
    queries=pq.read_table(raw/'queries-00000-of-00001-daeedab899d3c839.parquet').to_pylist()
    relations=pq.read_table(raw/'dev-00000-of-00001-57fb84a4aceaa695.parquet').to_pylist()
    eligible={r['qid'] for r in relations if r['pid'] in ids and r['score']>0}
    candidates=sorted(queries,key=lambda q:hashlib.sha256(('rfc01-cmedqa-unused|'+q['id']).encode()).hexdigest())
    selected=[];exclusions=Counter()
    for row in candidates:
        if row['id'] in prior_ids:exclusions['prior_id']+=1;continue
        if row['id'] not in eligible:exclusions['no_positive_in_fixed_corpus']+=1;continue
        try:
            if parse(row['text'])['explicit_constraints']:
                exclusions['explicit_constraints']+=1;continue
        except QuerySyntaxError:
            exclusions['invalid_syntax']+=1;continue
        tokens=grams(row['text'])
        if near(tokens):exclusions['near_observed_or_selected_query']+=1;continue
        if len(selected)<256:
            selected.append(dict(query_id=row['id'],text=row['text']))
            remember(tokens)
        else:exclusions['over_predeclared_cap']+=1
    if not selected:raise ValueError('no eligible unused queries')
    qids={r['query_id'] for r in selected}
    (out/'queries.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
    (out/'qrels.tsv').write_text(''.join(f"{r['qid']}\t0\t{r['pid']}\t{r['score']}\n" for r in relations if r['qid'] in qids and r['pid'] in ids))
    result=dict(model_sha256=frozen['model_sha256'],queries=len(selected),documents=len(ids),
                old_query_ids=len(prior_ids),exclusions=dict(exclusions),
                history_sha256={str(p):sha(p) for p in paths},source_manifest=manifest,
                output_sha256={p.name:sha(p) for p in out.iterdir()},
                selection='predeclared hash order; no retrieval scores; answerability-conditioned on fixed 20k corpus; no near duplicates of prior or selected queries',
                limitation=frozen['limitation'],no_tuning_on_these_labels=True)
    (out/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('history-root','model','freeze','out'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(prepare(args.history_root,args.model,args.freeze,args.out),ensure_ascii=False,indent=2))
