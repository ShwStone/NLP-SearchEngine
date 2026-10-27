"""Build a small research table from frozen lexical/dense candidate captures.

Recompute BM25 using the current upstream scorer; rebuild equal RRF with c=60.
Historical candidate captures and labels are explicitly reused development data.
"""

import argparse
import hashlib
import json
from pathlib import Path

from contracts import analyzer
from evaluation.evaluate import _metric_for_query
from offline.disk_index import DiskIndex
from online.fusion import rrf
from online.router import QueryFeatures, RuleRouter
from online.router_features import FEATURE_VERSION, FEATURE_NAMES, extract_statistics
from contracts.retrieval import RetrievalConfig
from online.scorers import bm25


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_rows(path):
    # str.splitlines() also splits U+2028/U+0085 inside valid JSON strings.
    with Path(path).open(encoding='utf-8') as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def metrics(ids, relevance):
    ndcg, recall, mrr, ap, _ = _metric_for_query(relevance, ids, 10, 100)
    return {'ndcg': ndcg, 'recall': recall, 'mrr': mrr, 'map': ap}


def prepare(index, capture, audit, qrels, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reader = DiskIndex(index)
    identities = {reader.doc(i).doc_id: i for i in range(reader.n_docs)}
    external = {i: doc_id for doc_id, i in identities.items()}
    groups = json.loads(Path(audit).read_text())['groups']
    relevance = {}
    for line in Path(qrels).read_text().splitlines():
        qid, _, docid, grade = line.split()
        if qid in groups:
            relevance.setdefault(qid, {})[docid] = int(grade)
    router = RuleRouter(RetrievalConfig())
    count = matches = 0
    with Path(capture).open() as source, (out/'table.jsonl').open('w') as dest:
        seen = set()
        for line in source:
            row = json.loads(line)
            qid = row['query_id']
            if qid in seen or qid not in groups:
                raise ValueError('duplicate query or missing audit group')
            seen.add(qid)
            terms = [t.term for t in analyzer.analyze(row['text'])]
            ranked = sorted(bm25(reader, terms).items(), key=lambda r: (-r[1], r[0]))[:200]
            lexical = [external[i] for i, _ in ranked]
            if lexical != [hit['doc_id'] for hit in row['lexical']]:
                raise ValueError('recomputed BM25 candidate identities/order differ from capture')
            matches += 1
            dense = [hit['doc_id'] for hit in row['dense']]
            combined = rrf([[i for i, _ in ranked], [identities[d] for d in dense]], c=60.)[:200]
            hybrid = [external[i] for i, _ in combined]
            stats = extract_statistics(row['text'], reader, terms, ranked)
            gap = stats['relative_gap'] if len(ranked)>1 else None
            decision = router.decide(QueryFeatures(False, stats['n_terms'], stats['vocab_coverage'], len(ranked), gap))
            record = dict(query_id=qid, group=groups[qid], features=stats,
                          rules_use_vector=decision.use_vector, forced_vector=not ranked,
                          lexical=metrics(lexical, relevance[qid]), dense=metrics(dense, relevance[qid]),
                          hybrid=metrics(hybrid, relevance[qid]))
            dest.write(json.dumps(record, ensure_ascii=False)+'\n')
            count += 1
            if count%100 == 0:
                print('prepared', count, flush=True)
    original = json.loads(Path(capture).with_suffix('.provenance.json').read_text())
    provenance = dict(feature_version=FEATURE_VERSION, feature_names=list(FEATURE_NAMES), queries=count,
                      bm25_candidate_parity=matches, candidate_k=200, rrf_c=60., fusion='equal-weight RRF',
                      source_manifest=reader.manifest(),
                      vector_manifest=original['index_info']['retrieval']['vector_manifest'],
                      vector_search_params=original['index_info']['retrieval']['vector_search_params'],
                      source_sha256={str(p):sha(p) for p in (capture, audit, qrels)},
                      table_sha256=sha(out/'table.jsonl'),
                      dense_source='historical frozen BGE/ANN capture; ef=128; no fresh dense runs',
                      status='reused observed development labels; not a new independent holdout')
    (out/'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('index', 'capture', 'audit', 'qrels', 'out'):
        p.add_argument('--'+name, required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(prepare(**vars(args)), ensure_ascii=False, indent=2))
