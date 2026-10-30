"""Check all canonical development pair inputs against a document-only cache."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .gte_document_tokens import DocumentTokenPairs
from .prepare_bm25_router import jsonl_rows, sha


def run(a):
    from transformers import AutoTokenizer
    captures = {r['query_id']: r for r in jsonl_rows(a.capture)}
    canonical = json.loads(a.canonical.with_name('summary.json').read_text())
    ids = canonical['protocol']['query_ids']
    if set(ids) != set(captures) or canonical['logits_sha256'] != sha(a.canonical):
        raise ValueError('canonical query scope or scores changed')
    documents = {r['doc_id']: (r['docid'], r['title'] + '\n' + r['text'])
                 for r in jsonl_rows(a.documents)}
    original = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    cache = DocumentTokenPairs(original, [v[1] for v in documents.values()], max_length=256)
    pairs, digest, originals_ms, cached_ms = 0, hashlib.sha256(), [], []
    for index, q in enumerate(ids):
        for k in (28, 40):
            docs = sorted((r['doc_id'] for r in captures[q]['dense'][:k]), key=lambda d: documents[d][0])
            texts = [documents[d][1] for d in docs]
            query = captures[q]['text']
            tick = time.perf_counter()
            expected = original([query] * k, texts, padding=False, truncation=True, max_length=256)
            originals_ms.append((time.perf_counter() - tick) * 1000)
            tick = time.perf_counter()
            actual = cache([query] * k, texts, padding=False, truncation=True, max_length=256)
            cached_ms.append((time.perf_counter() - tick) * 1000)
            if expected != actual:
                raise ValueError('BGE cached pair token IDs or masks differ: ' + q)
            digest.update(json.dumps([q, k, actual], separators=(',', ':')).encode())
            pairs += k
        if (index + 1) % 250 == 0:
            print('BGE tokenizer parity', index + 1, '/', len(ids), flush=True)
    import numpy as np
    report = dict(kind='canonical-BGE-document-token-parity-v1', queries=len(ids),
                  query_document_pairs=pairs, corpus_documents=len(documents),
                  cached_distinct_texts=len(cache.cache), cache_build_seconds=cache.build_seconds,
                  exact_all_input_ids_masks_special_tokens_and_truncation=True,
                  pair_input_sha256=digest.hexdigest(),
                  mean_encoding_ms=dict(original=float(np.mean(originals_ms)), cached=float(np.mean(cached_ms))),
                  cache_fallbacks=cache.fallback_calls, query_predictions_cached=False,
                  inputs={str(p.resolve()): sha(p) for p in
                          (a.capture, a.canonical, a.documents, a.model / 'model_inventory.json')},
                  code_sha256={str(p.resolve()): sha(p) for p in
                               (Path(__file__), Path(__file__).with_name('gte_document_tokens.py'))},
                  raw_qrels_read=False, confirmation_read=False, http_evaluated=False, goal_complete=False)
    a.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('capture', 'canonical', 'documents', 'model', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    run(p.parse_args())
