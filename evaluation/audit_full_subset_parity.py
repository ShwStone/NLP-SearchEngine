"""Stream full corpus to prove exact frozen-subset field/order coverage."""
import argparse
import hashlib
import json
from pathlib import Path
from evaluation.prepare_bm25_router import jsonl_rows, sha


def audit(full, subset, output, expected_full=8096668, expected_subset=100000):
    wanted = {}
    order = []
    for row in jsonl_rows(subset):
        if row['doc_id'] in wanted:
            raise ValueError('duplicate subset ID')
        wanted[row['doc_id']] = row
        order.append(row['doc_id'])
    if len(wanted) != expected_subset:
        raise ValueError('unexpected subset count')
    found = set()
    found_order = []
    count = 0
    digest = hashlib.sha256()
    with Path(full).open('rb') as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row['doc_id']
            if doc_id in wanted:
                if doc_id in found:
                    raise ValueError('subset ID repeated in full corpus')
                if any(row[k] != wanted[doc_id][k] for k in ('doc_id', 'title', 'text')):
                    raise ValueError(f'full/subset document mismatch: {doc_id}')
                found.add(doc_id)
                found_order.append(doc_id)
            count += 1
            if count % 1000000 == 0:
                print('scanned', count, 'subset matches', len(found), flush=True)
    if count != expected_full or found != set(wanted) or found_order != order:
        raise ValueError('full count, subset coverage or relative order mismatch')
    report = {'status': 'verified', 'full_documents': count, 'subset_documents': len(found),
              'full_corpus_sha256': digest.hexdigest(), 'subset_corpus_sha256': sha(subset),
              'exact_fields_and_relative_order': True,
              'scope': 'frozen subset coverage; not an all-document official archive identity check'}
    with Path(output).open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    for name in ('full', 'subset', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(audit(**vars(a)), indent=2))
