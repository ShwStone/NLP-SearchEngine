"""Verify archived hashes and recompute routing savings/loss from per-query rows.

This checks historical evidence arithmetic, without training or an online engine.
It does not make the old experiments an independent evaluation of this PR.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parent/'router_evidence'


def _equal(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f'evidence mismatch: {actual} != {expected}')


def replay(root=ROOT):
    root = Path(root)
    inventory = json.loads((root/'inventory.json').read_text())
    for name, expected in inventory['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('evidence checksum mismatch: '+name)
    rows = [json.loads(line) for line in (root/'round3-outer-predictions.jsonl').read_text().splitlines()]
    summary = json.loads((root/'round3-nested-cv.json').read_text())['outer']
    if len(rows) != 997 or len({r['query_id'] for r in rows}) != len(rows):
        raise ValueError('invalid round3 query identities')
    loss = [r['gain']*int(not r['use_lexical']) for r in rows]
    _equal(statistics.mean(loss), summary['mean_loss'])
    _equal(statistics.mean(not r['use_lexical'] for r in rows), summary['saving'])
    _equal(statistics.mean(r['fixed_ndcg10']-l for r, l in zip(rows, loss)), summary['adaptive_ndcg10'])
    report = {'round3': {k: summary[k] for k in ('mean_loss', 'saving', 'adaptive_ndcg10')}}
    for prefix in ('round4-proxy', 'round4-time'):
        summary = json.loads((root/(prefix+'-summary.json')).read_text())
        groups = defaultdict(list)
        for line in (root/(prefix+'-predictions.jsonl')).read_text().splitlines():
            row = json.loads(line)
            groups[row['protocol']].append(row)
        if set(groups) != set(summary['results']):
            raise ValueError('missing research protocols')
        report[prefix] = {}
        for protocol, rows in groups.items():
            if len(rows) != 997 or len({r['query_id'] for r in rows}) != len(rows):
                raise ValueError('invalid round4 query identities')
            expected = summary['results'][protocol]
            loss = [r['gain']*int(not r['use_bm25']) for r in rows]
            mean = statistics.mean(loss)
            upper = mean+1.645*statistics.stdev(loss)/math.sqrt(len(rows))
            saving = statistics.mean(not r['use_bm25'] for r in rows)
            work = sum(r['work']*int(not r['use_bm25']) for r in rows)/sum(r['work'] for r in rows)
            for actual, key in ((mean, 'mean_loss'), (upper, 'loss_upper95'), (saving, 'bm25_call_saving'),
                                (work, 'posting_work_saving' if prefix.endswith('proxy') else 'lexical_stage_ms_saving')):
                _equal(actual, expected[key])
            report[prefix][protocol] = dict(mean_loss=mean, bm25_call_saving=saving,
                                            meets_budget=expected['meets_budget'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence), ensure_ascii=False, indent=2))
