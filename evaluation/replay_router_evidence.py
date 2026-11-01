"""Validate retained Router summaries and external evidence bindings.

Per-query predictions and request logs are intentionally excluded from Git.
This entry point validates the compact summaries committed here and reports
which local/external artifacts are required for the optional full replay.
"""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent / 'router_evidence'


def replay(root=ROOT):
    root = Path(root)
    inventory = json.loads((root / 'inventory.json').read_text())
    checked = {}
    for name, expected in inventory['files'].items():
        path = root / name
        if not path.is_file():
            raise ValueError('retained evidence file missing: ' + name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('evidence checksum mismatch: ' + name)
        checked[name] = actual
    external = dict(
        historical_round3_predictions='evaluation/router_evidence/round3-outer-predictions.jsonl',
        historical_round3_nested='evaluation/router_evidence/round3-nested-cv.json',
        historical_round4_predictions=(
            'evaluation/router_evidence/round4-proxy-predictions.jsonl and '
            'evaluation/router_evidence/round4-time-predictions.jsonl'),
        note='Per-query files are intentionally excluded from Git; retrieve them from the local artifacts archive before full replay.',
    )
    return dict(retained_summary_hashes=checked, per_query_replay_available=False,
                external_evidence_required=external,
                historical_results_are_not_current_confirmation=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(replay(args.evidence), ensure_ascii=False, indent=2))
