"""Independent structural audit for the aggregate CE ablation report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .prepare_bm25_router import sha


REQUIRED = {
    "CE-0", "CE-28", "CE-40", "CE-28 + learned ranker",
    "CE-40 + learned ranker", "always-cheap", "always-heavy",
    "fixed-depth-28", "fixed-depth-40", "Router",
}


def audit(root: Path):
    root = Path(root)
    summary_path = root / "summary.json"
    protocol_path = root / "protocol.json"
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    actions = summary.get("actions", {})
    errors = []
    if set(actions) != REQUIRED:
        errors.append(f"action set differs: {sorted(set(actions) ^ REQUIRED)}")
    if protocol.get("large_query_records_written") is not False:
        errors.append("protocol does not forbid large query records")
    if summary.get("quality_cost_conclusion", {}).get("fixed_ce_gain_separated_from_router") is not True:
        errors.append("fixed CE and Router attribution is not explicit")
    if sum(summary.get("router_action_counts", {}).values()) != actions.get("Router", {}).get("n"):
        errors.append("Router action counts do not cover report queries")
    for name in REQUIRED:
        row = actions.get(name, {})
        for key in ("ndcg", "recall", "mean_ce_pairs", "heavy_rate", "paired_ci_vs_CE0", "cost"):
            if key not in row:
                errors.append(f"{name} missing {key}")
        if row.get("cost"):
            for key in ("ce_component_p50_ms", "ce_component_p95_ms", "e2e_p50_ms", "e2e_p95_ms"):
                if key not in row["cost"]:
                    errors.append(f"{name} missing cost.{key}")
    for path in root.rglob("*"):
        if path.is_file() and path.name.endswith((".jsonl", ".jsonl.gz", ".json.gz")):
            errors.append(f"query-level record present in report directory: {path.name}")
    for path, digest in summary.get("inputs", {}).items():
        if not Path(path).exists() or sha(Path(path)) != digest:
            errors.append(f"input hash mismatch: {path}")
    result = {
        "kind": "independent-aggregate-ce-ablation-audit-v1",
        "summary_sha256": sha(summary_path),
        "protocol_sha256": sha(protocol_path),
        "required_actions": sorted(REQUIRED),
        "errors": errors,
        "passed": not errors,
        "query_level_records_checked_absent": True,
    }
    (root / "independent-audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    audit(p.parse_args().root)
