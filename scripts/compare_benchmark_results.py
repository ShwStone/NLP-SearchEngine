"""Compare ranked IDs and rounded scores in two benchmark_http.py receipts."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--mode", default="lexical")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    left = {row["qid"]: row for row in baseline["measurements"] if row["mode"] == args.mode}
    right = {row["qid"]: row for row in candidate["measurements"] if row["mode"] == args.mode}
    if set(left) != set(right):
        raise ValueError("receipt query IDs differ")
    rows = []
    for qid in sorted(left):
        a, b = left[qid], right[qid]
        ids_a = [item["doc_id"] for item in a["results"]]
        ids_b = [item["doc_id"] for item in b["results"]]
        score_delta = max((abs(x["score"] - y["score"])
                           for x, y in zip(a["results"], b["results"])), default=0.0)
        rows.append({"qid": qid, "ids_equal": ids_a == ids_b,
                     "max_rounded_score_delta": score_delta,
                     "baseline_server_ms": a["server_ms"], "candidate_server_ms": b["server_ms"],
                     "baseline_client_ms": a["client_ms"], "candidate_client_ms": b["client_ms"]})
    receipt = {
        "schema_version": 1,
        "mode": args.mode,
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "queries": len(rows),
        "all_ids_equal": all(row["ids_equal"] for row in rows),
        "max_rounded_score_delta": max((row["max_rounded_score_delta"] for row in rows), default=0.0),
        "per_query": rows,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: value for key, value in receipt.items() if key != "per_query"}, indent=2))
    return 0 if receipt["all_ids_equal"] and receipt["max_rounded_score_delta"] <= 2e-6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
