"""Audit the frozen full-corpus CE-33/40 comparison."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import re

import numpy as np

from evaluation.dureader_policy_features import FrozenLinear
from evaluation.delivery_backend import READER_HASHES
from evaluation.evaluate import load_qrels, load_queries
from evaluation.prepare_bm25_router import metrics
from evaluation.router_metrics import group_bootstrap, percentile
from online.parser import parse


GUARD = "quality-guard-40"
FIXED = "fixed-pair-budget"
POLICIES = (GUARD, FIXED)
AUDIT_CONFIG = Path(__file__).resolve().parents[1] / "docs/rfc0001/full-quality-guard-http-v1.json"
CAPTURE_RECEIPT = "capture-before-scoring.json"
GUARD_SOURCE_FILES = (
    "evaluation/benchmark_dureader_policies_http.py", "evaluation/dureader_policy_features.py",
    "evaluation/delivery_backend.py", "evaluation/study_rerank_actions.py", "online/vector_index.py",
    "online/scorers.py", "online/compiled_bm25.py", "online/parser.py", "online/fusion.py",
    "contracts/analyzer.py", "adapters/embedding.py", "adapters/model_files.py",
)


def verify_bytes(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"Frozen input differs: {Path(path).name}")
    return raw


def validate_runtime(runtime, expected):
    if not isinstance(runtime, dict):
        raise ValueError("Missing full-corpus runtime identity")
    for field in ("documents", "compact_manifest_sha256", "vector_manifest_sha256", "corpus_fingerprint"):
        if type(runtime.get(field)) is not type(expected[field]) or runtime[field] != expected[field]:
            raise ValueError(f"Captured full-corpus identity differs: {field}")
    query = runtime.get("query_encoder", {})
    ce = runtime.get("cross_encoder", {})
    snapshot = runtime.get("declared_snapshot_files", {})
    ann = runtime.get("ann")
    if (runtime.get("collection") != "real_bge_official8096668"
            or not isinstance(snapshot, dict) or list(snapshot.values()) != [expected["snapshot_sha256"]]
            or not isinstance(ann, dict) or ann.get("exact") is not False
            or ann != {"hnsw_ef": 128, "exact": False, "timeout_s": 30.0}
            or runtime.get("candidate_k") != 200 or runtime.get("rrf_c") != 60.0
            or runtime.get("lexical_backend") != "frozen-rfc4-compact-bounded-float64"
            or not isinstance(query, dict) or query.get("inventory_sha256") != expected["query_inventory_sha256"]
            or query.get("implementation") != "bge-cls-torch" or query.get("dtype") != "float32"
            or re.fullmatch(r"cuda(?::[0-9]+)?", str(query.get("device"))) is None
            or ce != {"inventory_sha256": expected["cross_encoder_inventory_sha256"],
                      "model_id": "BAAI/bge-reranker-base", "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
                      "max_length": 256, "batch_size": 20, "device": "cuda", "dtype": "torch.float16"}):
        raise ValueError("Captured retrieval or model runtime differs from the frozen recapture")


def runtime_evidence(folder, requests_sha, config, capture_sha, attestation_path, attestation_sha,
                     restore_path, witness_path):
    """Bind a recapture to separately recorded pre-scoring and restore receipts."""
    for digest in (capture_sha, attestation_sha):
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Trusted receipt SHA256 must be lowercase hexadecimal")
    captured = json.loads(verify_bytes(folder / CAPTURE_RECEIPT, capture_sha))
    attested = json.loads(verify_bytes(attestation_path, attestation_sha))
    if (not isinstance(captured, dict) or type(captured.get("schema_version")) is not int
            or captured["schema_version"] != 2
            or captured.get("status") != "PAIRED_REQUESTS_CHECKED_BEFORE_SCORING"
            or captured.get("requests_sha256") != requests_sha
            or captured.get("models_sha256") != config["training_model_sha256"]
            or captured.get("queries_sha256") != config["holdout_sha256"]["queries"]
            or captured.get("queries") != 808
            or captured.get("threshold") != config["threshold"]
            or captured.get("policy_order") not in (list(POLICIES), list(reversed(POLICIES)))):
        raise ValueError("Pre-scoring receipt differs from the audited requests or inputs")
    runtime, sources = captured.get("runtime"), captured.get("runtime_source_sha256")
    validate_runtime(runtime, config["recapture"])
    frozen_readers = {f"rfc4/offline/{filename}": digest for filename, digest in READER_HASHES.items()}
    if (not isinstance(sources, dict) or set(sources) != set(GUARD_SOURCE_FILES) | set(frozen_readers)
            or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values())
            or any(sources[name] != digest for name, digest in frozen_readers.items())):
        raise ValueError("Missing runtime code identity")
    if (not isinstance(attested, dict) or type(attested.get("schema_version")) is not int
            or attested["schema_version"] != 1
            or attested.get("status") != "VERIFIED_ISOLATED_RESTORE_AND_CAPTURE"
            or attested.get("capture_sha256") != capture_sha
            or json.dumps(attested.get("runtime"), sort_keys=True) != json.dumps(runtime, sort_keys=True)
            or attested.get("runtime_source_sha256") != sources):
        raise ValueError("Independent runtime attestation differs from the captured run")
    restore_sha = attested.get("restore_receipt_sha256")
    if not isinstance(restore_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", restore_sha):
        raise ValueError("Independent attestation requires a verified restore receipt digest")
    witness_sha = attested.get("live_restore_witness_sha256")
    if not isinstance(witness_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", witness_sha):
        raise ValueError("Independent attestation requires a live restore witness digest")
    witness = json.loads(verify_bytes(witness_path, witness_sha))
    instance = witness.get("restored_instance") if isinstance(witness, dict) else None
    if (not isinstance(instance, str) or re.fullmatch(r"[A-Za-z0-9_-]{16,128}", instance) is None
            or witness.get("schema_version") != 1 or witness.get("benchmark_instance") != instance
            or witness.get("capture_sha256") != capture_sha
            or witness.get("native_restore_sha256") != restore_sha
            or witness.get("snapshot_sha256") != config["recapture"]["snapshot_sha256"]
            or witness.get("collection") != runtime["collection"]
            or witness.get("exact_points") != runtime["documents"]):
        raise ValueError("Witness does not bind the isolated restored collection to the captured run")
    restore = json.loads(verify_bytes(restore_path, restore_sha))
    frozen = config["recapture"]
    snapshot_name = next(iter(runtime["declared_snapshot_files"]))
    expected = {
        "official/compact/compact.json": frozen["compact_manifest_sha256"],
        "official/vector/vector_manifest.json": frozen["vector_manifest_sha256"],
        f"official/vector/{snapshot_name}": frozen["snapshot_sha256"],
    }
    members = restore.get("payload") if isinstance(restore, dict) else None
    if (not isinstance(members, list) or restore.get("status") != "OFFICIAL_CE_INPUTS_VERIFIED"
            or restore.get("manifest_sha256") != frozen["delivery_manifest_sha256"]
            or restore.get("documents") != frozen["documents"]):
        raise ValueError("Restore receipt does not identify the frozen native delivery")
    for path, digest in expected.items():
        matched = [row for row in members if isinstance(row, dict) and row.get("path") == path]
        if len(matched) != 1 or matched[0].get("sha256") != digest:
            raise ValueError("Restore receipt is missing the captured compact index or snapshot")
    return {"capture_sha256": capture_sha, "attestation_sha256": attestation_sha,
            "restore_receipt_sha256": restore_sha, "live_restore_witness_sha256": witness_sha,
            "runtime": runtime,
            "runtime_source_sha256": sources}


def valid_ranking(ranking):
    return (isinstance(ranking, list) and 33 <= len(ranking) <= 100
            and all(isinstance(doc, str) and doc for doc in ranking)
            and len(ranking) == len(set(ranking)))


def request_pairs(rows, queries, gate, threshold, *, historical=False):
    """Return exact audit pairs, in frozen query order."""
    by_key = {}
    for row in rows:
        key = row["query_id"], row["policy"]
        if key[0] not in queries or key[1] not in POLICIES or key in by_key:
            raise ValueError("Unexpected or duplicate query/policy pair")
        expected_explicit = bool(parse(queries[key[0]]).get("explicit_constraints", False))
        fallback = row.get("fallback", False if historical else None)
        explicit = row.get("explicit_constraints", expected_explicit if historical else None)
        if row.get("unsupported") or fallback is not False:
            raise ValueError("Missing success evidence or fallback occurred")
        if explicit is not expected_explicit or row.get("vector_attempt") is not (not expected_explicit):
            raise ValueError("HTTP query constraints differ from the frozen input")
        if type(row["gate_heavy"]) is not bool:
            raise ValueError("Missing or invalid routing evidence")
        latency = row["client_wall_ms"]
        if type(latency) not in (int, float) or not math.isfinite(latency) or latency < 0:
            raise ValueError("Invalid HTTP latency")
        ranking = row["results"]
        if not valid_ranking(ranking):
            raise ValueError("Invalid ranking for the fixed-depth comparison")
        if key[1] == GUARD or not historical:
            features = row.get("gate_features")
            if (not isinstance(features, list) or len(features) != 4
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in features)):
                raise ValueError("Invalid frozen gate features")
        if key[1] == GUARD:
            heavy = bool(not expected_explicit and gate.predict([features])[0] > threshold)
        else:
            heavy = False
        depth = 40 if heavy else 33
        if row["gate_heavy"] is not heavy or type(row["ce_pairs"]) is not int or row["ce_pairs"] != depth or len(ranking) < depth:
            raise ValueError("Action or CE work differs from the declared comparison")
        if not historical:
            candidates = row.get("pre_rerank_ids")
            if (not valid_ranking(candidates) or len(candidates) != len(ranking)
                    or candidates[depth:] != ranking[depth:]
                    or set(candidates[:depth]) != set(ranking[:depth])):
                raise ValueError("Reranking changed documents outside the declared prefix")
        by_key[key] = row
    if len(by_key) != len(queries) * len(POLICIES):
        raise ValueError("Incomplete paired HTTP capture")
    if not historical and any(by_key[qid, GUARD]["pre_rerank_ids"] != by_key[qid, FIXED]["pre_rerank_ids"]
                              for qid in queries):
        raise ValueError("Paired HTTP requests did not start from the same candidate order")
    if not historical and any(by_key[qid, GUARD]["gate_features"] != by_key[qid, FIXED]["gate_features"]
                              for qid in queries):
        raise ValueError("Paired HTTP requests used different BM25 gate features")
    return {policy: [by_key[qid, policy] for qid in queries] for policy in POLICIES}


def audit(folder, qrels_path, models_path, queries_path, requests_sha256, *,
          capture_sha256=None, runtime_attestation=None, restore_receipt=None, live_restore_witness=None):
    folder = Path(folder)
    config = json.loads(AUDIT_CONFIG.read_bytes())
    original_run = requests_sha256 == config["http_requests_sha256"]
    identity_args = (capture_sha256, runtime_attestation, restore_receipt, live_restore_witness)
    if any(value is not None for value in identity_args) and any(value is None for value in identity_args):
        raise ValueError("Bound audit requires all capture, attestation, restore and witness files")
    if original_run and runtime_attestation is not None:
        raise ValueError("Runtime attestation is only supported for a fresh capture")
    requests = verify_bytes(folder / "requests.jsonl", requests_sha256)
    model = json.loads(verify_bytes(models_path, config["training_model_sha256"]))
    frozen = config["holdout_sha256"]
    verify_bytes(queries_path, frozen["queries"])
    queries = {row.qid: row.query for row in load_queries(queries_path)}
    verify_bytes(queries_path, frozen["queries"])
    if len(queries) != 808:
        raise ValueError("Frozen holdout query count differs")
    runtime = (None if runtime_attestation is None else
               runtime_evidence(folder, requests_sha256, config, capture_sha256, *runtime_attestation,
                                restore_receipt, live_restore_witness))
    runs = request_pairs((json.loads(line) for line in requests.splitlines() if line.strip()),
                         queries, FrozenLinear(model["gate"]["model"]), config["threshold"],
                         historical=original_run)
    result = {
        "queries": len(queries),
        "quality_guard_source": "historical" if original_run else "unbound_recapture",
        "heavy_count": sum(row["gate_heavy"] for row in runs[GUARD]),
        "ce_pairs_mean": {arm: float(np.mean([row["ce_pairs"] for row in runs[arm]])) for arm in POLICIES},
        "requests_sha256": requests_sha256, "inputs_sha256": frozen,
        "training_model_sha256": config["training_model_sha256"],
        "fallback_field_required": not original_run,
    }
    if not original_run:
        result["matched_candidate_order_verified"] = True
    if runtime is not None:
        result.update(quality_guard_source="runtime_attested_recapture", runtime_evidence=runtime)
    if result["ce_pairs_mean"][GUARD] > config["cost_gate"]["max_mean_ce_pairs"]:
        raise ValueError("Absolute CE cost gate failed")
    verify_bytes(qrels_path, frozen["qrels"])
    qrels = load_qrels(qrels_path)
    verify_bytes(qrels_path, frozen["qrels"])
    if set(queries) != set(qrels):
        raise ValueError("Frozen holdout query/relevance coverage differs")
    scored = {arm: [metrics(row["results"], qrels[qid]) for qid, row in zip(queries, rows, strict=True)]
              for arm, rows in runs.items()}
    query_ids = np.asarray(list(queries))
    for metric in ("ndcg", "recall", "mrr", "map"):
        delta = np.array([a[metric] - b[metric]
                          for a, b in zip(scored[GUARD], scored[FIXED], strict=True)])
        bootstrap = group_bootstrap(delta, query_ids)
        bootstrap["unit"] = "paired queries"
        result[metric] = {"delta": float(delta.mean()), "paired_query_ci95": bootstrap}
    result["quality_gate_passed"] = all(result[name]["paired_query_ci95"]["ci95"][0] > 0 for name in ("ndcg", "recall"))
    result["latency_ms"] = {
        arm: {label: percentile([row["client_wall_ms"] for row in rows], quantile)
              for label, quantile in (("p50", .5), ("p95", .95))}
        for arm, rows in runs.items()
    }
    filename = {
        "historical": "audit-verified.json",
        "unbound_recapture": "audit-unbound.json",
        "runtime_attested_recapture": "audit-runtime-attested.json",
    }[result["quality_guard_source"]]
    with (folder / filename).open("x", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("folder", "qrels", "models", "queries"):
        cli.add_argument("--" + name, type=Path, required=True)
    cli.add_argument("--requests-sha256", required=True, help="Digest recorded before scoring")
    cli.add_argument("--capture-sha256", help="Independently recorded pre-scoring receipt digest")
    cli.add_argument("--runtime-attestation", nargs=2, metavar=("RECEIPT", "TRUSTED_SHA256"),
                     help="Independent verification of the actual restore and capture")
    cli.add_argument("--restore-receipt", type=Path, help="Native archive restore ACCEPTED.json")
    cli.add_argument("--live-restore-witness", type=Path, help="Independent same-instance witness JSON")
    args = cli.parse_args()
    result = audit(args.folder, args.qrels, args.models, args.queries, args.requests_sha256,
                   capture_sha256=args.capture_sha256, runtime_attestation=args.runtime_attestation,
                   restore_receipt=args.restore_receipt, live_restore_witness=args.live_restore_witness)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["quality_guard_source"] == "unbound_recapture":
        return 3
    return 0 if result["quality_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
