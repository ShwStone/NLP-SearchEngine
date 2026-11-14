from hashlib import sha256
import json
import sys
from types import SimpleNamespace

import pytest

import evaluation.audit_quality_guard_http as quality_audit
import evaluation.benchmark_dureader_policies_http as quality_benchmark
from evaluation.audit_quality_guard_http import FIXED, GUARD, request_pairs
from evaluation.benchmark_dureader_policies_http import capture_parity, policy_ce_budget, quality_guard_heavy


def test_quality_guard_uses_frozen_threshold_and_constraint_fallback():
    assert quality_guard_heavy(0.154, 0.153)
    assert not quality_guard_heavy(0.153, 0.153)
    assert not quality_guard_heavy(0.9, 0.153, explicit_constraints=True)


def test_quality_guard_budget_is_fixed_or_heavy():
    assert policy_ce_budget('quality-guard-40', 33, False) == 33
    assert policy_ce_budget('quality-guard-40', 33, True) == 40
    assert policy_ce_budget('fixed-pair-budget', 33, False) == 33
    assert policy_ce_budget('ce28', 33, False) == 28


def test_quality_guard_audit_requires_exact_fixed_control_and_unique_pairs():
    queries = {"q": "apple"}
    gate = SimpleNamespace(predict=lambda features: [features[0][0]])
    evidence = {
        "query_id": "q", "client_wall_ms": 1, "fallback": False,
        "explicit_constraints": False, "vector_attempt": True,
        "gate_features": [0.2, 0, 0, 5], "results": [str(i) for i in range(40)],
        "pre_rerank_ids": [str(i) for i in range(40)],
    }
    guard = {**evidence, "policy": GUARD, "gate_heavy": True, "ce_pairs": 40}
    fixed = {**evidence, "policy": FIXED, "gate_heavy": False, "ce_pairs": 33}
    assert request_pairs([fixed, guard], queries, gate, 0.153)[FIXED] == [fixed]
    for bad in ({**fixed, "ce_pairs": 40}, {**fixed, "fallback": True},
                {**fixed, "vector_attempt": None}, {**fixed, "query_id": "different"}, guard):
        with pytest.raises(ValueError):
            request_pairs([guard, bad], queries, gate, 0.153)

    historical = {key: value for key, value in fixed.items() if key not in ("fallback", "explicit_constraints")}
    with pytest.raises(ValueError):
        request_pairs([guard, historical], queries, gate, 0.153)
    assert request_pairs([guard, historical], queries, gate, 0.153, historical=True)[FIXED] == [historical]


def test_guard_audit_rejects_different_candidates_and_changes_outside_ce_prefix():
    queries = {"q": "apple"}
    gate = SimpleNamespace(predict=lambda features: [features[0][0]])
    ordered = [str(i) for i in range(45)]
    common = {
        "query_id": "q", "client_wall_ms": 1, "fallback": False,
        "explicit_constraints": False, "vector_attempt": True,
        "gate_features": [0.2, 1, 0, 45], "pre_rerank_ids": ordered,
    }
    guard = dict(common, policy=GUARD, gate_heavy=True, ce_pairs=40,
                 results=[*reversed(ordered[:40]), *ordered[40:]])
    fixed = dict(common, policy=FIXED, gate_heavy=False, ce_pairs=33, results=ordered)
    assert request_pairs([guard, fixed], queries, gate, 0.153)[GUARD] == [guard]

    changed_prefix = [ordered[1], ordered[0], *ordered[2:]]
    with pytest.raises(ValueError, match="same candidate order"):
        request_pairs([guard, dict(fixed, pre_rerank_ids=changed_prefix)], queries, gate, 0.153)
    with pytest.raises(ValueError, match="different BM25 gate features"):
        request_pairs([guard, dict(fixed, gate_features=[0.2, 1, 0, 44])], queries, gate, 0.153)
    for variant in (
        dict(fixed, pre_rerank_ids=None),
        dict(fixed, results=[*ordered[:32], ordered[33], ordered[32], *ordered[34:]]),
        dict(fixed, results=["replacement", *ordered[1:]]),
    ):
        with pytest.raises(ValueError, match="outside the declared prefix"):
            request_pairs([guard, variant], queries, gate, 0.153)


def test_existing_benchmark_policies_retain_capture_parity():
    recorded = {"heavy": False, "actions": {"bm25": ["d"], "learned-router": ["d"]}}
    row = {"results": ["d"], "gate_heavy": False}
    assert capture_parity(row, "bm25", recorded) is True
    assert capture_parity({**row, "results": ["other"]}, "bm25", recorded) is False
    assert capture_parity(row, GUARD, recorded) is None
    with pytest.raises(ValueError, match="Router decision"):
        capture_parity({**row, "gate_heavy": True}, "learned-router", recorded)


def test_guard_benchmark_checks_full_scope_before_loading_indexes(tmp_path, monkeypatch):
    model = tmp_path / "models.json"
    model.write_text("{}", encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        '{"protocol": {"fixed_pair_budget": 33}}', encoding="utf-8"
    )
    queries = tmp_path / "queries.tsv"
    queries.write_text("q\tapple\n", encoding="utf-8")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "training_model_sha256": sha256(model.read_bytes()).hexdigest(),
        "holdout_sha256": {"queries": sha256(queries.read_bytes()).hexdigest()},
        "threshold": 0.5,
    }), encoding="utf-8")
    monkeypatch.setattr(quality_benchmark, "AUDIT_CONFIG", config)
    output = tmp_path / "output"
    args = dict(index=None, vector_manifest=None, model_path=None, ce_model=None, url=None,
                queries=queries, qrels=None, policy_dir=tmp_path, output=output)
    for policies, limit in ((GUARD, 0), (f"{GUARD},{FIXED},{FIXED}", 0), (f"{GUARD},{FIXED}", 1)):
        with pytest.raises(ValueError, match="requires exactly the two frozen arms"):
            quality_benchmark.benchmark(**args, policies=policies, limit=limit)
    with pytest.raises(ValueError, match="query records do not match the frozen holdout"):
        quality_benchmark.benchmark(**args, policies=f"{GUARD},{FIXED}")
    assert not output.exists()


@pytest.mark.parametrize(
    ("historical", "expected_source", "expected_exit"),
    [
        (True, "historical", 0), (False, "unbound_recapture", 3),
        (False, "runtime_attested_recapture", 0), (False, "wrong_attestation", None),
        (False, "wrong_restore", None), (False, "wrong_witness", None),
        (False, None, None),
    ],
)
def test_quality_guard_cli_distinguishes_historical_unbound_and_attested_captures(
    tmp_path, monkeypatch, capsys, historical, expected_source, expected_exit
):
    queries = tmp_path / "queries.tsv"
    qrels = tmp_path / "qrels.txt"
    models = tmp_path / "models.json"
    requests = tmp_path / "requests.jsonl"
    model = {"mean": [0] * 4, "scale": [1] * 4, "coef": [1, 0, 0, 0], "intercept": 0}
    models.write_text(json.dumps({"gate": {"model": model}}), encoding="utf-8")
    qids = [f"q{i}" for i in range(808)]
    queries.write_text("".join(f"{qid}\tapple\n" for qid in qids), encoding="utf-8")
    qrels.write_text("".join(f"{qid} d39 1\n" for qid in qids), encoding="utf-8")
    ranking = [f"d{i}" for i in range(40)]
    with requests.open("w", encoding="utf-8") as handle:
        for index, qid in enumerate(qids):
            heavy = index < 100
            evidence = {"query_id": qid, "vector_attempt": True, "client_wall_ms": 1}
            if not historical:
                evidence.update(explicit_constraints=False, fallback=expected_source is None,
                                pre_rerank_ids=ranking)
            fixed = dict(evidence, policy=FIXED, gate_heavy=False, ce_pairs=33, results=ranking,
                         gate_features=[float(heavy), 0, 0, 40])
            guard = dict(
                evidence, policy=GUARD, gate_heavy=heavy, ce_pairs=40 if heavy else 33,
                results=[ranking[-1], *ranking[:-1]] if heavy else ranking,
                gate_features=[float(heavy), 0, 0, 40],
            )
            for row in (fixed, guard):
                handle.write(json.dumps(row) + "\n")
    request_digest = sha256(requests.read_bytes()).hexdigest()
    config = tmp_path / "config.json"
    frozen = json.loads(quality_audit.AUDIT_CONFIG.read_bytes())["recapture"]
    config.write_text(json.dumps({
        "training_model_sha256": sha256(models.read_bytes()).hexdigest(),
        "holdout_sha256": {
            "queries": sha256(queries.read_bytes()).hexdigest(),
            "qrels": sha256(qrels.read_bytes()).hexdigest(),
        },
        "http_requests_sha256": request_digest if historical else "0" * 64,
        "threshold": 0.5,
        "cost_gate": {"max_mean_ce_pairs": 36.49},
        "recapture": frozen,
    }), encoding="utf-8")
    monkeypatch.setattr(quality_audit, "AUDIT_CONFIG", config)
    args = [
        "audit_quality_guard_http", "--folder", str(tmp_path), "--qrels", str(qrels),
        "--models", str(models), "--queries", str(queries), "--requests-sha256", request_digest,
    ]
    if expected_source in ("runtime_attested_recapture", "wrong_attestation", "wrong_restore", "wrong_witness"):
        runtime = {
            key: frozen[key] for key in
            ("documents", "compact_manifest_sha256", "vector_manifest_sha256", "corpus_fingerprint")
        }
        runtime.update(
            collection="real_bge_official8096668",
            declared_snapshot_files={"frozen.snapshot": frozen["snapshot_sha256"]},
            ann={"hnsw_ef": 128, "exact": False, "timeout_s": 30.0},
            lexical_backend="frozen-rfc4-compact-bounded-float64",
            candidate_k=200, rrf_c=60.0,
            query_encoder={"inventory_sha256": frozen["query_inventory_sha256"],
                           "device": "cuda:0", "dtype": "float32", "implementation": "bge-cls-torch"},
            cross_encoder={"inventory_sha256": frozen["cross_encoder_inventory_sha256"],
                           "model_id": "BAAI/bge-reranker-base", "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
                           "max_length": 256, "batch_size": 20, "device": "cuda", "dtype": "torch.float16"},
        )
        sources = {name: "1" * 64 for name in quality_audit.GUARD_SOURCE_FILES}
        sources.update({f"rfc4/offline/{name}": digest for name, digest in quality_audit.READER_HASHES.items()})
        capture = tmp_path / quality_audit.CAPTURE_RECEIPT
        capture.write_text(json.dumps({
            "schema_version": 2, "status": "PAIRED_REQUESTS_CHECKED_BEFORE_SCORING",
            "requests_sha256": request_digest, "queries_sha256": sha256(queries.read_bytes()).hexdigest(),
            "models_sha256": sha256(models.read_bytes()).hexdigest(),
            "queries": 808, "threshold": 0.5, "policy_order": [GUARD, FIXED],
            "runtime": runtime, "runtime_source_sha256": sources,
        }), encoding="utf-8")
        capture_sha = sha256(capture.read_bytes()).hexdigest()
        restore = tmp_path / "restore.json"
        restore.write_text(json.dumps({
            "status": "OFFICIAL_CE_INPUTS_VERIFIED", "documents": frozen["documents"],
            "manifest_sha256": frozen["delivery_manifest_sha256"],
            "payload": [
                {"path": "official/compact/compact.json", "sha256": frozen["compact_manifest_sha256"]},
                {"path": "official/vector/vector_manifest.json", "sha256": frozen["vector_manifest_sha256"]},
                {"path": "official/vector/frozen.snapshot",
                 "sha256": "0" * 64 if expected_source == "wrong_restore" else frozen["snapshot_sha256"]},
            ],
        }), encoding="utf-8")
        witness = tmp_path / "witness.json"
        witness.write_text(json.dumps({
            "schema_version": 1, "restored_instance": "fixture_restore_instance",
            "benchmark_instance": ("different_fixture_instance" if expected_source == "wrong_witness"
                                   else "fixture_restore_instance"),
            "capture_sha256": capture_sha, "native_restore_sha256": sha256(restore.read_bytes()).hexdigest(),
            "snapshot_sha256": frozen["snapshot_sha256"], "collection": runtime["collection"],
            "exact_points": frozen["documents"],
        }), encoding="utf-8")
        attestation = tmp_path / "independent.json"
        attestation.write_text(json.dumps({
            "schema_version": 1, "status": "VERIFIED_ISOLATED_RESTORE_AND_CAPTURE",
            "capture_sha256": capture_sha, "restore_receipt_sha256": sha256(restore.read_bytes()).hexdigest(),
            "live_restore_witness_sha256": sha256(witness.read_bytes()).hexdigest(),
            "runtime": ({**runtime, "corpus_fingerprint": "0" * 64}
                        if expected_source == "wrong_attestation" else runtime),
            "runtime_source_sha256": sources,
        }), encoding="utf-8")
        args.extend(["--capture-sha256", capture_sha, "--runtime-attestation",
                     str(attestation), sha256(attestation.read_bytes()).hexdigest(),
                     "--restore-receipt", str(restore), "--live-restore-witness", str(witness)])
    monkeypatch.setattr(sys, "argv", args)

    if expected_source is None:
        qrels.unlink()
        with pytest.raises(ValueError, match="fallback occurred"):
            quality_audit.main()
        assert not list(tmp_path.glob("audit-*.json"))
        return
    failures = {"wrong_attestation": "Independent runtime attestation differs",
                "wrong_restore": "Restore receipt is missing",
                "wrong_witness": "Witness does not bind"}
    if expected_source in failures:
        qrels.unlink()
        with pytest.raises(ValueError, match=failures[expected_source]):
            quality_audit.main()
        assert not list(tmp_path.glob("audit-*.json"))
        return
    assert quality_audit.main() == expected_exit
    filename = {"historical": "audit-verified.json", "unbound_recapture": "audit-unbound.json",
                "runtime_attested_recapture": "audit-runtime-attested.json"}[expected_source]
    result = json.loads((tmp_path / filename).read_text(encoding="utf-8"))
    assert (tmp_path / "audit-verified.json").is_file() is historical
    assert result == json.loads(capsys.readouterr().out)
    assert result["quality_guard_source"] == expected_source
    assert result["fallback_field_required"] is not historical
    assert result.get("matched_candidate_order_verified") is (None if historical else True)
    assert ("runtime_evidence" in result) is (expected_source == "runtime_attested_recapture")
    assert result["quality_gate_passed"] is True
    assert result["ce_pairs_mean"] == {GUARD: pytest.approx(33 + 700 / 808), FIXED: 33.0}
