"""Read the frozen RFC-0002 evidence and print aggregate audit results."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
import unicodedata
from urllib.parse import urlsplit

from . import clean_news
from .clean_news import NEAR_HAMMING_MAX, extract_article, is_near_duplicate, signature


# Anchors from evaluation/RESOURCES.md and the local frozen evidence.
RECEIPT_SHA256 = "e73861130be4da1ee1a1eef772e4d52bbb5837126a265b6650aadebade8e99a6"
SELECTED_MANIFEST_SHA256 = "f14f7511da1a928bb4d6f7be4f49c159891cf1998bda97979b36448cb4436a0d"
QUERY_EXPORT_MANIFEST_SHA256 = "92814c7852e0af3d70ff1b539ca3f57d9c56b622dce1fd6351d9925965df91f3"
QRELS_MANIFEST_SHA256 = "fdcdc39b91388b7e48b58fbbbbfecab7206f756542ead1d49a6904912a1c5214"
OFFICIAL_FREEZE_SHA256 = "7e8e1ad7ea7f47a6ac4279ae92009d7c1a61682c4a58185266ac387005273c3a"
RUN_STATE_SHA256 = "6f03d13cfd69e5984bcaf7e916d207179978962e136ecd8155af3ed65ba0eed9"
NEWS_ARMS = ("lexical", "dense", "hybrid", "adaptive")


def require(condition, description):
    if not condition:
        raise ValueError(description)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hash(path, expected):
    require(sha256(path) == expected, f"SHA-256 mismatch for {path.name}")


def read_json(path, expected_hash):
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == expected_hash,
            f"SHA-256 mismatch for {path.name}")
    return json.loads(payload)


def jsonl(path):
    with path.open(encoding="utf-8") as source:
        yield from map(json.loads, source)


def distribution(counts):
    return {"min": min(counts), "median": median(counts), "max": max(counts)}


def residual_duplicates(bodies):
    """Count exact and nonexact near-duplicate pairs under the crawl rules."""
    require(NEAR_HAMMING_MAX < 4, "The four-band search cannot cover the configured threshold")
    exact_counts = Counter()
    bands = defaultdict(list)
    signatures = []
    exact_pairs = near_pairs = candidate_pairs = 0
    for body in bodies:
        current = signature(body)
        exact_pairs += exact_counts[current.normalized_text_sha256]
        exact_counts[current.normalized_text_sha256] += 1
        keys = [(i, (current.simhash64 >> (16 * i)) & 0xffff) for i in range(4)]
        # At most three differing bits leave at least one of four bands intact.
        candidates = {index for key in keys for index in bands[key]}
        candidate_pairs += len(candidates)
        near_pairs += sum(
            previous.normalized_text_sha256 != current.normalized_text_sha256
            and is_near_duplicate(previous, current)
            for previous in (signatures[index] for index in candidates)
        )
        for key in keys:
            bands[key].append(len(signatures))
        signatures.append(current)
    return {"documents": len(signatures), "normalized_exact_pairs": exact_pairs,
            "nonexact_near_pairs": near_pairs, "band_candidate_pairs": candidate_pairs}


def raw_extraction_stats(directory, sidecar, documents, max_bytes):
    """Replay archived accepted pages from byte-verified captures and emit aggregates."""
    counts = Counter()
    selectors = Counter()
    ordered_digest = hashlib.sha256()
    raw_root = directory.resolve() / "raw"
    for row in jsonl(sidecar):
        if row["outcome"] != "success":
            continue
        doc_id = f"crawl-{row['representative_seq']:06d}"
        url = row["source_url"]
        candidates = [capture for capture in row["captures"] if capture["source_url"] == url]
        if len(candidates) != 1:
            counts["missing_or_ambiguous_capture"] += 1
            continue
        capture, = candidates
        raw_digest = capture["http_bytes_sha256"]
        require(re.fullmatch(r"[a-f0-9]{64}", raw_digest), "Invalid capture SHA-256")
        relative = f"raw/{raw_digest[:2]}/{raw_digest}.html.gz"
        require(capture["cache_path"] == relative, "Unexpected raw cache member path")
        path = directory / relative
        require(path.resolve().is_relative_to(raw_root) and not path.is_symlink(), "Untrusted raw cache path")
        if not path.is_file():
            counts["missing_raw_file"] += 1
            continue
        try:
            with gzip.open(path, "rb") as stream:
                payload = stream.read(max_bytes + 1)
        except (OSError, EOFError):
            counts["raw_read_failure"] += 1
            continue
        if len(payload) > max_bytes or hashlib.sha256(payload).hexdigest() != raw_digest:
            counts["raw_hash_or_size_difference"] += 1
            continue
        article = extract_article(payload.decode(capture["charset"] or "utf-8", errors="replace"), url)
        meta = asdict(article.metadata)
        meta["filter_reasons"] = list(article.metadata.filter_reasons)
        doc = documents[doc_id]
        if (not article.accepted or (article.title, article.text) != (doc["title"], doc["text"])
                or meta != row["clean_metadata"]):
            counts["extraction_or_metadata_difference"] += 1
            continue
        counts["byte_hash_and_extraction_pass"] += 1
        selectors[meta["body_selector"]] += 1
        ordered_digest.update(f"{doc_id}\0{raw_digest}\0{meta['cleaned_body_sha256']}\n".encode())
    require(sum(counts.values()) == len(documents), "Raw audit did not cover the accepted records")
    return {"accepted_documents": len(documents), "counts": dict(counts), "selectors": dict(selectors),
            "ordered_raw_and_body_binding_sha256": ordered_digest.hexdigest()}


def corpus_stats(directory, receipt, *, visit_sidecar=None):
    corpus = directory / "corpus.jsonl"
    sidecar = directory / "article_sidecar.jsonl"
    check_hash(corpus, receipt["corpus_sha256"])
    check_hash(sidecar, receipt["sidecar_sha256"])
    require(receipt["status"] == "TARGET_REACHED", "Crawl did not finish")
    documents = {}
    for position, row in enumerate(jsonl(corpus), 1):
        require(row["doc_id"] == f"crawl-{position:06d}" and row["title"] and row["text"],
                "Invalid document identity, order or content")
        documents[row["doc_id"]] = row
    require(len(documents) == receipt["accepted_unique_documents"] == 20_000,
            "Unexpected accepted document count")
    accepted = set()
    outcomes = Counter()
    hosts = Counter()
    selectors = Counter()
    captures = 0
    for row in jsonl(sidecar):
        if visit_sidecar is not None:
            visit_sidecar(row)
        outcome = row["outcome"]
        outcomes[outcome] += 1
        captures += len(row["captures"])
        if outcome == "success":
            doc_id = f"crawl-{row['representative_seq']:06d}"
            require(doc_id in documents and doc_id not in accepted, "Invalid accepted sidecar identity")
            accepted.add(doc_id)
            metadata = row["clean_metadata"]
            body_hash = hashlib.sha256(documents[doc_id]["text"].encode()).hexdigest()
            require(body_hash == metadata["cleaned_body_sha256"], "Sidecar cleaned body mismatch")
            hosts[urlsplit(row["source_url"]).hostname] += 1
            selectors[metadata["body_selector"]] += 1
    require(accepted == set(documents) and outcomes == receipt["outcomes"],
            "Sidecar coverage or crawl dispositions differ from receipt")
    require(sum(outcomes.values()) == receipt["attempts"] and captures == receipt["raw_capture_count"],
            "Sidecar attempt or capture count differs from receipt")
    return documents, {
        "documents": len(documents), "attempt_outcomes": dict(outcomes),
        "attempts": receipt["attempts"], "http_requests": receipt["requests"], "raw_captures": captures,
        "accepted_source_hosts": dict(hosts), "accepted_body_selectors": dict(selectors),
        "body_codepoints": distribution([len(row["text"]) for row in documents.values()]),
    }


def normalize(text):
    return "".join(character for character in unicodedata.normalize("NFKC", text).lower()
                   if character.isalnum())


def query_signature(text):
    normalized = normalize(text)
    bigrams = {normalized[i:i + 2] for i in range(len(normalized) - 1)}
    return normalized, bigrams or {normalized}


def overlap(left, right):
    left = [query_signature(text) for text in left]
    right = [query_signature(text) for text in right]
    require(all(normalized for normalized, _ in left + right), "Empty normalized query")
    exact = near = 0
    for left_text, left_grams in left:
        for right_text, right_grams in right:
            if left_text == right_text:
                exact += 1
            elif 5 * len(left_grams & right_grams) >= 4 * len(left_grams | right_grams):
                near += 1
    return {"compared_pairs": len(left) * len(right), "normalized_exact": exact,
            "nonexact_bigram_jaccard_ge_0_8": near}


def query_sets(directory, documents):
    manifest = read_json(directory / "selected-200.manifest.json", SELECTED_MANIFEST_SHA256)
    selected = directory / "selected-200.source-supported.jsonl"
    check_hash(selected, manifest["selected_queries_sha256"])
    rows = list(jsonl(selected))
    queries = {row["query_id"]: row for row in rows}
    require(len(queries) == len(rows) == 200, "Unexpected selected query IDs")
    for row in rows:
        anchor = documents[row["source_doc_id"]]
        require(row["source_title"] == anchor["title"], "Query anchor title mismatch")
    sorted_ids = sorted(queries, key=lambda query_id: hashlib.sha256(
        f"{manifest['split_seed']}\0{query_id}".encode()).hexdigest())
    splits = {"dev": sorted_ids[:50], "test": sorted_ids[50:]}
    require(splits == manifest["splits"], "Frozen query split does not replay")
    exports = directory / "queries-only"
    exports_manifest = read_json(exports / "manifest.json", QUERY_EXPORT_MANIFEST_SHA256)
    require(exports_manifest["selected_jsonl_sha256"] == manifest["selected_queries_sha256"],
            "Query exports refer to another selection")
    identities = exports_manifest["current_source_rows_verified"]
    require(len(identities) == len(queries) and {row["query_id"] for row in identities} == set(queries),
            "Query anchor identity coverage mismatch")
    for identity in identities:
        qid = identity["query_id"]
        doc_id = queries[qid]["source_doc_id"]
        anchor = documents[doc_id]
        payload = json.dumps([doc_id, anchor["title"], anchor["text"]],
                             ensure_ascii=False, separators=(",", ":")).encode()
        require(identity["source_doc_id"] == doc_id and
                identity["source_identity_sha256"] == hashlib.sha256(payload).hexdigest(),
                "Query anchor document identity mismatch")
    for name, ids in {"all": list(queries), **splits}.items():
        filename = f"{name}.queries.tsv"
        check_hash(exports / filename, exports_manifest["files"][filename])
        exported = [line.split("\t", 1) for line in (exports / filename).read_text(encoding="utf-8").splitlines()]
        require({tuple(row) for row in exported} == {(qid, queries[qid]["query"]) for qid in ids}
                and len(exported) == len(ids), "Query export differs from selected rows")
    return queries, splits, manifest["selected_queries_sha256"]


def official_queries(directory):
    manifest = read_json(directory / "freeze.json", OFFICIAL_FREEZE_SHA256)
    require(manifest["source_split"] == "official dev", "Expected official-dev lineage")
    queries = {}
    for split, expected_count in (("train", 172), ("dev", 1020), ("test", 808)):
        filename = f"queries.{split}.jsonl"
        path = directory / filename
        check_hash(path, manifest["files"][filename]["sha256"])
        count_before = len(queries)
        for row in jsonl(path):
            require(row["qid"] not in queries, "Repeated official-dev query ID")
            queries[row["qid"]] = row["query"]
        require(len(queries) - count_before == expected_count, "Unexpected official-dev query count")
    return queries, manifest


def exact_body_overlap(path, documents, freeze):
    news_bodies = {row["text"] for row in documents.values()}
    digest = hashlib.sha256()
    matched = set()
    matches = count = 0
    with path.open("rb") as source:
        for count, raw in enumerate(source, 1):
            digest.update(raw)
            body = json.loads(raw)["text"]
            if body in news_bodies:
                matches += 1
                matched.add(body)
    require(count == freeze["documents"] and digest.hexdigest() == freeze["corpus_sha256"],
            "Official corpus document count or SHA-256 differs from freeze")
    return {"official_documents": count, "official_matching_documents": matches,
            "news_documents_with_exact_match": sum(row["text"] in matched for row in documents.values()),
            "official_corpus_sha256": digest.hexdigest()}


def qrel_stats(directory, pool_directory, queries, splits, documents, query_hash, corpus_hash,
               *, all_pairs=False):
    manifest = read_json(directory / "manifest.json", QRELS_MANIFEST_SHA256)
    require(manifest["corpus_sha256"] == corpus_hash and manifest["selected_queries_sha256"] == query_hash,
            "Qrels refer to different inputs")
    pool_path = pool_directory / "manifest.json"
    pool = read_json(pool_path, manifest["pool_manifest_sha256"])
    provenance = pool_directory / "pool-provenance.jsonl"
    check_hash(provenance, pool["files"][provenance.name])
    pool_rows = [(row["query_id"], row["doc_id"]) for row in jsonl(provenance)]
    require(len(set(pool_rows)) == len(pool_rows) == manifest["pool_pairs"] == pool["pool_pairs"],
            "Pool coverage mismatch")
    results = {}
    pair_sets = {}
    for name, ids in {"all": list(queries), **splits}.items():
        filename = f"{name}.qrels.tsv"
        path = directory / filename
        check_hash(path, manifest["files"][filename])
        rows = {}
        with path.open(encoding="utf-8") as source:
            for line in source:
                qid, iteration, doc_id, grade = line.split()
                require(iteration == "0" and (qid, doc_id) not in rows and doc_id in documents
                        and grade in ("0", "1", "2"), "Invalid qrels row")
                rows[qid, doc_id] = int(grade)
        pair_sets[name] = rows
        judged = Counter(qid for qid, _ in rows)
        positive = Counter(qid for (qid, _), grade in rows.items() if grade > 0)
        grades = Counter(str(grade) for grade in rows.values())
        require(set(judged) == set(ids) and grades == manifest["grade_counts"][name],
                "Qrels query or grade coverage differs from manifest")
        results[name] = {"queries": len(judged), "pairs": len(rows), "grades": dict(grades),
                         "judged_per_query": distribution(list(judged.values())),
                         "positive_per_query": distribution([positive[qid] for qid in ids]),
                         "queries_without_positive": len(set(ids) - positive.keys())}
    require(pair_sets["all"] == pair_sets["dev"] | pair_sets["test"] and
            set(pair_sets["all"]) == set(pool_rows), "Qrels do not exactly cover the split pool")
    judgment_path = directory / "judgments.jsonl"
    check_hash(judgment_path, manifest["files"][judgment_path.name])
    judgments = list(jsonl(judgment_path))
    require(len(judgments) == len(pair_sets["all"]) and
            {(row["query_id"], row["doc_id"]): row["relevance"] for row in judgments} == pair_sets["all"],
            "Source judgments differ from pooled qrels")
    return results, pair_sets["all" if all_pairs else "test"]


def load_frozen_rankings(directory, qrel_hash, test_ids):
    state = read_json(directory / "run-state.json", RUN_STATE_SHA256)
    require(state["qrels_manifest_sha256"] == qrel_hash, "TREC runs refer to other qrels")
    runs = {}
    for arm in NEWS_ARMS:
        path = directory / f"{arm}.trec"
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == state["arms"][arm]["sha256"],
                f"SHA-256 mismatch for {path.name}")
        run = {}
        seen = set()
        for line in payload.decode("utf-8").splitlines():
            qid, iteration, doc_id, rank, score, _ = line.split()
            ranked = run.setdefault(qid, [])
            require(iteration == "Q0" and int(rank) == len(ranked) + 1
                    and math.isfinite(float(score)) and (qid, doc_id) not in seen,
                    "Invalid TREC row, order or repeated document")
            seen.add((qid, doc_id))
            ranked.append(doc_id)
        require(set(run) == set(test_ids) and all(len(ranked) == 100 for ranked in run.values()),
                "TREC result depth or query coverage mismatch")
        runs[arm] = run
    return state, runs


def coverage_for_rankings(runs, judged_pairs):
    return {
        arm: {
            cutoff: {"judged": sum((qid, doc_id) in judged_pairs for qid, ranked in run.items()
                                  for doc_id in ranked[:cutoff]),
                     "slots": cutoff * len(run)}
            for cutoff in (10, 100)
        } for arm, run in runs.items()
    }


def ranked_coverage(directory, qrels, qrel_hash, test_ids):
    _, runs = load_frozen_rankings(directory, qrel_hash, test_ids)
    return coverage_for_rankings(runs, qrels)


def audit(root, official_corpus=None, *, check_residual_duplicates=False, check_raw_extraction=False):
    crawl = root / "news-clean-20k-20261102"
    receipt = read_json(crawl / "receipt.json", RECEIPT_SHA256)
    documents, corpus = corpus_stats(crawl, receipt)
    queries, splits, query_hash = query_sets(root / "news-query-design-20261102", documents)
    official, official_freeze = official_queries(root / "official-dev-frozen-splits-20261102")
    dev, test = ([queries[qid]["query"] for qid in splits[name]] for name in ("dev", "test"))
    qrels, test_qrels = qrel_stats(
        root / "news-formal-pooled-qrels-20261104",
        root / "news-formal-http-pool-rm3-resume-20261103" / "pool",
        queries, splits, documents, query_hash, receipt["corpus_sha256"])
    results = {
        "corpus": corpus,
        "query_types": dict(Counter(row["query_type"] for row in queries.values())),
        "overlap": {"news_dev_vs_test": overlap(dev, test),
                    "news_vs_official_dev": overlap(dev + test, official.values()),
                    "source_document_overlap_dev_vs_test": len(
                        {queries[qid]["source_doc_id"] for qid in splits["dev"]} &
                        {queries[qid]["source_doc_id"] for qid in splits["test"]})},
        "qrels": qrels,
        "formal_test_ranked_coverage": ranked_coverage(
            root / "news-formal-quality-test-20261104" / "formal-test",
            test_qrels, QRELS_MANIFEST_SHA256, splits["test"]),
        "unicode_version": unicodedata.unidata_version,
    }
    results["official_corpus_exact_body_overlap"] = (
        exact_body_overlap(official_corpus, documents, official_freeze)
        if official_corpus is not None else {"status": "not_requested"})
    if check_residual_duplicates:
        results["corpus"]["residual_duplicates"] = {
            "corpus_sha256": receipt["corpus_sha256"],
            **residual_duplicates(row["text"] for row in documents.values()),
        }
    if check_raw_extraction:
        sidecar = crawl / "article_sidecar.jsonl"
        extractor = Path(clean_news.__file__)
        source_sha = sha256(extractor)
        binding = receipt["binding"]
        require(clean_news.RULE_VERSION == binding["rule_version"], "Raw extraction rule version changed")
        results["corpus"]["raw_extraction"] = raw_extraction_stats(
            crawl, sidecar, documents, binding["config"]["max_bytes"])
        results["corpus"]["raw_extraction"].update(
            source_sha256=source_sha,
            original_source_sha256=binding["clean_source_sha256"],
            rule_version=clean_news.RULE_VERSION,
        )
        check_hash(extractor, source_sha)
        check_hash(sidecar, receipt["sidecar_sha256"])
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("--official-corpus", type=Path, help="Optional frozen official full JSONL")
    parser.add_argument("--residual-duplicates", action="store_true",
                        help="Check every final news document under the crawl deduplication rules")
    parser.add_argument("--replay-raw-extraction", action="store_true",
                        help="Re-extract every accepted article from the private HTTP response cache")
    args = parser.parse_args()
    result = audit(args.evidence_root, args.official_corpus, check_residual_duplicates=args.residual_duplicates,
                   check_raw_extraction=args.replay_raw_extraction)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    if args.replay_raw_extraction:
        raw = result["corpus"]["raw_extraction"]
        if raw["counts"].get("byte_hash_and_extraction_pass", 0) != raw["accepted_documents"]:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
