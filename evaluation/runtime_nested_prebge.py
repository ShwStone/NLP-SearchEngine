"""Live pre-CE gain routing for the audited v4 outer models.

The serving manifest contains model configuration and hashes only, never
training targets, qrels, captured features or saved query predictions.
"""
import json
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import ThreadpoolController

from contracts import analyzer
from contracts.api import ApiError, ResultItem
from online import parser
from online.highlight import indexed_highlights
from online.router_features import extract_statistics_v2
from .bge_candidate_support import bge_support, FEATURE_NAMES as SUPPORT_NAMES
from .prepare_bm25_router import sha
from .runtime_bge_calibration import BGECalibrationRouter, zscore
from .runtime_native_router import candidate_matrix
from .study_candidate_ranking import FEATURES, TEXT_FEATURES
from .tensorrt_cross_encoder import TensorRTCrossEncoder
from .stable_dense_topk import stable_dense_topk


def gain_features(stats, bm25_names, lexical, dense):
    ds = [float(s) for _, s in dense]
    ls = [float(s) for _, s in lexical]
    if len(ds) < 40:
        raise ValueError('gain router requires 40 dense candidates')
    gate = [stats['query_chars'], stats['n_terms'], stats['relative_gap'],
            ds[0] - ds[1], ls[0] - ls[1] if len(ls) > 1 else 0.,
            stats['candidate_hits'], stats['vocab_coverage']]
    overlaps = [len({d for d, _ in dense[:k]} & {d for d, _ in lexical[:k]}) / k
                for k in (10, 20)]
    return np.array([*gate, *[stats[n] for n in bm25_names],
                     float(np.mean(ds[:10])), float(np.std(ds[:10])), ds[9], ds[19], ds[39],
                     float(np.mean(ls[:10])) if ls else 0.,
                     float(np.std(ls[:10])) if ls else 0., *overlaps])


class NestedGainPreBGERouter(BGECalibrationRouter):
    def __init__(self, base, backend, study_dir, fold, *, export_path, model_path,
                 scorer_batch_size=20, document_token_parity=None):
        self.base, self.reader, self.backend = base, base.reader, backend
        root = Path(study_dir)
        serving = json.loads((root / 'study.json').read_text())
        protocol = serving['protocol']
        if protocol['kind'] != 'audited-nested-preBGE-serving-v1':
            raise ValueError('unsupported nested serving manifest')
        self.ce_max_length = protocol['online_scorer']['max_length']
        self.fold = next(r for r in serving['folds'] if r['fold'] == fold)
        if not hasattr(backend, 'nested_prebge_scorer'):
            backend.identity['discarded_startup_scorer'] = dict(backend.reranker.info)
            backend.reranker = None
            backend.torch.cuda.empty_cache()
            backend.nested_prebge_scorer = TensorRTCrossEncoder(
                export_path, model_path, max_length=self.ce_max_length, batch_size=scorer_batch_size)
            backend.identity['reranker'] = backend.nested_prebge_scorer.info
        backend.reranker = backend.nested_prebge_scorer
        if backend.reranker.info != protocol['online_scorer']:
            raise ValueError('canonical and live cross-encoder differ')
        if backend.identity['support_matrix_sha256'] != protocol['support_sha256']:
            raise ValueError('canonical and live support matrix differ')
        if document_token_parity is not None:
            report_path = Path(document_token_parity)
            report = json.loads(report_path.read_text())
            cache_source = Path(__file__).with_name('gte_document_tokens.py').resolve()
            doc_digest = self.reader.manifest()['files']['docs.jsonl']
            if (not report['exact_all_input_ids_masks_special_tokens_and_truncation'] or
                    report.get('max_length', 256) != self.ce_max_length or
                    doc_digest not in report['inputs'].values() or
                    sha(Path(model_path) / 'model_inventory.json') not in report['inputs'].values() or
                    report['code_sha256'].get(str(cache_source)) != sha(cache_source)):
                raise ValueError('document-token cache lacks matching all-pair input evidence')
            if not hasattr(backend, 'nested_prebge_document_tokens'):
                from .gte_document_tokens import DocumentTokenPairs
                backend.nested_prebge_document_tokens = DocumentTokenPairs(
                    backend.reranker.tokenizer, backend.texts, max_length=self.ce_max_length)
                backend.reranker.tokenizer = backend.nested_prebge_document_tokens
                backend.identity['BGE_document_token_cache'] = dict(
                    input_parity_sha256=sha(report_path),
                    corpus_sha256=doc_digest, query_prediction_cache=False,
                    source_sha256=sha(Path(__file__).with_name('gte_document_tokens.py')))
        self.bm25_names = protocol['bm25_names']
        expected = list(FEATURES) + list(TEXT_FEATURES)
        expected += ['bge_logit', 'bge_z', 'bge_rank_reciprocal', 'bge_gap']
        expected += list(SUPPORT_NAMES) + self.bm25_names
        if protocol['features'] != expected:
            raise ValueError('serving ranker feature order differs')
        loaded = {}
        for kind in ('gate', 'action'):
            path = root / self.fold[kind + '_file']
            if sha(path) != self.fold[kind + '_sha256']:
                raise ValueError('serving model changed: ' + kind)
            loaded[kind] = joblib.load(path)
        self.gate, self.model = loaded['gate'], loaded['action']
        # Reuse discovered libraries; constructing threadpool_limits on every
        # prediction repeatedly walks loaded shared libraries in the request.
        if not hasattr(backend, 'nested_prebge_threads'):
            backend.nested_prebge_threads = ThreadpoolController()
        self.threads = backend.nested_prebge_threads
        self.threshold, self.weight = self.fold['threshold'], self.fold['weight']
        self.identity = dict(serving_sha256=sha(root / 'study.json'), fold=fold,
                             gate_sha256=self.fold['gate_sha256'], action_sha256=self.fold['action_sha256'],
                             source_study_sha256=protocol['source_study_sha256'], backend=backend.identity,
                             gate_before_ce=True, one_cross_encoder_per_query=True,
                             label_assets_loaded=False, cached_query_predictions=False,
                             cached_query_vectors=False)

    def search(self, request):
        parsed = parser.parse(request.query)
        if request.mode != 'adaptive' or parsed.get('explicit_constraints', False):
            return super().search(request)
        if request.offset + request.k > 200:
            raise ApiError('research offset + k must not exceed 200')
        backend, torch = self.backend, self.backend.torch
        with backend.lock:
            started = time.perf_counter()
            timings = {}
            terms = [t.term for raw in parsed['terms'] for t in analyzer.analyze(raw)]
            tick = time.perf_counter()
            lexical = [(d, round(s, 6)) for d, s in backend.compiled.topk(terms, 200)]
            timings['lexical'] = (time.perf_counter() - tick) * 1000
            stats_future = backend.stats_executor.submit(
                extract_statistics_v2, request.query, self.reader, terms, lexical)
            tick = time.perf_counter()
            query = backend.native.encode([request.query], 64)[0]
            similarity = backend.matrix @ query['dense']
            dense = stable_dense_topk(similarity, 200)
            ids = [d for d, _ in dense]
            torch.cuda.synchronize()
            timings['native_encode_search'] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            stats = stats_future.result()
            gx = gain_features(stats, self.bm25_names, lexical, dense)
            with self.threads.limit(limits=1):
                gain = float(self.gate.predict(gx[None])[0])
            heavy = gain > self.threshold
            timings['pre_ce_gate'] = (time.perf_counter() - tick) * 1000
            pool = sorted(ids[:40 if heavy else 28])
            raw_future = None
            if heavy:
                raw_future = backend.feature_executor.submit(
                    candidate_matrix, request.query, backend.texts, pool, lexical, dense,
                    backend.normalized_texts, backend.feature_extension)
            tick = time.perf_counter()
            logits, _ = backend.reranker.score(request.query, [backend.texts[d] for d in pool])
            timings['rerank'] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            teacher = np.array(logits)
            normalized = zscore(teacher)
            prediction = normalized
            if heavy:
                ranks = np.empty(len(pool))
                ranks[np.argsort(-teacher, kind='stable')] = np.arange(1, len(pool) + 1)
                extra = np.column_stack((teacher, normalized, 1. / ranks, teacher.max() - teacher))
                support = bge_support(pool, teacher, [d for d, _ in lexical], ids, backend.support)
                bm25 = np.tile([stats[n] for n in self.bm25_names], (len(pool), 1))
                x = np.column_stack((raw_future.result(), extra, support, bm25)).astype('float32')
                with self.threads.limit(limits=1):
                    prob = np.clip(self.model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
                prediction = self.weight * normalized + (1 - self.weight) * zscore(np.log(prob / (1 - prob)))
            ranking = [pool[j] for j in sorted(range(len(pool)), key=lambda j: (-prediction[j], pool[j]))]
            rank_values = dict(zip(pool, prediction.tolist()))
            floor = float(prediction.min()) - 1
            for d in ids + sorted({d for d, _ in lexical}):
                if d not in rank_values:
                    rank_values[d] = floor - len(ranking) * 1e-3
                    ranking.append(d)
                if len(ranking) >= 200:
                    break
            timings['calibration'] = (time.perf_counter() - tick) * 1000
            tick = time.perf_counter()
            items = []
            for rank, d in enumerate(ranking[request.offset:request.offset + request.k], request.offset + 1):
                doc = self.reader.doc(d)
                items.append(ResultItem(doc.doc_id, rank, rank_values[d], doc.title, doc.text[:80],
                                        indexed_highlights(self.reader, doc, set(terms))).to_dict())
            timings['render'] = (time.perf_counter() - tick) * 1000
            return dict(research_api_version='nested-preBGE-1',
                        took_ms=(time.perf_counter() - started) * 1000, results=items,
                        parsed=parsed, expanded=[], timings_ms=timings,
                        routing=dict(requested_mode='adaptive', executed_mode='nested-preBGE',
                                     score_kind='learned-rank', vector_attempted=True, vector_used=True,
                                     fallback=False, candidate_k=200, policy_version='nested-preBGE-1'),
                        research_route=dict(action='heavy' if heavy else 'cheap', ce_max_length=self.ce_max_length,
                                            ce_pairs=len(pool), fold=self.fold['fold'], gate_before_ce=True,
                                            delegated=False, one_cross_encoder=True),
                        audit=dict(native_top20=[self.reader.doc(d).doc_id for d in ids[:20]],
                                   gate_features=gx.tolist(), predicted_gain=gain,
                                   teacher_logits={self.reader.doc(d).doc_id: v for d, v in zip(pool, logits)}))
