"""Nested grouped development study for a cheap new candidate-ranking action.

The model sees retrieval scores/ranks and text matching statistics, never query
or document IDs or labels as input. This is an offline feasibility study, not
an online Router deployment or independent confirmation result.
"""

import argparse
import json
import math
import time
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from contracts import analyzer
from .prepare_bm25_router import jsonl_rows, metrics, sha
from .train_bm25_router import group_bootstrap


FEATURES = (
    'lexical_rank_reciprocal', 'dense_rank_reciprocal',
    'lexical_rrf', 'dense_rrf', 'in_both', 'lexical_score_fraction',
    'dense_score', 'dense_top_gap', 'lexical_top_gap',
    'query_chars_log', 'query_terms_log', 'document_chars_log',
    'term_coverage', 'character_coverage', 'exact_query', 'prefix_term_coverage',
)
SPECS = (
    dict(max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=10.),
    dict(max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=10.),
)
EXTENDED_SPECS = SPECS + (
    dict(max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.),
    dict(max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.),
    dict(max_leaf_nodes=63, min_samples_leaf=40, l2_regularization=10.),
)
RANKING_SPECS = tuple(dict(kind='ranker', objective=objective, num_leaves=leaves,
                         min_child_samples=40, reg_lambda=10., n_estimators=100)
                      for objective in ('lambdarank', 'rank_xendcg') for leaves in (15,31))
RICH_RANKING_SPECS = tuple(dict(kind='ranker', objective=objective, num_leaves=leaves,
                         min_child_samples=child, reg_lambda=reg, n_estimators=estimators)
                      for objective in ('lambdarank', 'rank_xendcg')
                      for leaves,child,reg,estimators in ((31,20,1.,180),(63,20,1.,180),(63,40,10.,180)))
RECALL_SPECS = tuple(dict(kind='recall-ranker', recall_weight=recall_weight, num_leaves=15,
                         min_child_samples=40, reg_lambda=10., n_estimators=60)
                     for recall_weight in (.5, .9))
TEXT_FEATURES = ('raw_bm25_score','raw_bm25_top','raw_dense_top','lexical_log_rank',
    'dense_log_rank','query_trigram_coverage','query_fourgram_coverage','query_sixgram_coverage',
    'normalized_query_substring','mean_matching_term_count','max_matching_term_count',
    'earliest_matching_term_fraction','matching_term_span_fraction')


def normalize(text):
    return ''.join(char for char in unicodedata.normalize('NFKC',text).lower() if char.isalnum())


def text_features(query, text, lr, dr, ls, lexical_top, dense_top, *, query_terms=None,
                  normalized_document=None, normalized_query=None, query_chunks=None):
    terms = set(token.term for token in analyzer.analyze(query)) if query_terms is None else query_terms
    positions = [text.find(term) for term in terms if term in text]
    counts = [text.count(term) for term in terms]
    def coverage(length):
        chunks = ({query[i:i+length] for i in range(len(query)-length+1)}
                  if query_chunks is None else query_chunks[length])
        return sum(chunk in text for chunk in chunks)/len(chunks) if chunks else float(query in text)
    normalized_query=normalize(query) if normalized_query is None else normalized_query
    normalized_text=normalize(text) if normalized_document is None else normalized_document
    return [ls,lexical_top,dense_top,math.log1p(lr) if lr else math.log1p(1000),
        math.log1p(dr) if dr else math.log1p(1000), coverage(3),coverage(4),coverage(6),
        float(bool(normalized_query) and normalized_query in normalized_text),
        sum(counts)/max(len(terms),1),float(max(counts,default=0)),
        min(positions)/max(len(text),1) if positions else 1.,
        (max(positions)-min(positions))/max(len(text),1) if positions else 1.]


def features(query, text, lexical_rank, dense_rank, lexical_score, dense_score,
             lexical_top, dense_top, *, query_terms=None):
    terms = set(token.term for token in analyzer.analyze(query)) if query_terms is None else query_terms
    terms_in_text = sum(term in text for term in terms)
    chars = set(query)
    return [
        1./lexical_rank if lexical_rank else 0.,
        1./dense_rank if dense_rank else 0.,
        1./(60+lexical_rank) if lexical_rank else 0.,
        1./(60+dense_rank) if dense_rank else 0.,
        float(bool(lexical_rank and dense_rank)),
        lexical_score/max(abs(lexical_top), 1e-12) if lexical_rank else 0.,
        dense_score if dense_rank else -1.,
        dense_top-dense_score if dense_rank else 2.,
        1.-lexical_score/max(abs(lexical_top), 1e-12) if lexical_rank else 1.,
        math.log1p(len(query)), math.log1p(len(terms)), math.log1p(len(text)),
        terms_in_text/max(len(terms), 1),
        sum(char in text for char in chars)/max(len(chars), 1),
        float(query in text),
        sum(term in text[:100] for term in terms)/max(len(terms), 1),
    ]


def fit(spec, x, y, sizes, selected, offsets, relevance_totals=None):
    indices = np.concatenate([np.arange(offsets[i], offsets[i+1]) for i in selected])
    if spec.get('kind') in ('ranker', 'recall-ranker'):
        from lightgbm import LGBMRanker
        settings = {key:value for key,value in spec.items() if key!='kind'}
        weights = None
        if spec['kind'] == 'recall-ranker':
            from .recall_rank_loss import RecallNDCGLambda
            settings['objective'] = RecallNDCGLambda(settings.pop('recall_weight'))
            if relevance_totals is None:
                raise ValueError('recall objective requires training relevance totals')
            weights = np.concatenate([np.full(sizes[i], relevance_totals[i], dtype=float) for i in selected])
        model = LGBMRanker(**settings, learning_rate=.05, random_state=20261028,
            n_jobs=4, verbosity=-1, lambdarank_truncation_level=10, deterministic=True, force_col_wise=True)
        model.fit(x[indices],y[indices].astype(int),group=sizes[selected].tolist(),sample_weight=weights)
        return model
    # Each query has equal total weight, independent of its candidate count.
    weights = np.concatenate([np.full(sizes[i], 1./sizes[i]) for i in selected])
    weights *= len(weights)/weights.sum()
    model = HistGradientBoostingClassifier(**spec, max_iter=60, learning_rate=.1,
        early_stopping=False, random_state=20261028)
    model.fit(x[indices], y[indices], sample_weight=weights)
    return model


def score(model, x, selected, offsets, docs, relevance, query_ids, identities, rankings=None):
    result = {}
    for i in selected:
        xx = x[offsets[i]:offsets[i+1]]
        predicted = model.predict_proba(xx)[:, 1] if hasattr(model,'predict_proba') else model.predict(xx)
        ranked = [docs[i][j] for j in sorted(range(len(predicted)),
                  key=lambda j: (-predicted[j], identities[docs[i][j]]))]
        result[i] = metrics(ranked, relevance[query_ids[i]])
        if rankings is not None:
            rankings[i] = ranked[:100]
    return result


def objective(results):
    return float(np.mean([.5*(value['ndcg']+value['recall']) for value in results.values()]))


def study(table, capture, qrels, documents, out, teacher_logits=None, extended_grid=False, ranking_objective=False,
          extra_text_features=False, recall_objective=False, support_features=None, augmented_features=None,
          second_teacher_logits=None, late_features=None, export_dataset=False, prepare_only=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    source = json.loads(Path(table).with_name('provenance.json').read_text())
    if sha(documents) != source['source_manifest']['files']['docs.jsonl']:
        raise ValueError('document index identity mismatch')
    rows = list(jsonl_rows(table))
    by_id = {row['query_id']: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate query IDs')
    teacher = None
    feature_names = list(FEATURES)
    if teacher_logits is not None:
        teacher_logits = Path(teacher_logits)
        teacher = {row['query_id']: row for row in jsonl_rows(teacher_logits)}
        teacher_report = json.loads(teacher_logits.with_name('summary.json').read_text())
        if sha(teacher_logits) != teacher_report['logits_sha256']:
            raise ValueError('teacher logits checksum mismatch')
        if teacher_report['protocol']['kind'] != 'frozen-reranking-development-feasibility':
            raise ValueError('teacher must be a frozen, unsupervised scorer')
        if set(teacher) != set(by_id):
            raise ValueError('teacher does not cover the development queries')
        if teacher_report['protocol']['inputs'] != {str(p):sha(p) for p in (table,capture,qrels,documents)}:
            raise ValueError('teacher development input identity mismatch')
        feature_names.extend(('teacher_logit', 'teacher_rank_reciprocal', 'teacher_top_gap', 'teacher_scored'))
    second_teacher=None
    if second_teacher_logits is not None:
        if teacher is None:
            raise ValueError('secondary scorer requires a declared primary frozen scorer')
        second_teacher_logits=Path(second_teacher_logits)
        second_report=json.loads(second_teacher_logits.with_name('summary.json').read_text())
        if (sha(second_teacher_logits)!=second_report['logits_sha256'] or
                second_report['protocol']['kind']!='frozen-reranking-development-feasibility' or
                second_report['protocol']['inputs']!={str(p):sha(p) for p in (table,capture,qrels,documents)}):
            raise ValueError('secondary frozen scorer provenance mismatch')
        second_teacher={row['query_id']:row for row in jsonl_rows(second_teacher_logits)}
        if set(second_teacher)!=set(by_id):
            raise ValueError('secondary scorer query coverage mismatch')
        feature_names.extend(('secondary_logit','secondary_rank_reciprocal','secondary_top_gap','secondary_scored'))
    if extra_text_features:
        feature_names.extend(TEXT_FEATURES)
    support=None
    if support_features is not None:
        from .prepare_candidate_support import FEATURE_NAMES as SUPPORT_FEATURE_NAMES
        support_features=Path(support_features)
        support_protocol=json.loads(support_features.with_name('protocol.json').read_text())
        if (support_protocol.get('kind')!='query-local-candidate-support-v1' or
                support_protocol.get('labels_read') is not False or
                support_protocol.get('feature_file_sha256')!=sha(support_features) or
                support_protocol.get('feature_names')!=list(SUPPORT_FEATURE_NAMES)):
            raise ValueError('invalid label-free candidate support artifact')
        if teacher_logits is None or support_protocol['inputs']!={str(p):sha(p) for p in (table,capture,documents,teacher_logits)}:
            raise ValueError('candidate support input identity mismatch')
        support={row['query_id']:row['features'] for row in jsonl_rows(support_features)}
        if set(support)!=set(by_id):
            raise ValueError('candidate support/query coverage mismatch')
        feature_names.extend(SUPPORT_FEATURE_NAMES)
    augmented=None
    if augmented_features is not None:
        from .prepare_feedback_features import FEATURE_NAMES as AUGMENTED_FEATURE_NAMES
        augmented_features=Path(augmented_features)
        augmented_protocol=json.loads(augmented_features.with_name('protocol.json').read_text())
        if (augmented_protocol.get('kind')!='feedback-candidate-features-v1' or
                augmented_protocol.get('labels_read') is not False or
                augmented_protocol.get('feature_file_sha256')!=sha(augmented_features) or
                augmented_protocol.get('feature_names')!=list(AUGMENTED_FEATURE_NAMES)):
            raise ValueError('invalid augmented feature artifact')
        for path in (table,capture,documents,teacher_logits):
            matches=path is not None and augmented_protocol['inputs'].get(str(path))==sha(path)
            if not matches and path==teacher_logits and teacher_logits is not None:
                # A declared child scorer may fill scores for this exact candidate
                # artifact while preserving its original parent logits.
                matches=(teacher_report['protocol'].get('parent_logits_sha256') in augmented_protocol['inputs'].values()
                         and teacher_report['protocol'].get('augmented_candidate_file_sha256')==sha(augmented_features))
            if not matches:
                raise ValueError('augmented feature source identity mismatch')
        if support is not None:
            raise ValueError('support for original candidates cannot be silently reused on new candidates')
        augmented={row['query_id']:row['features'] for row in jsonl_rows(augmented_features)}
        if set(augmented)!=set(by_id):
            raise ValueError('augmented feature/query coverage mismatch')
        feature_names.extend(AUGMENTED_FEATURE_NAMES)
    late=None
    if late_features is not None:
        from .study_int8_late_interaction import FEATURE_NAMES as LATE_FEATURE_NAMES
        late_features=Path(late_features)
        late_protocol=json.loads(late_features.with_name('protocol.json').read_text())
        if (late_protocol.get('kind')!='m3-int8-late-interaction-v1' or
                late_protocol.get('qrels_read_for_features') is not False or
                late_protocol.get('feature_file_sha256')!=sha(late_features) or
                late_protocol.get('feature_names')!=list(LATE_FEATURE_NAMES)):
            raise ValueError('invalid frozen late-interaction feature artifact')
        for path in (table,capture,qrels,documents):
            if late_protocol['inputs'].get(str(path))!=sha(path):
                raise ValueError('late-interaction feature data identity mismatch')
        late={r['query_id']:r['features'] for r in jsonl_rows(late_features)}
        if set(late)!=set(by_id):
            raise ValueError('late-interaction query coverage mismatch')
        feature_names.extend(LATE_FEATURE_NAMES)
    # Only development labels are ever read.
    relevance = {}
    with Path(qrels).open(encoding='utf-8') as handle:
        for line in handle:
            query_id, _, doc_id, grade = line.split()
            if query_id in by_id:
                relevance.setdefault(query_id, {})[doc_id] = float(grade)
    if set(relevance) != set(by_id):
        raise ValueError('missing development labels')
    texts, identities = {}, {}
    for row in jsonl_rows(documents):
        texts[row['doc_id']] = row['title']+'\n'+row['text']
        identities[row['doc_id']] = row['docid']
    x, y, docs, query_ids, offsets = [], [], [], [], [0]
    seen = set()
    for row in jsonl_rows(capture):
        query_id = row['query_id']
        if query_id not in by_id:
            continue
        if query_id in seen:
            raise ValueError('duplicate captured query')
        seen.add(query_id)
        lexical = {hit['doc_id']: (rank, hit['score']) for rank, hit in enumerate(row['lexical'], 1)}
        dense = {hit['doc_id']: (rank, hit['score']) for rank, hit in enumerate(row['dense'], 1)}
        candidates = sorted(set(lexical) | set(dense), key=identities.__getitem__)
        if augmented is not None:
            if not set(candidates)<=set(augmented[query_id]):
                raise ValueError('augmentation dropped original candidates')
            candidates=sorted(augmented[query_id],key=identities.__getitem__)
        lexical_top = row['lexical'][0]['score'] if row['lexical'] else 0.
        dense_top = row['dense'][0]['score'] if row['dense'] else 0.
        teacher_scores = teacher[query_id]['logits'] if teacher is not None else {}
        teacher_ranks = {d:i for i,d in enumerate(sorted(teacher_scores, key=lambda d:(-teacher_scores[d], identities[d])),1)}
        teacher_top = max(teacher_scores.values()) if teacher_scores else 0.
        secondary_scores=second_teacher[query_id]['logits'] if second_teacher is not None else {}
        secondary_ranks={d:i for i,d in enumerate(sorted(secondary_scores,key=lambda d:(-secondary_scores[d],identities[d])),1)}
        secondary_top=max(secondary_scores.values()) if secondary_scores else 0.
        if support is not None and set(support[query_id])!=set(candidates):
            raise ValueError('candidate support/document identity mismatch')
        if late is not None and set(late[query_id])!=set(candidates):
            raise ValueError('late-interaction candidate identity mismatch')
        for doc_id in candidates:
            lr, ls = lexical.get(doc_id, (0, 0.))
            dr, ds = dense.get(doc_id, (0, -1.))
            candidate_features = features(row['text'], texts[doc_id], lr, dr, ls, ds, lexical_top, dense_top)
            if teacher is not None:
                candidate_features.extend((teacher_scores.get(doc_id, -30.),
                    1./teacher_ranks[doc_id] if doc_id in teacher_ranks else 0.,
                    teacher_top-teacher_scores.get(doc_id, -30.), float(doc_id in teacher_scores)))
            if second_teacher is not None:
                candidate_features.extend((secondary_scores.get(doc_id,-30.),
                    1./secondary_ranks[doc_id] if doc_id in secondary_ranks else 0.,
                    secondary_top-secondary_scores.get(doc_id,-30.),float(doc_id in secondary_scores)))
            if extra_text_features:
                candidate_features.extend(text_features(row['text'],texts[doc_id],lr,dr,ls,lexical_top,dense_top))
            if support is not None:
                if len(support[query_id][doc_id])!=len(SUPPORT_FEATURE_NAMES):
                    raise ValueError('candidate support feature dimensions mismatch')
                candidate_features.extend(support[query_id][doc_id])
            if augmented is not None:
                values=augmented[query_id][doc_id]
                if len(values)!=len(AUGMENTED_FEATURE_NAMES):
                    raise ValueError('augmented feature dimensions mismatch')
                candidate_features.extend(values)
            if late is not None:
                values=late[query_id][doc_id]
                if len(values)!=len(LATE_FEATURE_NAMES):
                    raise ValueError('late-interaction feature dimensions mismatch')
                candidate_features.extend(values)
            x.append(candidate_features)
            y.append(relevance[query_id].get(doc_id, 0)>0)
        docs.append(candidates)
        query_ids.append(query_id)
        offsets.append(len(x))
        if len(query_ids)%100 == 0:
            print('features', len(query_ids), flush=True)
    if seen != set(by_id):
        raise ValueError('capture does not cover development queries')
    del texts
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=bool)
    if not np.isfinite(x).all() or x.shape[1] != len(feature_names):
        raise ValueError('invalid feature matrix')
    sizes = np.diff(offsets)
    groups = np.array([by_id[q]['group'] for q in query_ids])
    relevance_totals = np.array([sum(grade > 0 for grade in relevance[q].values()) for q in query_ids])
    specs = RECALL_SPECS if recall_objective else RANKING_SPECS if ranking_objective else EXTENDED_SPECS if extended_grid else SPECS
    protocol = dict(kind='candidate-ranking-feasibility', features=feature_names,
        specs=specs, max_iter=60, seed=20261028, outer_folds=5, inner_folds=3,
        inputs={str(p): sha(p) for p in (table, capture, qrels, documents)},
        queries=len(query_ids), candidates=len(x), selection='mean of query nDCG@10 and Recall@10 on inner folds',
        scope='supervised development study; no online routing or independent confirmation claim',
        confirmation_read=False)
    if teacher_logits is not None:
        protocol['inputs'][str(teacher_logits)] = sha(teacher_logits)
        protocol['teacher_scoring_cost_excluded_from_offline_training'] = True
        protocol['teacher'] = teacher_report['protocol']['encoder']
    if second_teacher_logits is not None:
        protocol['inputs'][str(second_teacher_logits)]=sha(second_teacher_logits)
        protocol['secondary_teacher']=second_report['protocol']['encoder']
        protocol['secondary_scoring_cost_must_be_included_in_online_evaluation']=True
    if support_features is not None:
        protocol['inputs'][str(support_features)]=sha(support_features)
    if augmented_features is not None:
        protocol['inputs'][str(augmented_features)]=sha(augmented_features)
        protocol['augmentation']=augmented_protocol
    if late_features is not None:
        protocol['inputs'][str(late_features)]=sha(late_features)
        protocol['late_encoder']=late_protocol['encoder']
        protocol['late_feature_cache_cost_excluded_from_offline_training']=True
    if ranking_objective or recall_objective:
        import lightgbm
        protocol['lightgbm_version'] = lightgbm.__version__
        protocol['max_iter'] = 60 if recall_objective else 100
    if recall_objective:
        protocol['recall_loss'] = 'query-local sigmoid pairs weighted by a mixture of exact nDCG@10 and Recall@10 swap deltas'
        protocol['label_normalizer_not_feature'] = 'training query total known positives; never included in the input feature matrix'
    protocol['code_sha256'] = {str(p):sha(p) for p in (Path(__file__), Path(__file__).with_name('recall_rank_loss.py'))}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    if export_dataset or prepare_only:
        from .ranking_dataset import write_dataset
        write_dataset(out,x,y,offsets,docs,query_ids,groups,relevance,identities,feature_names,by_id)
    if prepare_only:
        print(json.dumps(dict(prepared_queries=len(query_ids),candidates=len(x),dataset=str(out/'dataset.json'))),flush=True)
        return dict(prepared=True,goal_complete=False)
    outer, folds, rankings = {}, [], {}
    for fold, (training, validation) in enumerate(GroupKFold(5).split(query_ids, groups=groups)):
        if set(groups[training]) & set(groups[validation]):
            raise ValueError('outer group leakage')
        trials = []
        for spec in specs:
            inner = {}
            for train_relative, valid_relative in GroupKFold(3).split(training, groups=groups[training]):
                train, valid = training[train_relative], training[valid_relative]
                model = fit(spec, x, y, sizes, train, offsets, relevance_totals)
                inner.update(score(model, x, valid, offsets, docs, relevance, query_ids, identities))
            trials.append(dict(spec=spec, inner_objective=objective(inner)))
        selected = max(trials, key=lambda trial: trial['inner_objective'])['spec']
        model = fit(selected, x, y, sizes, training, offsets, relevance_totals)
        heldout = score(model, x, validation, offsets, docs, relevance, query_ids, identities, rankings)
        outer.update(heldout)
        folds.append(dict(fold=fold, trials=trials, selected=selected,
                          heldout_ids=[query_ids[i] for i in validation], training_ids=[query_ids[i] for i in training]))
        if hasattr(model,'booster_'):
            filename = f'fold-{fold}-model.txt'
            model.booster_.save_model(str(out/filename))
            folds[-1].update(model_file=filename,model_sha256=sha(out/filename))
        print('outer', fold, 'heldout objective', objective(heldout), 'selected', selected, flush=True)
    summary = {}
    for metric in ('ndcg', 'recall'):
        values = np.array([outer[i][metric] for i in range(len(query_ids))])
        baseline = np.array([by_id[q]['hybrid'][metric] for q in query_ids])
        summary[metric] = dict(value=float(values.mean()), baseline=float(baseline.mean()),
            relative_lift=float(values.mean()/baseline.mean()-1),
            delta_ci=group_bootstrap(values-baseline, groups),
            passes_20_percent=bool(values.mean()/baseline.mean()>1.2))
    result = dict(protocol=protocol, outer=summary, folds=folds,
        quality_gate_passed=all(v['passes_20_percent'] and v['delta_ci']['ci95'][0]>0 for v in summary.values()),
        http_evaluated=False, goal_complete=False, elapsed_seconds=time.perf_counter()-start)
    (out/'study.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (out/'outer-predictions.jsonl').write_text(''.join(json.dumps(dict(query_id=query_ids[i], group=str(groups[i]), metrics=outer[i],ranked=rankings[i]))+'\n' for i in range(len(query_ids))))
    print(json.dumps({k:v for k,v in result.items() if k not in ('folds','protocol')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('table', 'capture', 'qrels', 'documents', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--teacher-logits',type=Path)
    parser.add_argument('--extended-grid',action='store_true')
    parser.add_argument('--ranking-objective',action='store_true')
    parser.add_argument('--extra-text-features',action='store_true')
    parser.add_argument('--recall-objective',action='store_true')
    parser.add_argument('--support-features',type=Path)
    parser.add_argument('--augmented-features',type=Path)
    parser.add_argument('--second-teacher-logits',type=Path)
    parser.add_argument('--late-features',type=Path)
    parser.add_argument('--export-dataset',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    study(**vars(parser.parse_args()))
