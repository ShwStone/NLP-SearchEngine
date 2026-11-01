"""Explicit research runtime for frozen native-first length routing.

The production HTTP v2 contract is not changed. Research responses identify
learned scores and the selected CE length; constraints/forced modes delegate to
the normal engine. No query IDs, cached predictions or qrels are runtime inputs.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

import numpy as np

from contracts import analyzer
from contracts.api import ApiError,ResultItem
from online import parser
from online.compiled_bm25 import CompiledBM25
from online.highlight import build_highlights,indexed_highlights
from online.router_features import extract_statistics_v2
from .prepare_bm25_router import sha
from .prepare_candidate_support import FEATURE_NAMES as SUPPORT_NAMES,support_features,support_features_fast
from .study_candidate_ranking import FEATURES,TEXT_FEATURES,features,text_features,normalize
from .study_nested_quality_router import action_confidence,replay_gate


def candidate_matrix(query,texts,candidates,lexical,dense,normalized_documents=None,native_extension=None):
    """29 original raw features in deterministic docid order; no labels."""
    lr={d:(i,s) for i,(d,s) in enumerate(lexical,1)}
    dr={d:(i,s) for i,(d,s) in enumerate(dense,1)}
    lt=lexical[0][1] if lexical else 0.;dt=dense[0][1] if dense else 0.
    terms={token.term for token in analyzer.analyze(query)}
    normalized_query=normalize(query)
    chunks={n:{query[i:i+n] for i in range(len(query)-n+1)} for n in (3,4,6)}
    if native_extension is not None:
        if normalized_documents is None:raise ValueError('native candidate features need the checked document normalization cache')
        retrieval=np.array([[*lr.get(d,(0,0.)),*dr.get(d,(0,-1.))] for d in candidates],dtype=np.float64).reshape(-1,4)
        # Native ABI uses lexical rank, dense rank, lexical score, dense score.
        retrieval=np.ascontiguousarray(retrieval[:,[0,2,1,3]])
        output=np.empty((len(candidates),29),dtype=np.float32)
        native_extension.fill(query,texts,normalized_documents,list(candidates),retrieval,list(terms),list(set(query)),
            [list(chunks[n]) for n in (3,4,6)],normalized_query,output,lt,dt)
        return output
    matrix=[]
    for d in candidates:
        rank_l,ls=lr.get(d,(0,0.));rank_d,ds=dr.get(d,(0,-1.))
        matrix.append(features(query,texts[d],rank_l,rank_d,ls,ds,lt,dt,query_terms=terms)+
            text_features(query,texts[d],rank_l,rank_d,ls,lt,dt,query_terms=terms,
                normalized_document=normalized_documents[d] if normalized_documents is not None else None,
                normalized_query=normalized_query,query_chunks=chunks))
    return np.array(matrix,dtype=np.float32)


def ranking_matrix(base,candidates,scores,lexical,dense,document_vectors,*,fast_support=False):
    teacher=sorted(scores,key=lambda d:(-scores[d],d))
    ranks={d:i for i,d in enumerate(teacher,1)};top=scores[teacher[0]]
    extra=np.array([[scores.get(d,-30.),1./ranks[d] if d in ranks else 0.,
                    top-scores.get(d,-30.),float(d in scores)] for d in candidates],dtype=np.float32)
    anchors=list(enumerate(teacher[:5]));teacher_count=len(anchors)
    dense_slot=lexical_slot=None
    if dense:dense_slot=len(anchors);anchors.append((0,dense[0][0]))
    if lexical:lexical_slot=len(anchors);anchors.append((0,lexical[0][0]))
    similarity=document_vectors[candidates]@document_vectors[[d for _,d in anchors]].T
    implementation=support_features_fast if fast_support else support_features
    support=implementation(similarity,candidates,anchors,teacher_count,dense_slot=dense_slot,lexical_slot=lexical_slot)
    return np.column_stack((base[:,:len(FEATURES)],extra,base[:,len(FEATURES):],support)).astype(np.float32)


class NativeRouterBackend:
    """One resident GPU backend shared by research engines; serial requests."""

    def __init__(self,reader,native_model_path,reranker_path,vectors_dir,support_vectors,*,scorer_batch_size=16,
                 query_graph=False,scorer_graph=False,feature_extension=None,fast_support=False):
        import torch
        from .study_m3_late_interaction import FrozenM3
        from .study_rerank_actions import FrozenCrossEncoder
        self.reader=reader;self.lock=threading.Lock();self.torch=torch
        self.stats_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='router-stats')
        self.feature_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='router-features')
        self.native=FrozenM3(native_model_path,'cuda',include_projections=False)
        if query_graph:
            from .cuda_graph_native_query import NativeQueryGraph
            self.native=NativeQueryGraph(self.native);self.native.warm()
        if scorer_graph:
            from .cuda_graph_cross_encoder import CUDAGraphCrossEncoder
            self.reranker=CUDAGraphCrossEncoder(reranker_path,batch_size=scorer_batch_size,max_length=256)
            self.reranker.warm()
        else:self.reranker=FrozenCrossEncoder(reranker_path,device='cuda',batch_size=scorer_batch_size,max_length=256,batch_strategy='length')
        self.feature_extension=None;extension_identity=None;self.fast_support=fast_support
        if feature_extension is not None:
            from .build_native_features import load
            self.feature_extension,extension_identity=load(feature_extension)
        self.compiled=CompiledBM25(reader)
        self.texts=[reader.doc(i).title+'\n'+reader.doc(i).text for i in range(reader.n_docs)]
        self.normalized_texts=[normalize(text) for text in self.texts]
        vectors_dir=Path(vectors_dir);support_vectors=Path(support_vectors)
        native_report=json.loads((vectors_dir/'summary.json').read_text())
        native_protocol=native_report['protocol']
        if native_protocol['encoder']!=self.native.info or native_protocol['document_count']!=reader.n_docs:
            raise ValueError('native runtime encoder/index identity mismatch')
        native_docs=next(Path(p) for p in native_protocol['inputs'] if Path(p).name=='docs.jsonl')
        if native_protocol['inputs'][str(native_docs)]!=reader.manifest()['files']['docs.jsonl']:
            raise ValueError('native runtime document index mismatch')
        if sha(vectors_dir/'vectors.npy')!=native_report['files']['vectors.npy']:
            raise ValueError('native runtime vector checksum mismatch')
        support_report=json.loads(support_vectors.with_name('protocol.json').read_text())
        if (support_report['vector_sha256']!=sha(support_vectors) or
                support_report['inputs'].get(str(native_docs))!=reader.manifest()['files']['docs.jsonl']):
            raise ValueError('runtime support vector identity mismatch')
        matrix=np.load(vectors_dir/'vectors.npy',mmap_mode='r',allow_pickle=False)
        self.support=np.load(support_vectors,mmap_mode='r',allow_pickle=False)
        if matrix.shape!=(reader.n_docs,1024) or self.support.shape!=(reader.n_docs,512):
            raise ValueError('runtime vector matrix dimensions mismatch')
        torch.backends.cuda.matmul.allow_tf32=False
        self.matrix=torch.tensor(np.array(matrix),device='cuda')
        self.identity=dict(native_encoder=self.native.info,reranker=self.reranker.info,
            document_matrix_sha256=native_report['files']['vectors.npy'],support_matrix_sha256=support_report['vector_sha256'],
            native_search='exact float32 full corpus, TF32 disabled',query_batch_size=1)
        self.identity['scorer_batch_size_changed_from_training']=scorer_batch_size!=16
        self.identity['document_text_normalization_cache']='query independent normalized document strings only'
        self.identity['highlights']='immutable posting positions and persisted offsets, equivalent to shared analyzer'
        self.identity.update(native_query_runtime=getattr(self.native,'runtime_info',dict(kind='original eager native')),
            feature_extension=extension_identity,fast_support=fast_support)


class NativeLengthRouter:
    def __init__(self,base,backend,study_dir,fold):
        import lightgbm
        self.base=base;self.backend=backend;self.reader=base.reader
        self.study_dir=Path(study_dir)
        study_file=self.study_dir/('bundle.json' if fold=='final' else 'study.json')
        study=json.loads(study_file.read_text())
        self.fold=next(row for row in study['folds'] if row['fold']==fold)
        if study['protocol'].get('gate_features_dataset') is None:
            raise ValueError('length runtime requires CE-free native proxy gate features')
        gate_file=self.study_dir/self.fold['gate_file']
        if sha(gate_file)!=self.fold['gate_sha256']:raise ValueError('runtime gate checksum mismatch')
        self.gate=json.loads(gate_file.read_text())
        self.models={};self.lengths={};self.specs={};self.feature_names={}
        for action in ('cheap','heavy'):
            selection=self.fold[action+'_selection'];family=selection['family']
            source=Path(study['protocol'][action+'_families'][family])
            if sha(source/'dataset.json')!=study['protocol']['inputs'][str(source/'dataset.json')]:
                raise ValueError('runtime action dataset identity changed')
            protocol=json.loads((source/'protocol.json').read_text())
            names=protocol['features'];self.feature_names[action]=names
            expected_names=list(FEATURES)+['teacher_logit','teacher_rank_reciprocal','teacher_top_gap','teacher_scored']+list(TEXT_FEATURES)+list(SUPPORT_NAMES)
            if names!=expected_names:
                raise ValueError('unsupported native length action feature schema')
            encoder=protocol['teacher']
            if any(encoder.get(k)!=backend.reranker.info.get(k) for k in encoder if k not in ('max_length','batch_size')):
                raise ValueError('runtime action frozen scorer configuration mismatch')
            self.lengths[action]=encoder['max_length'];self.specs[action]=selection['spec']
            model=self.fold[action+'_model']
            if 'model_file' in model:
                path=self.study_dir/model['model_file']
                if sha(path)!=model['model_sha256']:raise ValueError('runtime ranker checksum mismatch')
                self.models[action]=lightgbm.Booster(model_file=str(path))
            else:self.models[action]=None
        proxy=Path(study['protocol']['gate_features_dataset'])
        proxy_protocol=json.loads((proxy/'protocol.json').read_text())
        if proxy_protocol['features']!=list(FEATURES)+list(TEXT_FEATURES):
            raise ValueError('unsupported native proxy feature schema')
        self.proxy_names=proxy_protocol['features']
        # The sorted BM25 order is part of the persisted gate feature schema.
        self.bm25_names=sorted(extract_statistics_v2('x',self.reader,[],[]))
        self.identity=dict(study_sha256=sha(study_file),fold=fold,
            gate_sha256=self.fold['gate_sha256'],lengths=self.lengths,research_only=True,backend=backend.identity)

    def search(self,request):
        parsed=parser.parse(request.query)
        if request.mode!='adaptive' or parsed.get('explicit_constraints',False):
            from dataclasses import replace
            delegated=self.base.search(replace(request,mode='lexical') if parsed.get('explicit_constraints') and request.mode=='adaptive' else request)
            result=delegated.to_dict();result['research_route']=dict(delegated=True)
            return result
        if request.offset+request.k>200:raise ApiError('research offset + k must not exceed 200')
        backend=self.backend;torch=backend.torch
        with backend.lock:
            started=time.perf_counter();timings={}
            terms=[t.term for raw in parsed['terms'] for t in analyzer.analyze(raw)]
            tick=time.perf_counter();lexical=backend.compiled.topk(terms,200)
            # The frozen development capture stores public BM25 scores at six
            # decimals. Preserve that feature representation, while retaining
            # the exact reference scorer's document order.
            lexical=[(d,round(score,6)) for d,score in lexical]
            timings['lexical']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();torch.cuda.synchronize()
            query=backend.native.encode([request.query],64)[0]
            similarity=backend.matrix@query['dense']
            order=torch.argsort(similarity,descending=True,stable=True)[:200]
            values=similarity[order].cpu().tolist();ids=order.cpu().tolist()
            dense=list(zip(ids,values));torch.cuda.synchronize()
            timings['native_encode_search']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();candidates=sorted({d for d,_ in lexical}|set(ids))
            base=candidate_matrix(request.query,backend.texts,candidates,lexical,dense,backend.normalized_texts,backend.feature_extension)
            stats=extract_statistics_v2(request.query,self.reader,terms,lexical)
            dr_index=self.proxy_names.index('dense_rank_reciprocal')
            score_index=self.proxy_names.index('dense_score')
            proxy_scores=np.where(base[:,dr_index]>0,base[:,score_index],-np.inf)
            proxy_order=sorted(np.flatnonzero(np.isfinite(proxy_scores)),key=lambda i:(-proxy_scores[i],candidates[i]))
            confidence=action_confidence(proxy_scores,base,self.proxy_names,proxy_order)
            gx=np.array([[stats[name] for name in self.bm25_names]+confidence],dtype=float)
            if getattr(self, 'hash_gate', None) is not None:
                import hashlib
                digest=int(hashlib.sha256((self.hash_gate['salt']+request.query).encode()).hexdigest(),16)/float(1<<256)
                heavy=bool(digest >= float(self.hash_gate['cheap_rate']))
            elif self.gate['model'] is None:heavy=self.gate['selection']['constant']
            else:heavy=float(replay_gate(self.gate['model'],gx).mean())>self.gate['selection']['threshold']
            action='heavy' if heavy else 'cheap'
            timings['raw_features_gate']=(time.perf_counter()-tick)*1000
            # Only the selected CE action runs, after the pre-CE gate decision.
            tick=time.perf_counter();pool=sorted(ids[:20]);backend.reranker.max_length=self.lengths[action]
            logits,_=backend.reranker.score(request.query,[backend.texts[d] for d in pool])
            scores=dict(zip(pool,logits))
            timings['rerank']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter()
            if self.models[action] is None:
                if self.specs[action]['kind']!='fixed-teacher':raise ValueError('unsupported fixed research action')
                ranking=sorted(scores,key=lambda d:(-scores[d],d));rank_values={d:scores[d] for d in ranking}
            else:
                x=ranking_matrix(base,candidates,scores,lexical,dense,backend.support,fast_support=backend.fast_support)
                predicted=self.models[action].predict(x,num_threads=1)
                ranked=sorted(range(len(predicted)),key=lambda i:(-predicted[i],candidates[i]))[:200]
                ranking=[candidates[i] for i in ranked];rank_values={candidates[i]:float(predicted[i]) for i in ranked}
            timings['support_calibrate']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();items=[]
            for rank,d in enumerate(ranking[request.offset:request.offset+request.k],request.offset+1):
                doc=self.reader.doc(d)
                items.append(ResultItem(doc.doc_id,rank,rank_values[d],doc.title,doc.text[:80],indexed_highlights(self.reader,doc,set(terms))).to_dict())
            timings['render']=(time.perf_counter()-tick)*1000
            return dict(research_api_version='native-length-router-1',api_contract='experimental learned scores; separate from production v2',
                took_ms=(time.perf_counter()-started)*1000,results=items,parsed=parsed,expanded=[],timings_ms=timings,
                routing=dict(requested_mode='adaptive',executed_mode='native-ce-ranker',score_kind='learned-rank',
                    vector_attempted=True,vector_used=True,fallback=False,candidate_k=200,policy_version='research-native-length-router-1'),
                research_route=dict(action=action,ce_max_length=self.lengths[action],ce_pairs=len(pool),fold=self.fold['fold'],
                    gate_before_ce=True,delegated=False),
                audit=dict(native_top20=[self.reader.doc(d).doc_id for d in ids[:20]],
                    teacher_logits={self.reader.doc(d).doc_id:value for d,value in scores.items()}))
