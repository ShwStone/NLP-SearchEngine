"""Research-only BGE score calibration with live retrieval and no query cache."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from contracts import analyzer
from contracts.api import ApiError, ResultItem
from online import parser
from online.highlight import indexed_highlights
from online.router_features import extract_statistics_v2
from .prepare_bm25_router import sha
from .runtime_native_router import candidate_matrix
from .study_candidate_ranking import FEATURES, TEXT_FEATURES


def zscore(values):
    values=np.asarray(values,dtype=float)
    return (values-values.mean())/max(float(values.std()),1e-8)


class BGECalibrationRouter:
    def __init__(self,base,backend,study_dir,fold):
        self.base=base;self.reader=base.reader;self.backend=backend
        root=Path(study_dir);study=json.loads((root/'study.json').read_text())
        protocol=study['protocol'];self.fold=next(r for r in study['folds'] if r['fold']==fold)
        if protocol.get('scored_subset_proxy'):
            raise ValueError('proxy pool scores require fresh scoring before serving')
        expected=protocol['online_scorer']
        if expected!=backend.reranker.info:
            raise ValueError('BGE runtime/scored action identity mismatch')
        self.selection=self.fold['selection']
        path=root/self.fold['model_file']
        if sha(path)!=self.fold['model_sha256']:raise ValueError('calibration model checksum differs')
        self.model=joblib.load(path)
        raw_names=list(FEATURES)+list(TEXT_FEATURES)
        self.support_names=protocol.get('bge_support_features',[])
        if self.support_names:
            from .bge_candidate_support import FEATURE_NAMES
            if self.support_names!=list(FEATURE_NAMES) or backend.identity['support_matrix_sha256'] not in protocol['inputs'].values():
                raise ValueError('BGE support feature/matrix identity differs')
        self.bm25_names=protocol['features'][len(raw_names)+4+len(self.support_names):]
        self.identity=dict(study_sha256=sha(root/'study.json'),fold=fold,
            model_sha256=sha(path),selection=self.selection,backend=backend.identity,
            gate_before_ce=False,one_cross_encoder_per_query=True,
            cached_query_vectors=False,cached_query_predictions=False,
            label_assets_loaded=False)

    def search(self,request):
        parsed=parser.parse(request.query)
        if request.mode!='adaptive' or parsed.get('explicit_constraints',False):
            from dataclasses import replace
            response=self.base.search(replace(request,mode='lexical') if parsed.get('explicit_constraints') and request.mode=='adaptive' else request)
            result=response.to_dict();result['research_route']=dict(delegated=True);return result
        if request.offset+request.k>200:raise ApiError('research offset + k must not exceed 200')
        backend=self.backend;torch=backend.torch
        with backend.lock:
            started=time.perf_counter();timings={}
            terms=[t.term for raw in parsed['terms'] for t in analyzer.analyze(raw)]
            tick=time.perf_counter()
            lexical=[(d,round(s,6)) for d,s in backend.compiled.topk(terms,200)]
            timings['lexical']=(time.perf_counter()-tick)*1000
            stats_future=backend.stats_executor.submit(extract_statistics_v2,request.query,backend.reader,terms,lexical)
            tick=time.perf_counter();torch.cuda.synchronize()
            query=backend.native.encode([request.query],64)[0]
            similarity=backend.matrix@query['dense']
            order=torch.argsort(similarity,descending=True,stable=True)[:200]
            ids=order.cpu().tolist();values=similarity[order].cpu().tolist();dense=list(zip(ids,values))
            torch.cuda.synchronize();timings['native_encode_search']=(time.perf_counter()-tick)*1000
            pool=sorted(ids[:40])
            raw_future=backend.feature_executor.submit(candidate_matrix,request.query,backend.texts,pool,lexical,dense,
                backend.normalized_texts,backend.feature_extension)
            tick=time.perf_counter();pool=sorted(ids[:40])
            logits,_=backend.reranker.score(request.query,[backend.texts[d] for d in pool])
            timings['rerank']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();teacher=np.array(logits);normalized=zscore(teacher)
            stats=stats_future.result()
            zsorted=np.sort(normalized)[::-1]
            gate_values=[zsorted[0]-zsorted[1],zsorted[9]-zsorted[10],stats['query_chars'],stats['relative_gap']]
            policy=self.selection.get('policy',{'kind':'constant'})
            heavy=policy['kind']=='constant' or policy['direction']*gate_values[policy['feature']]>policy['direction']*policy['threshold']
            predicted=normalized
            if heavy:
                raw=raw_future.result()
                ranked=sorted(range(40),key=lambda j:(-teacher[j],pool[j]))
                ranks=np.empty(40);ranks[ranked]=np.arange(1,41)
                extra=np.column_stack((teacher,normalized,1./ranks,teacher.max()-teacher))
                bm25=np.tile([stats[name] for name in self.bm25_names],(40,1))
                parts=[raw,extra]
                if self.support_names:
                    from .bge_candidate_support import bge_support
                    parts.append(bge_support(pool,teacher,[d for d,_ in lexical],ids,backend.support))
                parts.append(bm25)
                x=np.column_stack(parts).astype('float32')
                with threadpool_limits(limits=1):
                    probability=np.clip(self.model.predict_proba(x)[:,1],1e-7,1-1e-7)
                learned=zscore(np.log(probability/(1-probability)))
                weight=self.selection['weight'];predicted=weight*normalized+(1-weight)*learned
            ranking=[pool[j] for j in sorted(range(40),key=lambda j:(-predicted[j],pool[j]))]
            rank_values=dict(zip(pool,predicted.tolist()))
            # Keep the requested result count; unseen candidates follow scored
            # candidates so the audited top10 and top40 remain unchanged.
            floor=float(predicted.min())-1
            remaining=sorted(set(ids)|{d for d,_ in lexical})
            for d in ids+remaining:
                if d not in rank_values:
                    rank_values[d]=floor-len(ranking)*1e-3;ranking.append(d)
                if len(ranking)>=200:break
            timings['calibration_gate']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();items=[]
            for rank,d in enumerate(ranking[request.offset:request.offset+request.k],request.offset+1):
                doc=self.reader.doc(d)
                items.append(ResultItem(doc.doc_id,rank,rank_values[d],doc.title,doc.text[:80],
                    indexed_highlights(self.reader,doc,set(terms))).to_dict())
            timings['render']=(time.perf_counter()-tick)*1000
            return dict(research_api_version='BGE-calibration-1',took_ms=(time.perf_counter()-started)*1000,
                results=items,parsed=parsed,expanded=[],timings_ms=timings,
                routing=dict(requested_mode='adaptive',executed_mode='BGE-calibration',score_kind='learned-rank',
                    vector_attempted=True,vector_used=True,fallback=False,candidate_k=200,
                    policy_version='research-BGE-calibration-1'),
                research_route=dict(action='heavy' if heavy else 'cheap',ce_max_length=256,ce_pairs=40,
                    fold=self.fold['fold'],gate_before_ce=False,delegated=False,one_cross_encoder=True),
                audit=dict(native_top20=[self.reader.doc(d).doc_id for d in ids[:20]],
                    teacher_logits={self.reader.doc(d).doc_id:value for d,value in zip(pool,logits)}))
