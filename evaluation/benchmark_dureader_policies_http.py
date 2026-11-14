"""Real retrieval plus CE and frozen learned policy over loopback HTTP.

No candidate/result replay. Each request executes actual lexical/dense retrieval
as appropriate. Client wall time includes serialization and response parsing.
Run in isolation for interpretable latency; background load is disclosed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

import numpy as np

from contracts import analyzer
from contracts.api import SearchRequest
from contracts.retrieval import RetrievalConfig,VectorManifest
from evaluation.dureader_policy_features import FrozenLinear,gate_features,rerank
from evaluation.prepare_bm25_router import jsonl_rows,metrics
from evaluation.router_metrics import percentile
from evaluation.study_rerank_actions import FrozenCrossEncoder
from contracts.index_format import sha256_of
from offline.disk_index import DiskIndex
from online.engine import SearchEngine
from online.fusion import rrf
from online.parser import parse
from online.router import QueryFeatures,RuleRouter
from online.vector_index import QdrantVectorIndex
from online.scorers import boolean_docs
from evaluation.prece_router import FrozenTreeEnsemble, FrozenScalarTree, extract_features


def quality_guard_heavy(score, threshold, explicit_constraints=False):
    """Return the frozen cheap-feature action without inspecting CE output."""
    return bool(not explicit_constraints and float(score) > float(threshold))


def policy_ce_budget(policy, fixed_budget, heavy):
    if policy in ('same-cost-prece-router','same-cost-calibrated-router'): return 40 if heavy else 20
    if policy=='same-cost-ce28-router': return 40 if heavy else 28
    if policy=='same-cost-ce24-router': return 40 if heavy else 24
    if policy=='ce20': return 20
    if policy=='ce24': return 24
    if policy.startswith('ce28'): return 28
    if policy.startswith('ce40') or heavy: return 40
    if policy in ('fixed-pair-budget', 'quality-guard-40'): return fixed_budget
    return 0


def benchmark(index,vector_manifest,model_path,ce_model,url,queries,qrels,
              policy_dir,output,policies='bm25,dense,ce0,old-rules-router,ce28,ce40,ce28-ranker,ce40-ranker,learned-router,fixed-pair-budget',
              limit=0,reader_source=None,quality_threshold=0.153,prece_model=None,scalar_model=None):
    from adapters.embedding import BGEEncoder
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    runtime_sources={str(Path(__file__).name):sha256_of(__file__)}
    for path in ['evaluation/dureader_policy_features.py','online/vector_index.py','online/scorers.py']:
        runtime_sources[path]=sha256_of(Path(__file__).resolve().parents[1]/path)
    bundle=json.loads((Path(policy_dir)/'models.json').read_text())
    source_predictions={r['query_id']:r for r in jsonl_rows(Path(policy_dir)/'dev-predictions.jsonl')}
    rows=list(jsonl_rows(queries))
    eligible=[r.get('query_id',r.get('qid')) for r in rows
              if not parse(r.get('query',r.get('text'))).get('explicit_constraints',False)]
    heavy_count=sum(r['heavy'] for r in source_predictions.values())
    random_heavy={f'random-{seed}':set(sorted(eligible,key=lambda q:
                  hashlib.sha256(f'{seed}:{q}'.encode()).digest())[:heavy_count])
                  for seed in (20261113,20261114)}
    if limit:rows=rows[:limit]
    selected=policies.split(',')
    allowed={'bm25','dense','ce0','ce20','ce24','old-rules-router','ce28','ce40','ce28-ranker','ce40-ranker','learned-router','fixed-pair-budget','quality-guard-40','same-cost-prece-router','same-cost-ce28-router','same-cost-ce24-router','same-cost-calibrated-router'}
    allowed.update({'random-20261113','random-20261114'})
    if not selected or set(selected)-allowed:raise ValueError('invalid policies')
    accelerator=None
    if reader_source:
        from evaluation.delivery_backend import load_reader,BoundedPackedBM25
        reader=load_reader(reader_source,index)
        accelerator=BoundedPackedBM25(reader)
    else:reader=DiskIndex(index)
    manifest=VectorManifest.from_dict(json.loads(Path(vector_manifest).read_text()))
    encoder=BGEEncoder(manifest.embedding,str(model_path),'cuda',batch_size=32)
    vectors=QdrantVectorIndex(manifest,url,ef=128)
    lexical_engine=SearchEngine(reader)
    ce=FrozenCrossEncoder(ce_model,device='cuda',batch_size=20,max_length=256)
    gate=FrozenLinear(bundle['gate']['model']);threshold=bundle['gate']['threshold']
    prece=FrozenTreeEnsemble.load(prece_model) if prece_model else None
    scalar=FrozenScalarTree.load(scalar_model) if scalar_model else None
    if ('same-cost-prece-router' in policies or 'same-cost-ce28-router' in policies or 'same-cost-ce24-router' in policies) and prece is None: raise ValueError('same-cost-prece-router requires --prece-model')
    if 'same-cost-calibrated-router' in policies and scalar is None: raise ValueError('same-cost-calibrated-router requires --scalar-model')
    rankers={b:(FrozenLinear(bundle['rankers'][str(b)]['model'])
                if bundle['rankers'][str(b)]['model'] is not None else None) for b in (28,40)}
    rules=RuleRouter(RetrievalConfig())
    fixed=int(json.loads((Path(policy_dir)/'summary.json').read_text())['protocol']['fixed_pair_budget'])
    identity=None if reader_source else {reader.doc(i).doc_id:i for i in range(reader.n_docs)}
    def search(text,policy,qid):
        constrained=parse(text).get('explicit_constraints',False)
        if constrained and policy=='dense':return {'unsupported':True}
        lexical=[]
        if policy!='dense':
            if accelerator is None:
                response=lexical_engine.search(SearchRequest(query=text,mode='lexical',k=200))
                lexical=[{'doc_id':h.doc_id,'docid':identity[h.doc_id],'score':h.score} for h in response.results]
            else:
                parsed=parse(text)
                scoring_terms=[t.term for raw in parsed['terms'] for t in analyzer.analyze(raw)]
                candidates=boolean_docs(reader,parsed['ast'],analyzer.analyze) if constrained else None
                hits=accelerator.topk(scoring_terms,200,candidate_docs=candidates)
                if candidates is not None:
                    scores=dict(hits)
                    hits=sorted(((i,scores.get(i,0.))for i in candidates),key=lambda h:(-h[1],h[0]))[:200]
                lexical=[{'doc_id':reader.external_doc_id(i),'docid':i,'score':s}for i,s in hits]
        row={'text':text,'lexical':lexical,'dense':[],'explicit_constraints':constrained}
        route_start=time.perf_counter()
        terms={t.term for raw in parse(text)['terms'] for t in analyzer.analyze(raw)}
        dfs={t:reader.df(t) for t in terms}
        features=gate_features(row,dfs)
        use_vector=policy!='bm25' and not constrained
        if policy=='old-rules-router':
            use_vector=rules.decide(QueryFeatures(constrained,int(features[0]),float(features[1]),
                int(features[3]),float(features[2]) if features[3]>1 else None)).use_vector
        # This policy uses only online BM25 statistics; the threshold is frozen
        # from the training distribution and is never selected from qrels.
        policy_threshold = quality_threshold if policy=='quality-guard-40' else threshold
        prece_features=extract_features(row).tolist() if prece is not None else None
        prece_score=prece.predict_one(prece_features) if prece_features is not None else None
        if policy in ('same-cost-prece-router','same-cost-ce28-router','same-cost-ce24-router'):
            heavy=quality_guard_heavy(prece_score, prece.threshold, constrained)
        elif policy=='same-cost-calibrated-router':
            prece_score=prece.predict_one(extract_features(row)) if prece is not None else 0.0
            calibrated_score=scalar.predict_one(prece_score)
            heavy=quality_guard_heavy(calibrated_score, scalar.threshold, constrained)
        else:
            heavy=policy in ('learned-router','quality-guard-40') and quality_guard_heavy(gate.predict([features])[0], policy_threshold, constrained)
        if policy.startswith('random-'):
            if qid not in source_predictions:raise ValueError('random control requires frozen evaluation qid')
            heavy=not constrained and qid in random_heavy[policy]
        router_ms=(time.perf_counter()-route_start)*1000
        if use_vector:
            hits=vectors.search(encoder.encode_query(text),200)
            for hit in hits:
                actual=(reader.external_doc_id(hit.docid) if reader_source else reader.doc(hit.docid).doc_id)
                if actual!=hit.doc_id:raise ValueError('dense identity mismatch')
            row['dense']=[{'doc_id':h.doc_id,'docid':h.docid,'score':h.score} for h in hits]
        if policy=='dense':pool=row['dense']
        elif not use_vector:pool=lexical
        else:
            ext={h['docid']:h['doc_id'] for h in lexical+row['dense']}
            pool=[{'docid':i,'doc_id':ext[i],'score':s} for i,s in
                  rrf([[h['docid'] for h in lexical],[h['docid'] for h in row['dense']]],60.)[:200]]
        row['pool']=[h['doc_id'] for h in pool[:100]]
        row['global_docids']=[h['docid'] for h in pool[:100]]
        budget=policy_ce_budget(policy, fixed, heavy)
        elapsed=0.;results=row['pool']
        if budget:
            chosen=row['pool'][:budget]
            documents=[]
            for docid in row['global_docids'][:budget]:
                doc=reader.document_text(docid) if reader_source else reader.doc(docid)
                documents.append(doc.title+'\n'+doc.text)
            scores,elapsed=ce.score(text,documents)
            row[f'ce{budget}']={'scores':scores}
            results=rerank(row,budget,rankers[budget] if policy.endswith('-ranker') else None)
        return {'results':results,'heavy':budget>0,'ce_pairs':min(budget,len(row['pool'])),
                'ce_ms':elapsed,'vector_attempt':use_vector,'gate_heavy':bool(heavy),
                'gate_features':features.tolist(),'prece_features':prece_features,'prece_score':prece_score,'router_ms':router_ms}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body['policy'] not in selected:raise ValueError('invalid policy')
                result=search(body['query'],body['policy'],body.get('query_id'));status=200
            except Exception as error:result={'error':str(error)};status=500
            raw=json.dumps(result,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    endpoint=f'http://127.0.0.1:{server.server_port}/search'
    def request(text,policy,qid):
        start=time.perf_counter()
        try:
            response=json.load(opener.open(urllib.request.Request(endpoint,
                  json.dumps({'query':text,'policy':policy,'query_id':qid}).encode(),{'Content-Type':'application/json'}),timeout=180))
        except urllib.error.HTTPError as error:
            detail=error.read().decode(errors='replace'); raise RuntimeError(f'HTTP policy {policy} failed: {detail}') from error
        response['client_wall_ms']=(time.perf_counter()-start)*1000
        return response
    output.mkdir(parents=True)
    records=[]
    try:
        warm=next(r for r in rows if not parse(r.get('query',r.get('text'))).get('explicit_constraints',False))
        for policy in selected:request(warm.get('query',warm.get('text')),policy,warm.get('query_id',warm.get('qid')))
        with (output/'requests.pending.jsonl').open('x',encoding='utf-8') as handle:
            for i,row in enumerate(rows):
                qid=row.get('query_id',row.get('qid'));text=row.get('query',row.get('text'))
                # Rotate action order to reduce systematic per-policy warm bias.
                order=selected[i%len(selected):]+selected[:i%len(selected)]
                for policy in order:
                    response=request(text,policy,qid)
                    response.update(query_id=qid,policy=policy)
                    if not response.get('unsupported'):
                        if policy=='learned-router':
                            expected=source_predictions[qid]
                            if response['gate_heavy']!=expected['heavy']:
                                raise ValueError('train/capture/serve Router decision mismatch')
                            # ANN and numerical precision may shift tied candidates.
                            response['exact_ids_match_capture']=response['results']==expected['actions'][policy][:100]
                        else:
                            response['exact_ids_match_capture']=None
                    records.append(response);handle.write(json.dumps(response,ensure_ascii=False)+'\n');handle.flush()
                if (i+1)%50==0:print('HTTP policies',i+1,'/',len(rows),flush=True)
    finally:server.shutdown();server.server_close()
    (output/'requests.pending.jsonl').rename(output/'requests.jsonl')
    prediction_sha=sha256_of(output/'requests.jsonl')
    labels={}
    for line in Path(qrels).read_text().splitlines():
        q,_,d,g=line.split();labels.setdefault(q,{})[d]=float(g)
    summary={}
    for policy in selected:
        rs=[r for r in records if r['policy']==policy and not r.get('unsupported')]
        values=[metrics(r['results'],labels[r['query_id']]) for r in rs]
        summary[policy]={'queries':len(rs),'metrics':{k:statistics.mean(v[k] for v in values) for k in values[0]},
                         'e2e_ms':{k:percentile([r['client_wall_ms'] for r in rs],p) for k,p in [('p50',.5),('p95',.95)]},
                         'ce_ms':{k:percentile([r['ce_ms'] for r in rs],p) for k,p in [('p50',.5),('p95',.95)]},
                         'ce_pairs_mean':statistics.mean(r['ce_pairs'] for r in rs),
                         'heavy_rate':statistics.mean(r['gate_heavy'] if policy in ('quality-guard-40','same-cost-prece-router','same-cost-ce28-router','same-cost-ce24-router','same-cost-calibrated-router') else r['heavy'] for r in rs),
                         'vector_attempt_rate':statistics.mean(r['vector_attempt'] for r in rs),
                         'router_ms':{k:percentile([r['router_ms'] for r in rs],p) for k,p in [('p50',.5),('p95',.95)]},
                         'exact_ids_match_capture_rate':(statistics.mean(r['exact_ids_match_capture'] for r in rs) if policy=='learned-router' else None)}
    summary['protocol']={'queries':len(rows),'partial_pilot':bool(limit),'qrels_read_after_requests':True,
                         'requests_sha256_before_labels':prediction_sha,
                         'models_sha256':sha256_of(Path(policy_dir)/'models.json'),
                         'vector_manifest_sha256':sha256_of(vector_manifest),
                         'candidate_replay':False,'search_params':{'hnsw_ef':128,'exact':False},
                         'lexical_backend':'frozen-rfc4-compact-bounded-float64' if reader_source else 'DiskIndex-reference',
                         'runtime_source_sha256':runtime_sources,'cpu_affinity':sorted(os.sched_getaffinity(0)),
                         'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','TOKENIZERS_PARALLELISM']},
                         'qrels_sha256':sha256_of(qrels),'queries_sha256':sha256_of(queries),
                         'latency_scope':'real loopback client wall; concurrent batch jobs may affect latency'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('index','vector-manifest','model-path','ce-model','url','queries','qrels','policy-dir','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--policies',default='bm25,dense,ce0,old-rules-router,ce28,ce40,ce28-ranker,ce40-ranker,learned-router,fixed-pair-budget')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--reader-source')
    p.add_argument('--quality-threshold',type=float,default=0.153)
    p.add_argument('--prece-model')
    p.add_argument('--scalar-model')
    print(json.dumps(benchmark(**vars(p.parse_args())),indent=2))
