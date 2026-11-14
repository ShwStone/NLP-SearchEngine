"""Frozen native BGE-M3 late-interaction pilot, with label-free projections.

Document token embeddings are computed once and cached in RAM for this pilot.
Per-query time includes encoding and candidate matching, but excludes first
retrieval and HTTP. Random projections are fixed before labels are read.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .fetch_bge_m3 import MODEL_ID,REVISION,FILES
from .prepare_bm25_router import jsonl_rows,metrics,sha
from .router_metrics import percentile
from .train_bm25_router import group_bootstrap


def maxsim_numpy(query,document):
    query,document=np.asarray(query),np.asarray(document)
    if (query.ndim!=2 or document.ndim!=2 or query.shape[1]!=document.shape[1]
            or not len(query) or not len(document) or not np.isfinite(query).all()
            or not np.isfinite(document).all()):
        raise ValueError('invalid late interaction token vectors')
    return float(np.max(query@document.T,axis=1).mean())


class FrozenM3:
    def __init__(self,path,device,include_projections=True):
        import torch
        from transformers import AutoModel,AutoTokenizer
        path=Path(path)
        inventory=json.loads((path/'model_inventory.json').read_text())
        if (inventory['model_id'],inventory['revision'])!=(MODEL_ID,'modelscope:'+REVISION):
            raise ValueError('late interaction model identity mismatch')
        for name,(digest,_) in FILES.items():
            if sha(path/name)!=digest:
                raise ValueError('late interaction model checksum mismatch')
        self.torch,self.device=torch,device
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,trust_remote_code=False)
        self.model=AutoModel.from_pretrained(path,local_files_only=True,trust_remote_code=False,
            use_safetensors=False,weights_only=True,torch_dtype=torch.float16).to(device).eval()
        hidden=self.model.config.hidden_size
        self.colbert=torch.nn.Linear(hidden,hidden).to(device,dtype=torch.float16)
        self.sparse=torch.nn.Linear(hidden,1).to(device,dtype=torch.float16)
        for module,name in ((self.colbert,'colbert_linear.pt'),(self.sparse,'sparse_linear.pt')):
            module.load_state_dict(torch.load(path/name,map_location='cpu',weights_only=True),strict=True)
            module.eval()
        generator=torch.Generator(device='cpu').manual_seed(20261028)
        self.projection=(torch.randn(hidden,256,generator=generator,dtype=torch.float32).to(device,dtype=torch.float16)
                         if include_projections else None)
        self.info=dict(model_id=MODEL_ID,revision='modelscope:'+REVISION,inventory_sha256=sha(path/'model_inventory.json'),
            document_max_tokens=512,query_max_tokens=64,colbert_dimensions=[1024,256,128],
            special_tokens='skip CLS; keep other nonpadding contextual token vectors; sparse excludes IDs 0/1/2/3',
            projection='fixed Gaussian seed 20261028, first128 columns for 128 dimensions; L2 normalize',
            query_prefix='',document_template='title + newline + text')
        if not include_projections:
            self.info.update(colbert_dimensions=[1024],projection=None)

    def encode(self,texts,max_length):
        torch=self.torch
        inputs=self.tokenizer(texts,padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(self.device)
        with torch.inference_mode():
            hidden=self.model(**inputs).last_hidden_state
            dense=torch.nn.functional.normalize(hidden[:,0].float(),dim=-1)
            full=torch.nn.functional.normalize(self.colbert(hidden[:,1:]).float(),dim=-1)
            variants={1024:full.half()}
            if self.projection is not None:
                projected=full.half()@self.projection
                variants.update({256:torch.nn.functional.normalize(projected.float(),dim=-1).half(),
                                 128:torch.nn.functional.normalize(projected[:,:,:128].float(),dim=-1).half()})
            weights=torch.relu(self.sparse(hidden).squeeze(-1).float()).cpu().numpy()
            ids=inputs['input_ids'].cpu().numpy()
            lengths=inputs['attention_mask'].sum(dim=1).cpu().tolist()
            rows=[]
            for i,length in enumerate(lengths):
                lexical={}
                for token,value in zip(ids[i,:length].tolist(),weights[i,:length].tolist()):
                    if token not in (0,1,2,3) and value>0:
                        lexical[token]=max(lexical.get(token,0.),value)
                rows.append(dict(dense=dense[i],sparse=lexical,
                                 colbert={dim:value[i,:int(length)-1] for dim,value in variants.items()}))
            return rows


def run(args):
    import torch
    args.out.mkdir(parents=True,exist_ok=False)
    parent=json.loads(args.candidates.with_name('summary.json').read_text())
    if sha(args.candidates)!=parent['logits_sha256']:
        raise ValueError('late interaction pilot candidates changed')
    for path in (args.table,args.capture,args.documents,args.qrels):
        if parent['protocol']['inputs'].get(str(path))!=sha(path):
            raise ValueError('late interaction pilot source mismatch')
    if len(parent['protocol']['queries'])!=96:
        raise ValueError('late interaction feasibility protocol requires the existing 96-query pilot')
    table={r['query_id']:r for r in jsonl_rows(args.table)}
    rows=list(jsonl_rows(args.candidates))
    selected={r['query_id'] for r in rows}
    queries={r['query_id']:r['text'] for r in jsonl_rows(args.capture) if r['query_id'] in selected}
    all_docs={r['doc_id']:(r['docid'],r['title']+'\n'+r['text']) for r in jsonl_rows(args.documents)}
    docids=sorted({d for row in rows for d in row['logits']},key=lambda d:all_docs[d][0])
    protocol=dict(kind='m3-late-interaction-development-pilot',queries=parent['protocol']['queries'],documents=len(docids),
        candidate_selection='same maximum pool as frozen M3 96-query pilot; no labels',
        inputs={str(p):sha(p) for p in (args.table,args.capture,args.documents,args.qrels,args.candidates)},
        qrels_read_for_features=False,independent_confirmation=False,http_evaluated=False,goal_complete=False,
        cache='ephemeral document-token vectors in RAM; cache build reported separately from per-query matching',
        scoring='mean over query tokens of max document-token cosine; sparse lexical dot product; dense CLS cosine',
        combined_weights=dict(dense=1.,sparse=.3,colbert=1.))
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    encoder=FrozenM3(args.model_path,args.device);protocol['encoder']=encoder.info
    np.save(args.out/'projection.npy',encoder.projection.float().cpu().numpy(),allow_pickle=False)
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    lengths=encoder.tokenizer([all_docs[d][1] for d in docids],padding=False,truncation=True,max_length=512)['input_ids']
    order=sorted(range(len(docids)),key=lambda i:(len(lengths[i]),i))
    del lengths
    cache={};cache_bytes=0
    torch.cuda.synchronize();build_started=time.perf_counter()
    for offset in range(0,len(order),16):
        batch=[docids[i] for i in order[offset:offset+16]]
        encoded=encoder.encode([all_docs[d][1] for d in batch],512)
        for d,value in zip(batch,encoded):
            vectors={dim:array.detach().cpu().numpy() for dim,array in value['colbert'].items()}
            size=sum(array.nbytes for array in vectors.values())
            cache_bytes+=size
            if cache_bytes>12*1024**3:
                raise MemoryError('pilot document cache exceeded predeclared 12GiB limit')
            cache[d]=dict(dense=value['dense'].cpu().numpy(),sparse=value['sparse'],colbert=vectors)
        if (offset//16+1)%100==0:
            print('late interaction encoded documents',min(offset+16,len(order)),'/',len(order),flush=True)
    torch.cuda.synchronize();build_seconds=time.perf_counter()-build_started
    del encoded
    observations=[]
    with (args.out/'scores.jsonl').open('w') as handle:
        for index,row in enumerate(rows):
            q=row['query_id'];candidates=sorted(row['logits'],key=lambda d:all_docs[d][0])
            torch.cuda.synchronize();started=time.perf_counter()
            query=encoder.encode([queries[q]],64)[0]
            features={d:dict(dense=float(np.dot(query['dense'].cpu().numpy(),cache[d]['dense'])),
                sparse=sum(value*cache[d]['sparse'].get(token,0.) for token,value in query['sparse'].items())) for d in candidates}
            for dim in (1024,256,128):
                lengths=[len(cache[d]['colbert'][dim]) for d in candidates]
                max_length=max(lengths)
                buffer=np.zeros((len(candidates),max_length,dim),dtype=np.float16)
                for j,d in enumerate(candidates):
                    buffer[j,:lengths[j]]=cache[d]['colbert'][dim]
                documents=torch.as_tensor(buffer,device=args.device)
                scores=torch.einsum('qd,bld->bql',query['colbert'][dim],documents).float()
                mask=torch.arange(max_length,device=args.device)[None,:]>=torch.as_tensor(lengths,device=args.device)[:,None]
                scores.masked_fill_(mask[:,None,:],float('-inf'))
                values=scores.max(dim=-1).values.mean(dim=-1).cpu().tolist()
                for d,value in zip(candidates,values):features[d]['colbert_'+str(dim)]=value
                del documents,scores,buffer
            torch.cuda.synchronize();elapsed=(time.perf_counter()-started)*1000
            record=dict(query_id=q,group=table[q]['group'],features=features,query_and_three_variant_match_ms=elapsed)
            handle.write(json.dumps(record)+'\n');handle.flush();observations.append(record)
            if (index+1)%8==0:print('late interaction scored queries',index+1,'/',len(rows),flush=True)
    # Metrics are first computed after all frozen feature generation has finished.
    relevance={}
    with args.qrels.open() as handle:
        for line in handle:
            q,_,d,g=line.split()
            if q in selected:relevance.setdefault(q,{})[d]=float(g)
    if set(relevance)!=selected:
        raise ValueError('late interaction pilot qrels incomplete')
    variants=('dense','sparse','colbert_1024','colbert_256','colbert_128','combined_1024','combined_256','combined_128')
    summaries={};groups=np.array([r['group'] for r in observations])
    for variant in variants:
        quality=[]
        for row in observations:
            q=row['query_id']
            def score(d):
                values=row['features'][d]
                if variant.startswith('combined_'):
                    return (values['dense']+.3*values['sparse']+values['colbert_'+variant.split('_')[1]])/2.3
                return values[variant]
            ranked=sorted(row['features'],key=lambda d:(-score(d),all_docs[d][0]))
            quality.append(metrics(ranked,relevance[q]))
        result={}
        for metric in ('ndcg','recall'):
            values=np.array([r[metric] for r in quality]);baseline=np.array([table[r['query_id']]['hybrid'][metric] for r in observations])
            result[metric]=dict(value=float(values.mean()),relative_lift=float(values.mean()/baseline.mean()-1),
                                delta_ci=group_bootstrap(values-baseline,groups))
        summaries[variant]=result
    times=[r['query_and_three_variant_match_ms'] for r in observations]
    result=dict(protocol=protocol,variants=summaries,document_cache_bytes=cache_bytes,document_cache_build_seconds=build_seconds,
        three_variant_query_match_ms=dict(mean=float(np.mean(times)),p95=percentile(times,.95)),
        scores_sha256=sha(args.out/'scores.jsonl'),projection_sha256=sha(args.out/'projection.npy'),
        limitations='pilot only; per-query measurement includes all three dimensional variants and RAM-to-GPU copy; excludes retrieval/render/HTTP',goal_complete=False)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='protocol'},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('table','capture','documents','qrels','candidates','model-path','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--device',default='cuda')
    run(parser.parse_args())
