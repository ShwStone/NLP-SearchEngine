"""Label-free query-local document support from frozen retrieved anchors.

Hashing uses the shared analyzer and has no vocabulary or query-label fitting.
All anchors are chosen by a frozen scorer or raw retrieval ranks. Output is an
optional ranking feature artifact, not an alternative candidate source.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from contracts import analyzer
from .prepare_bm25_router import jsonl_rows, sha


FEATURE_NAMES = ('teacher_top1_cosine', 'teacher_other_top3_max_cosine',
                 'teacher_other_top5_weighted_cosine', 'dense_top1_cosine',
                 'lexical_top1_cosine', 'teacher_top5_other_support_count')


def tokens(text):
    return [token.term for token in analyzer.analyze(text[:1024])]


def support_features(similarity, candidates, anchors, teacher_anchor_count, *, dense_slot=None, lexical_slot=None):
    result = np.zeros((len(candidates),len(FEATURE_NAMES)),dtype=np.float32)
    if not anchors:
        return result
    # For each candidate, other-anchor features exclude its own document.
    for i, doc_id in enumerate(candidates):
        teacher_other = [(rank,float(similarity[i,j])) for j,(rank,anchor) in enumerate(anchors[:teacher_anchor_count])
                         if anchor != doc_id]
        result[i,0] = similarity[i,0] if teacher_anchor_count else 0.
        result[i,1] = max((value for rank,value in teacher_other if rank<3),default=0.)
        total = sum(1./(rank+1) for rank,value in teacher_other)
        result[i,2] = sum(value/(rank+1) for rank,value in teacher_other)/total if total else 0.
        result[i,3] = similarity[i,dense_slot] if dense_slot is not None else 0.
        result[i,4] = similarity[i,lexical_slot] if lexical_slot is not None else 0.
        result[i,5] = sum(value>=.5 for rank,value in teacher_other)
    return result


def support_features_fast(similarity,candidates,anchors,teacher_anchor_count,*,dense_slot=None,lexical_slot=None):
    """Opt-in vectorized version, preserving float64 weighted arithmetic."""
    result=np.zeros((len(candidates),len(FEATURE_NAMES)),dtype=np.float32)
    if not anchors:return result
    if teacher_anchor_count:
        teacher=anchors[:teacher_anchor_count]
        values=similarity[:,:teacher_anchor_count].astype(np.float64)
        others=np.asarray(candidates)[:,None]!=np.asarray([d for _,d in teacher])[None,:]
        ranks=np.array([rank for rank,_ in teacher])
        weights=1./(ranks.astype(np.float64)+1.)
        result[:,0]=similarity[:,0]
        maximum=np.where(others&(ranks[None,:]<3),values,-np.inf).max(axis=1)
        result[:,1]=np.where(np.isneginf(maximum),0.,maximum)
        # Match the reference's division (not multiplication by reciprocal)
        # and left-to-right summation before its final float32 conversion.
        total=np.zeros(len(candidates),dtype=np.float64)
        numerator=np.zeros(len(candidates),dtype=np.float64)
        for j,rank in enumerate(ranks):
            total+=np.where(others[:,j],weights[j],0.)
            numerator+=np.where(others[:,j],values[:,j]/float(rank+1),0.)
        result[:,2]=np.divide(numerator,total,out=np.zeros_like(total),where=total>0)
        result[:,5]=np.sum(others&(values>=.5),axis=1)
    if dense_slot is not None:result[:,3]=similarity[:,dense_slot]
    if lexical_slot is not None:result[:,4]=similarity[:,lexical_slot]
    return result


def prepare(table,capture,documents,teacher_logits,out,document_vectors=None,allow_partial_teacher=False):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    provenance=json.loads(Path(table).with_name('provenance.json').read_text())
    if sha(documents)!=provenance['source_manifest']['files']['docs.jsonl']:
        raise ValueError('candidate support document identity mismatch')
    teacher_report=json.loads(Path(teacher_logits).with_name('summary.json').read_text())
    if sha(teacher_logits)!=teacher_report['logits_sha256'] or teacher_report['protocol']['kind']!='frozen-reranking-development-feasibility':
        raise ValueError('candidate support requires verified frozen teacher logits')
    for path in (table,capture,documents):
        if teacher_report['protocol']['inputs'].get(str(path))!=sha(path):
            raise ValueError('candidate support source identity mismatch')
    allowed={row['query_id'] for row in jsonl_rows(table)}
    teacher={row['query_id']:row['logits'] for row in jsonl_rows(teacher_logits)}
    if set(teacher)!=allowed:
        raise ValueError('teacher/query identity mismatch')
    docs=list(jsonl_rows(documents))
    identities={row['doc_id']:i for i,row in enumerate(docs)}
    if document_vectors is None:
        vectorizer=HashingVectorizer(analyzer=tokens,n_features=2**18,alternate_sign=False,
                                   norm='l2',dtype=np.float32)
        matrix=vectorizer.transform(row['title']+'\n'+row['text'] for row in docs)
    else:
        document_vectors=Path(document_vectors)
        vector_protocol=json.loads(document_vectors.with_name('protocol.json').read_text())
        if (vector_protocol['kind']!='frozen-document-vector-export-v1' or
                vector_protocol['vector_sha256']!=sha(document_vectors) or
                vector_protocol['inputs'].get(str(documents))!=sha(documents) or
                vector_protocol['manifest']!=provenance['vector_manifest']):
            raise ValueError('frozen support vector provenance mismatch')
        matrix=np.load(document_vectors,mmap_mode='r',allow_pickle=False)
        if matrix.shape!=(len(docs),provenance['vector_manifest']['embedding']['dimension']):
            raise ValueError('frozen support vector shape mismatch')
        identities={row['doc_id']:row['docid'] for row in docs}
    del docs
    seen=set()
    with (out/'features.jsonl').open('w') as handle:
        for row in jsonl_rows(capture):
            q=row['query_id']
            if q not in allowed:
                continue
            if q in seen:
                raise ValueError('duplicate candidate support query')
            seen.add(q)
            lexical=[hit['doc_id'] for hit in row['lexical']]
            dense=[hit['doc_id'] for hit in row['dense']]
            candidates=sorted(set(lexical)|set(dense),key=identities.__getitem__)
            if (not set(teacher[q])<=set(candidates) or
                    (not allow_partial_teacher and set(teacher[q]) != set(candidates))):
                raise ValueError('support experiment requires full candidate teacher coverage')
            best=sorted(teacher[q],key=lambda d:(-teacher[q][d],identities[d]))[:5]
            # Distinct slots are retained even when retrieval anchors coincide.
            anchors=list(enumerate(best))
            dense_slot=lexical_slot=None
            if dense:
                dense_slot=len(anchors)
                anchors.append((0,dense[0]))
            if lexical:
                lexical_slot=len(anchors)
                anchors.append((0,lexical[0]))
            sim=matrix[[identities[d] for d in candidates]]@matrix[[identities[d] for _,d in anchors]].T
            if document_vectors is None:
                sim=sim.toarray()
            values=support_features(sim,candidates,anchors,len(best),dense_slot=dense_slot,lexical_slot=lexical_slot)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite candidate support feature')
            handle.write(json.dumps(dict(query_id=q,features=dict(zip(candidates,values.tolist()))))+'\n')
            if len(seen)%100==0:
                print('support features',len(seen),flush=True)
    if seen!=allowed:
        raise ValueError('candidate support query coverage incomplete')
    protocol=dict(kind='query-local-candidate-support-v1',feature_names=FEATURE_NAMES,
        queries=len(seen),inputs={str(p):sha(p) for p in (table,capture,documents,teacher_logits)},
        feature_file_sha256=sha(out/'features.jsonl'),hashing_dimensions=2**18,document_max_chars=1024,
        analyzer=analyzer.config_hash(),anchor='top5 frozen teacher plus top1 dense and lexical',
        labels_read=False,confirmation_read=False,latency_includes_teacher=False,
        scope='optional label-free ranking feature, offline extraction cost not HTTP latency')
    protocol['partial_teacher_allowed']=allow_partial_teacher
    protocol['teacher_anchor_scope']='only actually scored frozen teacher candidates; no imputation for anchor selection'
    if document_vectors is not None:
        protocol.update(representation='frozen normalized document embedding',hashing_dimensions=None,
                        document_max_chars=None,vector_export_sha256=sha(document_vectors),
                        vector_export_protocol_sha256=sha(document_vectors.with_name('protocol.json')))
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps(protocol,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('table','capture','documents','teacher-logits','out'):
        parser.add_argument('--'+name,required=True,type=Path)
    parser.add_argument('--document-vectors',type=Path)
    parser.add_argument('--allow-partial-teacher',action='store_true')
    prepare(**vars(parser.parse_args()))
