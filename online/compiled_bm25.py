"""Explicit opt-in NumPy BM25 top-K for immutable IndexReader instances.

Keeps float64 term operations in the reference order, including repeated terms
and zero/negative weights. It caches posting IDs/TFs, never query results. The
default scorer registry and SearchEngine remain unchanged.
"""

import math

import numpy as np

from contracts.index_format import BM25Spec


class CompiledBM25:
    def __init__(self,reader):
        self.reader=reader
        self.spec=BM25Spec(**reader.manifest().get('bm25',{}))
        self.lengths=np.fromiter((reader.doc_length(i) for i in range(reader.n_docs)),dtype=np.float64,count=reader.n_docs)
        self.norm=self.spec.k1*(1-self.spec.b+self.spec.b*self.lengths/reader.avgdl)
        self.postings={}

    def _posting(self,term):
        if term not in self.postings:
            rows=self.reader.postings(term)
            if not rows:self.postings[term]=None
            else:
                ids=np.fromiter((r[0] for r in rows),dtype=np.int64,count=len(rows))
                tf=np.fromiter((r[1] for r in rows),dtype=np.float64,count=len(rows))
                df=self.reader.df(term);n=self.reader.n_docs
                idf=math.log(1+(n-df+.5)/(df+.5))
                self.postings[term]=(ids,tf,idf)
        return self.postings[term]

    def score_array(self,terms,*,match_all=False,candidate_docs=None,term_weights=None):
        values=np.zeros(self.reader.n_docs,dtype=np.float64)
        counts=np.zeros(self.reader.n_docs,dtype=np.int32)
        weights=term_weights or {}
        allowed=None
        if candidate_docs is not None:
            allowed=np.zeros(self.reader.n_docs,dtype=bool)
            # Unknown IDs cannot occur in postings, matching the reference set.
            ids=[i for i in candidate_docs if type(i) is int and 0<=i<self.reader.n_docs]
            allowed[ids]=True
        for term in terms:
            posting=self._posting(term)
            if posting is None:continue
            ids,tf,idf=posting
            contribution=weights.get(term,1.)*idf*tf*(self.spec.k1+1)/(tf+self.norm[ids])
            values[ids]+=contribution
            counts[ids]+=1
        included=counts>0
        if allowed is not None:included&=allowed
        elif match_all:included&=counts==len(terms)
        return values,np.flatnonzero(included)

    def score(self,terms,**kwargs):
        values,ids=self.score_array(terms,**kwargs)
        return {int(i):float(values[i]) for i in ids}

    def topk(self,terms,k,**kwargs):
        if type(k) is not int or k<1:raise ValueError('BM25 top-K must be a positive integer')
        values,ids=self.score_array(terms,**kwargs)
        if len(ids)>k:
            boundary=np.partition(values[ids],len(ids)-k)[len(ids)-k]
            greater=ids[values[ids]>boundary]
            ties=ids[values[ids]==boundary]
            ids=np.concatenate((greater,ties[:k-len(greater)]))
        order=np.lexsort((ids,-values[ids]))
        return [(int(i),float(values[i])) for i in ids[order]]
