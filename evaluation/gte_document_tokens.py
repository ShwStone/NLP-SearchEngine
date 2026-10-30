"""Query-independent document tokens for the exact frozen GTE pair template.

Only the scorer's existing repeated-query, unpadded, right-truncated pair API
is optimized. Other calls fall back to the original tokenizer. Query tokens
are recomputed for each call; model inputs and predictions are never cached.
"""
import time

class DocumentTokenPairs:
    def __init__(self,tokenizer,documents,max_length=512):
        if (tokenizer.truncation_side!='right' or tokenizer.padding_side!='right' or
                list(tokenizer.model_input_names)!=['input_ids','attention_mask'] or
                tokenizer.num_special_tokens_to_add(pair=True)!=4 or max_length<32):
            raise ValueError('unsupported tokenizer/template for document-token cache')
        self.original=tokenizer;self.max_length=max_length;self.enabled=True
        self.cache={};self.cached_calls=0;self.fallback_calls=0;self.last_encoding_ms=None
        started=time.perf_counter();documents=list(dict.fromkeys(documents))
        for start in range(0,len(documents),256):
            batch=documents[start:start+256]
            encoded=tokenizer(batch,add_special_tokens=False,padding=False,truncation=True,
                              max_length=max_length,return_attention_mask=False)
            self.cache.update(zip(batch,encoded['input_ids']))
        self.build_seconds=time.perf_counter()-started

    def __getattr__(self,name):
        return getattr(self.original,name)

    def __call__(self,queries,documents=None,**kwargs):
        tick=time.perf_counter()
        supported=(self.enabled and isinstance(queries,list) and isinstance(documents,list) and
                   len(queries)==len(documents) and len(queries)>0 and
                   all(isinstance(q,str) and q==queries[0] for q in queries) and
                   kwargs==dict(padding=False,truncation=True,max_length=self.max_length) and
                   all(d in self.cache and self.cache[d] for d in documents))
        if not supported:
            if self.enabled:self.fallback_calls+=1
            value=self.original(queries,documents,**kwargs)
        else:
            query=self.original.encode(queries[0],add_special_tokens=False,truncation=True,
                                       max_length=self.max_length)
            if not query:
                self.fallback_calls+=1;value=self.original(queries,documents,**kwargs)
            else:
                rows=[]
                for d in documents:
                    document=self.cache[d];qlen,dlen=len(query),len(document)
                    remove=max(0,qlen+dlen+4-self.max_length)
                    # The pinned tokenizer's right-side longest-first rule:
                    # remove the length difference first, then split remaining
                    # removals, with an odd final removal assigned to sequence2.
                    first=min(abs(qlen-dlen),remove);rest=remove-first
                    qremove=(first if qlen>dlen else 0)+rest//2
                    dremove=(first if qlen<=dlen else 0)+rest-rest//2
                    rows.append(self.original.build_inputs_with_special_tokens(
                        query[:qlen-qremove],document[:dlen-dremove]))
                value=dict(input_ids=rows,attention_mask=[[1]*len(r) for r in rows])
                self.cached_calls+=1
        self.last_encoding_ms=(time.perf_counter()-tick)*1000
        return value
