"""Shape-bounded CUDA Graph native query encoder, with actual per-call inputs.

Only single, unpadded queries are supported. Attention mask removal during
tracing matches that contract; document encoding and padded batches use the
original FrozenM3. No query vectors or query results are memoized.
"""

import torch


class NativeQueryGraph:
    def __init__(self,base):
        self.base=base;self.info=base.info;self.graphs={}
        self.torch=torch;self.device=base.device;self.tokenizer=base.tokenizer
        if base.device!='cuda':raise ValueError('native query graphs require CUDA')
        class Dense(torch.nn.Module):
            def __init__(self,model):super().__init__();self.model=model
            def forward(self,input_ids):
                hidden=self.model(input_ids=input_ids,attention_mask=torch.ones_like(input_ids)).last_hidden_state
                return torch.nn.functional.normalize(hidden[:,0].float(),dim=-1)
        # A single tokenized query has an all-one attention mask. Eliminate
        # that mask in each traced SDPA call, avoiding data-dependent host
        # checks while retaining the original unpadded attention semantics.
        def no_padding_mask(module,args):return (args[0],None,*args[2:])
        handles=[layer.attention.self.register_forward_pre_hook(no_padding_mask) for layer in base.model.encoder.layer]
        dummy=torch.tensor([[0]+[10]*14+[2]],device='cuda',dtype=torch.long)
        try:
            with torch.inference_mode():
                self.model=torch.jit.trace(Dense(base.model).eval(),(dummy,),check_trace=False,strict=False)
        finally:
            for handle in handles:handle.remove()
        self.runtime_info=dict(kind='torch-jit-cuda-graph-native-query',batch_size=1,max_tokens=64,
            validation='CPU attention mask must be all ones; one actual query per call',
            attention='same unpadded SDPA semantics as original single-query encoder',
            outputs='normalized CLS dense only; unused sparse/late heads are not computed',
            query_result_cache=False)

    def _forward(self,input_ids):
        shape=tuple(input_ids.shape)
        if shape[0]!=1 or not 2<=shape[1]<=64:raise ValueError('query graph input shape outside supported scope')
        if shape not in self.graphs:
            static=input_ids.clone();stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.model(static)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):output=self.model(static)
            self.graphs[shape]=(graph,static,output)
        graph,static,output=self.graphs[shape];static.copy_(input_ids);graph.replay()
        return output

    def warm(self):
        with torch.inference_mode():
            for length in range(2,65):
                ids=torch.tensor([[0]+[10]*(length-2)+[2]],device='cuda',dtype=torch.long)
                self._forward(ids)
        torch.cuda.synchronize()

    def encode(self,texts,max_length):
        if len(texts)!=1 or max_length!=64:raise ValueError('native query graph accepts one query with the frozen 64-token limit')
        inputs=self.tokenizer(texts,padding=True,truncation=True,max_length=64,return_tensors='pt')
        if not bool(torch.all(inputs['attention_mask']==1)):
            raise ValueError('padded query inputs cannot use the unpadded native graph')
        with torch.inference_mode():dense=self._forward(inputs['input_ids'].to('cuda'))[0]
        return [dict(dense=dense)]
