"""Optional shape-bounded CUDA Graph replay of pinned Torch CE weights.

Captures model computation only. Every call copies its actual query/document
input tensors, so this is kernel replay, never a cache of query predictions.
Broadcast-mask preparation avoids data-dependent host checks during capture;
its numerical effects must be audited before any quality/HTTP claim.
"""

import math
import time

from .study_rerank_actions import FrozenCrossEncoder,length_batches


class CUDAGraphCrossEncoder(FrozenCrossEncoder):
    def __init__(self,path,*,max_length=256,batch_size=20):
        super().__init__(path,device='cuda',batch_size=batch_size,max_length=max_length,batch_strategy='length')
        if batch_size!=20 or max_length not in (192,256):raise ValueError('unsupported CUDA Graph CE configuration')
        self.model.roberta.attn_implementation='eager'
        self.graphs={}
        self.info.update(runtime_backend='torch-cuda-graph',
            attention_mask_preparation='broadcast extended mask, SDPA attention module retained',
            graph_shapes=[[20,192],[20,256]],math_changed_from_training=True)

    def _forward(self,inputs):
        shape=tuple(inputs['input_ids'].shape)
        if shape not in ((20,192),(20,256)):
            return self.model(**inputs).logits.view(-1)
        torch=self.torch
        if shape not in self.graphs:
            static={name:value.clone() for name,value in inputs.items()}
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):self.model(**static).logits
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output=self.model(**static).logits.view(-1)
            self.graphs[shape]=(graph,static,output)
        graph,static,output=self.graphs[shape]
        if set(static)!=set(inputs):raise ValueError('CUDA Graph input tensor schema changed')
        for name,value in inputs.items():static[name].copy_(value)
        graph.replay()
        return output

    def score(self,query,texts):
        torch=self.torch;torch.cuda.synchronize();tick=time.perf_counter()
        encoded=self.tokenizer([query]*len(texts),texts,padding=False,truncation=True,max_length=self.max_length) if texts else {}
        values=[None]*len(texts)
        with torch.inference_mode():
            for indices in length_batches([len(r) for r in encoded.get('input_ids',[])],self.batch_size):
                batch={name:[rows[i] for i in indices] for name,rows in encoded.items()}
                inputs=self.tokenizer.pad(batch,padding=True,return_tensors='pt').to('cuda')
                logits=self._forward(inputs).float().cpu().tolist()
                if len(logits)!=len(indices) or any(not math.isfinite(v) for v in logits):
                    raise ValueError('invalid CUDA Graph CE logits')
                for i,value in zip(indices,logits):values[i]=float(value)
        torch.cuda.synchronize()
        return values,(time.perf_counter()-tick)*1000

    def warm(self):
        previous=self.max_length
        try:
            for length in (192,256):
                self.max_length=length
                self.score('模型预热',['医学说明文本 '*300]*20)
        finally:self.max_length=previous
