"""Checked CUDA-only TensorRT scorer for the original full BGE model.

The engine changes floating point execution. Its logits require fresh grouped
development validation; this class makes no ranking-equivalence claim.
"""
import json
from pathlib import Path
import time

import numpy as np

from .export_bge_tensorrt_input import position_ids
from .prepare_bm25_router import sha
from .study_rerank_actions import length_batches


class TensorRTCrossEncoder:
    def __init__(self,engine_path,model_path,*,max_length=256,batch_size=20):
        import torch
        import tensorrt as trt
        from transformers import AutoTokenizer
        engine_path,model_path=Path(engine_path),Path(model_path)
        report=json.loads((engine_path/'summary.json').read_text())
        protocol=report['protocol'];encoder=protocol.get('source_encoder',protocol.get('encoder'))
        if protocol['kind'] not in ('full-BGE-FP16-TensorRT-engine-v1','full-BGE-INT8-TensorRT-engine-v1'):
            raise ValueError('unsupported TensorRT engine')
        if sha(engine_path/'engine.plan')!=report['engine_sha256']:
            raise ValueError('TensorRT engine checksum mismatch')
        if (protocol['tensorrt']!=trt.__version__ or
                ('compute_capability' in protocol and protocol['compute_capability']!=list(torch.cuda.get_device_capability(0))) or
                ('gpu' in protocol and protocol['gpu']!=torch.cuda.get_device_name(0))):
            raise ValueError('TensorRT engine device/runtime differs')
        if sha(model_path/'model_inventory.json')!=encoder['inventory_sha256']:
            raise ValueError('TensorRT source model identity mismatch')
        inventory=json.loads((model_path/'model_inventory.json').read_text())
        for name,digest in inventory['files'].items():
            if sha(model_path/name)!=digest:raise ValueError('TensorRT tokenizer/model checksum mismatch: '+name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True,trust_remote_code=False)
        if self.tokenizer.padding_side!='right':
            raise ValueError('TensorRT manual padding requires a right-padding tokenizer')
        self.logger=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(self.logger,'')
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine((engine_path/'engine.plan').read_bytes())
        if self.engine is None:raise ValueError('TensorRT engine deserialization failed')
        self.context=self.engine.create_execution_context()
        if self.context is None:raise ValueError('TensorRT execution context unavailable')
        self.torch=torch;self.trt=trt;self.stream=torch.cuda.Stream()
        self.input_names=['input_ids','attention_mask','position_ids']
        inputs={self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.INPUT}
        outputs={self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))==trt.TensorIOMode.OUTPUT}
        if inputs!=set(self.input_names) or outputs!={'logits'}:
            raise ValueError('TensorRT engine IO schema mismatch')
        self.dtypes={trt.int64:torch.int64,trt.int32:torch.int32,trt.float16:torch.float16,trt.float32:torch.float32}
        for name in self.input_names+['logits']:
            if self.engine.get_tensor_location(name)!=trt.TensorLocation.DEVICE:
                raise ValueError('TensorRT expected CUDA IO bindings')
            dtype=self.engine.get_tensor_dtype(name)
            if dtype not in self.dtypes:raise ValueError('TensorRT unsupported IO dtype')
            if name in inputs and dtype not in (trt.int64,trt.int32):
                raise ValueError('TensorRT token inputs must be integers')
        self.max_length=max_length;self.batch_size=batch_size;self.batch_strategy='length'
        self.info=dict(encoder,max_length=max_length,batch_size=batch_size,
            runtime_backend='tensorrt-cuda',runtime_version=trt.__version__,
            engine_sha256=report['engine_sha256'],onnx_sha256=protocol.get('ONNX_sha256',protocol.get('source_model_sha256')),
            math_changed_from_training=True,query_prediction_cache=False,
            io_dtypes={name:str(self.engine.get_tensor_dtype(name)) for name in self.input_names+['logits']})

    def score(self,query,texts):
        torch=self.torch
        if type(self.max_length) is not int or not 32<=self.max_length<=256:
            raise ValueError('TensorRT sequence length outside verified profile')
        max_batch=int(self.engine.get_tensor_profile_shape('input_ids',0)[2][0])
        if type(self.batch_size) is not int or not 1<=self.batch_size<=max_batch:
            raise ValueError('TensorRT batch size outside verified profile')
        tick=time.perf_counter()
        encoded=self.tokenizer([query]*len(texts),texts,padding=False,truncation=True,max_length=self.max_length) if texts else {}
        values=[None]*len(texts);stream=self.stream;pending=[]
        with torch.cuda.stream(stream),torch.inference_mode():
            for indices in length_batches([len(r) for r in encoded.get('input_ids',[])],self.batch_size):
                rows=[encoded['input_ids'][i] for i in indices]
                max_len=max(len(row) for row in rows)
                pad_id=int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 1)
                input_ids=np.full((len(rows),max_len),pad_id,dtype=np.int64)
                attention=np.zeros((len(rows),max_len),dtype=np.int64)
                for row_i,row in enumerate(rows):
                    input_ids[row_i,:len(row)]=row;attention[row_i,:len(row)]=1
                arrays={'input_ids':input_ids,'attention_mask':attention,
                    'position_ids':position_ids(input_ids)}
                buffers={}
                for name in self.input_names:
                    tensor=torch.from_numpy(np.ascontiguousarray(arrays[name])).to(
                        device='cuda',dtype=self.dtypes[self.engine.get_tensor_dtype(name)]).contiguous()
                    buffers[name]=tensor
                    if not self.context.set_input_shape(name,tuple(tensor.shape)):
                        raise ValueError('TensorRT input shape rejected')
                    if not self.context.set_tensor_address(name,tensor.data_ptr()):
                        raise ValueError('TensorRT input address rejected')
                missing=self.context.infer_shapes()
                if missing:raise ValueError('TensorRT input shapes unresolved: '+str(missing))
                shape=tuple(self.context.get_tensor_shape('logits'))
                if not shape or any(d<=0 for d in shape):raise ValueError('TensorRT output shape unresolved')
                output=torch.empty(shape,device='cuda',dtype=self.dtypes[self.engine.get_tensor_dtype('logits')])
                buffers['logits']=output
                if not self.context.set_tensor_address('logits',output.data_ptr()):
                    raise ValueError('TensorRT output address rejected')
                if not self.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError('TensorRT execution failed')
                # Keep every input allocation alive until the queued kernels
                # have consumed it, including when the context changes shape.
                pending.append((indices,output,buffers))
            stream.synchronize()
            for indices,output,_buffers in pending:
                logits=output.reshape(-1).float().cpu().numpy()
                if len(logits)!=len(indices) or not np.isfinite(logits).all():
                    raise ValueError('invalid TensorRT logits')
                for i,value in zip(indices,logits):values[i]=float(value)
        return values,(time.perf_counter()-tick)*1000


class AdaptiveTensorRTCrossEncoder:
    """Use a shorter profile only when it is semantically exact for a request."""
    def __init__(self,short_scorer,long_scorer,short_max=192):
        if short_scorer.tokenizer.get_vocab()!=long_scorer.tokenizer.get_vocab():
            raise ValueError('adaptive TensorRT tokenizers differ')
        self.short,self.long=short_scorer,long_scorer;self.short_max=int(short_max)
        self.tokenizer=long_scorer.tokenizer
        self.info=dict(long_scorer.info, runtime_backend='tensorrt-cuda-adaptive-profile',
            short_engine_sha256=short_scorer.info['engine_sha256'],long_engine_sha256=long_scorer.info['engine_sha256'],
            adaptive_short_max=self.short_max,math_changed_from_training=True)

    def score(self,query,texts):
        encoded=self.tokenizer([query]*len(texts),texts,padding=False,truncation=True,max_length=256) if texts else {}
        maximum=max((len(r) for r in encoded.get('input_ids',[])),default=0)
        # The short engine receives exactly the same untruncated pair inputs;
        # its max profile is therefore an implementation optimization only.
        if maximum<=self.short_max:
            return self.short.score(query,texts)
        return self.long.score(query,texts)
