"""Expose exact XLM-R positions as an input; remove the dynamic CumSum path."""
import argparse
import json
from pathlib import Path

import numpy as np

from .prepare_bm25_router import sha


def position_ids(input_ids,padding_id=1):
    ids=np.asarray(input_ids)
    if ids.ndim!=2 or not np.issubdtype(ids.dtype,np.integer):
        raise ValueError('position IDs require a two-dimensional integer token matrix')
    mask=(ids!=padding_id).astype(np.int64)
    return np.cumsum(mask,axis=1,dtype=np.int64)*mask+padding_id


def run(a):
    import onnx
    from onnx import helper,TensorProto
    from onnx.utils import Extractor
    parent=json.loads((a.source/'summary.json').read_text())
    if sha(a.source/'model.onnx')!=parent['model_sha256']:raise ValueError('source export changed')
    a.out.mkdir(parents=True,exist_ok=False)
    model=onnx.load(a.source/'model.onnx')
    gathers=[n for n in model.graph.node if n.op_type=='Gather' and 'position_embeddings' in n.name]
    if len(gathers)!=1 or len(gathers[0].input)!=2:raise ValueError('unexpected position embedding graph')
    gathers[0].input[1]='position_ids'
    model.graph.input.append(helper.make_tensor_value_info('position_ids',TensorProto.INT64,['batch','sequence']))
    input_names=['input_ids','attention_mask','position_ids']
    exported=Extractor(model).extract_model(input_names,['logits'])
    if any(n.op_type=='CumSum' for n in exported.graph.node):raise ValueError('position CumSum was not pruned')
    target=a.out/'model.onnx';onnx.save(exported,target);onnx.checker.check_model(str(target))
    protocol=dict(kind='BGE-FP16-three-input-ONNX-v1',parent_export_sha256=sha(a.source/'summary.json'),
        parent_model_sha256=parent['model_sha256'],encoder=parent['protocol']['encoder'],
        parent_inventory_sha256=parent['protocol']['source_model_inventory_sha256'],
        inputs=input_names,outputs=['logits'],position_semantics='XLM-R mask(input_ids!=1), cumulative count * mask + 1',
        weights_changed=False,source_sha256=sha(Path(__file__)),quality_evaluated=False,http_evaluated=False,goal_complete=False)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(a.out/'source.py').write_bytes(Path(__file__).read_bytes())
    result=dict(protocol=protocol,model_sha256=sha(target),nodes=len(exported.graph.node),goal_complete=False)
    (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
