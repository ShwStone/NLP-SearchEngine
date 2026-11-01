"""Create a new complete freeze for an independently refitted nested pre-BGE final model."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .prepare_bm25_router import jsonl_rows,sha


def run(a):
    if a.out.exists() or a.confirmation_dir.exists():raise ValueError('freeze and confirmation directories must be new')
    study=json.loads((a.study/'study.json').read_text());final=json.loads((a.final/'study.json').read_text())
    training=json.loads((a.final/'training.json').read_text())
    http=json.loads((a.http/'summary.json').read_text());audit=json.loads((a.http/'independent-audit.json').read_text())
    smoke=json.loads((a.smoke/'summary.json').read_text());fa=json.loads((a.final/'independent-audit.json').read_text())
    if not study['quality_point_gate_passed'] or not audit['development_accepted'] or audit['summary_sha256']!=sha(a.http/'summary.json'):
        raise ValueError('audited development quality/HTTP gate failed or changed')
    if final['protocol']['source_study_sha256']!=sha(a.study/'study.json') or fa['study_sha256']!=sha(a.final/'study.json'):
        raise ValueError('final model not prepared/audited from this development study')
    if not smoke['top_scored_pool_and_actions_exact'] or smoke['model_sha256']!=sha(a.final/'study.json'):
        raise ValueError('final-model HTTP smoke failed')
    for path,digest in http['protocol']['code_sha256'].items():
        if sha(path)!=digest:raise ValueError('qualified serving source drift: '+path)
    for protocol in (study['protocol'],final['protocol'],dict(inputs=training['inputs'])):
        for path,digest in protocol.get('inputs',{}).items():
            if sha(path)!=digest:raise ValueError('development input drift: '+path)
        for path,digest in protocol.get('code_sha256',{}).items():
            if sha(path)!=digest:raise ValueError('development source drift: '+path)
    # Historical configuration supplies paths only; every file/history hash below is regenerated.
    base=json.loads(a.base_freeze.read_text());assets=json.loads((a.assets/'assets.json').read_text())
    runtime={k:v for k,v in base['runtime'].items() if not k.startswith('gte_')}
    runtime['tensorrt_engine']=str(a.engine.resolve())
    engine=json.loads((a.engine/'summary.json').read_text());encoder=final['protocol']['online_scorer']
    if encoder['engine_sha256']!=sha(a.engine/'engine.plan') or engine['engine_sha256']!=encoder['engine_sha256']:
        raise ValueError('final scorer/engine binding differs')
    repo=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=False)
    # Query catalogs supply text for observed IDs; catalogs alone are not
    # predictions and do not cause all unscored queries to become training data.
    catalog=repo/'artifacts/rfc01/twenty-percent/mmarco-expanded-development-features-r1/raw-query-source.jsonl'
    catalog_rows={r['query_id']:r['text'] for r in jsonl_rows(catalog)}
    query_files=set();observation_names={'captures.jsonl','capture.jsonl','logits.jsonl','outer-predictions.jsonl','requests.jsonl','quality-replay.jsonl','predictions.jsonl','observations.jsonl'}
    for root in (a.history_root/'artifacts',repo/'artifacts'):
        for path in root.rglob('*.jsonl'):
            if path.name.startswith('queries') or path.name in observation_names:query_files.add(path.resolve())
    observed=set();known=set();source_hashes={}
    for path in sorted(query_files):
        source_hashes[str(path)]=sha(path)
        for row in jsonl_rows(path):
            q=row.get('query_id',row.get('qid'))
            if q is None:continue
            q=str(q);observed.add(q);text=row.get('text',row.get('query'))
            if isinstance(text,str) and text:known.add((q,text))
    for q in observed:
        if q in catalog_rows:known.add((q,catalog_rows[q]))
    known_ids={q for q,_ in known}
    for q in observed-known_ids:known.add((q,''))
    history=a.out/'observed-query-history.jsonl'
    history.write_text(''.join(json.dumps(dict(query_id=q,text=text),ensure_ascii=False)+'\n' for q,text in sorted(known)))
    history_manifest=a.out/'history-sources.json'
    history_manifest.write_text(json.dumps(dict(inputs=source_hashes,catalog=str(catalog.resolve()),catalog_sha256=sha(catalog),
        observed_ids=len(observed),history_rows=len(known),coverage='all queries*.jsonl and captured/scored/predicted HTTP query IDs in both artifact roots',
        quality_or_relevance_used=False),indent=2)+'\n')
    print('Frozen exposure inventory:',len(observed),'IDs;',len(query_files),'files;',len(set(catalog_rows)-observed),'catalog IDs remain before near-duplicate/answerability filtering',flush=True)
    packages=['torch','transformers','tokenizers','numpy','scikit-learn','qdrant-client','safetensors','pyarrow','scipy','joblib','threadpoolctl']
    environment=dict(python=sys.version,executable=sys.executable,dependencies={name:importlib.metadata.version(name) for name in packages},
        environment={k:os.environ.get(k) for k in ('PYTHONPATH','LD_LIBRARY_PATH','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','TRANSFORMERS_OFFLINE')},
        hardware=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip(),
        cgroup_limits={k:Path('/sys/fs/cgroup',k).read_text().strip() for k in ('memory.max','memory.high','cpu.max')})
    ep=a.out/'environment.json';ep.write_text(json.dumps(environment,indent=2)+'\n')
    index=Path(assets['index']);im=json.loads((index/'manifest.json').read_text());vm_path=Path(assets['vector_manifest']);vm=json.loads(vm_path.read_text())
    files={Path(path) for path in source_hashes}|{catalog,history,history_manifest,ep,a.base_freeze,vm_path,index/'manifest.json'}
    files.update(index/name for name in im['files']);files.update(vm_path.parent/name for name in vm['snapshot_files'])
    parent_action=next(Path(p).parent for p in study['protocol']['inputs'] if Path(p).name=='study.json')
    for root in (a.study,parent_action,a.final,a.http,a.smoke,a.engine):
        files.update(p for p in root.rglob('*') if p.is_file() and 'snapshots' not in p.parts)
    for p in (Path(assets['native_vectors'])/'vectors.npy',Path(assets['support_vectors'])):
        files.add(p);files.update(p.parent.glob('*.json'))
    files.add(a.assets/'assets.json')
    for protocol in (study['protocol'],final['protocol'],dict(inputs=training['inputs'])):
        files.update(Path(path) for path in protocol.get('inputs',{}));files.update(Path(path) for path in protocol.get('code_sha256',{}))
        for path in protocol.get('inputs',{}):
            if Path(path).name=='dataset.json':
                manifest=json.loads(Path(path).read_text());files.update(Path(path).parent/name for name in manifest['files'])
    for package in ('evaluation','online','offline','contracts'):files.update((repo/package).glob('*.py'))
    files.update((a.history_root/'adapters').glob('*.py'));files.add(repo/'evaluation/native_candidate_features.cpp')
    files.update(Path(runtime['feature_extension']).glob('*'));files.add(Path(runtime['qdrant_binary']));files.add(Path(runtime['previous_router']))
    for key in ('native_model','small_model','reranker_model'):
        root=Path(runtime[key]);inventory=root/'model_inventory.json';inv=json.loads(inventory.read_text());files.add(inventory)
        for name,digest in inv['files'].items():
            if sha(root/name)!=digest:raise ValueError('model file drift: '+str(root/name))
            files.add(root/name)
    # Bind the isolated binary runtime, including CUDA libraries and wrapper.
    deps=Path('/root/autodl-tmp/router-tensorrt-cu11-deps')
    files.update(p for p in deps.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update(Path('/root/miniconda3/lib/python3.10/site-packages/torch/lib').glob('*.so*'))
    from threadpoolctl import threadpool_info
    files.update(Path(r['filepath']) for r in threadpool_info())
    raw={k:base['confirmation'][k] for k in ('raw_queries','raw_qrels')};files.update(Path(path) for path in raw.values())
    raw_manifest_candidates=[a.history_root/'artifacts/round3/mmarco-data/manifest.json',
                             repo.parent/'NLP-SearchEngine'/'artifacts/round3/mmarco-data/manifest.json',
                             repo/'../NLP-SearchEngine/artifacts/round3/mmarco-data/manifest.json']
    raw_manifest=next((p for p in raw_manifest_candidates if p.exists()),None)
    if raw_manifest is None:raise ValueError('pinned MMarco raw manifest is missing')
    files.add(raw_manifest)
    pinned=json.loads(raw_manifest.read_text())['source_sha256']
    for path in raw.values():
        if sha(path)!=pinned[Path(path).name]:raise ValueError('pinned MMarco raw source changed')
    if vm['n_docs']!=106813 or im['n_docs']!=vm['n_docs']:raise ValueError('fixed MMarco corpus size differs')
    frozen={str(p.resolve()):sha(p) for p in files if p.is_file()}
    storage={str(p.resolve()):sha(p) for p in Path(assets['storage']).rglob('*') if p.is_file()}
    if not storage:raise ValueError('owned database storage is empty')
    result=dict(kind='bge-calibration-router-independent-confirmation-freeze-v1',recipe_kind='fully-nested-preBGE-final-v1',protocol_version=2,created_epoch=time.time(),
        files=frozen,storage_files_before_start=storage,history_sha256={str(history.resolve()):sha(history)},history_source_manifest=str(history_manifest.resolve()),
        environment_path=str(ep.resolve()),model_path=str((a.final/'study.json').resolve()),model_sha256=sha(a.final/'study.json'),
        online_scorer=encoder,development_study_sha256=sha(a.study/'study.json'),http_development_sha256=sha(a.http/'summary.json'),
        corpus=dict(index=str(index.resolve()),vector_manifest=str(vm_path.resolve()),storage=assets['storage'],native_vectors=assets['native_vectors'],support_vectors=assets['support_vectors'],
            documents=vm['n_docs'],corpus_sha256=vm['corpus_sha256']),runtime=runtime,
        confirmation=dict(directory=str(a.confirmation_dir.resolve()),hash_salt='rfc01-twenty-percent-MMARCO-nested-preBGE-seq160-2499-v1|',
            limit=256,minimum_queries=128,repeats=3,near_duplicate_threshold=.8,**raw,
            selection='fixed SHA256 order; fixed-corpus answerability only; exclude all observed IDs/text near duplicates; one use',
            source='fresh pinned MMarco queries after fully nested2499 preBGE-seq160 development; fixed106813-document corpus',labels_may_not_be_used_for_tuning=True,one_acceptance_run=True),
        acceptance=dict(ce_max_length=encoder['max_length'],ndcg_relative_lift_strictly_greater_than=.2,recall_relative_lift_strictly_greater_than=.2,paired_group_bootstrap_lower_bounds_positive=True,
            http_p95_ratio_max=1.05,require_nonconstant_confirmation=True,report_fixed_dense_and_previous_router=True),
        limitation='Repeated exploratory development on2499 MMarco queries. All final action/gate models are refitted on full development. Only a fresh once-consumed confirmation tests generalization; development success does not imply goal completion.',
        independent_confirmation_read=False,goal_complete=False)
    (a.out/'FREEZE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(freeze=str(a.out/'FREEZE.json'),bound_files=len(frozen),storage_files=len(storage),confirmation_selected=False),indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('study','final','http','smoke','engine','assets','base-freeze','history-root','confirmation-dir','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
