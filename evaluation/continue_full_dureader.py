"""Continue full-corpus capture->fusion->CE->nested evaluation sequentially.

Consumes only completed hashed protocols. Missing output is waited on only
while the explicitly identified producer remains live. No automatic shutdown,
PR publication, or restart of an incomplete upstream job is performed here.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from contracts.index_format import sha256_of
from evaluation.merge_full_captures import read_capture
from evaluation.train_dureader_policies import read_scores


def wait_protocol(directory,pid,command_token):
    directory=Path(directory)
    while not(directory/'protocol.json').exists():
        proc=Path(f'/proc/{pid}/cmdline')
        if not proc.exists() or command_token.encode() not in proc.read_bytes():
            raise RuntimeError(f'producer {pid} stopped before publishing {directory}')
        print('waiting for live producer',pid,directory,flush=True)
        time.sleep(15)
    return read_capture(directory)


def execute(module,kwargs):
    cmd=[sys.executable,'-u','-m',module]
    for key,value in kwargs.items():cmd.extend(['--'+key,str(value)])
    print('stage command',json.dumps(cmd),flush=True)
    subprocess.run(cmd,check=True)


def run(root,train_capture_pid,dev_supervisor_pid):
    root=Path(root)
    state=root/'dureader-full-experiment-v1'
    state.mkdir(exist_ok=True)
    # One orchestrator at a time; completed artifacts are reused only with
    # independent hash verification, never overwritten with partial outputs.
    import fcntl
    with(state/'.pipeline.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for split,pid,token in [('train',train_capture_pid,'evaluation.capture_compact_bm25'),
                                ('dev',dev_supervisor_pid,'evaluation.capture_compact_bm25')]:
            lexical=root/f'dureader-full-compact-bm25-{split}-v1'
            wait_protocol(lexical,pid,token)
            dense=root/f'dureader-full-dense-capture-{split}-v1'
            read_capture(dense)
            merged=state/f'{split}-merged'
            if not merged.exists():
                execute('evaluation.merge_full_captures',{'lexical-dir':lexical,'dense-dir':dense,'output':merged})
            else:read_capture(merged)
            scores=state/f'{split}-ce'
            if not scores.exists():
                execute('evaluation.capture_ce_actions',{'capture-dir':merged,
                    'corpus':root/'dureader-full-corpus-v1.jsonl',
                    'model':root/'NLP-SearchEngine/.runtime/models/bge-reranker-base',
                    'output':scores,'expected-corpus-sha256':'bc67f3ccfe7423c039f04c7a5e7f6eae07a7f722087edf0caa7cfc694a8c45e8'})
            else:read_scores(scores)
            (state/f'{split}-stage-complete.json').write_text(json.dumps({
                'merged_protocol_sha256':sha256_of(merged/'protocol.json'),
                'ce_protocol_sha256':sha256_of(scores/'protocol.json')},indent=2))
        statistics=state/'combined-query-statistics.json'
        tr=json.loads((root/'dureader-full-global-statistics-train-v1.json').read_text())
        dv=json.loads((root/'dureader-full-global-statistics-dev-v1.json').read_text())
        if tr['n_docs']!=dv['n_docs'] or tr['shards']!=dv['shards']:
            raise ValueError('train/dev global statistics corpus identities differ')
        overlap=set(tr['dfs'])&set(dv['dfs'])
        if any(tr['dfs'][t]!=dv['dfs'][t]for t in overlap):raise ValueError('global df mismatch')
        combined={'dfs':dict(tr['dfs'],**dv['dfs']),'n_docs':tr['n_docs'],
                  'source_sha256':[sha256_of(root/f'dureader-full-global-statistics-{s}-v1.json')for s in ['train','dev']]}
        if statistics.exists():
            if json.loads(statistics.read_text())!=combined:raise ValueError('combined statistics changed')
        else:statistics.write_text(json.dumps(combined,indent=2))
        output=state/'nested-policies'
        if not output.exists():
            execute('evaluation.train_dureader_policies',{'train-scores':state/'train-ce','dev-scores':state/'dev-ce',
                'train-qrels':root/'dureader-mini-100k-v1.0/qrels.train.tsv',
                'dev-qrels':root/'dureader-mini-100k-v1.0/qrels.dev.tsv',
                'statistics':statistics,'output':output})
        if not(output/'summary.json').exists():raise RuntimeError('nested experiment incomplete')
        summary=json.loads((output/'summary.json').read_text())
        if summary['protocol']['dev_queries']!=1020 or summary['protocol']['train_queries']!=172:
            raise ValueError('incomplete full nested query coverage')
        (state/'offline-stages-complete.json').write_text(json.dumps({
            'status':'offline_stages_complete','summary_sha256':sha256_of(output/'summary.json'),
            'full_goal_complete':False,'http_and_independent_parity_pending':True},indent=2))
        print('full offline stages complete; HTTP, independent audit and PR delivery remain',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--train-capture-pid',type=int,required=True)
    p.add_argument('--dev-supervisor-pid',type=int,required=True)
    run(**vars(p.parse_args()))
