import json
import pytest

from contracts.index_format import sha256_of
from evaluation import train_dureader_policies as mod


def test_nested_oof_coverage_and_dev_predictions_frozen_before_labels(tmp_path,monkeypatch):
    def write_scores(name,n):
        p=tmp_path/name;p.mkdir();rows=[];qrels=[]
        for i in range(n):
            a,b=f'{name}-doc-{i}-a',f'{name}-doc-{i}-b'
            rows.append({'query_id':f'{name}-{i}','text':f'apple {i}',
                         'explicit_constraints':False,'pool':[a,b],'global_docids':[i*2,i*2+1],
                         'lexical':[{'doc_id':a,'score':10.},{'doc_id':b,'score':9.}],
                         'dense':[{'doc_id':b,'score':.9}],
                         'ce28':{'scores':[0.,1.],'pairs':2,'component_ms':1.},
                         'ce40':{'scores':[0.,1.],'pairs':2,'component_ms':2.}})
            qrels.append(f'{name}-{i} 0 {b} 1')
        (p/'scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (p/'protocol.json').write_text(json.dumps({'scores_sha256':sha256_of(p/'scores.jsonl'),
            'qrels_read':False,'queries':n,'corpus_sha256':'a'*64,'ce':{'fixture':True}}))
        q=tmp_path/(name+'.qrels');q.write_text('\n'.join(qrels)+'\n');return p,q
    train,tq=write_scores('train',16);dev,dq=write_scores('dev',4)
    stats=tmp_path/'stats';stats.write_text(json.dumps({'dfs':{'apple':2,**{str(i):1 for i in range(16)}}}))
    output=tmp_path/'out'; original=mod.qrels
    def guarded(path):
        if path==dq:
            assert (output/'dev-predictions.jsonl').exists()
            assert (output/'models.json').exists()
        return original(path)
    monkeypatch.setattr(mod,'qrels',guarded)
    monkeypatch.setattr(mod,'group_bootstrap',lambda v,g:{'mean':float(v.mean()),'ci95':[float(v.mean())]*2})
    result=mod.train(train,dev,tq,dq,stats,output)
    oof=[json.loads(l) for l in (output/'train-oof.jsonl').open()]
    assert len(oof)==16 and len({r['query_id'] for r in oof})==16
    folds=json.loads((output/'nested-selection.json').read_text())
    assert len(folds)==4
    for f in folds:assert not set(f['train_qids'])&set(f['validation_qids'])
    assert result['protocol']['dev_prediction_sha256_before_labels']==sha256_of(output/'dev-predictions.jsonl')
    assert result['random-20261113']['ce_pairs_mean']==result['learned-router']['ce_pairs_mean']
    assert result['fixed-pair-budget']['ce_component_ms'] is None
    assert result['learned-router']['e2e_http_ms'] is None
    from evaluation.audit_dureader_policies import audit
    audited=audit(train,dev,tq,dq,stats,output,tmp_path/'independent-audit')
    assert audited['all_aggregate_quality_metrics_recomputed']
    assert audited['outer_group_exclusion_verified']
    from evaluation.audit_dureader_inner_selection import audit as audit_inner
    inner=audit_inner(train,tq,stats,output,tmp_path/'inner-audit')
    assert inner['recorded_trials_independently_recomputed']==70
    assert inner['dev_labels_read']is False
    (output/'models.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact hash mismatch'):
        audit(train,dev,tq,dq,stats,output,tmp_path/'corrupt-audit')
    assert not(tmp_path/'corrupt-audit').exists()
