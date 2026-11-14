import json
from contracts.index_format import sha256_of
from evaluation import evaluate_fixed_ce_control as mod


def test_fixed_control_uses_separate_scores_without_changing_router(tmp_path,monkeypatch):
    score=tmp_path/'score';score.mkdir();policy=tmp_path/'policy';policy.mkdir()
    row={'query_id':'q','text':'apple','pool':['a','b'],'global_docids':[0,1],
         'ce1':{'scores':[1.],'pairs':1,'component_ms':7.}}
    (score/'scores.jsonl').write_text(json.dumps(row)+'\n')
    (score/'protocol.json').write_text(json.dumps({'queries':1,'qrels_read':False,
        'scores_sha256':sha256_of(score/'scores.jsonl'),'budgets':[1]}))
    (policy/'summary.json').write_text(json.dumps({'protocol':{'fixed_pair_budget':1}}))
    (policy/'dev-predictions.jsonl').write_text(json.dumps({'query_id':'q','actions':{
        'learned-router':['b','a'],'fixed-pair-budget':['a','b']}})+'\n')
    labels=tmp_path/'labels';labels.write_text('q 0 b 1\n')
    monkeypatch.setattr(mod,'group_bootstrap',lambda v,g:{'mean':float(v.mean())})
    result=mod.evaluate(score,policy,labels,tmp_path/'out')
    assert result['protocol']['separately_scored']
    assert not result['protocol']['model_or_threshold_changed']
    assert result['ce_component_ms']['p50']==7.
    assert result['fixed_ce_pairs']==1
    assert result['changed_rankings_vs_reused_ce40_prefix']==0
