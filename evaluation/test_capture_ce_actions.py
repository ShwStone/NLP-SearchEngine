import json
import pytest
from contracts.index_format import sha256_of
from evaluation import capture_ce_actions as mod


def test_independent_budget_execution_and_corpus_identity(tmp_path,monkeypatch):
    corpus=tmp_path/'corpus'
    corpus.write_text(''.join(json.dumps({'doc_id':str(i),'title':'','text':f'text-{i}'})+'\n'for i in range(3)))
    source=tmp_path/'capture';source.mkdir()
    row={'query_id':'q','text':'apple','explicit_constraints':False,'lexical':[], 'dense':[],
         'hybrid':[{'doc_id':str(i),'docid':i,'score':1.}for i in range(3)]}
    (source/'capture.jsonl').write_text(json.dumps(row)+'\n')
    (source/'protocol.json').write_text(json.dumps({'queries':1,'documents':3,'queries_sha256':'a'*64,
        'capture_sha256':sha256_of(source/'capture.jsonl')}))
    calls=[]
    class Scorer:
        info={'fixture':True}
        def __init__(self,*args,**kwargs):pass
        def score(self,query,texts):
            calls.append(len(texts));return list(range(len(texts))),float(len(texts))
    monkeypatch.setattr(mod,'FrozenCrossEncoder',Scorer)
    result=mod.score(source,corpus,tmp_path/'model',tmp_path/'scored',sha256_of(corpus),budgets='1,2')
    assert calls==[1,2,1,2]
    assert result['budgets']==[1,2]and result['qrels_read']is False
    saved=json.loads((tmp_path/'scored/scores.jsonl').read_text())
    assert saved['ce1']['pairs']==1 and saved['ce2']['pairs']==2
    assert saved['ce1']['component_ms']==1. and saved['ce2']['component_ms']==2.
    with pytest.raises(ValueError,match='corpus identity'):
        mod.score(source,corpus,tmp_path/'model',tmp_path/'invalid','0'*64,budgets='2')
    assert not(tmp_path/'invalid').exists()
