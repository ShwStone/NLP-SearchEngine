import json
import pytest
from contracts.index_format import sha256_of
from evaluation.audit_capture_parity import audit


def test_independent_ranking_parity_rejects_order_and_score_changes(tmp_path):
    def write(name,hits):
        p=tmp_path/name;p.mkdir()
        row={'query_id':'q','text':'apple','explicit_constraints':False,'lexical':hits}
        (p/'capture.jsonl').write_text(json.dumps(row)+'\n')
        (p/'protocol.json').write_text(json.dumps({'queries':1,'documents':2,'k':2,
            'queries_sha256':'a'*64,'capture_sha256':sha256_of(p/'capture.jsonl')}))
        return p
    hits=[{'docid':0,'doc_id':'a','score':1.},{'docid':1,'doc_id':'b','score':.5}]
    ref=write('ref',hits);same=write('same',hits)
    assert audit(ref,same,tmp_path/'ok')['ordered_candidates_checked']==2
    changed=write('changed',hits[::-1])
    with pytest.raises(ValueError,match='ranking mismatch'):audit(ref,changed,tmp_path/'bad')
    scores=write('scores',[dict(hits[0],score=1.1),hits[1]])
    with pytest.raises(ValueError,match='score mismatch'):audit(ref,scores,tmp_path/'badscore')
    assert not(tmp_path/'badscore').exists()
