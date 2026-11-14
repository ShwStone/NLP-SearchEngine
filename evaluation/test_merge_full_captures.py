import json

import pytest

from contracts.index_format import sha256_of
from evaluation.merge_full_captures import merge


def test_keyed_merge_global_ties_constraints_and_stale_identity(tmp_path):
    def write(name, mode, reverse=False):
        p = tmp_path/name; p.mkdir()
        rows = [{'query_id': 'q1', 'text': '苹果', 'explicit_constraints': False,
                 mode: [{'docid': i, 'doc_id': str(i), 'score': 3-i} for i in ([2,1] if mode=='dense' else [1,2])]},
                {'query_id': 'q2', 'text': '苹果 AND 香蕉', 'explicit_constraints': True,
                 mode: [] if mode=='dense' else [{'docid': 0,'doc_id': '0','score': 1.}]}]
        if reverse: rows.reverse()
        (p/'capture.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        proto={'queries':2,'documents':3,'k':2,'queries_sha256':'a'*64,'qrels_read':False,
               'capture_sha256':sha256_of(p/'capture.jsonl')}
        (p/'protocol.json').write_text(json.dumps(proto));return p
    l=write('lexical','lexical'); d=write('dense','dense',True)
    out=tmp_path/'merged'; result=merge(l,d,out)
    rows=[json.loads(x) for x in (out/'capture.jsonl').read_text().splitlines()]
    assert [h['docid'] for h in rows[0]['hybrid']]==[1,2]
    assert rows[1]['hybrid']==rows[1]['lexical']
    assert result['qrels_read'] is False
    with (d/'capture.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='SHA256'):
        merge(l,d,tmp_path/'corrupt')
    assert not (tmp_path/'corrupt').exists()
