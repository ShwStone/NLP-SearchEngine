"""Offline group preflight, atomic build and lightweight document-source checks."""

import json
from pathlib import Path
import tempfile

from contracts import analyzer
from offline.build import build
from offline.disk_documents import DiskDocuments
from offline.prepare import prepare


ROOT=Path(__file__).resolve().parents[1]


def _raises(kind, fn):
    try: fn()
    except kind:return
    raise AssertionError('expected '+kind.__name__)


def test_prepare_stable_ids_and_explicit_exclusions():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);source=root/'source.jsonl'
        source.write_text('\n'.join([
            json.dumps({'doc_id':'a','title':'A','text':'心肌梗死'}),
            json.dumps({'doc_id':'bad','title':'B','text':''}),
            '{bad json',
            json.dumps({'doc_id':'b','title':'B','text':'phone water'}),
        ])+'\n',encoding='utf-8')
        _raises(ValueError,lambda:prepare(source,root/'strict'))
        report=prepare(source,root/'prepared',invalid='skip')
        assert report['source_records']==4 and report['accepted']==2 and report['excluded']==2
        rows=[json.loads(x) for x in (root/'prepared/corpus.jsonl').read_text().splitlines()]
        mapping=[json.loads(x) for x in (root/'prepared/id_map.jsonl').read_text().splitlines()]
        assert [r['doc_id'] for r in rows]==['a','b']
        assert mapping==[{'doc_id':'a','docid':0,'source_line':1},{'doc_id':'b','docid':1,'source_line':4}]
        reasons={json.loads(x)['reason'] for x in (root/'prepared/excluded.jsonl').read_text().splitlines()}
        assert reasons=={'empty_body','invalid_json'}
        _raises(ValueError,lambda:prepare(source,root/'prepared'))


def test_prepare_duplicate_ids_are_always_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        source=Path(tmp)/'source.jsonl';source.write_text('\n'.join([
            json.dumps({'doc_id':'same','title':'','text':'a'}),
            json.dumps({'doc_id':'same','title':'','text':'b'}),
        ])+'\n')
        _raises(ValueError,lambda:prepare(source,Path(tmp)/'out',invalid='skip'))


def test_streaming_build_is_atomic_and_document_source_is_narrow():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);corpus=root/'corpus.jsonl';corpus.write_text(
            ''.join(json.dumps({'doc_id':f'd{i}','title':'','text':text},ensure_ascii=False)+'\n'
                    for i,text in enumerate(('心肌梗死','phone water','camera lens'))))
        out=root/'index';manifest=build(corpus,out,'offline-test')
        assert manifest.n_docs==3 and (out/'manifest.json').is_file()
        with DiskDocuments(out) as source:
            assert source.n_docs==3
            assert source.doc(0).doc_id=='d0' and source.doc(0).offsets==[]
            assert source.doc(2).text=='camera lens'
        _raises(ValueError,lambda:build(corpus,out,'overwrite'))
        broken=root/'broken.jsonl';broken.write_text('{bad\n')
        target=root/'broken-index'
        _raises(ValueError,lambda:build(broken,target,'broken'))
        assert not target.exists()


def test_build_matches_analyzer_token_lengths_and_reproducible_files():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);corpus=root/'corpus.jsonl';rows=[{'doc_id':'a','title':'','text':'😀心肌梗死'}]
        corpus.write_text(json.dumps(rows[0],ensure_ascii=False)+'\n')
        first=build(corpus,root/'one','test');second=build(corpus,root/'two','test')
        assert first.to_dict()==second.to_dict()
        for name in ('docs.jsonl','postings.jsonl','offsets.jsonl','stats.json'):
            assert (root/'one'/name).read_bytes()==(root/'two'/name).read_bytes()
        assert first.n_docs==1
        assert first.avgdl==len(analyzer.analyze(rows[0]['text']))


ALL_TESTS=[value for name,value in sorted(globals().items()) if name.startswith('test_')]
