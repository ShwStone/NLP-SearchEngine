import io,json,tarfile,hashlib
import pytest
from evaluation import audit_official_full_source as mod


def test_official_archive_extracted_files_and_normalization_all_match(tmp_path,monkeypatch):
    root=tmp_path/'official';(root/'passage-collection').mkdir(parents=True);(root/'dev').mkdir()
    mapping={str(i):f'id-{i}'for i in range(4)}
    (root/'passage-collection/passage2id.map.json').write_text(json.dumps(mapping))
    for i in range(4):
        (root/f'passage-collection/part-{i:02d}').write_text(f'{i}\ttitle\t正文{i}\n', encoding='utf-8')
    for name in ('dev.json','q2qid.dev.json'):(root/'dev'/name).write_text('{}')
    archive=tmp_path/'archive.tar.gz'
    with tarfile.open(archive,'w:gz')as tar:
        for path in root.rglob('*'):
            if path.is_file():tar.add(path,arcname='dureader-retrieval-baseline-dataset/'+str(path.relative_to(root)))
    monkeypatch.setattr(mod,'ARCHIVE_SHA256',hashlib.sha256(archive.read_bytes()).hexdigest())
    corpus=tmp_path/'corpus'
    rows=[{'doc_id':f'id-{i}','title':'','text':f'正文{i}'}for i in range(4)]
    corpus.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    result=mod.audit(archive,root,corpus,tmp_path/'ok',expected_documents=4)
    assert result['all_official_normalized_rows_verified']
    rows[2]['text']='changed';corpus.write_text(''.join(json.dumps(r)+'\n'for r in rows))
    with pytest.raises(ValueError,match='normalized row mismatch'):
        mod.audit(archive,root,corpus,tmp_path/'bad-row',expected_documents=4)
    (root/'passage-collection/part-00').write_text('modified')
    with pytest.raises(ValueError,match='member mismatch'):
        mod.audit(archive,root,corpus,tmp_path/'bad-extract',expected_documents=4)
    assert not(tmp_path/'bad-row').exists()
    assert not(tmp_path/'bad-extract').exists()
