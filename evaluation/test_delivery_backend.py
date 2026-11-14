import pytest
from evaluation.delivery_backend import BoundedPackedBM25
from online.memory_index import MemoryIndex
from online.scorers import bm25


def test_bounded_scoring_matches_reference_repeated_terms_and_filters():
    reader=MemoryIndex.from_corpus([{'doc_id':str(i),'title':'','text':s}
        for i,s in enumerate(['apple apple banana','banana','apple pear','pear banana'])])
    scorer=BoundedPackedBM25(reader)
    for terms in [['apple'],['apple','apple','banana'],['missing'],[]]:
        for kwargs in [{},{'candidate_docs':{1,2}},{'term_weights':{'apple':-1.}}]:
            expected=sorted(bm25(reader,terms,**kwargs).items(),key=lambda h:(-h[1],h[0]))[:2]
            actual=scorer.topk(terms,2,**kwargs)
            assert [d for d,_ in actual]==[d for d,_ in expected]
            assert [s for _,s in actual]==pytest.approx([s for _,s in expected],abs=1e-12)
    assert not scorer.postings
