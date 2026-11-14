import numpy as np
import pytest

from evaluation.dureader_policy_features import gate_features,query_groups,rank_features,rerank,FrozenLinear


def test_connected_query_groups_and_rerank_features_do_not_read_labels():
    rows=[{'query_id':'a','text':' A B '},{'query_id':'b','text':'ＡＢ'},
          {'query_id':'c','text':'other'},{'query_id':'d','text':'unique'}]
    labels={'a':{'x':1},'b':{'y':1},'c':{'y':1},'d':{'z':1}}
    groups=query_groups(rows,labels)
    assert groups[0]==groups[1]==groups[2]
    assert groups[3]!=groups[0]
    r={'text':'apple','lexical':[{'doc_id':'a','score':10},{'doc_id':'b','score':5}],
       'dense':[{'doc_id':'b','score':.9}], 'pool':['a','b'], 'global_docids':[8,2],
       'ce28':{'scores':[1.,1.]}}
    f=gate_features(r,{'apple':2})
    assert f.tolist()==[1.,1.,.5,2.]
    with pytest.raises(ValueError,match='missing'):
        gate_features(r,{})
    features=rank_features(r,28)
    assert features.shape==(2,6)
    assert features[0].tolist()==[1.,1.,1.,0.,1.,0.]
    assert rerank(r,28)==['b','a']
    r['ce28']['scores']=[1.]
    with pytest.raises(ValueError,match='cardinality'):
        rank_features(r,28)


def test_frozen_linear_rejects_invalid_scaling_and_feature_dimensions():
    spec={'mean':[1.,2.],'scale':[2.,4.],'coef':[3.,5.],'intercept':7.}
    model=FrozenLinear(spec)
    assert model.predict([[3.,6.]]).tolist()==[15.]
    with pytest.raises(ValueError):FrozenLinear(dict(spec,scale=[0.,4.]))
    with pytest.raises(ValueError):model.predict([[1.]])
    with pytest.raises(ValueError):model.predict([[float('nan'),1.]])
