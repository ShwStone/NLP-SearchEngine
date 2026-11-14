import json

import numpy as np

from evaluation.prece_router import (
    FEATURE_NAMES,
    FrozenScalarTree,
    FrozenTreeEnsemble,
    extract_features,
)


def test_pre_ce_features_have_frozen_schema_and_handle_empty_pools():
    row = {
        'text': '心肌梗死 的 治疗',
        'lexical': [],
        'dense': [],
        'pool': [],
    }
    values = extract_features(row)
    assert values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(values).all()
    assert values[-1] == 200.0


def test_frozen_tree_inference_is_portable(tmp_path):
    spec = {
        'threshold': 0.0,
        'feature_names': list(FEATURE_NAMES),
        'trees': [{
            'feature': [0, -2, -2],
            'threshold': [1.0, -2.0, -2.0],
            'left': [1, -1, -1],
            'right': [2, -1, -1],
            'value': [0.0, 0.25, 0.75],
        }],
    }
    path = tmp_path / 'tree.json'
    path.write_text(json.dumps(spec))
    model = FrozenTreeEnsemble.load(path)
    assert model.predict_one(np.zeros(len(FEATURE_NAMES))) == 0.25
    assert model.predict_one(np.ones(len(FEATURE_NAMES)) * 2) == 0.75

    scalar_path = tmp_path / 'scalar.json'
    scalar_path.write_text(json.dumps({
        'threshold': 0.0,
        'trees': [{
            'feature': [0, -2, -2],
            'threshold': [1.0, -2.0, -2.0],
            'left': [1, -1, -1],
            'right': [2, -1, -1],
            'value': [0.0, 0.1, 0.9],
        }],
    }))
    scalar = FrozenScalarTree.load(scalar_path)
    assert scalar.predict_one(0.0) == 0.1
    assert scalar.predict_one(2.0) == 0.9
