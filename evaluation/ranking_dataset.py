"""Checked development matrices for genuinely nested action/Router fitting.

These contain raw frozen features and scoped development labels, never global
OOF supervised predictions. Each outer fold must fit its own action models.
"""

import json
from pathlib import Path

import numpy as np

from .prepare_bm25_router import sha


def write_dataset(out, x, y, offsets, docs, query_ids, groups, relevance,
                  identities, feature_names, table):
    out = Path(out)
    np.save(out/'features.npy', x, allow_pickle=False)
    np.save(out/'labels.npy', y, allow_pickle=False)
    metadata = dict(query_ids=query_ids, groups=groups.tolist(), offsets=list(offsets),
        docs=docs, relevance={q:relevance[q] for q in query_ids},
        identities={d:identities[d] for row in docs for d in row},
        feature_names=feature_names, bm25_feature_names=sorted(table[query_ids[0]]['features']),
        bm25_features=[table[q]['features'] for q in query_ids],
        baselines={name:[table[q][name] for q in query_ids] for name in ('lexical','dense','hybrid')})
    (out/'dataset-metadata.json').write_text(json.dumps(metadata)+'\n')
    manifest = dict(kind='frozen-candidate-feature-dataset-v1', confirmation_read=False,
        supervised_predictions_in_features=False,
        files={name:sha(out/name) for name in ('features.npy','labels.npy','dataset-metadata.json','protocol.json')})
    (out/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')


def load_dataset(path):
    path = Path(path)
    manifest = json.loads((path/'dataset.json').read_text())
    if (manifest['kind']!='frozen-candidate-feature-dataset-v1' or
            manifest.get('confirmation_read') is not False or
            manifest.get('supervised_predictions_in_features') is not False):
        raise ValueError('invalid raw candidate dataset provenance')
    for name in ('features.npy','labels.npy','dataset-metadata.json','protocol.json'):
        if manifest['files'].get(name)!=sha(path/name):
            raise ValueError('candidate dataset content hash mismatch')
    data = json.loads((path/'dataset-metadata.json').read_text())
    x = np.load(path/'features.npy',mmap_mode='r',allow_pickle=False)
    y = np.load(path/'labels.npy',mmap_mode='r',allow_pickle=False)
    offsets = np.array(data['offsets'],dtype=int)
    count = len(data['query_ids'])
    if (len(set(data['query_ids']))!=count or len(data['groups'])!=count or len(data['docs'])!=count
            or len(offsets)!=count+1 or offsets[0]!=0 or offsets[-1]!=len(x)
            or (np.diff(offsets)<=0).any() or y.shape!=(len(x),)
            or x.shape!=(len(y),len(data['feature_names'])) or not np.isfinite(x).all()):
        raise ValueError('invalid candidate dataset dimensions')
    for i, docs in enumerate(data['docs']):
        if len(docs)!=offsets[i+1]-offsets[i] or len(set(docs))!=len(docs):
            raise ValueError('invalid candidate ordering')
        expected=np.array([data['relevance'][data['query_ids'][i]].get(d,0)>0 for d in docs])
        if not np.array_equal(y[offsets[i]:offsets[i+1]],expected):
            raise ValueError('candidate labels differ from scoped relevance')
    return dict(x=x,y=y,offsets=offsets,sizes=np.diff(offsets),
                groups=np.array(data['groups']),metadata=data,manifest_sha256=sha(path/'dataset.json'))
