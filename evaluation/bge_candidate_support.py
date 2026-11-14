"""Label-free semantic support anchored exclusively by the current BGE scores."""
import numpy as np

from .prepare_candidate_support import FEATURE_NAMES,support_features_fast


def bge_support(candidates,scores,lexical,dense,vectors):
    if len(candidates)!=len(scores) or len(set(candidates))!=len(candidates):
        raise ValueError('BGE support requires unique aligned candidate scores')
    best=sorted(range(len(candidates)),key=lambda i:(-scores[i],candidates[i]))[:5]
    anchors=[(rank,candidates[i]) for rank,i in enumerate(best)]
    count=len(anchors);dense_slot=lexical_slot=None
    if dense:dense_slot=len(anchors);anchors.append((0,dense[0]))
    if lexical:lexical_slot=len(anchors);anchors.append((0,lexical[0]))
    sim=vectors[list(candidates)]@vectors[[d for _,d in anchors]].T
    return support_features_fast(sim,candidates,anchors,count,dense_slot=dense_slot,lexical_slot=lexical_slot)
