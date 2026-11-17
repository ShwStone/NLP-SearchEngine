"""Optional Numba decoder for compact-v1 variable-length posting blocks.

The pure-Python decoder in packed_index.py remains the fallback. This module is
imported lazily so the index and HTTP service retain their zero-dependency mode.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _decode(data, df, n_docs):
    docids = np.empty(df, dtype=np.uint32)
    frequencies = np.empty(df, dtype=np.uint32)
    starts = np.empty(df, dtype=np.uint32)
    i = 0
    last = -1
    found = 0
    for row in range(df):
        if i >= len(data):
            break
        if i + 2 > len(data):
            raise ValueError("truncated posting header")
        docid = int(data[i])
        tf = int(data[i + 1])
        i += 2
        if not (last < docid < n_docs) or tf < 1 or i + tf > len(data):
            raise ValueError("invalid compact posting block")
        docids[row] = docid
        frequencies[row] = tf
        starts[row] = i
        found += 1
        i += tf
        last = docid
    if found != df or i != len(data):
        raise ValueError("compact posting count mismatch")
    return docids, frequencies, starts


def decode(data, df, n_docs):
    """Decode one immutable array('I') block without per-posting Python work."""
    raw = np.frombuffer(data, dtype=np.uint32)
    return _decode(raw, df, n_docs)
