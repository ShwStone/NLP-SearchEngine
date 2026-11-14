"""Load audited RFC4 reader sources explicitly; no silent checkout fallback."""
import importlib
import math
from pathlib import Path
import sys
import numpy as np

from contracts.index_format import sha256_of
from online.compiled_bm25 import CompiledBM25

READER_HASHES={
    'compact.py':'ddf5c37dfce30e4e9a4c24543a7321fa51bc1c67f33ceff14626fc984d23dc16',
    'packed_index.py':'3bf65da7a4c0b2b19c94d64cf8e703c79868c67038ccede4778ef0bbfaaeea06',
}


def load_reader(source,compact):
    import offline
    directory=Path(source).resolve()/'offline'
    for name,digest in READER_HASHES.items():
        if sha256_of(directory/name)!=digest:raise ValueError('frozen reader source mismatch: '+name)
    names = ('offline.compact', 'offline.packed_index')
    for name in names:
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != directory:
            raise ValueError('unrelated compact reader already imported')
    offline.__path__.insert(0, str(directory))
    try:
        for name in names:
            module = importlib.import_module(name)
            if Path(module.__file__).resolve().parent != directory:
                raise ValueError('unrelated compact reader already imported')
        return module.PackedIndex(compact, cache_bytes=64*1024*1024)
    finally:
        offline.__path__.remove(str(directory))


class BoundedPackedBM25(CompiledBM25):
    """Reference-order float64 scorer with bounded immutable posting cache.

    Numeric arrays reference cached packed blocks during each query only.
    The cache belongs to PackedIndex; no full-vocabulary NumPy cache grows.
    """
    def _posting(self,term):
        if hasattr(self.reader,'_block'):
            block=self.reader._block(term)
            if block is None:return None
            ids=np.frombuffer(block.docids,dtype=np.uint32)
            tf=np.frombuffer(block.frequencies,dtype=np.uint32).astype(np.float64)
        else:
            rows=self.reader.postings(term)
            if not rows:return None
            ids=np.fromiter((r[0] for r in rows),dtype=np.int64,count=len(rows))
            tf=np.fromiter((r[1] for r in rows),dtype=np.float64,count=len(rows))
        idf=math.log(1+(self.reader.n_docs-self.reader.df(term)+.5)/(self.reader.df(term)+.5))
        return ids,tf,idf
