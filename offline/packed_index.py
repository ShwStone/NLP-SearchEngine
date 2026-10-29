"""Optional compact-v1 reader that avoids decoding unused position lists.

The cache budget covers retained blocks and their keys, not in-flight requests,
Python's allocator, document buffers, SQLite or total process RSS. A single block
larger than the budget is served but not retained. Concurrent requests for one
term share one load, including an oversized block.
"""

from array import array
from bisect import bisect_left
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
import os
import sys
import threading

from .compact import CompactIndex, _pread, integers


@dataclass(frozen=True)
class _Block:
    data: array
    docids: array
    frequencies: array
    starts: array

    def charge(self,term):
        return (sys.getsizeof(self)+sys.getsizeof(self.__dict__)+sys.getsizeof(term)
                +sum(sys.getsizeof(x) for x in (self.data,self.docids,self.frequencies,self.starts)))

    def positions(self,docid):
        i=bisect_left(self.docids,docid)
        if i==len(self.docids) or self.docids[i]!=docid:return []
        start=self.starts[i]
        return list(self.data[start:start+self.frequencies[i]])


class PackedIndex(CompactIndex):
    storage_format='compact-packed-v1'

    def __init__(self,root,cache_bytes=64*1024*1024):
        if type(cache_bytes) is not int or cache_bytes<0:raise ValueError('cache_bytes must be a nonnegative integer')
        super().__init__(root,cache_terms=0)
        # Keep the required list-returning interface without a second decoded cache.
        del self.postings
        self._budget=cache_bytes;self._cached_bytes=0
        self._blocks=OrderedDict();self._pending={};self._cache_lock=threading.RLock()
        self._stats=dict(hits=0,misses=0,coalesced=0,loads=0,evictions=0,oversized=0)

    def _load_block(self,term):
        row=self._term(term)
        if row is None:return None
        start,size,df=row;raw=_pread(self._files['postings.bin'],size,start)
        if len(raw)!=size:raise ValueError('truncated compact posting block')
        data=integers(raw);docids=array('I');frequencies=array('I');starts=array('I');i=0;last=-1
        while i<len(data):
            if i+2>len(data):raise ValueError('truncated posting header')
            docid,tf=data[i],data[i+1];i+=2
            if not last<docid<self.n_docs or tf<1 or i+tf>len(data):raise ValueError('invalid compact posting block')
            docids.append(docid);frequencies.append(tf);starts.append(i);i+=tf;last=docid
        if len(docids)!=df:raise ValueError('compact posting count mismatch')
        return _Block(data,docids,frequencies,starts)

    def _block(self,term):
        with self._cache_lock:
            if self._closed:raise ValueError('reader is closed')
            if term in self._blocks:
                self._stats['hits']+=1;self._blocks.move_to_end(term)
                return self._blocks[term][0]
            future=self._pending.get(term)
            owner=future is None
            if owner:
                future=Future();self._pending[term]=future;self._stats['misses']+=1
            else:self._stats['coalesced']+=1
        if not owner:return future.result()
        try:
            block=self._load_block(term)
            with self._cache_lock:
                self._stats['loads']+=1
                if block is not None and not self._closed:
                    charge=block.charge(term)
                    if charge<=self._budget:
                        while self._blocks and self._cached_bytes+charge>self._budget:
                            _,(_,removed)=self._blocks.popitem(last=False)
                            self._cached_bytes-=removed;self._stats['evictions']+=1
                        self._blocks[term]=(block,charge);self._cached_bytes+=charge
                    else:self._stats['oversized']+=1
            future.set_result(block)
            return block
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._cache_lock:self._pending.pop(term,None)

    def term_frequencies(self,term):
        """Optional fast path for scorers; preserves docid and summation order."""
        block=self._block(term)
        return zip(block.docids,block.frequencies) if block else iter(())

    def positions_for(self,term,docid):
        block=self._block(term)
        return block.positions(docid) if block else []

    def prepare_positions(self,terms):
        """Pin each term once per render, including non-retained oversized blocks."""
        blocks={term:self._block(term) for term in terms}
        def lookup(term,docid):
            block=blocks[term]
            return block.positions(docid) if block else []
        return lookup

    def postings(self,term):
        block=self._block(term)
        if block is None:return None
        return [(docid,tf,list(block.data[start:start+tf]))
                for docid,tf,start in zip(block.docids,block.frequencies,block.starts)]

    def cache_info(self):
        with self._cache_lock:
            return dict(self._stats,cached_bytes=self._cached_bytes,budget_bytes=self._budget,
                        terms=len(self._blocks),pending=len(self._pending))

    def clear_cache(self):
        with self._cache_lock:
            if self._pending:raise RuntimeError('cannot clear cache while terms are loading')
            self._blocks.clear();self._cached_bytes=0

    def close(self):
        if not hasattr(self,'_cache_lock'):return super().close()
        with self._cache_lock:
            if self._pending:raise RuntimeError('stop requests before closing reader')
            if not self._closed:
                self._blocks.clear();self._cached_bytes=0;self._term.cache_clear();self._db.close()
                for fd in self._files.values():os.close(fd)
                self._closed=True
