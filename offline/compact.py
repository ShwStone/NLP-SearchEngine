"""Validated, atomic conversion of lexical v1 JSONL to a compact disk reader.

The source lexical contract is unchanged; this is a separate derived artifact.
No numpy/SDK dependency. Read-only files plus bounded decoded-postings caching.
"""

import argparse
from array import array
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import struct
import sys
import tempfile
import threading
import time

from contracts import analyzer
from contracts.index_format import load_manifest, verify_files, sha256_of, REQUIRED_FILES
from contracts.reader import Document

PAIR = struct.Struct('<QQ')
_PREAD_LOCK = threading.RLock()


def _pread(fd: int, size: int, offset: int) -> bytes:
    """Portable positional read; Windows Python lacks ``os.pread``."""

    if hasattr(os, "pread"):
        return os.pread(fd, size, offset)
    with _PREAD_LOCK:
        os.lseek(fd, offset, os.SEEK_SET)
        return os.read(fd, size)


def u32(values):
    data = array('I', values)
    if data.itemsize != 4:
        raise RuntimeError('requires 32-bit unsigned int')
    if sys.byteorder != 'little':
        data.byteswap()
    return data.tobytes()


def integers(data):
    values = array('I')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def compile_index(source, target):
    source, target = Path(source), Path(target)
    manifest = load_manifest(source)
    if set(REQUIRED_FILES) - set(manifest.files):
        raise ValueError('source manifest requires all file checksums')
    verify_files(source, manifest)
    if manifest.analyzer_hash != analyzer.config_hash() or manifest.n_docs <= 0:
        raise ValueError('source analyzer/document count mismatch')
    if target.exists() and any(target.iterdir()):
        raise ValueError('compact target must be fresh')
    target.parent.mkdir(parents=True, exist_ok=True)
    stats = json.loads((source/'stats.json').read_text(encoding='utf-8'))
    lengths = stats['doc_lengths']
    if set(lengths) != {str(i) for i in range(manifest.n_docs)}:
        raise ValueError('source document lengths incomplete')
    lengths = [lengths[str(i)] for i in range(manifest.n_docs)]
    if any(type(n) is not int or n <= 0 for n in lengths):
        raise ValueError('compact source requires positive token lengths')
    if not math.isclose(sum(lengths)/len(lengths), manifest.avgdl, rel_tol=1e-9) or not math.isclose(stats['avgdl'],manifest.avgdl,rel_tol=1e-9):
        raise ValueError('source length statistics mismatch')
    with tempfile.TemporaryDirectory(prefix='.compact-', dir=target.parent) as tmp:
        stage = Path(tmp)
        ids, char_lengths, count = set(), [], 0
        with (source/'docs.jsonl').open('rb') as docs, (stage/'docs.jsonl').open('wb') as out, (stage/'docs.idx').open('wb') as index:
            for count, line in enumerate(docs, 1):
                row = json.loads(line)
                if (type(row['docid']) is not int or row['docid'] != count-1
                        or not isinstance(row['doc_id'],str) or not row['doc_id'] or row['doc_id'] in ids
                        or not isinstance(row['text'],str) or not isinstance(row['title'],str)):
                    raise ValueError('invalid/duplicate source document')
                ids.add(row['doc_id']);char_lengths.append(len(row['text']))
                index.write(PAIR.pack(out.tell(),len(line)));out.write(line)
        if count != manifest.n_docs:
            raise ValueError('source document count mismatch')
        del ids
        count = 0
        with (source/'offsets.jsonl').open(encoding='utf-8') as offsets, (stage/'offsets.bin').open('wb') as out, (stage/'offsets.idx').open('wb') as index:
            for count, line in enumerate(offsets, 1):
                row = json.loads(line);docid=count-1;spans=row['spans']
                if type(row['docid']) is not int or row['docid'] != docid or docid >= manifest.n_docs or len(spans)!=lengths[docid]:
                    raise ValueError('source offset count/order mismatch')
                flat=[]
                for i, span in enumerate(spans):
                    if (not isinstance(span,list) or len(span)!=4 or any(type(v) is not int for v in span)
                            or span[:2]!=[i,i+1] or not 0<=span[2]<span[3]<=char_lengths[docid]):
                        raise ValueError('invalid source offset span')
                    flat.extend(span[2:])
                index.write(PAIR.pack(out.tell(),len(flat)*4));out.write(u32(flat))
        if count != manifest.n_docs:
            raise ValueError('source offset document count mismatch')
        (stage/'lengths.bin').write_bytes(u32(lengths))
        totals=[0]*manifest.n_docs
        database=sqlite3.connect(stage/'terms.sqlite')
        try:
            database.execute('CREATE TABLE terms(term TEXT PRIMARY KEY, start INTEGER, size INTEGER, df INTEGER) WITHOUT ROWID')
            with (source/'postings.jsonl').open(encoding='utf-8') as postings, (stage/'postings.bin').open('wb') as out:
                terms=0;previous=None
                for line in postings:
                    row=json.loads(line);term=row['term'];docs=row['docs'];df=row['df']
                    if (not isinstance(term,str) or not term or previous is not None and term<=previous
                            or type(df) is not int or df<1 or df!=len(docs)):
                        raise ValueError('source term order/df mismatch')
                    previous=term;flat=[];last=-1
                    for docid,tf,positions in docs:
                        if (type(docid) is not int or type(tf) is not int or not last<docid<manifest.n_docs
                                or tf<1 or tf!=len(positions) or any(type(p) is not int for p in positions)
                                or not 0<=positions[0]<=positions[-1]<lengths[docid]
                                or any(a>=b for a,b in zip(positions,positions[1:]))):
                            raise ValueError('invalid source posting')
                        last=docid;totals[docid]+=tf;flat.extend([docid,tf]);flat.extend(positions)
                    data=u32(flat);start=out.tell();out.write(data)
                    database.execute('INSERT INTO terms VALUES(?,?,?,?)',(term,start,len(data),df));terms+=1
                    if terms%100000==0:print('compact terms',terms,flush=True)
            if totals!=lengths:
                raise ValueError('source posting token totals mismatch')
            database.commit()
        finally:database.close()
        files={p.name:sha256_of(p) for p in stage.iterdir()}
        metadata=dict(compact_version=1,source_manifest=manifest.to_dict(),files=files,terms=terms)
        (stage/'compact.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
        if target.exists():target.rmdir()
        stage.replace(target)
    return metadata


class CompactIndex:
    storage_format='compact-v1'
    def __init__(self, root, cache_terms=32):
        self.root=Path(root);metadata=json.loads((self.root/'compact.json').read_text(encoding='utf-8'))
        required={'docs.jsonl','docs.idx','offsets.bin','offsets.idx','lengths.bin','terms.sqlite','postings.bin'}
        if metadata.get('compact_version')!=1 or set(metadata['files'])!=required:
            raise ValueError('invalid compact artifact files/version')
        for name,digest in metadata['files'].items():
            if sha256_of(self.root/name)!=digest:raise ValueError('compact checksum mismatch: '+name)
        self._manifest=metadata['source_manifest'];self.n_docs=self._manifest['n_docs'];self.avgdl=self._manifest['avgdl']
        if self._manifest['analyzer']['config_hash']!=analyzer.config_hash():raise ValueError('compact analyzer mismatch')
        self._lengths=integers((self.root/'lengths.bin').read_bytes())
        if len(self._lengths)!=self.n_docs:raise ValueError('compact lengths mismatch')
        self._doc_index=(self.root/'docs.idx').read_bytes();self._offset_index=(self.root/'offsets.idx').read_bytes()
        if len(self._doc_index)!=self.n_docs*PAIR.size or len(self._offset_index)!=self.n_docs*PAIR.size:
            raise ValueError('compact row index mismatch')
        binary_flag = getattr(os, 'O_BINARY', 0)
        self._files={name:os.open(self.root/name,os.O_RDONLY | binary_flag) for name in ('docs.jsonl','offsets.bin','postings.bin')}
        self._db=sqlite3.connect('file:'+str((self.root/'terms.sqlite').resolve())+'?mode=ro&immutable=1',uri=True,check_same_thread=False)
        self._lock=threading.RLock()
        self._term=lru_cache(maxsize=4096)(self._lookup)
        self.postings=lru_cache(maxsize=cache_terms)(self._postings)
        self._closed=False

    def _lookup(self,term):
        with self._lock:
            return self._db.execute('SELECT start,size,df FROM terms WHERE term=?',(term,)).fetchone()

    def _postings(self,term):
        row=self._term(term)
        if row is None:return None
        start,size,df=row;data=integers(_pread(self._files['postings.bin'],size,start));result=[];i=0
        while i<len(data):
            docid,tf=data[i:i+2];i+=2;result.append((docid,tf,list(data[i:i+tf])));i+=tf
        if len(result)!=df:raise ValueError('compact posting count mismatch')
        return result

    def df(self,term):
        row=self._term(term)
        return row[2] if row else 0

    def doc_length(self,docid):return self._lengths[docid]

    def _document_row(self,docid):
        if type(docid) is not int or not 0<=docid<self.n_docs:raise IndexError('invalid docid')
        start,size=PAIR.unpack_from(self._doc_index,docid*PAIR.size)
        return json.loads(_pread(self._files['docs.jsonl'],size,start))

    def external_doc_id(self,docid):
        """Validate vector identity without materializing token-offset pairs."""
        return self._document_row(docid)['doc_id']

    def doc(self,docid):
        row=self._document_row(docid)
        start,size=PAIR.unpack_from(self._offset_index,docid*PAIR.size)
        data=integers(_pread(self._files['offsets.bin'],size,start))
        return Document(docid,row['doc_id'],row['title'],row['text'],list(zip(data[::2],data[1::2])))

    def vocab(self):
        connection=sqlite3.connect('file:'+str((self.root/'terms.sqlite').resolve())+'?mode=ro&immutable=1',uri=True)
        try:
            for row in connection.execute('SELECT term FROM terms ORDER BY term'):yield row[0]
        finally:connection.close()

    def manifest(self):return self._manifest

    def close(self):
        if not self._closed:
            self._term.cache_clear();self.postings.cache_clear();self._db.close()
            for fd in self._files.values():os.close(fd)
            self._closed=True

    def __enter__(self):return self
    def __exit__(self,*_):self.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();start=time.perf_counter();result=compile_index(args.source,args.out)
    print(json.dumps(dict(documents=result['source_manifest']['n_docs'],terms=result['terms'],seconds=time.perf_counter()-start),indent=2))


if __name__=='__main__':main()
