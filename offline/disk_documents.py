"""Read-only document source for vector building without loading lexical objects."""

import json
import threading
from array import array
from pathlib import Path

from contracts import analyzer
from contracts.index_format import DOCS, load_manifest, verify_files
from contracts.reader import Document


class DiskDocuments:
    """Keep two byte offsets per JSONL row; decode only the requested document.

    The bundle is immutable while open. Checksums cover every lexical artifact,
    but postings/offsets are never parsed or materialized in this process.
    """

    def __init__(self, index_dir):
        root = Path(index_dir)
        manifest = load_manifest(root)
        if manifest.analyzer_hash != analyzer.config_hash():
            raise ValueError("document source analyzer mismatch")
        verify_files(root, manifest)
        self.n_docs = manifest.n_docs
        self._starts, self._lengths = array('Q'), array('Q')
        self._lock = threading.Lock()
        self._file = (root / DOCS).open('rb')
        try:
            while True:
                start = self._file.tell()
                line = self._file.readline()
                if not line:
                    break
                self._starts.append(start)
                self._lengths.append(len(line))
            if len(self._starts) != self.n_docs:
                raise ValueError("document source count mismatch")
        except Exception:
            self.close()
            raise

    def doc(self, docid):
        if type(docid) is not int or not 0 <= docid < self.n_docs:
            raise IndexError("document ID outside source")
        with self._lock:
            self._file.seek(self._starts[docid])
            row = json.loads(self._file.read(self._lengths[docid]))
        if (not isinstance(row, dict) or type(row.get('docid')) is not int
                or row['docid'] != docid or not isinstance(row.get('doc_id'), str)
                or not row['doc_id'] or not isinstance(row.get('title'), str)
                or not isinstance(row.get('text'), str)):
            raise ValueError("invalid document source row")
        return Document(docid, row['doc_id'], row['title'], row['text'])

    def close(self):
        with self._lock:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
