# Full DuReader resource audit

The official archive is present and verified locally. The four passage files contain exactly 8,096,668 records:

```
part-00 2,024,167
part-01 2,024,167
part-02 2,024,167
part-03 2,024,167
```

The current repository BM25 builder materializes the complete corpus and all postings as Python objects before writing JSONL. The frozen 100k build produced a 896 MiB BM25 directory. A linear extrapolation to 8,096,668 documents is about 72 GiB of BM25 JSONL alone, before the Python postings graph, temporary objects, full dense vectors and Qdrant HNSW storage. The machine currently has 123 GiB free and 251 GiB RAM. A full dense build would additionally encode 8,096,668 documents (at least 16.6 GiB of raw float32 vectors before Qdrant/index overhead).

Because the implementation's in-memory postings representation scales with every token occurrence, a full build cannot be claimed safe on this host without a streaming/sharded index implementation and a fresh disk budget. No full-corpus result is therefore reported as if it were complete. The official archive, 100k reconstruction, 100k lexical/dense indexes, CE actions, and loopback HTTP evidence remain independently reproducible. The goal stays open until either a sharded full-corpus backend is implemented and audited or the maintainer explicitly accepts the 100k-only scope.

## Storage mount check

A second host volume exists at `/autodl-pub` (1.4 TiB free), but `mount` reports it as `xfs (ro)` and `/autodl-pub/data` as a read-only autofs mount. A write probe was rejected with `Read-only file system`. The only writable project volume is `/root/autodl-tmp`, with approximately 123 GiB free at the audit time. This rules out silently moving the full build to the larger read-only volume.

## Full BM25 attempt after disk expansion

On the follow-up run, `/root/autodl-tmp` expanded to 849 GiB with 462 GiB free, so the 7.43 GiB normalized full corpus was generated successfully (`8,096,668` JSONL records). A real `offline.build` full-corpus attempt was then started. The process reached approximately 30 GiB RSS and disappeared before creating an output index. The container cgroup reports `memory.max=32212254720` bytes (30 GiB); this confirms the current in-memory postings builder cannot complete the full corpus even after the disk expansion. A sharded/streaming index implementation is required for the remaining full-corpus task.
