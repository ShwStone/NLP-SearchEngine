"""Resume bounded-memory shard builds, verifying completed files before reuse.

Run as a module. An advisory process lock prevents overlapping launchers.
Shard-local statistics are build artifacts, not valid global BM25 scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

from contracts import analyzer
from contracts.index_format import REQUIRED_FILES, load_manifest, sha256_of
from evaluation.artifact_lock import exclusive_artifact_lock


def verified(directory, dataset):
    manifest = load_manifest(directory)
    if manifest.dataset != dataset or manifest.analyzer_hash != analyzer.config_hash():
        raise ValueError(f'{directory}: build identity mismatch')
    if manifest.bm25.k1 != 1.2 or manifest.bm25.b != .75:
        raise ValueError(f'{directory}: BM25 parameters mismatch')
    for name in REQUIRED_FILES:
        expected = manifest.files.get(name)
        if not expected or sha256_of(directory / name) != expected:
            raise ValueError(f'{directory}: missing or corrupt {name}')
    return manifest


def build_one(item, output):
    number, source = item
    directory = output / source.name
    dataset = f'dureader-full-shard-{number:03d}'
    started = time.monotonic()
    if (directory / 'manifest.json').exists():
        verified(directory, dataset)
        print('verified', source.name, flush=True)
        return
    print('building', source.name, flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'offline.build', '--corpus', str(source),
                    '--out', str(directory), '--dataset', dataset,
                    '--k1', '1.2', '--b', '0.75'], check=True)
    verified(directory, dataset)
    print('completed', source.name, round(time.monotonic() - started, 1), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--corpus-dir', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with exclusive_artifact_lock(args.out / '.build.lock'):
        parts = sorted(args.corpus_dir.glob('part-*'))
        if not parts:
            raise ValueError('no corpus shards')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda item: build_one(item, args.out), enumerate(parts)))
        manifests = [verified(args.out / p.name, f'dureader-full-shard-{i:03d}')
                     for i, p in enumerate(parts)]
        report = {'shards': len(parts), 'documents': sum(m.n_docs for m in manifests),
                  'files_sha256_verified': True, 'global_scoring_ready': False}
        (args.out / 'build-summary.json').write_text(
            json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
