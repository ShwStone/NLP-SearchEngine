"""Deterministic corpus preflight. Never reads queries/qrels to select docs."""

import json
from pathlib import Path
import tempfile

from contracts import analyzer
from contracts.index_format import sha256_of


def prepare(corpus, out, *, invalid='error'):
    if invalid not in ('error','skip'):
        raise ValueError('invalid policy must be error or skip')
    corpus,out=Path(corpus),Path(out)
    if out.exists():
        raise ValueError('preparation requires a new directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    seen=set();accepted=excluded=total=blank=0
    with tempfile.TemporaryDirectory(prefix='.'+out.name+'-',dir=out.parent) as tmp:
        stage=Path(tmp)
        with corpus.open(encoding='utf-8') as source, (stage/'corpus.jsonl').open('w',encoding='utf-8') as docs, \
                (stage/'id_map.jsonl').open('w',encoding='utf-8') as mapping, \
                (stage/'excluded.jsonl').open('w',encoding='utf-8') as exclusions:
            for line_number,line in enumerate(source,1):
                if not line.strip():
                    blank+=1;continue
                total+=1;row=None;reason=None
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    reason='invalid_json'
                if reason is None:
                    if not isinstance(row,dict):reason='not_object'
                    elif not isinstance(row.get('doc_id'),str) or not row['doc_id'].strip():reason='invalid_doc_id'
                    elif row['doc_id'] in seen:
                        # Ambiguous external identities are always fatal.
                        raise ValueError(f'duplicate doc_id at line {line_number}: {row["doc_id"]}')
                    else:
                        seen.add(row['doc_id'])
                        if not isinstance(row.get('text'),str) or not isinstance(row.get('title',''),str):reason='invalid_text_or_title'
                        elif not row['text'].strip():reason='empty_body'
                        elif not analyzer.analyze(row['text']):reason='zero_analyzer_tokens'
                if reason:
                    if invalid=='error':raise ValueError(f'corpus line {line_number}: {reason}')
                    excluded+=1
                    exclusions.write(json.dumps(dict(source_line=line_number,doc_id=row.get('doc_id') if isinstance(row,dict) else None,
                        reason=reason),ensure_ascii=False,sort_keys=True)+'\n')
                    continue
                normalized=dict(doc_id=row['doc_id'],title=row.get('title',''),text=row['text'])
                docs.write(json.dumps(normalized,ensure_ascii=False,sort_keys=True)+'\n')
                mapping.write(json.dumps(dict(docid=accepted,doc_id=row['doc_id'],source_line=line_number),ensure_ascii=False,sort_keys=True)+'\n')
                accepted+=1
        if accepted==0:raise ValueError('no valid documents')
        report=dict(schema_version=1,source_sha256=sha256_of(corpus),invalid_policy=invalid,
            source_records=total,accepted=accepted,excluded=excluded,blank_lines=blank,
            analyzer=analyzer.config(),analyzer_hash=analyzer.config_hash(),
            files={name:sha256_of(stage/name) for name in ('corpus.jsonl','id_map.jsonl','excluded.jsonl')})
        (stage/'source_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        stage.rename(out)
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate a frozen corpus without reading qrels")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--invalid", choices=("error", "skip"), default="error")
    args = parser.parse_args()
    print(json.dumps(prepare(args.corpus, args.out, invalid=args.invalid), ensure_ascii=False, indent=2))
