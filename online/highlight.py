"""高亮（在线组）。

输入文档 offsets 与命中的词项，输出 doc.text 上的字符区间。
UI 只做区间合并与 HTML 转义，不许自己再对文本做匹配。
"""

from contracts.api import Highlight
from contracts.reader import Document
from bisect import bisect_left


def build_highlights(doc: Document, matched_terms: set) -> list:
    """返回合并后的高亮区间，全部 kind="original"。扩展词的高亮由 engine 叠加 kind="expanded"。"""
    if not matched_terms:
        return []
    spans = []
    # 用分词器复切原文拿到每个 token 的词形，再对照 offsets（骨架版：直接逐 token 比对）
    from contracts import analyzer
    tokens = analyzer.analyze(doc.text)
    for t in tokens:
        if t.term in matched_terms:
            spans.append((t.start, t.end))
    return merge(spans)


def merge(spans: list, kind: str = "original") -> list:
    """合并重叠/相邻区间，返回 [Highlight]。"""
    out = []
    for s, e in sorted(spans):
        if out and s <= out[-1].end:
            out[-1] = Highlight(out[-1].start, max(out[-1].end, e), kind)
        else:
            out.append(Highlight(s, e, kind))
    return out


def indexed_highlights(reader,doc:Document,matched_terms:set) -> list:
    """Explicit opt-in highlights from immutable postings and token offsets.

    Uses the same analyzer's persisted positions instead of re-tokenizing the
    whole document. Readers without complete offsets use the reference path.
    """
    if len(doc.offsets)!=reader.doc_length(doc.docid):
        return build_highlights(doc,matched_terms)
    spans=[]
    for term in matched_terms:
        posting=reader.postings(term)
        if not posting:continue
        i=bisect_left(posting,doc.docid,key=lambda row:row[0])
        if i<len(posting) and posting[i][0]==doc.docid:
            spans.extend(doc.offsets[pos] for pos in posting[i][2])
    return merge(spans)
