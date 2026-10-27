"""在线组看索引的唯一视角：IndexReader 协议。

在线组不许直接读索引文件、不许拼文件路径，只能通过这个接口拿数据。
它有两个实现：offline.disk_index.DiskIndex（读真索引）与
online.memory_index.MemoryIndex（手工塞小数据）。
两者必须过同一套 conformance 测试（contracts/conformance.py）。
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol


@dataclass(frozen=True)
class Document:
    docid: int          # 内部编号（0..n-1，打分/排序用）
    doc_id: str         # 数据集里的外部 id（评测输出用）
    title: str
    text: str
    offsets: list = field(default_factory=list)  # 每个 token 的 (char_start, char_end)，高亮用


class DocumentSource(Protocol):
    """Narrow offline encoding input; does not require postings or offsets."""
    n_docs: int

    def doc(self, docid: int) -> Document: ...


class IndexReader(Protocol):
    n_docs: int
    avgdl: float

    def postings(self, term: str) -> Optional[list]:
        """词项的倒排表：[(docid, tf, [pos...]), ...]，按 docid 升序；词不存在返回 None。"""

    def df(self, term: str) -> int:
        """包含该词项的文档数；不存在返回 0。"""

    def doc_length(self, docid: int) -> int:
        """该文档索引时的 token 数。"""

    def doc(self, docid: int) -> Document:
        """取文档原文与 offsets。"""

    def vocab(self) -> Iterable[str]:
        """全词表（RM3 扫候选词用）。"""

    def manifest(self) -> dict:
        """索引的 manifest 字典（/index/info 直接回传）。"""
