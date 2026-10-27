"""唯一分词器实现（E0 冻结）。

离线组与在线组都只准 import 这里，谁也不许自己再写一份分词逻辑。
索引 manifest 里记录 config_hash()，加载索引时比对，不一致直接拒绝启动。

当前为骨架占位实现：ASCII 词小写化 + 中文连续串重叠二元组（bigram）。
E0 定稿时替换为 jieba（用 jieba.tokenize 拿偏移量），接口不变。
"""

import hashlib
import json
import re
from dataclasses import dataclass

NAME = "simple-bigram"
VERSION = "0.1.0"
PARAMS = {"lowercase": True, "cjk": "bigram"}

_CJK_RUN = re.compile(r"[一-鿿㐀-䶿]+")
_ASCII_WORD = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Token:
    """term 是用于索引/查询的词形；start/end 是原文的字符下标（左闭右开）。"""

    term: str
    start: int
    end: int


def analyze(text: str) -> list[Token]:
    """把文本切成带偏移量的 token 流。两侧共用的唯一入口。

    约束（conformance 测试会检查）：
    - token 按 start 升序；中文 bigram 的相邻 token 可以重叠；
    - 对 CJK 部分 term == text[start:end]；
      对 ASCII 部分 term == text[start:end].lower()。
    """
    tokens: list[Token] = []
    for m in _CJK_RUN.finditer(text):
        run = m.group(0)
        if len(run) == 1:
            tokens.append(Token(run, m.start(), m.end()))
        else:
            for i in range(len(run) - 1):
                tokens.append(Token(run[i : i + 2], m.start() + i, m.start() + i + 2))
    for m in _ASCII_WORD.finditer(text):
        tokens.append(Token(m.group(0).lower(), m.start(), m.end()))
    tokens.sort(key=lambda t: t.start)
    return tokens


def config() -> dict:
    return {"name": NAME, "version": VERSION, "params": PARAMS}


def config_hash() -> str:
    payload = json.dumps(config(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
