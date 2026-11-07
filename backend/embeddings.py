# -*- coding: utf-8 -*-
"""混合检索的向量重排（BM25 召回 → bge-small 语义重排）。

设计取舍：
- 不做全库向量索引（267–1200 段首次编码 3–15s，解析体验崩），只对 **BM25 前 40 段**做重排；
- 编码结果按段落哈希缓存（pickle），同一文档第二次检索 ~0ms；
- 模型/依赖缺失时 available() 返回 False，全链路自动退回 BM25，不影响线上；
- 跨语言：查询用「原问题 + 英文关键词」拼接（关键词已由 LLM 扩展生成）。
"""

from __future__ import annotations

import hashlib
import math
import os
import pickle

_CACHE_DIR = None
_MODEL = None
_FAILED = False
DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def set_cache_dir(d: str):
    global _CACHE_DIR
    _CACHE_DIR = d
    os.makedirs(d, exist_ok=True)


def _load_model():
    global _MODEL, _FAILED
    if _MODEL is not None or _FAILED:
        return _MODEL
    try:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(model_name=DEFAULT_MODEL)
    except Exception:
        _FAILED = True
        _MODEL = None
    return _MODEL


def available() -> bool:
    return _load_model() is not None


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


def _cache_path(doc_key: str) -> str:
    return os.path.join(_CACHE_DIR or ".", (doc_key or "doc") + ".pkl")


def _load_cache(doc_key: str) -> dict:
    p = _cache_path(doc_key)
    if os.path.exists(p):
        try:
            return pickle.load(open(p, "rb"))
        except Exception:
            return {}
    return {}


def _save_cache(doc_key: str, cache: dict):
    try:
        pickle.dump(cache, open(_cache_path(doc_key), "wb"))
    except Exception:
        pass


def rerank(doc_key: str, query: str, items: list[dict], topk: int = 40) -> list[dict]:
    """对 items 前 topk 条做语义重排；失败/不可用时原样返回。"""
    model = _load_model()
    if model is None or not items:
        return items
    pool = items[:topk]
    texts = [it.get("text") or "" for it in pool]
    cache = _load_cache(doc_key)
    todo = [(i, t) for i, t in enumerate(texts) if _hash(t) not in cache]
    if todo:
        try:
            vecs = list(model.embed([t[:1200] for _, t in todo]))
        except Exception:
            return items
        for (i, t), v in zip(todo, vecs):
            cache[_hash(t)] = [round(float(x), 5) for x in v]
        _save_cache(doc_key, cache)
    try:
        qv = list(model.embed([query[:600]]))[0]
    except Exception:
        return items

    def cos(a, b):
        s = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return s / (na * nb)

    scored = []
    for i, t in enumerate(texts):
        v = cache.get(_hash(t))
        if v is None:
            continue
        sim = cos(qv, v)
        scored.append((sim, pool[i]))
    if not scored:
        return items
    scored.sort(key=lambda x: -x[0])
    # 只保留重排后的前 topk，其余（未进入候选池的）保持原顺序附后
    head = [it for _, it in scored]
    tail = items[topk:]
    for sim, it in scored:
        it["semantic"] = round(sim, 4)
    return head + tail
