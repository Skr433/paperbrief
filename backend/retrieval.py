# -*- coding: utf-8 -*-
"""检索与问答：BM25 段落检索 + 意图增强，答案永远带原文锚点。

对应命题 1.2.1：单篇问答要"给出参考原文，让用户能发现幻觉"。
本项目把"必须给锚点"写成硬约束：任何回答都返回 paragraphs 里的 page/bbox。
"""

from __future__ import annotations

import math
import re

from pdf_parse import tokens
import embeddings

# 事实型证据包：数字类问题（参数量/上下文/归一化位置）BM25 常召回不到，
# 这里按关键词补充候选段落，交给大模型判断。
FACT_PATTERNS = [
    r"parameter", r"billion", r"million", r"model size", r"参数量", r"参数规模",
    r"context window", r"nctx", r"context length", r"maximum context", r"上下文",
    r"layer norm", r"layernorm", r"normalization", r"pre-norm", r"post-norm",
    r"encoder", r"decoder", r"bidirection", r"unidirection",
    r"gpt-1", r"gpt-2", r"gpt-3", r"bert",
    r"token", r"vocabulary", r"tokens? per",
]
_FACT_RE = [re.compile(p, re.I) for p in FACT_PATTERNS]


def build_evidence(parsed: dict, base_anchors: list[dict], extra: int = 6, question: str = "") -> list[dict]:
    """在 BM25 结果之外，补充"命中事实关键词"的段落（去重、限量）。"""
    out = list(base_anchors)
    seen = {(a.get("page"), (a.get("text") or "")[:40]) for a in out}
    cands = []
    for p in parsed.get("paragraphs", []):
        t = p.get("text") or ""
        if len(t) < 60:
            continue
        hits = sum(1 for r in _FACT_RE if r.search(t))
        if hits < 1:
            continue
        key = (p.get("page"), t[:40])
        if key in seen:
            continue
        # 数字邻近加权：上下文长度/参数量这类题，真正有用的是“关键词 + 数字”同句
        low = t.lower()
        ql = (question or "").lower()
        if re.search(r"(nctx|context window|context length|parameters?|billion|million)", low) and re.search(r"\d{2,4}", low):
            hits += 2
        # 按问题类型定向加权（避免参数量段落挤掉上下文段落）
        if re.search(r"上下文|context|长度|length|window", ql):
            if re.search(r"(nctx|context window|context length|context size)", low) and re.search(r"\d{3,4}", low):
                hits += 8
        if re.search(r"参数|parameter|规模|billion", ql):
            if re.search(r"(billion|million|parameters?|参数量)", low) and re.search(r"\d", low):
                hits += 5
        cands.append((hits, p))
    cands.sort(key=lambda x: (-x[0], x[1]["page"]))
    # 证据优先：命中强的段落放前面（否则被下游 top-k 截断时会被丢掉）
    front = [{"page": p["page"], "bbox": p["bbox"], "text": p["text"][:500], "score": 0.9}
             for _, p in cands[:extra]]
    return front + out


SYNONYMS = {
    "作者": ["author", "authors", "by"],
    "author": ["作者", "authors"],
    "机构": ["affiliation", "university", "institute", "google", "brain", "research"],
    "单位": ["affiliation", "university", "institute"],
    "超参": ["hyperparameter", "beta", "learning", "rate", "dropout", "warmup"],
    "超参数": ["hyperparameter", "beta", "learning", "rate", "dropout", "warmup"],
    "beta": ["β", "β1", "β2", "adam"],
    "β": ["beta", "adam"],
    "gpu": ["gpu", "p100", "nvidia", "tpu", "device", "cards"],
    "显卡": ["gpu", "p100", "nvidia"],
    "bleu": ["bleu", "score"],
    "参考文献": ["reference", "references", "bibliography"],
    "reference": ["参考文献", "references"],
    "上下文": ["context", "window", "token", "length"],
    "窗口": ["context", "window"],
    "参数量": ["parameter", "parameters", "billion", "million", "size"],
    "参数": ["parameter", "parameters", "hyperparameter"],
    "规模": ["scale", "size", "parameter", "billion"],
    "层归一化": ["layer", "normalization", "layernorm", "ln"],
    "归一化": ["normalization", "norm", "layernorm"],
    "layernorm": ["layer", "normalization"],
    "预训练": ["pretraining", "pre-training", "pretrained"],
    "微调": ["finetuning", "fine-tuning", "finetune"],
    "训练": ["train", "training", "trained", "steps", "epoch"],
    "数据集": ["dataset", "data", "corpus"],
    "准确率": ["accuracy", "acc"],
    "召回率": ["recall"],
    "延迟": ["latency", "time", "speed"],
    "推理": ["inference", "test"],
    "模型": ["model", "models"],
    "编码器": ["encoder"],
    "解码器": ["decoder"],
    "层数": ["layer", "layers", "depth"],
    "丢弃": ["drop", "dropped", "discard", "removed", "without"],
    "对比": ["compare", "comparison", "versus", "vs", "difference"],
    "差异": ["difference", "differ", "compare"],
    "性能": ["performance", "result"],
    "结果": ["result", "results", "performance"],
    "提升": ["improve", "improvement", "gain"],
    "亿": ["billion"],
}

METRIC_WORDS = ("bleu", "f1", "accuracy", "perplexity", "em", "rouge", "map")


def _expand(q: str) -> list[str]:
    ql = q.lower()
    toks = tokens(q)
    extra: list[str] = []
    for key, syns in SYNONYMS.items():
        if key in ql:
            extra.extend(syns)
    return toks + extra


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.n = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.n)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []
        for d in docs:
            cnt: dict[str, int] = {}
            for t in d:
                cnt[t] = cnt.get(t, 0) + 1
            self.tf.append(cnt)
            for t in cnt:
                self.df[t] = self.df.get(t, 0) + 1

    def score(self, query: list[str], i: int) -> float:
        dl = len(self.docs[i])
        s = 0.0
        for t in query:
            if t not in self.tf[i]:
                continue
            idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
            f = self.tf[i][t]
            s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s


def _sentences(text: str) -> list[str]:
    # 中英文混排分句（中文句末标点后常无空格）
    parts = re.split(r"(?<=[。！？；!?])\s*|(?<=\.)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def ask(parsed: dict, question: str, topk: int = 3, extra_terms: str = "", doc_key: str = "") -> dict:
    """返回 {answer, anchors:[{page,bbox,text,score}], mode}。"""
    paras = [p for p in parsed["paragraphs"] if len(p["text"]) > 40]
    ql = question.lower()
    q_tokens = _expand(question)
    if extra_terms:
        q_tokens = q_tokens + tokens(extra_terms)

    # --- 意图：作者 / 参考文献 走结构化字段，准确率更高
    if re.search(r"作者|author|谁写的", ql):
        meta = parsed["meta"]
        alist = meta.get("author_list") or []
        aff = meta.get("affiliations") or ""
        bbox = meta.get("authors_bbox") or [0, 0, 0, 0]
        m2 = re.search(r"第\s*(\d+)\s*(?:位)?\s*作者", question) or re.search(r"(\d+)(?:st|nd|rd|th)\s+author", ql)
        if m2 and alist:
            idx = int(m2.group(1)) - 1
            if 0 <= idx < len(alist):
                txt = f"第 {idx + 1} 作者：{alist[idx]}"
                if aff:
                    txt += f"\n机构（首页脚注）：{aff}"
                ctx = "作者（按顺序）：" + "、".join(alist)
                if aff:
                    ctx += f"\n机构：{aff}"
                return {
                    "mode": "author",
                    "answer": txt,
                    "anchors": [{"page": 1, "bbox": bbox, "text": ctx, "score": 1.0}],
                }
        if alist:
            txt = "作者（共 %d 位）：" % len(alist) + "、".join(alist)
            if aff:
                txt += "\n机构（首页脚注）：" + aff
            return {
                "mode": "author",
                "answer": txt,
                "anchors": [{"page": 1, "bbox": bbox, "text": meta.get("authors", "")[:300], "score": 1.0}],
            }
        if meta.get("authors"):
            return {
                "mode": "author",
                "answer": meta["authors"],
                "anchors": [{"page": 1, "bbox": bbox, "text": meta["authors"], "score": 1.0}],
            }

    if re.search(r"参考文献|reference|引用.*哪篇|bibliography", ql):
        refs = parsed.get("references", [])
        lines = q_tokens
        hits = [r for r in refs if any(t in r["text"].lower() for t in lines if len(t) > 3)]
        if hits:
            return {
                "mode": "reference",
                "answer": "\n".join(h["text"][:300] for h in hits[:3]),
                "anchors": [
                    {"page": h["page"], "bbox": h["bbox"], "text": h["text"][:300], "score": 0.9}
                    for h in hits[:3]
                ],
            }

    bm = BM25([tokens(p["text"]) for p in paras])
    scored = [(bm.score(q_tokens, i), i) for i in range(len(paras))]
    scored.sort(reverse=True)

    # 混合检索：BM25 前 40 段 → 向量语义重排（模型不可用时自动跳过）
    if embeddings.available() and len(paras) > topk:
        bm_scores = {i: sc for sc, i in scored}
        pool = [{"page": paras[i]["page"], "bbox": paras[i]["bbox"], "text": paras[i]["text"],
                 "_idx": i, "score": round(sc, 3)} for sc, i in scored[:20]]
        query = (question + " " + (extra_terms or "")).strip()
        reranked = embeddings.rerank(doc_key, query, pool, topk=20)
        order = [it.get("_idx") for it in reranked if it.get("_idx") is not None]
        if order:
            seen, rescored = set(), []
            for idx in order:
                if idx in seen:
                    continue
                seen.add(idx)
                rescored.append((bm_scores.get(idx, 0.0), idx))
            for sc2, i2 in scored:
                if i2 not in seen:
                    rescored.append((sc2, i2))
            scored = rescored

    # 只保留与最佳命中分数接近的段落，避免"凑满三条"带进无关证据
    top_score = scored[0][0] if scored else 0.0
    keep = [(sc, i) for sc, i in scored[:topk] if sc > 0 and (top_score == 0 or sc >= top_score * 0.45)]

    anchors = []
    answer_bits = []
    for sc, i in keep:
        p = paras[i]
        anchors.append(
            {"page": p["page"], "bbox": p["bbox"], "text": p["text"][:400], "score": round(sc, 3)}
        )
        sents = _sentences(p["text"])
        best = [s for s in sents if any(t in s.lower() for t in q_tokens)]
        answer_bits.append(" ".join((best or sents)[:2])[:300])

    if not anchors:
        return {"mode": "none", "answer": "未在论文中找到相关段落（无锚点，不出结论）。", "anchors": []}

    return {
        "mode": "rag",
        "answer": "\n".join(f"[第{a['page']}页] {a['text'][:200]}" for a in anchors),
        "anchors": anchors,
    }


def _first_page_bbox(parsed: dict, needle: str) -> list[float]:
    for p in parsed["paragraphs"]:
        if p["page"] == 1 and needle[:20] in p["text"]:
            return p["bbox"]
    return [0, 0, 0, 0]
