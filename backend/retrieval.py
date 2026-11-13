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
    # 数字型证据（参数/上下文）：多篇对比时补 "1.5 billion / nctx=2048" 一类段落
    for a in _fact_extra_anchors(parsed, question or ""):
        if not any(a["page"] == b.get("page") and (a.get("text") or "")[:40] == (b.get("text") or "")[:40] for b in out):
            out.append(a)
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




# 总结类问题：中文提问在英文原文里 BM25 常 0 命中，用摘要兜底（仍带锚点）
_SUMMARY_RE = re.compile(
    r"主要|概述|总结|简介|介绍|讲了什么|写了什么|说的什么|什么内容|主要内容|"
    r"贡献|创新点|解决什么|是什么|关于什么|summary|overview|contribution|about",
    re.I,
)


def _abstract_anchors(parsed: dict, limit: int = 2) -> list[dict]:
    """取首页摘要区域的长段落作为兜底证据（保持页码/bbox 可跳转）。"""
    cands = []
    for p in parsed.get("paragraphs", []):
        if p.get("page") != 1:
            continue
        text = p.get("text") or ""
        if len(text) < 180:
            continue
        cands.append(p)
        if len(cands) >= limit:
            break
    return [
        {"page": p["page"], "bbox": p["bbox"], "text": p["text"][:500], "score": 0.5}
        for p in cands
    ]



# 数字/参数/上下文类问题：BM25 常把附录排前面，按"数字+单位"补证据
_FACT_EXTRA = [
    (re.compile(r"参数|参数量|规模|billion|million|parameter|parameters|model size", re.I),
     re.compile(r"\d[\d,.]*\s*(billion|million|亿|B\b)", re.I)),
    (re.compile(r"上下文|context|窗口|长度|token|tokens", re.I),
     re.compile(r"(context|token)[^.]{0,90}\d{2,5}|\d{2,5}[^.]{0,40}(context|token)", re.I)),
]


def _fact_extra_anchors(parsed: dict, question: str, limit: int = 3) -> list[dict]:
    """按问题类型补充含数字的段落（保持 bbox 可跳转）。"""
    if not any(p.search(question) for p, _ in _FACT_EXTRA):
        return []
    pats = [pat for q, pat in _FACT_EXTRA if q.search(question)]
    out, seen = [], set()
    for p in parsed.get("paragraphs", []):
        text = p.get("text") or ""
        if len(text) < 40:
            continue
        if not any(pat.search(text) for pat in pats):
            continue
        key = (p["page"], p["bbox"][0] if p.get("bbox") else 0)
        if key in seen:
            continue
        seen.add(key)
        out.append({"page": p["page"], "bbox": p["bbox"], "text": text[:450], "score": 0.6, "fact": True})
        if len(out) >= limit:
            break
    return out

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _precision_anchors(anchors: list[dict], question: str, answer_text: str,
                       max_pages: int = 2, query_terms: list[str] | None = None) -> list[dict]:
    """锚点精度：问题词做 IDF 加权，每页保留最强一条，弱页丢弃。"""
    base = [a for a in anchors if a.get("page") and a.get("text")]
    if len(base) <= 1:
        return anchors

    def toks(s: str) -> set:
        out = {w.lower() for w in _WORD_RE.findall(s or "") if len(w) >= 3}
        out |= set(_NUM_RE.findall(s or ""))
        return out

    q = {w.lower() for w in _WORD_RE.findall(question or "") if len(w) >= 3}
    if query_terms:
        q |= {w.lower() for w in query_terms if len(w) >= 2}
    q_num = set(_NUM_RE.findall(question or ""))
    q |= q_num

    texts = [toks(a.get("text") or "") for a in base]
    n = len(base)
    df: dict = {}
    for ts in texts:
        for w in ts:
            df[w] = df.get(w, 0) + 1

    def weight(w: str) -> float:
        d = df.get(w, 0) or 1
        return 1.0 + (n / d)

    scored = []
    for a, ts in zip(base, texts):
        hit = (q & ts)
        s = sum(weight(w) for w in hit)
        if a.get("fact"):
            s += 3.0
        # 数值问题：锚点里有数字再加一点（但不能盖过问题词命中）
        if q_num and (q_num & set(_NUM_RE.findall(a.get("text") or ""))):
            s += 1.0
        scored.append((s, a))

    best_score = max(s for s, _ in scored)

    # 同页保留最强一条
    by_page: dict = {}
    for s, a in scored:
        pg = a["page"]
        if pg not in by_page or s > by_page[pg][0]:
            by_page[pg] = (s, a)
    pages = sorted(by_page.items(), key=lambda kv: -kv[1][0])

    keep = []
    # 数字型事实锚点强制保留（GPU/超参数等题不能被 BM25 噪声页挤掉）
    facts = [(s, a) for s, a in scored if a.get("fact")]
    if facts:
        fs, fa = max(facts, key=lambda x: x[0])
        keep.append((fs, fa))
    for pg, (s, a) in pages:
        if keep and a["page"] == keep[0][1]["page"]:
            continue
        if not facts and best_score > 0 and s < best_score * 0.72:
            continue
        keep.append((s, a))
        if len(keep) >= max_pages:
            break
    if not keep:
        keep = [pages[0][1]]
    return [a for _, a in keep]


_HW_Q = re.compile(r"GPU|显卡|显存|TPU|P100|V100", re.I)


def _hardware_anchors(parsed: dict, question: str, limit: int = 2) -> list[dict]:
    """硬件类问题：强制找含"数量+型号"的段落，避免被 Table 2 等页挤掉。"""
    if not _HW_Q.search(question or ""):
        return []
    pat = re.compile(r"\d+\s*(?:块|台)?\s*(?:NVIDIA\s*)?(?:P100|V100|GPU|TPU)", re.I)
    cand = []
    for p in parsed.get("paragraphs", []):
        txt = p.get("text") or ""
        if not pat.search(txt):
            continue
        # 正文实验段优先于摘要/表格页
        score = 1.0
        if p.get("page", 0) > 2:
            score += 1.0
        if re.search(r"trained our models|one machine|Table 3", txt, re.I):
            score += 2.0
        cand.append((score, {"page": p["page"], "bbox": p["bbox"], "text": txt[:450],
                             "score": score, "fact": True}))
    cand.sort(key=lambda x: -x[0])
    return [a for _, a in cand[:limit]]


_HP_Q = re.compile(r"Adam|β|beta|学习率|dropout|warmup|label smoothing", re.I)


def _hyper_anchors(parsed: dict, question: str, limit: int = 1) -> list[dict]:
    """超参数类问题：强制找含 Adam/β/0.9/0.98 的段落。"""
    if not _HP_Q.search(question or ""):
        return []
    out = []
    pat = re.compile(r"(β1|β2|beta1|beta2|0\.9[^0-9]|0\.98|epsilon|dropout\s*=\s*0\.|warmup\s*=\s*\d)", re.I)
    for p in parsed.get("paragraphs", []):
        txt = p.get("text") or ""
        if len(txt) < 40:
            continue
        if pat.search(txt):
            out.append({"page": p["page"], "bbox": p["bbox"], "text": txt[:450],
                        "score": 1.0, "fact": True})
            if len(out) >= limit:
                break
    return out


_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_num(s: str) -> int:
    """中文数字转 int：六→6，十一→11，二十三→23。"""
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if "十" in s:
        a, _, b = s.partition("十")
        return (_CN_NUM.get(a, 1) * 10) + (_CN_NUM.get(b, 0) if b else 0)
    return _CN_NUM.get(s, 0)


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
        m2 = re.search(r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:位)?\s*作者", question) or re.search(r"(\d+)(?:st|nd|rd|th)\s+author", ql)
        if m2 and alist:
            idx = _cn_num(m2.group(1)) - 1
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
        cand = [r for r in refs if any(t in r["text"].lower() for t in lines if len(t) > 3)]
        # 打分：命中词数越多越可信；要求至少 2 个词（单一 "layer" 会误召回 MoE 等条目）
        scored = [(sum(1 for t in set(lines) if len(t) > 3 and t in r["text"].lower()), r) for r in cand]
        scored.sort(key=lambda x: -x[0])
        hits = [r for sc, r in scored if sc >= 2] or [r for sc, r in scored[:2]]
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

    # 只保留接近最佳分数的段落
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

    for _a in _fact_extra_anchors(parsed, question) + _hardware_anchors(parsed, question) + _hyper_anchors(parsed, question):
        if not any(_a["page"] == b["page"] and _a["bbox"] == b["bbox"] for b in anchors):
            anchors.append(_a)
            answer_bits.append((_a["text"] or "")[:240])

    if not anchors:
        # 总结类问题（"这篇论文主要写了什么"）→ 摘要兜底，仍带页码锚点
        if _SUMMARY_RE.search(question):
            ab = _abstract_anchors(parsed)[:1]
            if ab:
                return {
                    "mode": "rag",
                    "answer": "这篇论文的主要内容（依据首页摘要）：\n" + ab[0]["text"][:400],
                    "anchors": ab,
                }
        return {"mode": "none", "answer": "未在论文中找到相关段落（无锚点，不出结论）。", "anchors": []}

    # 锚点去重：同页同 bbox 只留一条
    _seen, _ded = set(), []
    for a in anchors:
        key = (a.get("page"), tuple(round(float(v), 1) for v in (a.get("bbox") or [])))
        if key in _seen:
            continue
        _seen.add(key)
        _ded.append(a)
    anchors = _ded

    # 锚点精度：每页最强一条 + 丢掉弱相关页
    anchors = _precision_anchors(anchors, question, " ".join(answer_bits), query_terms=q_tokens)

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
