# -*- coding: utf-8 -*-
"""汇报包生成：把解析结果压成"可直接讲组会"的材料（对应命题 1.6 写作助手）。

无 LLM 也能跑：先用抽取式规则生成结构化材料（确定性、可复现、每条带锚点）；
配置了 LLM_API_KEY 时，再让大模型在"给定证据"基础上润色，不允许新增事实。
"""

from __future__ import annotations

import os
import re

CAPTION_STOP = re.compile(r"^((figure|fig\.|table|algorithm)\s*[\dA-Z]|(图|表)\s*\d)", re.I)
CONTRIB_PAT = re.compile(
    r"\b(we (propose|present|introduce|develop|design|demonstrate|show)|"
    r"this paper (proposes|presents|introduces))\b",
    re.I,
)
CONTRIB_CN = re.compile(r"(本文|论文|作者)(提出|设计|实现|构建|开发|给出|建立)|"
                        r"(提出|设计|实现|构建)了?(一种|一个|一套|通用|新型)|"
                        r"(实验|结果)(表明|显示|证明)|研究表明")
LIMIT_PAT = re.compile(r"(limitation|future work|we leave|remains? (an )?open|open (problem|question))", re.I)
LIMIT_CN = re.compile(r"(局限|不足|待改进|未来工作|后续工作|下一步|尚未|有待|进一步(研究|改进))")
METRIC_PAT = re.compile(r"\b(bleu|f1|accuracy|perplexity|gpu|p100|tpu|hours?|days?)\b", re.I)
METRIC_CN = re.compile(r"(准确率|精确率|召回率|F1|BLEU|延迟|时延|毫秒|秒级|提升|下降|达到|响应时间|吞吐)")
NUM_PAT = re.compile(r"\d")
# 非结果句：致谢/审稿人/作者日常等，不应进入「关键结果」
NOT_RESULT = re.compile(r"(spent|thank|grateful|would like to|we thank|acknowledg|reviewers?)", re.I)
# 术语表停用词：期刊/元数据/通用缩写，避免噪声
TERM_STOP = {
    "DOI", "ISSN", "CN", "GB", "JSON", "OS", "PP", "OL", "CONTENT", "IEEE", "DONE",
    "TEL", "EMAIL", "HTTP", "HTTPS", "URL", "PDF", "HTML", "XML", "THE", "AND", "FOR",
}


def _sents(text: str) -> list[str]:
    # 中英文混排分句：中文句末标点后通常没有空格，不能用 \s+ 作为必要条件
    parts = re.split(r"(?<=[。！？；!?])\s*|(?<=\.)\s+", text)
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _pick(paras: list[dict], para_pred, sent_pred, limit: int = 6) -> list[dict]:
    """段落级先筛，再在句级筛；避免"整段命中关键词就把无关句子也带出来"。"""
    out, seen = [], set()
    for p in paras:
        if not para_pred(p):
            continue
        for s in _sents(p["text"]):
            if len(out) >= limit:
                return out
            if not sent_pred(s):
                continue
            key = s[:60]
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": s[:300], "page": p["page"], "bbox": p["bbox"]})
    return out


def build_brief(parsed: dict) -> dict:
    paras = parsed["paragraphs"]
    body = [p for p in paras if p["kind"] == "body"]
    meta = parsed["meta"]
    text_all = " ".join(p["text"] for p in body)

    # 摘要：优先中文「摘 要：」或英文「Abstract:」，其次取正文前两段
    abstract = ""
    ab_idx = None
    for i, p in enumerate(paras):
        t = p["text"].strip()
        if (re.match(r"^摘\s*要\s*[:：]?", t)
                or re.match(r"^要\s*[:：]", t)
                or re.match(r"^abstract\s*[:：]?", t, re.I)):
            ab_idx = i
            break
    if ab_idx is not None:
        first = re.sub(r"^(摘\s*要|要|abstract)\s*[:：]?\s*", "", paras[ab_idx]["text"], flags=re.I)
        chunks = [first] if first else []
        for p in paras[ab_idx + 1: ab_idx + 5]:
            if p["kind"] == "heading" and chunks:
                break
            if re.match(r"^(关键词|关键字|key\s*words|中图分类号)", p["text"], re.I):
                break
            if p["kind"] == "body":
                chunks.append(p["text"])
        abstract = " ".join(chunks)
    if not abstract:
        abstract = " ".join(p["text"] for p in body[:2])
    # 摘录去尾注与关键词行
    abstract = re.split(r"[∗*]\s*Equal contribution|∗", abstract)[0]
    abstract = re.split(r"关键词|关键字|Key ?words", abstract)[0].strip()[:1500]

    contrib_pred = lambda t: bool(CONTRIB_PAT.search(t) or CONTRIB_CN.search(t))
    metric_pred = lambda t: bool(METRIC_PAT.search(t) or METRIC_CN.search(t)) and bool(NUM_PAT.search(t))
    limit_pred = lambda t: bool(LIMIT_PAT.search(t) or LIMIT_CN.search(t))

    # 中文论文：句子必须含中文，避免英文摘要与参考文献混入
    cjk_paper = bool(re.search(r"[\u4e00-\u9fff]", meta.get("title", "")))
    if cjk_paper:
        contrib_pred2 = lambda t: contrib_pred(t) and bool(re.search(r"[\u4e00-\u9fff]", t))
        metric_pred2 = lambda t: metric_pred(t) and bool(re.search(r"[\u4e00-\u9fff]", t))
        limit_pred2 = lambda t: limit_pred(t) and bool(re.search(r"[\u4e00-\u9fff]", t))
    else:
        contrib_pred2, metric_pred2, limit_pred2 = contrib_pred, metric_pred, limit_pred

    # 所有论文都排除"致谢/作者日常"这类非结果句
    metric_pred3 = lambda t: metric_pred2(t) and not NOT_RESULT.search(t)

    contributions = _pick(body[:120], lambda p: contrib_pred2(p["text"]), contrib_pred2, 6)
    contrib_keys = {c["text"] for c in contributions}

    metrics = [
        m for m in _pick(body, lambda p: metric_pred3(p["text"]), metric_pred3, 12)
        if m["text"] not in contrib_keys
    ][:6]

    limits = _pick(body, lambda p: limit_pred2(p["text"]), limit_pred2, 5)

    # 关键图表（带页码，可点）：英文 Figure 3: / 中文 图1、表2
    fig_pat = re.compile(
        r"^((figure|fig\.|table|algorithm)\s*[\dA-Z]+\s*[:.\-–—]|(图|表)\s*\d+\s*[\.:：、\s])", re.I
    )
    figs = []
    for p in paras:
        if p["kind"] == "caption" and fig_pat.match(p["text"].strip()):
            figs.append({"text": p["text"][:220], "page": p["page"], "bbox": p["bbox"]})
        if len(figs) >= 8:
            break

    # 术语表：大写缩写词频统计（过滤元数据噪声，且至少出现 2 次）
    term_page: dict[str, int] = {}
    freq: dict[str, int] = {}
    for p in body:
        if p["kind"] != "body":
            continue
        for m in re.findall(r"\b[A-Z][A-Z0-9]{1,7}\b", p["text"]):
            if m in TERM_STOP or re.fullmatch(r"[IVX]+", m):
                continue
            freq[m] = freq.get(m, 0) + 1
            term_page.setdefault(m, p["page"])
    terms = [
        {"term": t, "count": c, "first_page": term_page.get(t, 1)}
        for t, c in sorted(freq.items(), key=lambda x: -x[1])
        if c >= 2
    ][:12]

    # 章节树（给前端做导航）
    outline = [{"title": s["title"], "level": s["level"], "page": s["page"]} for s in parsed["sections"][:40]]

    # 5 分钟讲稿（模板化拼接，全部来自上面的证据）
    script = []
    script.append(
        {
            "title": "1. 一句话总结",
            "text": (abstract.split("。")[0] if "。" in abstract else abstract[:160]) or meta["title"],
            "page": 1,
            "bbox": paras[0]["bbox"] if paras else [0, 0, 0, 0],
        }
    )
    for c in contributions[:3]:
        script.append({"title": "2. 它做了什么", "text": c["text"], "page": c["page"], "bbox": c["bbox"]})
    for m in metrics[:2]:
        script.append({"title": "3. 结果有多好", "text": m["text"], "page": m["page"], "bbox": m["bbox"]})
    for l in limits[:1]:
        script.append({"title": "4. 局限与后续", "text": l["text"], "page": l["page"], "bbox": l["bbox"]})

    return {
        "meta": meta,
        "abstract": abstract,
        "contributions": contributions,
        "metrics": metrics,
        "limitations": limits,
        "figures": figs,
        "terms": terms,
        "outline": outline,
        "script": script,
        "references_preview": [
            {"text": r["text"][:200], "page": r["page"]} for r in parsed.get("references", [])[:8]
        ],
        "stats": parsed["stats"],
    }


def brief_to_markdown(brief: dict) -> str:
    m = brief["meta"]
    lines = [f"# {m['title']}", ""]
    if m.get("authors"):
        lines += [f"**作者**：{m['authors']}", ""]
    lines += [
        f"**文件**：{m['file']}　**页数**：{m['pages']}　**段落**：{brief['stats']['n_paragraphs']}　"
        f"**章节**：{brief['stats']['n_sections']}　**图表**：{brief['stats']['n_captions']}　"
        f"**参考文献**：{brief['stats']['n_references']}",
        "",
        "## 摘要",
        brief["abstract"] or "（未识别到摘要段）",
        "",
        "## 关键贡献（带页码）",
    ]
    for c in brief["contributions"]:
        lines.append(f"- {c['text']}  `p.{c['page']}`")
    lines += ["", "## 关键结果"]
    for m2 in brief["metrics"]:
        lines.append(f"- {m2['text']}  `p.{m2['page']}`")
    lines += ["", "## 局限与后续工作"]
    for l in brief["limitations"]:
        lines.append(f"- {l['text']}  `p.{l['page']}`")
    lines += ["", "## 关键图表"]
    for f in brief["figures"]:
        lines.append(f"- {f['text']}  `p.{f['page']}`")
    lines += ["", "## 术语表（高频缩写）"]
    lines.append("| 术语 | 出现次数 | 首次出现 |")
    lines.append("| --- | --- | --- |")
    for t in brief["terms"]:
        lines.append(f"| {t['term']} | {t['count']} | p.{t['first_page']} |")
    lines += ["", "## 5 分钟讲稿"]
    for s in brief["script"]:
        lines.append(f"**{s['title']}**：{s['text']}  `p.{s['page']}`")
    lines += ["", "## 章节导航"]
    for o in brief["outline"]:
        lines.append(f"{'　' * (o['level'] - 1)}- {o['title']}  `p.{o['page']}`")
    lines.append("")
    lines.append("> 本材料由 PaperBrief 自动生成，每条要点均标注页码，可点回原文核查。")
    return "\n".join(lines)
