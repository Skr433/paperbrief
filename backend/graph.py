# -*- coding: utf-8 -*-
"""引文图谱：直接以「PDF 里解析出的参考文献」为节点，用 Crossref 补全元数据与共被引关系。

为什么不用「先搜根论文再取 references」：
- 同名论文多（Crossref 里 "Attention Is All You Need" 有 2025 年的同名文章），根节点容易认错；
- 我们已经有 PyMuPDF 解析出的 40 条参考文献，直接查更准，也更符合"从原文出证据"的产品逻辑。

节点：根论文 + 最多 20 条参考文献
边：root→ref（cites）；参考文献之间若互相引用同一篇（共被引）则加 co-cite 边
排序：PageRank
"""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "paperbrief/0.7 (mailto:dev@example.com)"
CROSSREF = "https://api.crossref.org"


def _get(url: str, timeout: int = 20) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt == 0:
                time.sleep(2)
                continue
            raise
        except Exception:
            if attempt == 0:
                time.sleep(1)
                continue
            raise
    raise RuntimeError("crossref 请求失败")



def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()




def guess_title(ref_text: str) -> str:
    """未匹配 Crossref 时，启发式去掉开头作者串，留下题名。"""
    txt = re.sub(r"^\s*\[\d+\]\s*", "", ref_text or "").strip()
    m = re.match(r"^[A-Z][^.]{0,140}\.\s+(.+)$", txt)
    if m:
        txt = m.group(1)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:120]

def _match_item(ref_text: str, items: list[dict]) -> dict | None:
    """把一条参考文献文本匹配到 Crossref 结果：标题被包含 > 标题相似度。"""
    low = ref_text.lower()
    best, best_score = None, 0.0
    for it in items:
        title = _clean((it.get("title") or [""])[0])
        if not title:
            continue
        tl = title.lower()
        cited = it.get("is-referenced-by-count", 0) or 0
        if len(tl) > 12 and tl in low:
            score = 2.0 + min(cited, 5000) / 5000.0
        else:
            score = difflib.SequenceMatcher(None, low[:200], tl).ratio()
            if score < 0.55:
                continue
        if score > best_score:
            best, best_score = it, score
    return best


def _resolve_ref(ref_text: str) -> dict | None:
    """去掉 [n] 前缀与页码后查询 Crossref。"""
    txt = re.sub(r"^\s*\[\d+\]\s*", "", ref_text)
    txt = re.sub(r"\s*\d{1,4}\s*$", "", txt)
    q = urllib.parse.quote(_clean(txt)[:200])
    try:
        data = _get(f"{CROSSREF}/works?query.bibliographic={q}&rows=3"
                    f"&select=DOI,title,issued,is-referenced-by-count,author,reference")
    except Exception:
        return None
    items = data.get("message", {}).get("items", [])
    it = _match_item(ref_text, items)
    if not it:
        return None
    issued = (it.get("issued") or {}).get("date-parts") or [[None]]
    year = str(issued[0][0]) if issued and issued[0] and issued[0][0] else ""
    authors = it.get("author") or []
    first = ""
    if authors:
        a = authors[0]
        first = (a.get("family") or a.get("name") or "")[:40]
    return {
        "doi": (it.get("DOI") or "").lower(),
        "title": _clean((it.get("title") or [""])[0])[:140],
        "year": year,
        "cited_by": it.get("is-referenced-by-count", 0) or 0,
        "first_author": first,
        "refs": [((r.get("DOI") or "").lower()) for r in (it.get("reference") or []) if r.get("DOI")],
    }



# 主题聚类规则（按顺序匹配，先命中先归类；用于给节点上色）
THEME_RULES = [
    ("注意力/Transformer", ["attention", "transformer", "self-attention", "multi-head"]),
    ("机器翻译/NMT", ["translation", "nmt", "bleu", "wmt", "bilingual", "decoder"]),
    ("序列建模/RNN", ["rnn", "lstm", "long short-term", "recurrent", "sequence to sequence", "seq2seq", "gru", "memory network"]),
    ("优化与训练", ["optimiz", "adam", "sgd", "dropout", "regulariz", "learning rate", "batch norm", "layer norm", "normalization", "training"]),
    ("表示学习/嵌入", ["embedding", "word2vec", "glove", "representation", "vector space", "semantic"]),
    ("语言模型/预训练", ["language model", "gpt", "bert", "pre-train", "pretrain", "masked"]),
    ("记忆与推理", ["memory", "reasoning", "knowledge", "graph"]),
]
THEME_COLORS = {
    "注意力/Transformer": "#22d3ee",
    "机器翻译/NMT": "#3b82f6",
    "序列建模/RNN": "#a78bfa",
    "优化与训练": "#f59e0b",
    "表示学习/嵌入": "#34d399",
    "语言模型/预训练": "#f472b6",
    "记忆与推理": "#facc15",
    "其他/背景": "#64748b",
}


def classify_theme(text: str) -> str:
    low = (text or "").lower()
    for name, keys in THEME_RULES:
        if any(k in low for k in keys):
            return name
    return "其他/背景"


def pagerank(nodes: list[dict], edges: list[dict], damping: float = 0.85, iters: int = 30) -> dict:
    n = max(1, len(nodes))
    idx = {nd["id"]: i for i, nd in enumerate(nodes)}
    out_links: dict[str, list[str]] = {nd["id"]: [] for nd in nodes}
    for e in edges:
        if e["source"] in out_links and e["target"] in idx:
            out_links[e["source"]].append(e["target"])
    score = {nd["id"]: 1.0 / n for nd in nodes}
    for _ in range(iters):
        new = {nd["id"]: (1 - damping) / n for nd in nodes}
        for nd in nodes:
            outs = out_links[nd["id"]]
            if not outs:
                continue
            share = damping * score[nd["id"]] / len(outs)
            for t in outs:
                new[t] += share
        score = new
    return score




_STOP = set("the a an of and for to in on with by from at as is are was were be been using based via its their this that these those we our".split())


def _title_tokens(title: str) -> set:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (title or "").lower())
    return {w for w in words if w not in _STOP}


def add_similar_edges(nodes: list[dict], edges: list[dict], threshold: float = 0.16) -> int:
    """基于标题词重合度生成"相似关联"边（无外部引文数据时的可视化关系）。"""
    refs = [nd for nd in nodes if not nd.get("is_root")]
    toks = {nd["id"]: _title_tokens(nd.get("title", "")) for nd in refs}
    added = 0
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = refs[i], refs[j]
            ta, tb = toks[a["id"]], toks[b["id"]]
            if len(ta) < 2 or len(tb) < 2:
                continue
            inter = len(ta & tb)
            union = len(ta | tb)
            sim = inter / union if union else 0
            if sim >= threshold:
                edges.append({"source": a["id"], "target": b["id"], "type": "similar",
                              "weight": round(sim, 3)})
                added += 1
    return added


def build_graph(meta_title: str, references: list[dict], max_refs: int = 20) -> dict:
    """references: [{text,page,bbox}]，来自 PDF 解析结果。"""
    refs = [r for r in (references or []) if (r.get("text") or "").strip()][:max_refs]
    if not refs:
        return {"error": "该论文未识别到参考文献，无法构建图谱"}

    nodes = [{"id": "root", "is_root": True, "title": _clean(meta_title)[:140],
              "year": "", "doi": "", "cited_by": 0}]
    edges = []
    enriched: list[tuple] = []
    # 并发查询 Crossref（串行 20 条约 50s，并发约 5s）
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        resolved = list(ex.map(lambda r: _resolve_ref(_clean(r["text"])), refs))
    for i, r in enumerate(refs):
        text = _clean(r["text"])
        nid = f"r{i}"
        info = resolved[i]
        node = {
            "id": nid, "is_root": False, "page": r.get("page"),
            "title": (info or {}).get("title") or guess_title(text) or "(未匹配)",
            "year": (info or {}).get("year", ""),
            "doi": (info or {}).get("doi", ""),
            "cited_by": (info or {}).get("cited_by", 0),
            "authors": (info or {}).get("first_author", ""),
            "raw": text[:200],
            "matched": bool(info),
            "theme": classify_theme(((info or {}).get("title") or "") + " " + text),
        }
        nodes.append(node)
        edges.append({"source": "root", "target": nid, "type": "cites"})
        if info and info.get("refs"):
            enriched.append((nid, set(info["refs"])))

    # 共被引：两篇参考文献都引用了同一篇文献
    for a in range(len(enriched)):
        for b in range(a + 1, len(enriched)):
            nid_a, set_a = enriched[a]
            nid_b, set_b = enriched[b]
            if set_a & set_b:
                edges.append({"source": nid_a, "target": nid_b, "type": "co-cite"})

    add_similar_edges(nodes, edges)
    from collections import Counter
    theme_counts = Counter(nd.get("theme") or "其他/背景" for nd in nodes if not nd.get("is_root"))
    themes = [{"name": k, "count": v, "color": THEME_COLORS.get(k, "#64748b")}
              for k, v in theme_counts.most_common()]
    scores = pagerank(nodes, edges)
    for nd in nodes:
        nd["score"] = round(scores.get(nd["id"], 0.0), 4)
    nodes.sort(key=lambda x: (not x["is_root"], -(x.get("score") or 0)))
    matched = sum(1 for nd in nodes if nd.get("matched"))
    return {
        "root": {"title": _clean(meta_title), "year": ""},
        "nodes": nodes, "edges": edges,
        "matched": matched, "total_refs": len(refs), "themes": themes,
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Crossref",
    }
