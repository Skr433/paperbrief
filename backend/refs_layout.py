# -*- coding: utf-8 -*-
"""版式参考文献抽取：列边距判条目首行 + 跨页续行合并 + 页眉页脚过滤。

用于旧逻辑明显失效的文档（T5 作者起首风格、UDL 教材多栏）。
"""
import re


# 版式参考文献抽取（从 pdf_parse.py 拆出，保持单文件 <20KB）
_REF_HEAD = re.compile(r"^(references|bibliography|参\s*考\s*文\s*献)\s*[:：]?$", re.I)
_REF_KEY = re.compile(r"^\[[\w\-+]{1,12}\]\s|^\d{1,3}[.)]\s")
_REF_SKIP = re.compile(r"^(bib(liography)?|references|index|\d{1,4}|[ivxlcdm]{1,5})$", re.I)
_REF_FOOT = re.compile(r"draft:|please send errata|creative commons|downloaded from|^\d{1,4}$", re.I)


def _page_lines(page):
    d = page.get_text("dict")
    out = []
    for b in d.get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            spans = ln.get("spans", [])
            if not spans:
                continue
            text = "".join(s.get("text", "") for s in spans).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            out.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "text": text})
    return out


def _repeated_edges(doc):
    from collections import Counter
    cnt = Counter()
    n = len(doc)
    for i in range(n):
        page = doc[i]
        H = page.rect.height
        for l in _page_lines(page):
            if l["y0"] < 120 or l["y0"] > H - 80:
                cnt[l["text"].strip()[:80]] += 1
    return {t for t, c in cnt.items() if c > max(3, n * 0.3)}


def _refs_by_layout(doc) -> list[dict]:
    """按版式抽取参考文献：列边距判条目首行 + 跨页续行合并 + 页眉页脚过滤。"""
    n = len(doc)
    start = None
    start_y = 0.0
    for i in range(int(n * 0.2), n):
        lines = _page_lines(doc[i])
        hit = None
        for l in lines:
            if _REF_HEAD.match(l["text"].strip()):
                hit = l
                break
        if hit is not None:
            start = i
            start_y = hit["y0"]
            break
    if start is None:
        return []
    skip = _repeated_edges(doc)
    refs, seen = [], set()
    cur = None

    def flush():
        nonlocal cur
        if cur and len(cur["text"]) >= 30:
            key = re.sub(r"\s+", " ", cur["text"])[:60].lower()
            if key not in seen:
                seen.add(key)
                refs.append(cur)
        cur = None

    for pno in range(start, n):
        page = doc[pno]
        H = page.rect.height
        lines = []
        floor_y = (start_y - 2) if pno == start else 50
        for l in _page_lines(page):
            txt = l["text"].strip()
            if not (floor_y < l["y0"] < H - 35):
                continue
            if _REF_SKIP.match(txt) or _REF_FOOT.search(txt) or txt[:80] in skip:
                continue
            lines.append(l)
        if not lines:
            continue
        if any(re.match(r"^(index|索引)$", l["text"].strip(), re.I) for l in lines[:8]):
            break
        margins = []
        for l in sorted(lines, key=lambda x: x["x0"]):
            if not margins or l["x0"] - margins[-1] > 40:
                margins.append(l["x0"])

        def band_of(x0):
            best = margins[0]
            for m in margins:
                if x0 >= m - 1:
                    best = m
            return best

        for l in sorted(lines, key=lambda x: (band_of(x["x0"]), x["y0"])):
            m = band_of(l["x0"])
            is_start = (l["x0"] - m <= 3.5) or bool(_REF_KEY.match(l["text"]))
            if is_start and len(l["text"]) > 20:
                flush()
                cur = {"page": pno + 1,
                       "bbox": [round(l["x0"], 2), round(l["y0"], 2),
                                round(l["x1"], 2), round(l["y1"], 2)],
                       "text": l["text"]}
            elif cur is not None:
                cur["text"] = (cur["text"] + " " + l["text"])[:900]
        low = " ".join(l["text"].lower() for l in lines[:6])
        if pno > start + 1 and ("appendix" in low or "附录" in low):
            break
    flush()
    # 尾处理：无年份/短碎片并入上一条（跨页断行残留）
    final = []
    for r in refs:
        txt = r["text"].strip()
        if final and len(txt) < 200 and not re.search(r"\b(19|20)\d{2}\b", txt) \
                and "arxiv" not in txt.lower() and "http" not in txt.lower():
            final[-1]["text"] = (final[-1]["text"] + " " + txt)[:900]
        else:
            final.append(r)
    return final


def refs_by_layout(doc) -> list[dict]:
    return _refs_by_layout(doc)


def fallback_refs(doc) -> list[dict]:
    """参考文献回退：从出现 References/参考文献 的页开始，按 block 切分条目。"""
    start = None
    n = len(doc)
    # 从文末往前找"独立成行"的 References 标题页（避免正文里提到 references）
    for i in range(n - 1, max(0, int(n * 0.35)) - 1, -1):
        try:
            txt = doc[i].get_text("text") or ""
        except Exception:
            continue
        lines = [ln.strip().lower().strip(":：.") for ln in txt.splitlines()]
        if any(ln in ("references", "bibliography", "参考文献") or ln.startswith("references ") and len(ln) < 30 for ln in lines):
            start = i
            break
        if re.search(r"参\s*考\s*文\s*献", txt):
            start = i
            break
    if start is None:
        return []
    out, seen = [], set()
    started = False
    for pno in range(start, len(doc)):
        page = doc[pno]
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            continue
        for b in blocks:
            if len(b) < 5:
                continue
            text = (b[4] or "").strip()
            if len(text) < 30:
                continue
            bbox = [round(float(b[0]), 2), round(float(b[1]), 2),
                    round(float(b[2]), 2), round(float(b[3]), 2)]
            parts = re.split(r"(?=\[\d+\]\s)|(?=\n\d{1,3}[.)]\s)", "\n" + text)
            for part in parts:
                part = part.strip()
                if len(part) < 25:
                    continue
                if not started:
                    if len(part) > 40 and REF_LINE_RE.search(part):
                        started = True
                    else:
                        continue
                key = (pno, part[:40])
                if key in seen:
                    continue
                seen.add(key)
                if out and (part[:1].islower() or out[-1]["text"].endswith("-")) and len(part) < 200:
                    out[-1]["text"] = (out[-1]["text"] + " " + part)[:600]
                    continue
                out.append({"page": pno + 1, "bbox": bbox, "text": part[:600]})
                if len(out) >= 300:
                    return out
    return out
