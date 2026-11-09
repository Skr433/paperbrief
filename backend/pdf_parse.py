# -*- coding: utf-8 -*-
"""PDF 解析模块：把论文切成段落/章节/图表/参考文献，并保留坐标（锚点）。

设计原则（对应命题 1.1）：
- 一切解析结果都带 page + bbox，后续任何"引用/问答/汇报要点"都能点回原文；
- 段落是检索与定位的最小单位，**按 PyMuPDF 的 block 顺序合并，避免双栏串行**；
- 不依赖外部服务，离线可跑（PyMuPDF）。
"""

from __future__ import annotations

from refs_layout import fallback_refs, refs_by_layout
from figures import figures_from_drawings

import hashlib
import os
import re
import unicodedata
from collections import Counter

# 解析版本号：改动解析逻辑时递增，服务端会据此自动失效旧缓存（避免"改了代码还看到旧结果"）
PARSER_VERSION = "0.2.12"

AFFIL_NAMES = [
    "Google Brain", "Google Research", "Google", "University of Toronto", "University of Washington",
    "OpenAI", "DeepMind", "Microsoft Research", "Meta AI", "Facebook AI Research", "NVIDIA",
    "IBM Research", "Stanford University", "MIT", "Carnegie Mellon University", "UC Berkeley",
    "Tsinghua University", "Peking University",
]

def clean_affiliations(text: str) -> str:
    """把首页脚注里的机构去重成 "Google Brain / Google Research / University of Toronto"。"""
    if not text:
        return ""
    found = []
    low = text.lower()
    for name in AFFIL_NAMES:
        if name.lower() in low and name not in found:
            found.append(name)
    # 去掉被更长机构名包含的短名（如 "Google" 已被 "Google Brain" 覆盖）
    found = [n for n in found if not any(n != m and n.lower() in m.lower() for m in found)]
    if found:
        return " / ".join(found)
    seen, out = set(), []
    for w in text.split():
        if w not in seen:
            seen.add(w)
            out.append(w)
    return " ".join(out)[:200]
MATH_FONT_HINTS = ("CMMI", "CMSY", "CMEX", "CMR", "MSAM", "MSBM", "STIX", "Symbol", "Math")
CAPTION_RE = re.compile(r"^((figure|fig\.|table|algorithm|listing)\s*[\dA-Z]|(图|表)\s*\d)", re.I)
REF_HEADING_RE = re.compile(r"^(references|bibliography|参\s*考\s*文\s*献|引用文献)\s*[:：]?$", re.I)

# 参考文献行特征：编号开头，或含 arXiv/Proceedings 等出版信息
REF_LINE_RE = re.compile(
    r"^\[\d+\]|^\d+[.)]\s|arXiv|Proceedings|Conference on|pp\.|vol\.|et al\.|"
    r"University Press|Springer|IEEE|ACL|NeurIPS|ICML|ICLR|EMNLP|NAACL",
    re.I,
)
NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+){0,3})\s+[A-Z\u4e00-\u9fff]")
META_LINE_RE = re.compile(
    r"^(基金项目|作者简介|通信作者|收稿日期|DOI|Tel\.?|E-mail|中图分类号|文献标志码|文章编号|关键词|关键字)",
    re.I,
)
WATERMARK_RE = re.compile(
    r"^(arxiv:\d{4}\.\d{4,}v\d+)|(provided proper attribution)|(reproduce the tables and figures)|"
    r"(scholarly works)|(this work is licensed)|(creative commons)",
    re.I,
)
_TOKEN = re.compile(r"[A-Za-z0-9βγδαεζηθικλμνξοπρστυφχψω]+")

def norm_space(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").replace("\u00ad", "")
    s = re.sub(r"[ \t\u3000]+", " ", s)
    return s.strip()

def norm_text(s: str) -> str:
    s = norm_space(s).lower()
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def tokens(s: str) -> list[str]:
    """英文按词、模型名（gpt-1 等）保整体、中文按二元组切分。"""
    t = norm_space(s).lower()
    out = re.findall(r"[a-z]+-\d+", t)  # gpt-1 / gpt-3 / bert-base 这类模型名保留整体
    out += _TOKEN.findall(t)
    for seg in re.findall(r"[\u4e00-\u9fff]+", t):
        if len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i: i + 2] for i in range(len(seg) - 1))
    return out

def _page_blocks(page_dict: dict, page_h: float) -> list[dict]:
    """把一页拆成 block → lines，保持 PyMuPDF 的阅读顺序（不跨栏重排）。"""
    blocks = []
    for b in page_dict.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines = []
        for line in b.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            # 竖排水印/侧边文字（arxiv 印刷标记）不进入正文
            d = line.get("dir") or (1, 0)
            if abs(d[1]) > 0.1:
                continue
            text = "".join(sp["text"] for sp in spans)
            t = norm_space(text)
            if not t or WATERMARK_RE.match(t):
                continue
            bbox = [round(v, 2) for v in line["bbox"]]
            # 页眉页脚：靠边且短
            if (bbox[1] < page_h * 0.045 or bbox[3] > page_h * 0.96) and len(t) < 90:
                continue
            if re.fullmatch(r"[\d\s\-–—]+", t):
                continue
            size = round(max(sp["size"] for sp in spans), 1)
            font = " ".join(sp["font"] for sp in spans)
            lines.append(
                {
                    "text": t,
                    "bbox": bbox,
                    "size": size,
                    "font": font,
                    "math": any(h in font for h in MATH_FONT_HINTS),
                    "bold": bool(re.search(r"bold|black|semibold", font, re.I)),
                }
            )
        if lines:
            blocks.append({"bbox": [round(v, 2) for v in b["bbox"]], "lines": lines})
    return blocks

def _body_size(all_blocks: list[list[dict]]) -> float:
    cnt = Counter()
    for blocks in all_blocks[1:6]:
        for b in blocks:
            for ln in b["lines"]:
                if len(ln["text"]) >= 20:
                    cnt[ln["size"]] += len(ln["text"])
    if not cnt:
        for blocks in all_blocks[:2]:
            for b in blocks:
                for ln in b["lines"]:
                    cnt[ln["size"]] += max(1, len(ln["text"]))
    return cnt.most_common(1)[0][0] if cnt else 10.0

def _heading_level(text: str, size: float, body: float, bold: bool = False) -> int:
    t = norm_space(text)
    if not t or len(t) > 140:
        return 0
    if META_LINE_RE.match(t):
        return 0
    if CAPTION_RE.match(t):
        return 0
    if REF_HEADING_RE.match(t):
        return 1
    if size >= body * 1.45:
        return 1
    m = NUMBERED_HEADING_RE.match(t)
    if m:
        return min(t.split()[0].count(".") + 1, 3)
    if size >= body * 1.12:
        return 2
    # 粗体短行：Abstract / Introduction 这类无编号标题（中文需排除标点与数字）
    if bold and 2 <= len(t) <= 28 and not re.search(r"[：:；;，,、（(【\[]", t):
        return 2
    if t.isupper() and 3 <= len(t) <= 60:
        return 2
    return 0

def _blocks_to_paragraphs(blocks: list[dict], body: float) -> list[dict]:
    """每个 block 内部按行合并；块之间不合并（双栏安全）。"""
    out = []
    for b in blocks:
        cur = None
        for ln in b["lines"]:
            t = ln["text"]
            lvl = _heading_level(t, ln["size"], body, ln.get("bold", False))
            if lvl:
                if cur:
                    out.append(cur)
                    cur = None
                out.append({"kind": "heading", "level": lvl, "text": t, "bbox": ln["bbox"], "size": ln["size"]})
                continue
            if cur is None:
                cur = {"kind": "body", "text": t, "bbox": list(ln["bbox"]), "size": ln["size"],
                       "math": ln["math"]}
                continue
            if cur["text"].endswith("-"):
                cur["text"] = cur["text"][:-1] + t
            elif cur["math"] and ln["math"]:
                cur["text"] += " " + t
            else:
                cur["text"] += " " + t
            cur["bbox"][2] = max(cur["bbox"][2], ln["bbox"][2])
            cur["bbox"][3] = max(cur["bbox"][3], ln["bbox"][3])
            cur["math"] = cur["math"] and ln["math"]
        if cur:
            out.append(cur)
    for p in out:
        p["text"] = norm_space(p["text"])
    return out

# ---------------------------------------------------------------- 主入口

def parse_pdf(path: str) -> dict:
    import pymupdf

    doc = pymupdf.open(path)
    page_blocks = [_page_blocks(p.get_text("dict"), p.rect.height) for p in doc]
    body = _body_size(page_blocks)

    title = ""
    authors_line = ""
    author_list: list[str] = []
    affiliations = ""
    authors_bbox = [0, 0, 0, 0]
    AFFIL_RE = re.compile(r"(google|university|institute|research|brain|openai|deepmind|microsoft|meta|ibm|nvidia|college|academy|laborator)", re.I)
    if page_blocks and page_blocks[0]:
        first_lines = [ln for b in page_blocks[0] for ln in b["lines"]]
        first_lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        big = [ln for ln in first_lines if ln["size"] >= body * 1.4 and len(ln["text"]) >= 6]
        if big:
            tsize = max(ln["size"] for ln in big)
            tlines = [ln for ln in big if ln["size"] >= tsize - 0.6]
            title = norm_space(" ".join(ln["text"] for ln in tlines[:3]))
            last_y = max(ln["bbox"][3] for ln in tlines)
            below = [
                ln
                for ln in first_lines
                if ln["bbox"][1] > last_y - 2
                and body * 0.85 <= ln["size"] <= tsize * 0.95
                and 4 <= len(ln["text"]) <= 220
            ]
            auth_parts, affil_parts = [], []
            for ln in below:
                if re.search(r"abstract|摘要", ln["text"], re.I):
                    break
                if AFFIL_RE.search(ln["text"]) and len(ln["text"]) > 10 and "∗" not in ln["text"]:
                    affil_parts.append(ln["text"])
                    continue
                if len(auth_parts) < 10:
                    auth_parts.append(ln["text"])
                    authors_bbox = ln["bbox"] if authors_bbox == [0, 0, 0, 0] else authors_bbox
            authors_line = norm_space(" ".join(auth_parts))
            authors_line = re.sub(r"\S+@\S+", " ", authors_line)
            authors_line = norm_space(authors_line)[:400]
            affiliations = clean_affiliations(norm_space(" ".join(affil_parts)))[:200]
            # 作者列表：按 ∗ 等上标符号 / 逗号切分；遇到机构名即停止
            names = []
            for c in re.split(r"[∗\*†‡§¶]|\s*,\s*", authors_line):
                c = norm_space(c).strip(" ,;0123456789")
                if not c:
                    continue
                if AFFIL_RE.search(c):
                    break
                if 5 <= len(c) <= 60 and re.search(r"[A-Za-z]{2,}\s+[A-Za-z]", c):
                    names.append(c)
            author_list = names

            # 中文论文：标题含 CJK 时，用「标题下方短行」抽中文姓名（英文规则会抓到单位与摘要）
            if re.search(r"[\u4e00-\u9fff]", title):
                cjk_lines = []
                for ln in first_lines:
                    if ln["bbox"][1] <= last_y - 2:
                        continue
                    t = ln["text"]
                    if re.search(r"摘\s*要|关键词|Abstract|Key\s*words|中图分类号", t, re.I):
                        break
                    cjk_lines.append(t)
                    if len(cjk_lines) >= 8:
                        break
                cjk_names = []
                for t in cjk_lines:
                    if len(t) > 30:
                        continue
                    if re.search(r"大学|学院|中心|研究所|实验室|上海|北京|基金|作者简介|通信作者|收稿|DOI", t):
                        continue
                    for nm in re.findall(r"[\u4e00-\u9fff]{2,4}", t):
                        if nm not in cjk_names:
                            cjk_names.append(nm)
                if cjk_names:
                    author_list = cjk_names[:6]
                    authors_line = "、".join(author_list)
                    for ln in first_lines:
                        if re.search(r"大学|学院|研究所", ln["text"]) and len(ln["text"]) < 200:
                            affiliations = affiliations or norm_space(ln["text"])[:200]
                            break

    paragraphs: list[dict] = []
    sections: list[dict] = []
    captions: list[dict] = []
    figures: list[dict] = []
    _fig_seen = set()
    in_refs = False
    refs_start = 0

    for pno, (page, blocks) in enumerate(zip(doc, page_blocks), start=1):
        for p in _blocks_to_paragraphs(blocks, body):
            text = p["text"]
            if not text:
                continue
            item = {"page": pno, "bbox": p["bbox"], "text": text, "size": p.get("size", body)}
            if p["kind"] == "heading":
                stripped = text.strip()
                if REF_HEADING_RE.match(stripped):
                    # 只看文档后段，避免目录页/早期章节误开参考区
                    if pno > len(page_blocks) * 0.4:
                        in_refs = True
                        refs_start = pno
                    sections.append({"sid": len(sections), "title": text, "level": 1,
                                     "page": pno, "bbox": p["bbox"], "para_index": len(paragraphs)})
                    continue
                if in_refs and pno > refs_start + 1 and re.search(
                        r"appendix|appendices|supplementary|附录|^index$|索引", stripped, re.I):
                    in_refs = False
                sections.append({"sid": len(sections), "title": text, "level": p["level"],
                                 "page": pno, "bbox": p["bbox"], "para_index": len(paragraphs)})
                paragraphs.append({**item, "kind": "heading"})
                continue
            if CAPTION_RE.match(text):
                captions.append({**item})
                paragraphs.append({**item, "kind": "caption"})
            elif in_refs:
                paragraphs.append({**item, "kind": "reference"})
            else:
                paragraphs.append({**item, "kind": "body"})

        pw, ph = page.rect.width, page.rect.height
        for img in page.get_images(full=True):
            try:
                rects = page.get_image_rects(img[0])
            except Exception:
                rects = []
            for r in rects:
                x0, y0 = max(0.0, r.x0), max(0.0, r.y0)
                x1, y1 = min(pw, r.x1), min(ph, r.y1)
                if x1 - x0 > 40 and y1 - y0 > 40:
                    bbox = [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
                    key = (pno, tuple(bbox))
                    if key not in _fig_seen:
                        _fig_seen.add(key)
                        figures.append({"page": pno, "bbox": bbox})

    # 参考文献条目合并：无编号的续行并入上一条
    refs: list[dict] = []
    for p in paragraphs:
        if p["kind"] != "reference":
            continue
        t = p["text"]
        if re.match(r"^(\[\d+\]|\d+[.)]\s)", t) or not refs:
            refs.append({"page": p["page"], "bbox": p["bbox"], "text": t})
        else:
            refs[-1]["text"] += " " + t
            refs[-1]["text"] = refs[-1]["text"][:600]

    # 回退：部分论文 References 标题未被识别为 heading；从文末倒扫参考文献样式段落
    refs_source = "paragraphs"
    if len(refs) < 5:
        refs_source = "tail"
        tail = [p for p in paragraphs
                if p.get("kind") in ("body", "reference") and (p.get("text") or "").strip()]
        picked, misses = [], 0
        for p in reversed(tail):
            txt = p["text"].strip()
            if len(txt) > 30 and REF_LINE_RE.search(txt):
                picked.append({"page": p["page"], "bbox": p["bbox"], "text": txt[:600]})
                misses = 0
            else:
                misses += 1
                if misses >= 3 and picked:
                    break
                if misses >= 10 and not picked:
                    break
        if len(picked) >= 5:
            refs = list(reversed(picked))

    if len(refs) < 5:
        fb = fallback_refs(doc)
        if len(fb) >= 5:
            refs = fb
            refs_source = "fallback"

    # 旧逻辑明显失效（T5 作者起首、UDL 教材、附录混入）→ 版式抽取
    if len(refs) < 10 or (len(page_blocks) > 100 and len(refs) < 200) \
            or (refs_source != "paragraphs" and len(page_blocks) >= 8):
        try:
            _lay = refs_by_layout(doc)
            if len(_lay) > max(30, len(refs) * 2) or (
                    refs_source != "paragraphs" and len(_lay) >= 30):
                refs = _lay
        except Exception:
            pass

    meta = {
        "file": os.path.basename(path),
        "pages": len(page_blocks),
        "sha1": hashlib.sha1(open(path, "rb").read()).hexdigest()[:16],
        "size_kb": round(os.path.getsize(path) / 1024, 1),
        "title": title or os.path.splitext(os.path.basename(path))[0],
        "authors": authors_line,
        "author_list": author_list,
        "affiliations": affiliations,
        "authors_bbox": authors_bbox,
        "body_font_size": body,
    }
    doc.close()

    # 纯矢量图 PDF（T5 等）：raster 图为 0 时用 drawings 兜底
    if not figures:
        try:
            import pymupdf as _mp
            _vp = _mp.open(path)
            figures = figures_from_drawings(_vp)
            _vp.close()
        except Exception:
            figures = []

    captions = captions[:300]
    figures = figures[:300]
    return {
        "meta": meta,
        "sections": sections,
        "paragraphs": paragraphs,
        "captions": captions,
        "figures": figures,
        "references": refs[:300],
        "stats": {
            "n_paragraphs": len(paragraphs),
            "n_sections": len(sections),
            "n_captions": len(captions),
            "n_figures": len(figures),
            "n_references": len(refs),
            "n_chars": sum(len(p["text"]) for p in paragraphs),
        },
    }

# ---------------------------------------------------------------- 去重指纹

def _shingles(text: str, k: int = 5) -> list[str]:
    ws = norm_text(text).split()
    if len(ws) < k:
        return [" ".join(ws)] if ws else []
    return [" ".join(ws[i : i + k]) for i in range(len(ws) - k + 1)]

def fingerprint(parsed: dict, k: int = 5) -> list[int]:
    """64 位 MinHash 签名，用于近似判重（v1/v7 这类不同版本也能识别）。"""
    import random

    text = " ".join(p["text"] for p in parsed["paragraphs"] if p["kind"] in ("body", "reference"))[:200000]
    sh = _shingles(text, k)
    if not sh:
        return [0] * 64
    rnd = random.Random(20261031)
    params = [(rnd.randrange(1, 2**31 - 1), rnd.randrange(0, 2**31 - 1)) for _ in range(64)]
    hs = [(hash(s) & 0xFFFFFFFF) for s in sh]
    return [min((a * h + b) % (2**31 - 1) for h in hs) for a, b in params]

def jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    if not sig_a or not sig_b:
        return 0.0
    return sum(1 for x, y in zip(sig_a, sig_b) if x == y) / len(sig_a)
