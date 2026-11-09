# -*- coding: utf-8 -*-
"""T5 等纯矢量图 PDF 的补充检测：用 drawings 聚类出图区域（仅当 raster 图为 0 时启用）。"""
import re


def figures_from_drawings(doc, limit: int = 30) -> list[dict]:
    out, seen = [], set()
    for pno, page in enumerate(doc, start=1):
        try:
            drawings = page.get_drawings()
        except Exception:
            continue
        if len(drawings) < 20:
            continue
        rects = [d["rect"] for d in drawings if d.get("rect") is not None]
        if not rects:
            continue
        x0 = min(r.x0 for r in rects)
        y0 = min(r.y0 for r in rects)
        x1 = max(r.x1 for r in rects)
        y1 = max(r.y1 for r in rects)
        pw, ph = page.rect.width, page.rect.height
        area = (x1 - x0) * (y1 - y0) / max(pw * ph, 1)
        if area < 0.03:
            continue
        text = page.get_text("text") or ""
        if not re.search(r"(figure|fig\.)\s*\d+", text, re.I):
            continue
        bbox = [round(max(0.0, x0), 2), round(max(0.0, y0), 2),
                round(min(pw, x1), 2), round(min(ph, y1), 2)]
        key = (pno, tuple(bbox))
        if key in seen:
            continue
        seen.add(key)
        out.append({"page": pno, "bbox": bbox})
        if len(out) >= limit:
            break
    return out
