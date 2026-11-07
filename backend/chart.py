# -*- coding: utf-8 -*-
"""CSV → 图表 + 图注（命题 1.6.3「实验数据可视化」）。

零新依赖：用标准库 csv 解析、手写 SVG 绘图（服务器 2G 内存，装 matplotlib 不划算）。
同时生成一段可复现的 matplotlib 脚本，方便把图放回论文。
"""

from __future__ import annotations

import csv
import io
import math
import os
import re
from statistics import mean

PALETTE = ["#22d3ee", "#3b82f6", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#facc15"]


def _to_float(v):
    try:
        return float(str(v).strip().replace(",", ""))
    except Exception:
        return None


def parse_csv(path: str, max_rows: int = 5000) -> dict:
    raw = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    else:
        text = raw.decode("utf-8", "replace")
    sample = text[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except Exception:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(text), dialect))[:max_rows]
    rows = [r for r in rows if any((c or "").strip() for c in r)]
    if not rows:
        return {"error": "CSV 为空"}
    header = [c.strip() or f"col{i+1}" for i, c in enumerate(rows[0])]
    body = rows[1:]
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    columns = []
    for i, name in enumerate(header):
        vals = [r[i] for r in body]
        nums = [_to_float(v) for v in vals]
        numeric = sum(1 for x in nums if x is not None)
        columns.append({"name": name, "numeric": numeric >= max(2, len(vals) * 0.6),
                        "sample": vals[:3]})
    return {"columns": columns, "rows": body, "n_rows": len(body),
            "header": header}


def _summary(header, rows, x, ys) -> dict:
    xi = header.index(x)
    data = {}
    for y in ys:
        yi = header.index(y)
        pts = []
        for r in rows:
            xv = _to_float(r[xi])
            if xv is None:
                xv = float(len(pts))
            yv = _to_float(r[yi])
            if yv is None:
                continue
            pts.append((xv, yv))
        if pts:
            data[y] = pts
    return {"x": x, "series": {k: {"n": len(v), "min": min(p[1] for p in v),
                                   "max": max(p[1] for p in v),
                                   "mean": round(mean([p[1] for p in v]), 3),
                                   "first": v[0][1], "last": v[-1][1]} for k, v in data.items()}}


def build_svg(header, rows, x, ys, kind: str = "line", title: str = "") -> tuple:
    """返回 (svg 字符串, summary dict)。kind: line / bar / scatter。"""
    xi = header.index(x)
    series = []
    for idx, y in enumerate(ys):
        yi = header.index(y)
        pts = []
        for r in rows:
            yv = _to_float(r[yi])
            if yv is None:
                continue
            xv = _to_float(r[xi])
            pts.append((xv if xv is not None else float(len(pts)), yv, str(r[xi])))
        if pts:
            series.append({"name": y, "pts": pts, "color": PALETTE[idx % len(PALETTE)]})
    if not series:
        return "<svg xmlns='http://www.w3.org/2000/svg'></svg>", {"error": "没有可绘制的数值列"}

    W, H = 760, 420
    L, R, T, B = 64, 20, 34, 56
    pw, ph = W - L - R, H - T - B

    xs = [p[0] for s in series for p in s["pts"]]
    ys_all = [p[1] for s in series for p in s["pts"]]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys_all), max(ys_all)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1
    pad = (ymax - ymin) * 0.08 or 1
    ymin -= pad
    ymax += pad

    def sx(v):
        return L + (v - xmin) / (xmax - xmin) * pw

    def sy(v):
        return T + ph - (v - ymin) / (ymax - ymin) * ph

    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}' viewBox='0 0 {W} {H}' "
             f"font-family='-apple-system,Segoe UI,Microsoft YaHei,sans-serif'>",
             f"<rect width='{W}' height='{H}' fill='#0b1220' rx='10'/>"]
    # 网格 + y 轴刻度
    for i in range(5):
        f = i / 4
        yv = ymin + f * (ymax - ymin)
        yy = sy(yv)
        parts.append(f"<line x1='{L}' y1='{yy:.1f}' x2='{W-R}' y2='{yy:.1f}' stroke='#1e293b' stroke-width='1'/>")
        parts.append(f"<text x='{L-8}' y='{yy+4:.1f}' fill='#94a3b8' font-size='11' text-anchor='end'>{yv:.3g}</text>")
    # x 轴刻度
    for i in range(6):
        f = i / 5
        xv = xmin + f * (xmax - xmin)
        xx = sx(xv)
        label = f"{xv:.3g}"
        parts.append(f"<text x='{xx:.1f}' y='{T+ph+18}' fill='#94a3b8' font-size='11' text-anchor='middle'>{label}</text>")
    parts.append(f"<line x1='{L}' y1='{T+ph}' x2='{W-R}' y2='{T+ph}' stroke='#334155'/>")
    parts.append(f"<line x1='{L}' y1='{T}' x2='{L}' y2='{T+ph}' stroke='#334155'/>")
    parts.append(f"<text x='{L+pw/2:.0f}' y='{H-16}' fill='#cbd5e1' font-size='12' text-anchor='middle'>{x}</text>")
    if title:
        parts.append(f"<text x='{W/2:.0f}' y='22' fill='#e5e7eb' font-size='13' text-anchor='middle'>{title}</text>")

    for s in series:
        pts = sorted(s["pts"], key=lambda p: p[0])
        if kind == "bar":
            n = len(pts)
            bw = max(2.0, pw / max(1, n) * 0.7 / max(1, len(series)))
            for i, p in enumerate(pts):
                bx = L + (i + 0.5) / n * pw - bw / 2
                by = sy(p[1])
                parts.append(f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bw:.1f}' height='{T+ph-by:.1f}' "
                             f"fill='{s['color']}' opacity='0.85'/>")
        elif kind == "scatter":
            for p in pts:
                parts.append(f"<circle cx='{sx(p[0]):.1f}' cy='{sy(p[1]):.1f}' r='3.2' fill='{s['color']}' opacity='0.9'/>")
        else:
            path = " ".join(("M" if i == 0 else "L") + f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(pts))
            parts.append(f"<path d='{path}' fill='none' stroke='{s['color']}' stroke-width='2'/>")
            for p in pts[:200]:
                parts.append(f"<circle cx='{sx(p[0]):.1f}' cy='{sy(p[1]):.1f}' r='2.2' fill='{s['color']}'/>")

    # 图例
    lx = L + 6
    for s in series:
        parts.append(f"<rect x='{lx}' y='{T-24}' width='10' height='10' fill='{s['color']}' rx='2'/>")
        parts.append(f"<text x='{lx+14}' y='{T-15}' fill='#cbd5e1' font-size='11'>{s['name']}</text>")
        lx += 24 + len(s["name"]) * 7
    parts.append("</svg>")
    return "\n".join(parts), _summary(header, rows, x, ys)


def matplotlib_snippet(path_name: str, header, x, ys, kind) -> str:
    """生成可复现的 Python 绘图脚本（给用户贴进论文/复现）。"""
    cols = ", ".join([x] + list(ys))
    plot = "plot" if kind == "line" else ("bar" if kind == "bar" else "scatter")
    yargs = ", ".join([f"df['{y}']" for y in ys])
    if len(ys) > 1 and kind != "scatter":
        yargs = "[" + ", ".join([f"df['{y}']" for y in ys]) + "]"
    return f"""# PaperBrief 自动生成：可复现绘图脚本
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("{path_name}")
plt.figure(figsize=(7, 4))
df.{plot}(x='{x}', y={yargs})
plt.xlabel("{x}")
plt.ylabel("{' / '.join(ys)}")
plt.title("{path_name}")
plt.legend()
plt.tight_layout()
plt.savefig("figure.png", dpi=200)
plt.show()
# 数据列：{cols}
"""
