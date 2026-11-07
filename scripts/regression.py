# -*- coding: utf-8 -*-
"""9 题回归脚本（官方单篇 5 + 多篇 4），输出 Markdown 表，供技术文档直接引用。

用法：
    python scripts/regression.py --base http://1.14.98.248
    python scripts/regression.py            # 默认 http://127.0.0.1:8000
输出：控制台表格 + regression_YYYYMMDD.md
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.parse
import urllib.request

SINGLE = [
    ("①第 6 作者及机构", "这篇论文的第 6 作者是谁，在哪个公司或机构工作？"),
    ("②Adam β1/β2", "Adam 优化器的超参数 β1 和 β2 分别设置成了多少？"),
    ("③GPU 数/型号", "训练 Base 模型使用了多少块 GPU，具体是什么型号？"),
    ("④BLEU(big)", "Transformer (big) 在 English-to-German 上的 BLEU 分数是多少？"),
    ("⑤LayerNorm 文献", "请列出论文中引用 Layer Normalization 的参考文献。"),
]
MULTI = [
    ("⑥BERT 与 Transformer", "BERT 的架构和原始 Transformer 的 Encoder 有什么承袭关系？BERT 丢弃了哪个部分？"),
    ("⑦GPT 参数量", "对比 GPT-1、GPT-2、GPT-3 的模型参数量变化。"),
    ("⑧GPT-2 LayerNorm", "GPT-2 相比于 GPT-1，在 Transformer 的 LayerNorm 位置上做了什么关键调整？"),
    ("⑨上下文长度", "从 GPT-1 到 GPT-3，模型的最大上下文 Token 长度分别递增到了多少？"),
]


def _post(base, path, payload=None, timeout=240):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(base, path, timeout=240):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _clean(s, n=110):
    return (s or "").replace("\n", " ").replace("|", "/")[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    library = _get(base, "/api/library")["docs"]
    names = [d["name"] for d in library]
    if not names:
        print("库为空，无法回归")
        return 2

    def parse(name):
        return _post(base, "/api/parse?name=" + urllib.parse.quote(name))["key"]

    v7 = parse("1706.03762v7.pdf")
    bert = parse("N19-1423.pdf") if "N19-1423.pdf" in names else None
    gpt1 = parse("language_understanding_paper.pdf") if "language_understanding_paper.pdf" in names else None
    gpt3 = parse("2005.14165v4.pdf") if "2005.14165v4.pdf" in names else None

    rows = []
    for label, q in SINGLE:
        r = _post(base, "/api/ask?key=" + v7, {"q": q})
        pages = [a["page"] for a in r.get("anchors", [])]
        rows.append((label, q, r.get("mode", ""), pages, _clean(r.get("answer"))))

    pairs = [("⑥", bert, v7), ("⑦", gpt1, gpt3), ("⑧", gpt1, gpt3), ("⑨", gpt1, gpt3)]
    for (label, q), (_, k1, k2) in zip(MULTI, pairs):
        if not k1 or not k2:
            rows.append((label, q, "跳过（缺文档）", [], ""))
            continue
        r = _post(base, "/api/compare", {"k1": k1, "k2": k2, "q": q})
        pages = sorted({a["page"] for d in r.get("docs", []) for a in d.get("anchors", [])})
        rows.append((label, q, r.get("mode", ""), pages, _clean(r.get("answer"))))

    lines = ["# PaperBrief 官方 9 题回归（%s）" % base, "",
             "生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "",
             "| # | 问题 | 模式 | 锚点页 | 答案摘要 | 判定 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for label, q, mode, pages, ans in rows:
        pg = ",".join(str(p) for p in pages[:4]) if pages else "无"
        lines.append("| %s | %s | %s | %s | %s | 待人工判定 |" % (label, _clean(q, 40), mode, pg, ans))
    lines += ["", "> 判定列需人工核对答案是否正确（脚本只负责跑通与取证）。"]
    md = "\n".join(lines)

    print(md)
    out = args.out or ("regression_%s.md" % datetime.date.today().strftime("%Y%m%d"))
    open(out, "w", encoding="utf-8").write(md)
    print("\n已保存：%s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
