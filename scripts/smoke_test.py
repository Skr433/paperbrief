# -*- coding: utf-8 -*-
"""离线冒烟测试：解析官方测试集样例，检查汇报包与问答是否可用。

用法：python scripts/smoke_test.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))
for cand in (os.path.join(os.path.dirname(ROOT), "_pylibs"),):
    if os.path.isdir(cand):
        sys.path.insert(0, cand)

from pdf_parse import parse_pdf, fingerprint, jaccard  # noqa: E402
from brief import build_brief, brief_to_markdown  # noqa: E402
from retrieval import ask  # noqa: E402

PDF_DIR = os.environ.get("PAPERBRIEF_PDF_DIR") or os.path.join(
    os.path.dirname(ROOT), "ref_kxyzs", "pdf"
)

FILES = ["1706.03762v7.pdf", "1706.03762v1.pdf", "1706.03762v1_img.pdf", "N19-1423.pdf"]
QUESTIONS = [
    "这篇论文的第 6 作者是谁，在哪个公司或机构工作？",
    "Adam 优化器的超参数 β1 和 β2 分别设置成了多少？",
    "训练 Base 模型使用了多少块 GPU，具体是什么型号？",
    "Transformer (big) 在 English-to-German 上的 BLEU 分数是多少？",
]


def main():
    print("PDF_DIR:", PDF_DIR)
    parsed = {}
    for f in FILES:
        p = os.path.join(PDF_DIR, f)
        if not os.path.exists(p):
            print("skip (missing):", f)
            continue
        t0 = time.time()
        d = parse_pdf(p)
        d["fingerprint"] = fingerprint(d)
        parsed[f] = d
        print(
            "%-28s %5.1fs pages=%-4d paras=%-5d secs=%-4d caps=%-4d refs=%-4d"
            % (f, time.time() - t0, d["meta"]["pages"], d["stats"]["n_paragraphs"],
               d["stats"]["n_sections"], d["stats"]["n_captions"], d["stats"]["n_references"])
        )
        print("    title:", d["meta"]["title"][:90])
        print("    authors:", d["meta"]["authors"][:90])
        print("    outline:", [s["title"][:48] for s in d["sections"][:6]])

    if "1706.03762v7.pdf" in parsed and "1706.03762v1.pdf" in parsed:
        sim = jaccard(parsed["1706.03762v7.pdf"]["fingerprint"], parsed["1706.03762v1.pdf"]["fingerprint"])
        print("\n去重: v7 vs v1 相似度 = %.3f" % sim)

    if "1706.03762v7.pdf" in parsed:
        d = parsed["1706.03762v7.pdf"]
        b = build_brief(d)
        print("\n== 汇报包 ==")
        print("摘要:", b["abstract"][:160])
        print("贡献:", [c["text"][:60] for c in b["contributions"][:3]])
        print("结果:", [m["text"][:60] for m in b["metrics"][:3]])
        print("图表:", [f["text"][:50] for f in b["figures"][:3]])
        print("术语:", [t["term"] for t in b["terms"][:8]])
        print("讲稿条数:", len(b["script"]))
        md = brief_to_markdown(b)
        out = os.path.join(ROOT, "data", "brief_1706.03762v7.md")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "w", encoding="utf-8").write(md)
        print("markdown ->", out, len(md), "chars")

        print("\n== 官方示例问题 ==")
        for q in QUESTIONS:
            r = ask(d, q)
            print("-", q)
            print("  mode=%s anchors=%d" % (r["mode"], len(r["anchors"])))
            print("  raw:", r["answer"][:220].replace("\n", " | "))

    if "1706.03762v1_img.pdf" in parsed:
        print("\n== 扫描件 ==", "chars:", parsed["1706.03762v1_img.pdf"]["stats"]["n_chars"])


if __name__ == "__main__":
    main()
