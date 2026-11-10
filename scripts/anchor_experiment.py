# -*- coding: utf-8 -*-
"""锚点约束对照实验：A 纯 LLM / B RAG 无锚点要求 / C 本系统（锚点约束）。

用法（在本机）：
    $env:DEEPSEEK_API_KEY="sk-..."      # 或把 key 写进项目根目录 .deepseek_key
    python scripts/anchor_experiment.py

输出：04_资料/评测_1108/anchor_experiment.json + 控制台表
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for cand in (r"E:\deepseek\对话\_pylibs", r"C:\Users\ASUS\Desktop\paperbrief\05_deps\_pylibs",
             os.path.join(ROOT, "_pylibs")):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)
sys.path.insert(0, os.path.join(ROOT, "backend"))

PDF_DIR = r"E:\deepseek\对话\官方测试集_1108\pdf"
OUT = r"C:\Users\ASUS\Desktop\paperbrief\04_资料\评测_1108\anchor_experiment.json"

DOCS = {
    "att": "1706.03762v7.pdf",
    "bert": "N19-1423.pdf",
    "gpt3": "2005.14165v4.pdf",
}

# (doc, question, expected 关键数字/关键词)
QUESTIONS = [
    ("att", "这篇论文的第 6 作者是谁，在哪个机构工作？", ["Gomez", "Google"]),
    ("att", "Adam 优化器的超参数 β1 和 β2 分别设置成了多少？", ["0.9", "0.98"]),
    ("att", "训练 Base 模型使用了多少块 GPU，具体是什么型号？", ["8", "P100"]),
    ("att", "Transformer (big) 在 English-to-German 上的 BLEU 分数是多少？", ["28.4"]),
    ("att", "论文中 residual dropout 的丢弃率是多少？", ["0.1"]),
    ("att", "warmup 步数设置为多少？", ["4000"]),
    ("att", "label smoothing 的值是多少？", ["0.1"]),
    ("att", "Transformer big 模型训练了多少步？", ["300000"]),
    ("att", "论文一共有多少位作者？", ["8"]),
    ("bert", "BERT Base 有多少层 Transformer block？", ["12"]),
    ("bert", "BERT Large 的隐藏层维度是多少？", ["1024"]),
    ("bert", "BERT 预训练语料大约有多少词？", ["3.3"]),
    ("bert", "BERT 支持的最大输入序列长度是多少？", ["512"]),
    ("bert", "BERT Base 的参数量大约是多少？", ["110"]),
    ("bert", "BERT 预训练使用了几个任务？", ["2"]),
    ("gpt3", "GPT-3 最大模型有多少参数？", ["175"]),
    ("gpt3", "GPT-3 的上下文窗口是多少 token？", ["2048"]),
    ("gpt3", "GPT-3 训练用了多少 token？", ["300"]),
    ("gpt3", "GPT-3 最大模型有多少层？", ["96"]),
    ("gpt3", "GPT-3 论文标题是什么？", ["Language Models are Few-Shot Learners"]),
]


def get_key() -> str:
    key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        p = os.path.join(ROOT, ".deepseek_key")
        if os.path.exists(p):
            key = open(p, encoding="utf-8").read().strip()
    if not key:
        raise SystemExit("没有 DEEPSEEK_API_KEY：请设环境变量，或把 key 写入 .deepseek_key")
    return key


def main():
    os.environ["DEEPSEEK_API_KEY"] = get_key()
    os.environ.setdefault("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    import llm
    import pdf_parse
    import retrieval

    parsed, nums = {}, {}
    for tag, fn in DOCS.items():
        t0 = time.time()
        p = pdf_parse.parse_pdf(os.path.join(PDF_DIR, fn))
        parsed[tag] = p
        nums[tag] = {m for para in p["paragraphs"] for m in re.findall(r"\d+(?:\.\d+)?", para["text"])}
        print("[parse] %s %.1fs" % (fn, time.time() - t0))

    def norm(s: str) -> str:
        return re.sub(r"[,\s]", "", s.lower())

    def nums_in(text: str) -> set:
        return set(re.findall(r"\d+(?:\.\d+)?", text))

    rows = []
    for i, (tag, q, expected) in enumerate(QUESTIONS, 1):
        ctx_anchors = retrieval.ask(parsed[tag], q).get("anchors") or []
        ctx = "\n".join("[p.%s] %s" % (a["page"], a["text"][:600]) for a in ctx_anchors[:3])
        print("[%02d] %s | anchors=%s" % (i, q[:28], [a["page"] for a in ctx_anchors]))
        out = {}
        # A：纯 LLM，无原文
        pa = [{"role": "system", "content": "你是科研助手，凭已有知识回答，尽量给具体数字。"},
              {"role": "user", "content": q}]
        try:
            out["A"] = llm.chat(pa, temperature=0, timeout=60, max_tokens=300)
        except Exception as e:
            out["A"] = "ERROR " + str(e)[:80]
        # B：RAG 给原文，但不要求标页码
        pb = [{"role": "system", "content": "你是科研论文阅读助手。只能依据<原文>回答，不得编造；用中文，不超过120字。"},
              {"role": "user", "content": "<原文>\n%s\n</原文>\n\n问题：%s" % (ctx, q)}]
        try:
            out["B"] = llm.chat(pb, temperature=0, timeout=60, max_tokens=300)
        except Exception as e:
            out["B"] = "ERROR " + str(e)[:80]
        # C：本系统（锚点约束 + 页码）
        try:
            out["C"] = llm.answer_with_evidence(q, ctx_anchors)
        except Exception as e:
            out["C"] = "ERROR " + str(e)[:80]

        support = {}
        for arm in "ABC":
            ans = out[arm]
            ns = nums_in(ans)
            bad = {n for n in ns if n not in nums[tag]} if ns else set()
            support[arm] = {"nums": sorted(ns), "unsupported_nums": sorted(bad),
                            "has_unsupported": bool(bad)}

        rows.append({"i": i, "doc": DOCS[tag], "q": q, "expected": expected,
                     "answers": out, "support": support,
                     "anchors": [a["page"] for a in ctx_anchors]})
        print("   A:%s | B:%s | C:%s" % (
            "OK" if not support["A"]["has_unsupported"] else "无依据数字" + str(support["A"]["unsupported_nums"]),
            "OK" if not support["B"]["has_unsupported"] else "无依据数字" + str(support["B"]["unsupported_nums"]),
            "OK" if not support["C"]["has_unsupported"] else "无依据数字" + str(support["C"]["unsupported_nums"])))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rows, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    def rate(arm):
        bad = sum(1 for r in rows if r["support"][arm]["has_unsupported"])
        return bad, len(rows)

    print("\n=== 汇总（数字无依据率）===")
    for arm in "ABC":
        bad, n = rate(arm)
        print("%s: %d/%d = %.0f%%" % (arm, bad, n, 100.0 * bad / n))
    print("结果已写入:", OUT)


if __name__ == "__main__":
    main()
