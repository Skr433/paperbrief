# -*- coding: utf-8 -*-
"""fastembed 可行性 spike（1 小时封顶，只验证“2G 服务器能否跑向量检索”）。

用法（**独立 venv，不污染生产环境**）：
    python3 -m venv /tmp/embed-venv
    /tmp/embed-venv/bin/pip install -q fastembed
    /tmp/embed-venv/bin/python scripts/spike_fastembed.py --n 80

输出：模型加载耗时、编码耗时、内存峰值、向量维度，并给出 go / no-go 建议。
判定线（经验值）：内存峰值 < 900MB 且编码 80 段 < 2s → 可以上；否则放弃。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

SAMPLE = [
    "The Transformer is a sequence transduction model based entirely on attention, replacing recurrent layers.",
    "We trained our models on one machine with 8 NVIDIA P100 GPUs for 12 hours.",
    "BERT uses a bidirectional Transformer encoder pre-trained with masked language modeling.",
    "Layer normalization is applied before each sub-layer in the pre-norm architecture.",
    "We propose a sparse attention mechanism to reduce quadratic complexity for long contexts.",
    "The model achieves 28.4 BLEU on WMT 2014 English-to-German translation.",
    "Retrieval augmented generation combines a dense retriever with a sequence-to-sequence generator.",
    "We evaluate zero-shot performance across 42 benchmark tasks.",
]


def rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KB
    except Exception:
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except Exception:
            return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80, help="编码段落数")
    ap.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    args = ap.parse_args()

    try:
        from fastembed import TextEmbedding
    except Exception:
        print("[no-go] fastembed 未安装。先执行：")
        print("  python3 -m venv /tmp/embed-venv && /tmp/embed-venv/bin/pip install -q fastembed")
        return 2

    texts = (SAMPLE * ((args.n // len(SAMPLE)) + 1))[: args.n]
    print("[1/3] 加载模型：%s" % args.model)
    t0 = time.time()
    try:
        model = TextEmbedding(model_name=args.model)
    except Exception as e:
        print("[no-go] 模型加载失败：%s" % str(e)[:200])
        return 1
    load_s = time.time() - t0
    print("      加载耗时 %.1fs，内存峰值 %.0f MB" % (load_s, rss_mb()))

    print("[2/3] 编码 %d 段…" % len(texts))
    t0 = time.time()
    vecs = list(model.embed(texts))
    enc_s = time.time() - t0
    dim = len(vecs[0]) if vecs else 0
    print("      编码耗时 %.2fs（%.1f 段/秒），维度 %d，内存峰值 %.0f MB" % (
        enc_s, len(texts) / max(enc_s, 1e-6), dim, rss_mb()))

    print("[3/3] 判定")
    mem = rss_mb()
    ok = (mem < 900) and (enc_s < 2.0)
    print("      内存峰值 %.0f MB（线 900MB）；80 段编码 %.2fs（线 2s）" % (mem, enc_s))
    print("      结论：%s" % ("GO —— 可以在 2G 服务器上做混合检索" if ok else "NO-GO —— 资源超限，保持 BM25 + 跨语言扩展"))
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
