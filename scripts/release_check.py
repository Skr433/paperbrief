# -*- coding: utf-8 -*-
"""发布前自检：一次跑完健康、解析、问答、模块，输出 PASS/FAIL。

用法：
    python scripts/release_check.py --base http://1.14.98.248
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def _req(base, path, payload=None, timeout=300):
    data = json.dumps(payload).encode() if payload is not None else b""
    rq = urllib.request.Request(base + path, data=data, method="POST" if payload is not None else "GET",
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(rq, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0


def check(name, fn):
    try:
        detail = fn()
        results.append((name, PASS, detail))
        print("[PASS] %-28s %s" % (name, detail))
    except Exception as e:
        results.append((name, FAIL, str(e)[:160]))
        print("[FAIL] %-28s %s" % (name, str(e)[:160]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://1.14.98.248")
    args = ap.parse_args()
    base = args.base.rstrip("/")

    def health():
        h, dt = _req(base, "/api/health", timeout=30)
        assert h.get("ok") and h.get("llm"), "health 异常"
        return "parser=%s provider=%s %.2fs" % (h.get("parser_version"), h.get("provider"), dt)

    def parse_doc(name):
        def _f():
            d, dt = _req(base, "/api/parse?name=" + urllib.parse.quote(name), {})
            st = d["stats"]
            assert st["n_paragraphs"] > 0, "0 段落"
            return "refs=%s figs=%s %.1fs" % (st["n_references"], st["n_figures"], dt)
        return _f

    key = {}
    key_pair = {}

    def parse_attention():
        d, dt = _req(base, "/api/parse?name=" + urllib.parse.quote("1706.03762v7.pdf"), {})
        key["a"] = d["key"]
        st = d["stats"]
        assert st["n_references"] >= 30, "文献过少"
        return "refs=%s %.1fs" % (st["n_references"], dt)

    def ask_gpu():
        r, dt = _req(base, "/api/ask?key=" + key["a"], {"q": "训练 Base 模型使用了多少块 GPU，具体是什么型号？"})
        assert "P100" in (r.get("answer") or ""), "答案未命中 P100"
        assert r.get("anchors"), "无锚点"
        return "%.1fs anchors=%s" % (dt, [a["page"] for a in r["anchors"]])

    def ask_bleu():
        r, dt = _req(base, "/api/ask?key=" + key["a"], {"q": "Transformer (big) 在 English-to-German 上的 BLEU 分数是多少？"})
        assert "28.4" in (r.get("answer") or ""), "答案未命中 28.4"
        return "%.1fs" % dt

    def ask_summary():
        r, dt = _req(base, "/api/ask?key=" + key["a"], {"q": "这篇论文主要写了什么"})
        assert len(r.get("answer") or "") > 20, "总结为空"
        return "%.1fs" % dt

    def ask_stream():
        rq = urllib.request.Request(
            base + "/api/ask_stream?key=" + key["a"],
            data=json.dumps({"q": "训练 Base 模型使用了多少块 GPU，具体是什么型号？"}).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        deltas, done = 0, None
        t0 = time.time()
        with urllib.request.urlopen(rq, timeout=120) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev.get("type") == "delta":
                    deltas += 1
                elif ev.get("type") == "done":
                    done = ev
        assert deltas >= 3, "流式 token 太少"
        assert done and "P100" in (done.get("answer") or ""), "流式答案未命中"
        return "%d chunks %.1fs" % (deltas, time.time() - t0)

    def brief():
        r, dt = _req(base, "/api/brief?key=" + key["a"])
        assert r.get("contributions"), "无关键贡献"
        assert "summary" in r and "speech" in r, "缺 summary/speech 别名"
        return "contribs=%s ai=%s %.1fs" % (len(r["contributions"]), bool(r.get("ai_summary")), dt)

    def translate():
        r, dt = _req(base, "/api/translate?key=" + key["a"], {"page": 1})
        assert len(r.get("items") or []) > 0, "无译文"
        return "items=%s %.1fs" % (len(r["items"]), dt)

    def graph():
        r, dt = _req(base, "/api/graph?key=" + key["a"], timeout=60)
        assert len(r.get("nodes") or []) > 5, "节点过少"
        return "nodes=%s edges=%s %.1fs" % (len(r["nodes"]), len(r.get("edges") or []), dt)

    def chart():
        info, _ = _req(base, "/api/csv_sample", timeout=30)
        r, dt = _req(base, "/api/chart", {"name": info["name"], "x": "epoch", "ys": ["bleu"], "kind": "line"})
        assert r.get("svg"), "无 SVG"
        return "%.1fs" % dt

    def version_dedupe():
        v7, _ = _req(base, "/api/parse?name=" + urllib.parse.quote("1706.03762v7.pdf"), {})
        v1, _ = _req(base, "/api/parse?name=" + urllib.parse.quote("1706.03762v1.pdf"), {})
        dup = v1.get("duplicate") or {}
        assert dup.get("file"), "v1 未检测到 v7 重复"
        assert (dup.get("similarity") or 0) >= 0.6, "相似度过低"
        key_pair["v1"], key_pair["v7"] = v1["key"], v7["key"]
        return "v1→v7 相似度 %.2f" % dup["similarity"]

    def version_diff():
        r, dt = _req(base, "/api/diff?k1=%s&k2=%s" % (key_pair["v1"], key_pair["v7"]), timeout=120)
        assert r.get("summary"), "差异摘要为空"
        st = r.get("stats") or {}
        return "mode=%s +%s/-%s/改%s %.1fs" % (r.get("mode"), st.get("added"), st.get("removed"), st.get("changed"), dt)

    def error_path():
        try:
            _req(base, "/api/brief?key=not-exist", timeout=30)
            raise AssertionError("不存在 key 未报错")
        except urllib.error.HTTPError as e:
            assert e.code == 404, "错误码 %s" % e.code
            return "404"

    check("health", health)
    check("parse Attention", parse_attention)
    check("parse T5", parse_doc("1910.10683v4.pdf"))
    check("ask GPU", ask_gpu)
    check("ask BLEU", ask_bleu)
    check("ask 总结", ask_summary)
    check("ask 流式", ask_stream)
    check("brief", brief)
    check("translate", translate)
    check("graph", graph)
    check("chart", chart)
    check("版本去重 v1/v7", version_dedupe)
    check("版本差异 diff", version_diff)
    check("error 404", error_path)

    bad = [r for r in results if r[1] == FAIL]
    print("\n%d/%d PASS" % (len(results) - len(bad), len(results)))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
