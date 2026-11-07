# -*- coding: utf-8 -*-
"""HTTP 冒烟测试：模拟前端调用后端接口。"""
import json
import sys
import urllib.request

BASE = "http://127.0.0.1:8010"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=120) as r:
        return r.status, r.headers.get("content-type"), r.read()


def post(path, data=None, ctype="application/json"):
    body = data if isinstance(data, (bytes, type(None))) else json.dumps(data).encode()
    req = urllib.request.Request(BASE + path, data=body or b"", method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def main():
    st, ct, b = get("/api/library")
    lib = json.loads(b)
    print("library:", st, len(lib["docs"]), "篇 ->", [d["name"] for d in lib["docs"][:3]], "...")

    st, d = post("/api/parse?name=1706.03762v7.pdf")
    print("parse v7:", st, "cached=%s" % d["cached"], "key=%s" % d["key"],
          "pages=%s paras=%s secs=%s refs=%s" % (d["doc"]["pages"], d["doc"]["stats"]["n_paragraphs"],
                                                 d["doc"]["stats"]["n_sections"], d["doc"]["stats"]["n_references"]))
    print("  title:", d["doc"]["title"])
    key = d["key"]

    st, d2 = post("/api/parse?name=1706.03762v1.pdf")
    print("parse v1:", st, "duplicate =", d2.get("duplicate"))

    st, ct, b = get("/api/brief?key=" + key)
    brief = json.loads(b)
    print("brief:", st, "abstract=%d chars, contributions=%d, metrics=%d, figures=%d, terms=%d, script=%d"
          % (len(brief["abstract"]), len(brief["contributions"]), len(brief["metrics"]),
             len(brief["figures"]), len(brief["terms"]), len(brief["script"])))

    st, ct, b = get("/api/brief?key=" + key + "&format=md")
    print("brief.md:", st, ct, len(b), "bytes")

    for q in ["这篇论文的第 6 作者是谁，在哪个公司或机构工作？", "训练 Base 模型使用了多少块 GPU，具体是什么型号？"]:
        st, r = post("/api/ask?key=" + key, {"q": q})
        print("ask:", q, "->", r["mode"], "anchors=%d" % len(r["anchors"]))
        print("   ", r["answer"][:150].replace("\n", " "))

    st, ct, b = get("/api/page?key=%s&n=1" % key)
    print("page1:", st, ct, "%d bytes" % len(b))
    st, ct, b = get("/api/page?key=%s&n=8" % key)
    print("page8:", st, ct, "%d bytes" % len(b))


if __name__ == "__main__":
    main()
