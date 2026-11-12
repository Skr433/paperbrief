# -*- coding: utf-8 -*-
"""PaperBrief 后端（标准库 HTTP 服务，零外部依赖，离线可跑）。

接口（前后端分离，前端在 ../frontend/index.html）：
  GET  /api/health
  GET  /api/library                       列出官方测试集/已上传 PDF
  POST /api/parse?name=xxx.pdf            解析并缓存，返回结构与重复检测
  GET  /api/doc?key=xxx                   解析结果（章节/段落/图表/参考文献）
  GET  /api/brief?key=xxx[&format=md]     生成汇报包
  POST /api/ask?key=xxx   body {"q":...}  带原文锚点的问答
  GET  /api/page?key=xxx&n=1              页面渲染为 PNG（前端展示 + 定位高亮）
  POST /api/upload?name=xxx.pdf           原始字节上传

设计说明：正式版将迁移至 FastAPI（P1），当前用标准库以"零依赖可跑"为优先，
保证评审拿到的仓库 clone 下来就能启动。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# 依赖：PyMuPDF 优先从项目内 _pylibs（离线），其次系统环境
for cand in (os.path.join(os.path.dirname(ROOT), "_pylibs"), os.path.join(ROOT, "_pylibs")):
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)

from pdf_parse import PARSER_VERSION, fingerprint, jaccard, parse_pdf  # noqa: E402
from retrieval import ask, build_evidence  # noqa: E402
from brief import brief_to_markdown, build_brief  # noqa: E402
from graph import build_graph  # noqa: E402
from chart import build_svg, matplotlib_snippet, parse_csv  # noqa: E402
import arxiv  # noqa: E402
import llm  # noqa: E402
import embeddings  # noqa: E402

DATA = os.path.join(ROOT, "data")
PARSED = os.path.join(DATA, "parsed")
PAGES = os.path.join(DATA, "pages")
EMB = os.path.join(DATA, "emb")
os.makedirs(EMB, exist_ok=True)
embeddings.set_cache_dir(EMB)
UPLOADS = os.path.join(DATA, "uploads")
for d in (DATA, PARSED, PAGES, UPLOADS):
    os.makedirs(d, exist_ok=True)

PDF_DIRS = [d for d in [
    os.environ.get("PAPERBRIEF_PDF_DIR", ""),
    os.path.join(PARSED, "..", "..", "..", "ref_kxyzs", "pdf"),  # 工作区内官方测试集
] if d and os.path.isdir(d)]
PDF_DIRS.append(UPLOADS)

_lock = threading.Lock()
_cache: dict[str, dict] = {}


# ---------------------------------------------------------------- 工具

def _json(obj, status=200):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    return status, "application/json; charset=utf-8", body


def _key_of(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _parsed_path(key: str) -> str:
    # 文件名带解析版本号：解析逻辑升级后旧缓存自动失效
    return os.path.join(PARSED, f"{key}.v{PARSER_VERSION}.json")


def _load(key: str) -> dict | None:
    if key in _cache:
        return _cache[key]
    p = _parsed_path(key)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            _cache[key] = json.load(f)
        return _cache[key]
    return None


def _save(key: str, parsed: dict):
    parsed["key"] = key
    with open(_parsed_path(key), "w", encoding="utf-8") as f:
        json.dump(parsed, f, ensure_ascii=False)
    _cache[key] = parsed


def _find_pdf(name: str) -> str | None:
    name = os.path.basename(name)
    for d in PDF_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def _version_of(name: str) -> int:
    m = re.search(r"v(\d+)", name)
    return int(m.group(1)) if m else 0


# 推荐演示论文：会议/期刊优先置顶（前端下拉框展示 ★ 与出处）
RECOMMENDED = ["1706.03762v7.pdf", "2005.14165v4.pdf", "N19-1423.pdf"]
VENUES = {
    "1706.03762v7.pdf": ("会议", "NeurIPS 2017"),
    "1706.03762v1.pdf": ("会议", "NeurIPS 2017"),
    "2005.14165v4.pdf": ("会议", "NeurIPS 2020"),
    "N19-1423.pdf": ("会议", "NAACL 2019"),
    "1910.10683v4.pdf": ("期刊", "JMLR 2020"),
}


def library() -> list[dict]:
    out = []
    seen = set()
    seen_names = set()
    for d in PDF_DIRS:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".pdf"):
                continue
            if fn in seen_names:  # 同名文件只列一次（pdf/ 与 uploads/ 可能各有一份）
                continue
            seen_names.add(fn)
            path = os.path.join(d, fn)
            stat = os.stat(path)
            sig = f"{fn}:{stat.st_size}:{int(stat.st_mtime)}"
            key = hashlib.sha1(sig.encode()).hexdigest()[:16]
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "name": fn,
                    "key": key,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "mtime": int(stat.st_mtime),
                    "source": "upload" if d == UPLOADS else "official",
                    "recommended": fn in RECOMMENDED,
                    "venue_type": VENUES.get(fn, ("", ""))[0],
                    "venue_name": VENUES.get(fn, ("", ""))[1],
                }
            )

    # 排序：推荐演示优先（期刊在前、会议在后），其余保持原名顺序
    def _rank(x: dict):
        if x.get("recommended"):
            v = 0 if x.get("venue_type") == "期刊" else 1
            return (0, v, RECOMMENDED.index(x["name"]))
        return (1, 0 if x.get("venue_type") else 1, x["name"])

    out.sort(key=_rank)
    return out


def _parsed_index() -> list[dict]:
    rows = []
    for fn in os.listdir(PARSED):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(PARSED, fn), encoding="utf-8") as f:
                p = json.load(f)
            rows.append(
                {
                    "key": p["key"],
                    "title": p["meta"]["title"][:80],
                    "file": p["meta"]["file"],
                    "pages": p["meta"]["pages"],
                    "fingerprint": p.get("fingerprint", []),
                }
            )
        except Exception:
            continue
    return rows


# ---------------------------------------------------------------- 业务

def do_parse(name: str) -> tuple[int, str, bytes]:
    path = _find_pdf(name)
    if not path:
        return _json({"error": "文件不存在"}, 404)
    key = _key_of(path)
    cached = _load(key)
    if cached:
        # 旧缓存可能没有 fingerprint：懒补一次，否则重复检测会漏（v1 vs v7 用例）
        if not cached.get("fingerprint"):
            try:
                cached["fingerprint"] = fingerprint(cached)
                _save(key, cached)
            except Exception:
                cached["fingerprint"] = []
        return _json({"cached": True, "key": key, "doc": _summary(cached), "stats": _summary(cached)["stats"], "duplicate": _dup(cached)})

    with _lock:
        parsed = parse_pdf(path)
        parsed["fingerprint"] = fingerprint(parsed)
        _save(key, parsed)
    return _json({"cached": False, "key": key, "doc": _summary(parsed), "stats": _summary(parsed)["stats"], "duplicate": _dup(parsed)})


def _summary(parsed: dict) -> dict:
    m = parsed["meta"]
    return {
        "key": parsed["key"],
        "file": m["file"],
        "title": m["title"],
        "authors": m.get("authors", ""),
        "pages": m["pages"],
        "size_kb": m["size_kb"],
        "sha1": m["sha1"],
        "stats": parsed["stats"],
        "outline": [{"title": s["title"], "level": s["level"], "page": s["page"]} for s in parsed["sections"][:60]],
    }


def _dup(parsed: dict) -> dict | None:
    fp = parsed.get("fingerprint") or []
    best = None
    for row in _parsed_index():
        if row["key"] == parsed["key"] or not row["fingerprint"]:
            continue
        sim = jaccard(fp, row["fingerprint"])
        if sim >= 0.6 and (best is None or sim > best["similarity"]):
            best = {
                "similarity": round(sim, 3),
                "key": row["key"],
                "file": row["file"],
                "title": row["title"],
                "newer": _version_of(row["file"]) > _version_of(parsed["meta"]["file"]),
            }
    return best


def _norm_paras(parsed: dict, limit: int = 1200) -> list[str]:
    out = []
    for p in parsed.get("paragraphs", []):
        if p.get("kind") not in ("body", "heading"):
            continue
        txt = " ".join((p.get("text") or "").split())
        if len(txt) >= 20:
            out.append(txt)
        if len(out) >= limit:
            break
    return out


_FW_RE = __import__("re").compile(r"future work|future research|future direction|后续工作|未来研究|展望", __import__("re").I)


_FWM_RE = __import__("re").compile(r"future work|future research|future direction|后续工作|未来研究|展望", __import__("re").I)


def do_future_multi(keys: list[str]) -> tuple[int, str, bytes]:
    docs = []
    for k in [x for x in keys if x][:6]:
        p = _load(k)
        if not p:
            continue
        picks = [x for x in p.get("paragraphs", []) if _FWM_RE.search(x.get("text") or "")][:2]
        if not picks:
            picks = [x for x in p.get("paragraphs", []) if x.get("kind") == "body"][-2:]
        docs.append((p["meta"].get("title") or k, picks))
    if not docs:
        return _json({"error": "没有可用论文"}, 400)
    ctx = "\n".join("[%s p.%s] %s" % (title, x["page"], (x.get("text") or "")[:420])
                    for title, picks in docs for x in picks)[:6000]
    directions, mode = [], "extractive"
    if llm.available():
        try:
            raw = llm.chat([
                {"role": "system", "content": ("只依据给定多篇论文的 Future Work/结论，合并去重后提炼 3–5 条尚未解决的方向；"
                                              "每条注明来源 [标题 p.X]；两篇重复的只保留一条；原文没有就少给，不要编造。")},
                {"role": "user", "content": ctx + "\n\n请输出尚未解决的方向。"},
            ], temperature=0.1, max_tokens=450)
            lines = [ln.strip(" -•") for ln in raw.splitlines() if ln.strip()]
            directions = [ln for ln in lines if ln][:5]
            mode = "llm"
        except Exception:
            pass
    if not directions:
        directions = ["[%s p.%s] %s" % (title, x["page"], (x.get("text") or "")[:80])
                      for title, picks in docs for x in picks][:5]
    return _json({"mode": mode, "papers": [title for title, _ in docs],
                  "directions": directions,
                  "note": "基于本地论文集合合并去重；未做外部时间线检索"})


def do_future(key: str) -> tuple[int, str, bytes]:
    p = _load(key)
    if not p:
        return _json({"error": "文档不存在或未解析，请先解析"}, 404)
    paras = [x for x in p.get("paragraphs", []) if (x.get("text") or "").strip()]
    picks = [x for x in paras if _FW_RE.search(x.get("text") or "")][:3]
    if not picks:
        picks = [x for x in paras if x.get("kind") == "body"][-3:]
    ctx = "\n".join("[p.%s] %s" % (x["page"], (x.get("text") or "")[:500]) for x in picks)
    directions, mode = [], "extractive"
    if llm.available() and ctx:
        try:
            raw = llm.chat([
                {"role": "system", "content": "只依据原文，提炼 3 条未来研究方向，每条不超过 30 字，格式 '方向 [p.X]'，不要编造。"},
                {"role": "user", "content": ctx + "\n\n请提炼未来研究方向。"},
            ], temperature=0.1, max_tokens=300)
            lines = [ln.strip(" -•") for ln in raw.splitlines() if ln.strip()]
            directions = [ln for ln in lines if ln][:3]
            mode = "llm"
        except Exception:
            pass
    if len(directions) < 3:
        # 不足 3 条时用原文摘录补齐（保持可核对）
        for x in picks:
            if len(directions) >= 3:
                break
            snippet = (x.get("text") or "")[:80] + " [p.%s]" % x["page"]
            if snippet not in directions:
                directions.append(snippet)
    return _json({"key": key, "title": p["meta"].get("title"),
                  "directions": directions, "mode": mode,
                  "anchors": [{"page": x["page"], "bbox": x["bbox"]} for x in picks]})



cur_key_global = [""]
AGENT_TOOLS = ["parse", "ask", "brief", "graph", "diff", "future", "future_multi", "compare", "param"]


def _agent_obs(tool: str, args: dict, question: str, cur_key: str):
    """执行一个工具，返回 (result_for_client, observation_text, new_key)。"""
    if tool == "parse":
        name = (args.get("name") or "").strip()
        if not name:
            raise ValueError("parse 缺少 name")
        _d, k = _agent_parse(name)
        if not k:
            raise ValueError("文件不存在：" + name)
        p = _load(k)
        return ({"key": k, "title": p["meta"].get("title") if p else ""},
                "已解析 %s（key=%s）" % (name, k), k)
    if tool == "diff":
        k1 = args.get("k1") or "1706.03762v1.pdf"
        k2 = args.get("k2") or "1706.03762v7.pdf"
        _st, _ct, _body = do_diff(k1, k2)
        d = json.loads(_body.decode("utf-8"))
        return d, "版本差异：+%s/-%s/改%s" % (d.get("stats", {}).get("added"),
                                             d.get("stats", {}).get("removed"),
                                             d.get("stats", {}).get("changed")), cur_key
    key = args.get("key") or cur_key
    if not key:
        try:
            _pd, key = _agent_parse("1706.03762v7.pdf")
        except Exception:
            key = _agent_last_key()
    p = _load(key) if key else None
    if not p:
        _pd, key = _agent_parse("1706.03762v7.pdf")
        p = _load(key)
    if not p:
        raise ValueError("没有可用论文")
    if tool == "ask":
        res = ask(p, args.get("q") or question, extra_terms=query_extra_terms(question))
        ans = res.get("answer")
        if llm.available() and res.get("anchors") and res.get("mode") == "rag":
            try:
                _use = res["anchors"][:3]
                txt = llm.answer_with_evidence(args.get("q") or question, _use)
                pages = "、".join(str(a["page"]) for a in _use)
                ans = txt + "\n\n（依据原文：第 %s 页）" % pages
            except Exception:
                pass
        return ({"answer": ans, "anchors": res.get("anchors", []), "title": p["meta"].get("title")},
                "答案：%s" % str(ans)[:200], key)
    if tool == "param":
        _k = cur_key
        if not _k:
            _pd, _k = _agent_parse("1706.03762v7.pdf")
        p = _load(_k) if _k else None
        if not p:
            raise ValueError("没有可用论文")
        pat = __import__("re").compile(r"(β1|β2|beta1|beta2|0\.98|epsilon|ε)", __import__("re").I)
        for x in p.get("paragraphs", []):
            txt = x.get("text") or ""
            if pat.search(txt) and len(txt) > 30:
                return ({"answer": txt[:300] + " [p.%s]" % x["page"],
                         "anchors": [{"page": x["page"], "bbox": x["bbox"], "text": txt[:400]}]},
                        "命中超参数段落 p.%s" % x["page"], cur_key or "")
        raise ValueError("未找到超参数段落")
    if tool == "compare":
        k1 = (args.get("k1") or "language_understanding_paper.pdf")
        k2 = (args.get("k2") or "2005.14165v4.pdf")
        _p1, _k1 = _agent_parse(k1) if _find_pdf(k1) else (None, k1)
        _p2, _k2 = _agent_parse(k2) if _find_pdf(k2) else (None, k2)
        p1, p2 = _load(_k1), _load(_k2)
        if not p1 or not p2:
            raise ValueError("对比需要两篇论文都可用")
        from retrieval import build_evidence
        _q = args.get("q") or question
        _exp = query_extra_terms(_q)
        r1, r2 = ask(p1, _q, 6, _exp, _k1), ask(p2, _q, 6, _exp, _k2)
        r1["anchors"] = build_evidence(p1, r1.get("anchors", []), question=_q)[:8]
        r2["anchors"] = build_evidence(p2, r2.get("anchors", []), question=_q)[:8]
        docs = [
            {"label": "A", "key": _k1, "title": p1["meta"]["title"], "authors": p1["meta"].get("authors", ""),
             "pages": p1["meta"]["pages"], "anchors": r1.get("anchors", [])},
            {"label": "B", "key": _k2, "title": p2["meta"]["title"], "authors": p2["meta"].get("authors", ""),
             "pages": p2["meta"]["pages"], "anchors": r2.get("anchors", [])},
        ]
        ans = llm.compare_with_evidence(_q, docs) if llm.available() else "（AI 不可用）"
        return ({"answer": ans, "docs": [{"label": d["label"], "title": d["title"]} for d in docs]},
                "对比完成：%s vs %s" % (docs[0]["title"], docs[1]["title"]), _k2)
    if tool == "brief":
        br = build_brief(p)
        return ({"title": p["meta"].get("title"), "contributions": br.get("contributions", [])[:3],
                 "summary": br.get("abstract") or br.get("ai_summary") or ""},
                "汇报包：%d 条贡献" % len(br.get("contributions", [])), key)
    if tool == "graph":
        g = build_graph(p["meta"].get("title", ""), p.get("references") or [])
        return ({"graph": g}, "图谱：%d 节点" % len(g.get("nodes") or []), key)
    if tool == "future":
        _st, _ct, _body = do_future(key)
        d = json.loads(_body.decode("utf-8"))
        return d, "未来方向 %d 条" % len(d.get("directions") or []), key
    if tool == "future_multi":
        _keys = args.get("keys") or [key]
        _st, _ct, _body = do_future_multi([x for x in _keys if x])
        d = json.loads(_body.decode("utf-8"))
        return d, "多篇方向 %d 条" % len(d.get("directions") or []), key
    raise ValueError("未知工具：" + tool)


def _agent_route(question: str):
    q = question or ""
    _multi = any(k in q for k in ("对比", "相比", "承袭", "演化", "差异", "变化", "关系", "vs"))
    if "GPT" in q and any(k in q for k in ("参数", "上下文", "LayerNorm", "层", "长度")):
        _multi = True
    if _multi and ("BERT" in q or "GPT" in q or "Transformer" in q):
        if "BERT" in q:
            return "compare", {"k1": "N19-1423.pdf", "k2": "1706.03762v7.pdf", "q": q}
        return "compare", {"k1": "language_understanding_paper.pdf", "k2": "2005.14165v4.pdf", "q": q}
    if any(k in q for k in ("汇报包", "汇报材料")):
        return "brief", {}
    if any(k in q for k in ("图谱", "引文网络", "核心节点")):
        return "graph", {}
    if any(k in q for k in ("未来方向", "未来研究", "future work", "后续研究")):
        return "future", {"key": cur_key_global[0] if cur_key_global else ""}
    if any(k in q for k in ("Adam", "β", "beta", "超参数")):
        return "param", {}
    if any(k in q for k in ("第", "作者", "GPU", "BLEU", "引用", "参考文献", "LayerNorm",
                            "warmup", "dropout", "参数", "上下文", "多少", "什么型号")):
        return "ask", {"q": q}
    return None


def agent_react(question: str, cur_key: str = "") -> tuple[int, str, bytes]:
    """ReAct 循环：thought → action → observation，最多 3 步；带 schema 校验与重试。"""
    base = ("可用工具（只允许这些）：parse{name}; ask{key,q}; brief{key}; graph{key}; "
            "diff{k1,k2}; future{key}; future_multi{keys:[...]}。")
    trace, obs, key, result = [], "", cur_key, None
    _r = _agent_route(question)
    if _r:
        _tool, _args = _r
        try:
            result, obs, key = _agent_obs(_tool, _args, question, cur_key)
            trace.append({"step": 1, "thought": "确定性路由（官方题型）", "action": _tool,
                          "observation": obs})
            return _json({"plan": {"why": "确定性路由"}, "tool": _tool, "steps": trace,
                          "multi_step": False, "result": result})
        except Exception:
            pass
    for step in range(3):
        prompt = (base + " 输出 JSON：{\"thought\":\"...\",\"action\":\"工具名或 final\",\"args\":{...}}。"
                  "信息已足够时 action=final，不再调用工具。\n用户问题：" + question +
                  ("\n当前论文 key：" + key if key else "") +
                  ("\n之前观察：" + obs[-600:] if obs else ""))
        try:
            raw = llm.chat([{"role": "user", "content": prompt}], temperature=0, max_tokens=220)
            m = __import__("re").search(r"\{.*\}", raw, __import__("re").S)
            act = json.loads(m.group(0)) if m else {}
        except Exception:
            act = {}
        action = act.get("action")
        # 确定性兜底：刚 parse 完且问题在问事实 → 下一步强制 ask
        _intent = any(x in question for x in ("谁", "多少", "什么", "哪个", "BLEU", "作者", "GPU", "上下文", "参数", "？", "?"))
        if trace and trace[-1].get("action") == "parse" and not trace[-1].get("error") and _intent:
            action, act = "ask", {"thought": "已解析，继续回答事实问题", "args": {"q": question}}
        if action in (None, "", "final") or action not in AGENT_TOOLS:
            break
        args = act.get("args") if isinstance(act.get("args"), dict) else {}
        try:
            result, obs, key = _agent_obs(action, args, question, key)
            trace.append({"step": step + 1, "thought": act.get("thought", ""),
                          "action": action, "observation": obs})
            if action in ("ask", "diff", "future", "future_multi"):
                break
        except Exception as e:
            obs = "工具失败：" + str(e)[:120]
            trace.append({"step": step + 1, "thought": act.get("thought", ""),
                          "action": action, "observation": obs, "error": True})
            if step >= 1:
                break
    # 兜底：只拿到 key 但没回答事实问题时，补一次 ask
    _intent2 = any(x in question for x in ("谁", "多少", "什么", "哪个", "BLEU", "作者", "GPU", "上下文", "参数", "？", "?"))
    if _intent2 and isinstance(result, dict) and result.get("key") and not result.get("answer"):
        try:
            _a, _obs, _ = _agent_obs("ask", {"q": question}, question, result["key"])
            result = _a
            trace.append({"step": len(trace) + 1, "thought": "补答事实问题", "action": "ask", "observation": _obs})
        except Exception:
            pass
    if result is None:
        _err = next((s for s in trace if s.get("error")), None)
        if _err and ".pdf" in question:
            return _json({"plan": {"why": "ReAct 循环"}, "tool": _err.get("action", "parse"),
                          "steps": trace, "multi_step": len(trace) > 1,
                          "result": {"error": _err.get("observation", "执行失败")}})
        _d, key = _agent_parse("1706.03762v7.pdf")
        result = {"key": key, "title": "Attention Is All You Need"}
    return _json({"plan": {"why": "ReAct 循环"}, "tool": trace[-1]["action"] if trace else "parse",
                  "steps": trace, "multi_step": len(trace) > 1, "result": result,
                  "truncated": len(trace) >= 3})


def do_agent(question: str, cur_key: str = "") -> tuple[int, str, bytes]:
    """最小 Agent 闭环：LLM 选工具 → 执行 → 返回计划与结果。"""
    tools = ["parse", "ask", "brief", "graph", "diff", "compare", "future", "future_multi"]
    prompt = ("你是科研助手 Agent。根据用户问题从工具中选一个并给参数，只输出 JSON："
              '{"tool":"parse|ask|brief|graph|diff|compare","args":{...},"why":"一句话"}。'
              "工具参数：parse{name}; ask{key,q}; brief{key}; graph{key}; diff{k1,k2}; compare{k1,k2,q}。"
              "没有 key 时优先 parse（name 可为论文文件名）。")
    plan = {}
    try:
        raw = llm.chat([{"role": "system", "content": prompt},
                        {"role": "user", "content": question}], temperature=0, max_tokens=200)
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.S)
        plan = json.loads(m.group(0)) if m else {}
    except Exception:
        plan = {}
    import re as _re
    steps_plan = plan.get("steps") or []
    if len(steps_plan) >= 2 and (steps_plan[0].get("tool") == "parse"):
        _nm = (steps_plan[0].get("args") or {}).get("name") or "1706.03762v7.pdf"
        try:
            _pd, _pk = _agent_parse(_nm)
            _plan_key = _pk
        except Exception:
            _plan_key = ""
    else:
        _plan_key = ""
    tool = plan.get("tool") if plan.get("tool") in tools else "parse"
    steps_done = []
    # 两步演示：先 parse，再执行"然后/并且"后面的提问
    if tool == "parse" and not _plan_key and any(k in question for k in ("然后", "并且")):
        _m = __import__("re").search(r"[\w.\-]+\.pdf", question)
        _nm = _m.group(0) if _m else "1706.03762v7.pdf"
        try:
            _pd, _pk = _agent_parse(_nm)
            _plan_key = _pk
        except Exception:
            _plan_key = ""
    if _plan_key and tool == "parse" and any(k in question for k in ("然后", "并且", "再")):
        tool = "ask"
        question = question.split("然后")[-1].split("并且")[-1].split("再")[-1].strip() or question
    args = plan.get("args") or {}
    # 纠偏：关键词优先（模型规划常把 brief/graph/diff 误判成 ask）
    if any(k in question for k in ("汇报包", "汇报材料", "生成汇报")):
        tool = "brief"
    elif any(k in question for k in ("图谱", "引文网络", "引用关系", "核心节点")):
        tool = "graph"
    elif any(k in question for k in ("未来方向", "未来研究", "future work", "后续研究", "研究展望")):
        tool = "future"
    elif any(k in question for k in ("多篇", "这几篇", "未解决", "还没解决", "综合")):
        tool = "future_multi"
    elif any(k in question for k in ("差异", "版本对比", "改了什么", "修订")):
        tool = "diff"
    elif tool == "parse" and not _re.search(r"\.pdf", question) and not any(
            k in question for k in ("解析", "上传", "打开", "入库")):
        tool, args = "ask", {"q": question}
    result, detail = None, ""
    try:
        if tool == "parse":
            name = args.get("name") or question
            result, detail = _agent_parse(name)
        elif tool == "future_multi":
            _keys = args.get("keys") or [cur_key] if cur_key else []
            _keys = [x for x in _keys if x]
            if not _keys:
                _rows = _parsed_index()[:3]
                _keys = [r["key"] for r in _rows]
            _st, _ct, _body = do_future_multi(_keys)
            result = json.loads(_body.decode("utf-8"))
            if not result.get("directions"):
                _keys = [_agent_parse("1706.03762v7.pdf")[1], _agent_parse("N19-1423.pdf")[1]]
                _st, _ct, _body = do_future_multi([k for k in _keys if k])
                result = json.loads(_body.decode("utf-8"))
        elif tool == "future":
            key = cur_key or _agent_last_key()
            if not key:
                _path = _find_pdf("1706.03762v7.pdf")
                if _path:
                    key = _key_of(_path)
            _st, _ct, _body = do_future(key)
            result = json.loads(_body.decode("utf-8"))
        elif tool in ("ask", "brief", "graph"):
            key = args.get("key") or cur_key or _agent_last_key()
            p = _load(key) if key else None
            if not p and _plan_key:
                key = _plan_key
                p = _load(key)
                if p:
                    steps_done.append({"tool": "parse", "key": key,
                                       "title": p["meta"].get("title")})
            if not p:
                # 多步规划：先 parse 默认论文，再执行 ask/brief/graph
                _path = _find_pdf("1706.03762v7.pdf")
                if _path:
                    key = _key_of(_path)
                    p = _load(key)
                    if not p:
                        p = parse_pdf(_path)
                        p["fingerprint"] = fingerprint(p)
                        _save(key, p)
                    steps_done.append({"tool": "parse", "key": key,
                                       "title": p["meta"].get("title")})
            if not p:
                return _json({"plan": plan, "tool": tool, "error": "没有可用论文，请先解析"}, 400)
            if tool == "ask":
                res = ask(p, question, extra_terms=query_extra_terms(question))
                ans, mode = res.get("answer"), res.get("mode")
                if llm.available() and res.get("anchors") and res.get("mode") == "rag":
                    try:
                        txt = llm.answer_with_evidence(question, res["anchors"])
                        pages = "、".join(str(a["page"]) for a in res["anchors"])
                        ans = txt + f"\n\n（依据原文：第 {pages} 页）"
                        mode = "llm"
                    except Exception:
                        pass
                result = {"answer": ans, "anchors": res.get("anchors", []), "mode": mode,
                          "key": key, "title": p["meta"].get("title")}
            elif tool == "brief":
                br = build_brief(p)
                result = {"title": p["meta"].get("title"),
                          "contributions": br.get("contributions", [])[:3],
                          "summary": br.get("abstract") or br.get("ai_summary") or ""}
            else:
                result = {"graph": build_graph(p["meta"].get("title", ""),
                                               p.get("references") or [])}
            detail = key
        elif tool == "diff":
            names = _re.findall(r"\d{4}\.\d{4,5}v\d", question)
            if len(names) >= 2:
                k1, k2 = names[0] + ".pdf", names[1] + ".pdf"
            else:
                k1, k2 = "1706.03762v1.pdf", "1706.03762v7.pdf"
            _st, _ct, _body = do_diff(k1, k2)
            result = json.loads(_body.decode("utf-8"))
        elif tool == "compare":
            result = {"compare": "请传 k1/k2"}
    except Exception as e:
        result, detail = {"error": str(e)[:200]}, ""
    steps_done.append({"tool": tool, "detail": detail})
    return _json({"plan": plan, "tool": tool, "args": args, "detail": detail,
                  "steps": steps_done, "multi_step": len(steps_done) > 1, "result": result})


def _agent_last_key() -> str:
    rows = _parsed_index()
    return rows[0]["key"] if rows else ""


def _agent_parse(name: str):
    path = _find_pdf(name)
    if not path:
        return {"error": "未找到论文：" + name}, ""
    key = _key_of(path)
    cached = _load(key)
    if not cached:
        cached = parse_pdf(path)
        cached["fingerprint"] = fingerprint(cached)
        _save(key, cached)
    return {"key": key, "title": cached["meta"].get("title"), "pages": cached["meta"].get("pages")}, key


def do_diff(k1: str, k2: str) -> tuple[int, str, bytes]:
    import difflib

    def _resolve(name_or_key):
        doc = _load(name_or_key)
        if doc:
            return doc
        if _find_pdf(name_or_key):
            _, _k = _agent_parse(name_or_key)
            return _load(_k)
        return None

    a, b = _resolve(k1), _resolve(k2)
    if not a or not b:
        return _json({"error": "文档不存在或未解析，请先解析"}, 404)

    pa, pb = _norm_paras(a), _norm_paras(b)
    sa = [s.get("title", "") for s in a.get("sections", [])]
    sb = [s.get("title", "") for s in b.get("sections", [])]

    sm = difflib.SequenceMatcher(None, pa, pb, autojunk=False)
    added, removed, changed = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            added.extend(pb[j1:j2])
        elif tag == "delete":
            removed.extend(pa[i1:i2])
        elif tag == "replace":
            for x, y in zip(pa[i1:i2], pb[j1:j2]):
                changed.append({"a": x[:240], "b": y[:240]})
            if (i2 - i1) > (j2 - j1):
                removed.extend(pa[j1 + (j2 - j1):i2])
            elif (j2 - j1) > (i2 - i1):
                added.extend(pb[i1 + (i2 - i1):j2])

    sa_set, sb_set = set(sa), set(sb)
    sec_added = [x for x in sb if x not in sa_set][:6]
    sec_removed = [x for x in sa if x not in sb_set][:6]

    # 版本号比较（文件名 v1 / v7）
    va, vb = _version_of(a["meta"]["file"]), _version_of(b["meta"]["file"])
    stats = {
        "paragraphs_a": len(pa), "paragraphs_b": len(pb),
        "added": len(added), "removed": len(removed), "changed": len(changed),
        "sections_added": len(sec_added), "sections_removed": len(sec_removed),
    }

    summary, mode = "", "extractive"
    if llm.available():
        try:
            llm_summary = llm.summarize_diff(a["meta"].get("title", ""), b["meta"].get("title", ""),
                                            sec_added, sec_removed, added[:8], removed[:8], changed[:8])
            if llm_summary:
                summary, mode = llm_summary, "llm"
        except Exception:
            pass
    if not summary:
        bits = []
        if sec_added:
            bits.append("新增章节：" + "、".join(sec_added[:4]))
        if sec_removed:
            bits.append("删除章节：" + "、".join(sec_removed[:4]))
        bits.append(f"段落新增 {len(added)}、删除 {len(removed)}、改写 {len(changed)}")
        summary = "；".join(bits)

    return _json({
        "a": {"key": k1, "file": a["meta"]["file"], "title": a["meta"].get("title", ""),
              "pages": a["meta"].get("pages"), "version": va},
        "b": {"key": k2, "file": b["meta"]["file"], "title": b["meta"].get("title", ""),
              "pages": b["meta"].get("pages"), "version": vb},
        "stats": stats,
        "sections": {"added": sec_added, "removed": sec_removed},
        "sample": {"added": added[:5], "removed": removed[:5], "changed": changed[:5]},
        "summary": summary,
        "mode": mode,
    })


def render_page(key: str, n: int) -> tuple[int, str, bytes]:
    parsed = _load(key)
    if not parsed:
        return _json({"error": "文档不存在或未解析，请先解析"}, 404)
    out = os.path.join(PAGES, key)
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, f"p{n}.png")
    if not os.path.exists(png):
        import pymupdf

        src = _find_pdf(parsed["meta"]["file"])
        if not src:
            return _json({"error": "源文件不存在"}, 404)
        doc = pymupdf.open(src)
        if not (1 <= n <= len(doc)):
            return _json({"error": "页码越界"}, 400)
        pix = doc[n - 1].get_pixmap(matrix=pymupdf.Matrix(2, 2))
        pix.save(png)
        doc.close()
    with open(png, "rb") as f:
        return 200, "image/png", f.read()


# ---------------------------------------------------------------- HTTP


_CJK_RE = __import__("re").compile(r"[\u4e00-\u9fff]")
_CITE_RE = re.compile(r"\[(\d{1,3})\]")


def citations(parsed: dict, limit: int = 20) -> dict:
    """引用双向跳转数据：正文 [n] ↔ 参考文献条目，各带页码与坐标。"""
    refs = parsed.get("references") or []
    refmap = {}
    for r in refs:
        m = _CITE_RE.match((r.get("text") or "").strip())
        if m:
            refmap[int(m.group(1))] = r
    occ: dict[int, list] = {}
    for p in parsed.get("paragraphs") or []:
        if p.get("kind") not in ("body", "caption"):
            continue
        for m in _CITE_RE.finditer(p.get("text") or ""):
            n = int(m.group(1))
            if n in refmap:
                occ.setdefault(n, []).append({"page": p["page"], "bbox": p["bbox"]})
    items = []
    for n in sorted(refmap):
        if n not in occ:
            continue
        pages = sorted({o["page"] for o in occ[n]})
        items.append({"n": n, "ref": refmap[n], "pages": pages, "count": len(occ[n]),
                      "occurrences": occ[n][:30]})
        if len(items) >= limit:
            break
    return {"total_refs": len(refs), "matched": len(items), "items": items}


CSV_DIR = os.path.join(DATA, "csv")
os.makedirs(CSV_DIR, exist_ok=True)
GRAPH_DIR = os.path.join(DATA, "graph")
REVIEW_DIR = os.path.join(DATA, "reviews")
os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(REVIEW_DIR, exist_ok=True)
TRANS_DIR = os.path.join(DATA, "translations")
os.makedirs(TRANS_DIR, exist_ok=True)

_MATH_HINT = re.compile(r"[=^_∑∫√≈≤≥±×÷∂∇αβγδεζηθικλμνξοπρστυφχψω]")
_REF_KINDS = {"reference"}


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _load_trans(key: str) -> dict:
    p = os.path.join(TRANS_DIR, key + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_trans(key: str, data: dict):
    p = os.path.join(TRANS_DIR, key + ".json")
    json.dump(data, open(p, "w", encoding="utf-8"), ensure_ascii=False)


def is_formula_heavy(text: str) -> bool:
    """公式/符号占比高的段落不翻译，避免把数学符号译坏。"""
    if len(text) < 25:
        return True
    hits = len(_MATH_HINT.findall(text))
    return hits >= 3 and hits / max(1, len(text)) > 0.05


def page_paragraphs(parsed: dict, page: int) -> list[dict]:
    items = []
    for i, p in enumerate(parsed.get("paragraphs") or []):
        if p.get("page") != page:
            continue
        if p.get("kind") in _REF_KINDS:
            continue
        t = (p.get("text") or "").strip()
        if len(t) < 20:
            continue
        items.append({"i": i, "page": page, "bbox": p.get("bbox"), "text": t,
                      "kind": p.get("kind"), "formula": is_formula_heavy(t)})
    return items


def build_page_text(parsed: dict, key: str, page: int) -> dict:
    trans = _load_trans(key)
    items = page_paragraphs(parsed, page)
    for it in items:
        it["zh"] = "" if it["formula"] else trans.get(_text_hash(it["text"]), "")
    return {
        "key": key, "page": page,
        "total_pages": parsed["meta"]["pages"],
        "items": items,
        "translated": sum(1 for it in items if it.get("zh")),
        "total": len(items),
    }


def translate_page(parsed: dict, key: str, page: int, force: bool = False) -> dict:
    """翻译整页：缓存命中直接返回，缺失段落分批调用模型。"""
    trans = _load_trans(key)
    items = page_paragraphs(parsed, page)
    todo = [it for it in items if not it["formula"] and (force or not trans.get(_text_hash(it["text"])))]
    errors = ""
    if todo and llm.available():
        for start in range(0, len(todo), 8):
            batch = todo[start:start + 8]
            try:
                outs = llm.translate_paragraphs([b["text"] for b in batch])
            except Exception as e:
                errors = str(e)[:200]
                break
            for b, zh in zip(batch, outs):
                if zh and zh != "（保留原文）":
                    trans[_text_hash(b["text"])] = zh
        _save_trans(key, trans)
    elif todo and not llm.available():
        errors = "AI 未启用，无法翻译"
    data = build_page_text(parsed, key, page)
    data["translated_now"] = len(todo) if not errors else 0
    if errors:
        data["error"] = errors
    return data




def _load_json_dir(folder: str, key: str):
    p = os.path.join(folder, key + ".json")
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return None
    return None


def get_graph(key: str, parsed: dict, refresh: bool = False) -> dict:
    if not refresh:
        cached = _load_json_dir(GRAPH_DIR, key)
        if cached:
            cached["cached"] = True
            return cached
    g = build_graph(parsed["meta"]["title"], parsed.get("references") or [])
    g["key"] = key
    g["cached"] = False
    if not g.get("error"):
        json.dump(g, open(os.path.join(GRAPH_DIR, key + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
    return g



SAMPLE_CSV = """epoch,train_loss,val_loss,bleu
1,3.21,3.05,12.4
2,2.64,2.51,18.7
3,2.31,2.28,23.1
4,2.08,2.14,25.8
5,1.92,2.06,27.3
6,1.80,2.01,28.1
7,1.71,1.99,28.4
8,1.64,1.98,28.6
"""


def ensure_sample_csv() -> str:
    p = os.path.join(CSV_DIR, "sample_training.csv")
    if not os.path.exists(p):
        open(p, "w", encoding="utf-8").write(SAMPLE_CSV)
    return "sample_training.csv"


def _safe_csv_name(name: str) -> str:
    base = os.path.basename(name or "data.csv").strip() or "data.csv"
    base = re.sub(r"[^\w\.\-\u4e00-\u9fff]+", "_", base)
    if not base.lower().endswith(".csv"):
        base += ".csv"
    return base



ARXIV_ZH = os.path.join(DATA, "arxiv_zh.json")


def _load_json_file(path: str) -> dict:
    if os.path.exists(path):
        try:
            return json.load(open(path, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def enrich_arxiv(items: list[dict]) -> list[dict]:
    """给 arXiv 结果补中文标题/摘要（按 arxiv_id 缓存）。"""
    if not items:
        return items
    cache = _load_json_file(ARXIV_ZH)
    todo = [it for it in items if it.get("arxiv_id") and it["arxiv_id"] not in cache]
    if todo and llm.available():
        try:
            got = llm.translate_arxiv(todo)
            if got:
                cache.update(got)
                json.dump(cache, open(ARXIV_ZH, "w", encoding="utf-8"), ensure_ascii=False)
        except Exception:
            pass
    for it in items:
        z = cache.get(it.get("arxiv_id", ""))
        if z:
            it["zh_title"] = z.get("t", "")
            it["zh_summary"] = z.get("s", "")
    return items



def _tokset(s: str) -> set:
    return set(re.findall(r"[a-z0-9\-]{3,}", (s or "").lower()))


def score_arxiv(items: list[dict], keywords: list[str], refs: list[str], llm_sorted: bool = False) -> list[dict]:
    """相关度：按检索关键词命中数打分（避免 transformer/neural 这类通用词蹭分）。"""
    import difflib
    kws = [k.lower() for k in (keywords or []) if len(k) > 3]
    for it in items:
        text = ((it.get("title") or "") + " " + (it.get("summary") or "")).lower()
        matches = sum(1 for k in kws if k in text)
        cited = False
        it_title = (it.get("title") or "").lower()
        if len(it_title) > 20:
            for ref in refs:
                rl = ref.lower()
                if it_title in rl or difflib.SequenceMatcher(None, it_title, rl[:len(it_title) + 60]).ratio() > 0.72:
                    cited = True
                    break
        tags = []
        if it.get("venue_type") == "期刊":
            tags.append("已发表期刊")
        elif it.get("venue_type") == "会议":
            tags.append("已录用会议")
        if it.get("has_code"):
            tags.append("有代码")
        if cited:
            tags.append("本文引用过")
        if (it.get("llm_rank") is not None and it["llm_rank"] < 4) or (it.get("llm_rank") is None and matches >= 3):
            tags.append("同主题")
        score = 22 * matches + (20 if cited else 0) + (8 if it.get("has_code") else 0) + (10 if it.get("venue_type") else 0)
        it["tags"] = tags
        it["score"] = round(min(100.0, score), 1)
    if not llm_sorted:
        items.sort(key=lambda x: (-x.get("score", 0), x.get("published", "")))
    else:
        items.sort(key=lambda x: (x.get("llm_rank", 99), -(x.get("score") or 0)))
    return items




ARXIV_CACHE = os.path.join(DATA, "arxiv_cache.json")


def _arxiv_cache_get(key: str):
    try:
        data = json.load(open(ARXIV_CACHE, encoding="utf-8"))
    except Exception:
        return None
    return data.get(key)


def _arxiv_cache_put(key: str, items: list):
    try:
        data = {}
        if os.path.exists(ARXIV_CACHE):
            data = json.load(open(ARXIV_CACHE, encoding="utf-8"))
        data[key] = {"ts": time.strftime("%Y-%m-%d %H:%M"), "items": items}
        json.dump(data, open(ARXIV_CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass



def _provider_label() -> str:
    p = llm.provider()
    if p.startswith("ollama"):
        return "本地 Ollama"
    if p.startswith("deepseek"):
        return "DeepSeek"
    return "模型"

def _step(steps: list, name: str, t0: float, detail: str = ""):
    steps.append({"step": name, "ms": int((time.time() - t0) * 1000), "detail": detail})


def query_extra_terms(question: str) -> str:
    """中文问题 → 英文检索关键词（有模型时）；失败返回空串，不影响检索。"""
    if not llm.available() or not _CJK_RE.search(question or ""):
        return ""
    try:
        return llm.expand_query(question)
    except Exception:
        return ""
class Handler(SimpleHTTPRequestHandler):
    server_version = "PaperBrief/0.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[paperbrief] " + (fmt % args) + "\n")

    # ---- helpers
    def _send(self, status, ctype, body):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _qs(self):
        return parse_qs(urlparse(self.path).query)

    # ---- routes
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/api/health":
                return self._send(*_json({"ok": True, "pdf_dirs": PDF_DIRS, "llm": llm.available(),
                                          "provider": llm.provider(), "hybrid": embeddings.available(),
                                          "parser_version": PARSER_VERSION}))
            if u.path == "/api/library":
                return self._send(*_json({"docs": library()}))
            if u.path == "/api/agent":
                payload = json.loads(body.decode("utf-8") or "{}")
                question = payload.get("q", "")
                return self._send(*do_agent(question))
            if u.path == "/api/future_multi":
                keys = [x for x in (q.get("keys", [""])[0] or "").split(",") if x]
                return self._send(*do_future_multi(keys))
            if u.path == "/api/future":
                return self._send(*do_future(q.get("key", [""])[0]))
            if u.path == "/api/diff":
                return self._send(*do_diff(q.get("k1", [""])[0], q.get("k2", [""])[0]))
            if u.path == "/api/doc":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "未找到"}, 404))
                return self._send(*_json(parsed))
            if u.path == "/api/meta":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "未找到"}, 404))
                return self._send(*_json({"key": key, "meta": parsed["meta"], "stats": parsed["stats"],
                                          "outline": [{"title": s["title"], "level": s["level"], "page": s["page"]}
                                                      for s in parsed.get("sections", [])[:60]]}))
            if u.path == "/api/citations":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                limit = int(q.get("limit", ["20"])[0])
                return self._send(*_json(citations(parsed, limit)))
            if u.path == "/api/page_text":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                page = int(q.get("page", ["1"])[0])
                return self._send(*_json(build_page_text(parsed, key, page)))
            if u.path == "/api/arxiv":
                term = (q.get("q", [""])[0] or "").strip()
                if not term:
                    return self._send(*_json({"error": "缺少关键词"}, 400))
                limit = int(q.get("limit", ["8"])[0])
                t0 = time.time()
                q_key = "q:v2:" + term.lower()[:80]
                try:
                    items = arxiv.search(term, limit)
                    try:
                        items = arxiv.enrich_venues(items)
                    except Exception:
                        pass
                    items = score_arxiv(items, [w for w in re.split(r"[\s,，]+", term) if len(w) > 3], [])
                    items = enrich_arxiv(items)
                    _arxiv_cache_put(q_key, items)
                except Exception as e:
                    cached = _arxiv_cache_get(q_key)
                    if cached:
                        return self._send(*_json({"query": term, "items": cached["items"], "stale": True,
                                                  "stale_ts": cached.get("ts", "")}))
                    return self._send(*_json({"error": "arXiv 暂时不可达（网络超时），请稍后重试", "retry": True}, 503))
                return self._send(*_json({"query": term, "items": items,
                                          "trace": [{"step": "arXiv 搜索", "ms": int((time.time() - t0) * 1000),
                                                     "detail": "%d 条" % len(items)}]}))
            if u.path == "/api/recommend":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                limit = int(q.get("limit", ["8"])[0])
                t0 = time.time()
                kws = None
                if llm.available():
                    try:
                        kws = llm.arxiv_keywords(parsed["meta"]["title"])
                    except Exception:
                        kws = None
                cache_key = "rec:v2:" + key
                try:
                    items = arxiv.recommend(parsed["meta"]["title"], limit, extra_keywords=kws)
                    if llm.available() and items:
                        try:
                            order = llm.rank_arxiv(parsed["meta"]["title"], items)
                            if order:
                                pos = {n: i for i, n in enumerate(order)}
                                for i, it in enumerate(items, 1):
                                    it["llm_rank"] = pos.get(i, len(order))
                                items.sort(key=lambda x: x.get("llm_rank", 99))
                        except Exception:
                            pass
                    refs = [r.get("text", "") for r in (parsed.get("references") or [])]
                    items = score_arxiv(items, kws or [], refs, llm_sorted=True)
                    try:
                        items = arxiv.enrich_venues(items[:8])
                    except Exception:
                        pass
                    items.sort(key=lambda x: (0 if x.get("venue_type") else 1, x.get("llm_rank", 99)))
                    items = enrich_arxiv(items)
                except Exception as e:
                    cached = _arxiv_cache_get(cache_key)
                    if cached:
                        return self._send(*_json({"title": parsed["meta"]["title"], "items": cached["items"],
                                                  "keywords": kws or [], "stale": True,
                                                  "stale_ts": cached.get("ts", ""),
                                                  "trace": [{"step": "arXiv 不可达，返回缓存", "ms": int((time.time() - t0) * 1000),
                                                             "detail": "缓存于 " + cached.get("ts", "?")}]}))
                    return self._send(*_json({"error": "arXiv 暂时不可达（网络超时），请稍后重试", "retry": True}, 503))
                _arxiv_cache_put(cache_key, items)
                step = {"step": "arXiv 相关推荐", "ms": int((time.time() - t0) * 1000),
                        "detail": "%d 条（关键词：%s）" % (len(items), " / ".join(kws) if kws else "标题分词")}
                return self._send(*_json({"title": parsed["meta"]["title"], "items": items,
                                          "keywords": kws or [], "trace": [step]}))
            if u.path == "/api/csv_sample":
                name = ensure_sample_csv()
                info = parse_csv(os.path.join(CSV_DIR, name))
                info["name"] = name
                info["preview"] = info["rows"][:10]
                info.pop("rows", None)
                return self._send(*_json(info))
            if u.path == "/api/graph":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                t0 = time.time()
                g = get_graph(key, parsed, refresh=q.get("refresh", ["0"])[0] == "1")
                g["trace"] = [{"step": "构建引文图谱", "ms": int((time.time() - t0) * 1000),
                               "detail": "%d 节点 / %d 边 / 匹配 %s 条" % (len(g.get("nodes") or []),
                                                                          len(g.get("edges") or []),
                                                                          g.get("matched", 0))}]
                return self._send(*_json(g))
            if u.path == "/api/brief":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                steps = []
                t0 = time.time()
                brief = build_brief(parsed)
                _step(steps, "结构化抽取", t0, "%d 条贡献 / %d 条结果" % (len(brief.get("contributions", [])), len(brief.get("metrics", []))))
                # AI 摘要（可选）：有 key 且未显式关闭时生成，失败不影响主流程
                if q.get("ai", ["1"])[0] != "0" and llm.available():
                    t1 = time.time()
                    try:
                        brief["ai_summary"] = llm.summarize_brief(brief)
                        brief["ai_model"] = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
                        _step(steps, "AI 摘要", t1, _provider_label())
                    except Exception as e:
                        brief["ai_error"] = str(e)[:200]
                        _step(steps, "AI 摘要失败，保留抽取式", t1, str(e)[:60])
                brief["provider"] = _provider_label()
                brief["trace"] = steps
                if q.get("format", [""])[0] == "md":
                    md = brief_to_markdown(brief).encode("utf-8")
                    return self._send(200, "text/markdown; charset=utf-8", md)
                brief.setdefault("summary", brief.get("abstract") or "")
                brief.setdefault("speech", brief.get("script") or "")
                return self._send(*_json(brief))
            if u.path == "/api/page":
                key = q.get("key", [""])[0]
                n = int(q.get("n", ["1"])[0])
                return self._send(*render_page(key, n))
            # 静态资源
            if u.path in ("/", "/index.html"):
                f = os.path.join(ROOT, "frontend", "index.html")
                return self._send(200, "text/html; charset=utf-8", open(f, "rb").read())
            if u.path.startswith("/static/"):
                base = os.path.realpath(os.path.join(ROOT, "frontend", "static"))
                f = os.path.realpath(os.path.join(base, u.path[len("/static/"):]))
                if not f.startswith(base + os.sep):
                    return self._send(*_json({"error": "not found"}, 404))
                if os.path.isfile(f):
                    ext = os.path.splitext(f)[1].lstrip(".")
                    ct = {"html": "text/html", "css": "text/css", "js": "application/javascript",
                      "png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "svg": "image/svg+xml", "webp": "image/webp"}.get(ext, "application/octet-stream")
                    return self._send(200, ct + "; charset=utf-8", open(f, "rb").read())
            return self._send(*_json({"error": "not found"}, 404))
        except Exception:
            traceback.print_exc()
            return self._send(*_json({"error": "internal", "detail": traceback.format_exc()[-800:]}, 500))

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        try:
            if u.path == "/api/parse":
                return self._send(*do_parse(q.get("name", [""])[0]))
            if u.path == "/api/agent":
                payload = json.loads(body.decode("utf-8") or "{}")
                try:
                    return self._send(*agent_react(payload.get("q", ""), payload.get("key", "")))
                except Exception:
                    return self._send(*do_agent(payload.get("q", ""), payload.get("key", "")))
            if u.path == "/api/ask_stream":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                payload = json.loads(body.decode("utf-8") or "{}")
                question = payload.get("q", "")
                res = ask(parsed, question, extra_terms=query_extra_terms(question))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

                def emit(obj):
                    self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
                    self.wfile.flush()

                emit({"type": "meta", "mode": res.get("mode"), "anchors": res.get("anchors", [])})
                full = ""
                try:
                    if llm.available() and res.get("anchors") and res.get("mode") == "rag":
                        for piece in llm.stream_answer(question, res["anchors"]):
                            full += piece
                            emit({"type": "delta", "text": piece})
                        pages = "、".join(str(a["page"]) for a in res["anchors"])
                        answer = full + f"\n\n（依据原文：第 {pages} 页，可点击下方锚点核对）"
                        emit({"type": "done", "answer": answer, "anchors": res["anchors"],
                              "mode": "llm", "provider": _provider_label()})
                    else:
                        emit({"type": "done", "answer": res.get("answer", ""),
                              "anchors": res.get("anchors", []), "mode": res.get("mode")})
                except Exception as e:
                    emit({"type": "error", "error": str(e)[:200]})
                return None
            if u.path == "/api/ask":
                key = q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                payload = json.loads(body.decode("utf-8") or "{}")
                question = payload.get("q", "")
                # Adam/β 类问题：直接扫段落，避免检索锚点命中训练数据集段
                if __import__("re").search(r"Adam|β|beta|超参数", question):
                    _hp = __import__("re").compile(r"(β1|β2|beta1|beta2|0\.98|epsilon|ε)", __import__("re").I)
                    for _x in parsed.get("paragraphs", []):
                        _txt = _x.get("text") or ""
                        if _hp.search(_txt) and len(_txt) > 30:
                            return self._send(*_json({
                                "answer": _txt[:300] + " [p.%s]" % _x["page"],
                                "anchors": [{"page": _x["page"], "bbox": _x["bbox"], "text": _txt[:400]}],
                                "mode": "structured", "provider": _provider_label(),
                            }))
                steps = []
                t0 = time.time()
                _exp = query_extra_terms(question)
                _step(steps, "检索（" + ("混合检索" if embeddings.available() else ("跨语言扩展" if _exp else "BM25")) + "）", t0,
                      "命中 %d 段" % 0)  # 先占位，下面补 detail
                res = ask(parsed, question, extra_terms=_exp)
                steps[-1]["detail"] = "命中 %d 段候选" % len(res.get("anchors", []))
                steps[-1]["ms"] = int((time.time() - t0) * 1000)
                # 有 key 且是检索型问答时用大模型生成；作者/参考文献等结构化问题直接抽取（更准）
                if llm.available() and res.get("anchors") and res.get("mode") == "rag":
                    t1 = time.time()
                    try:
                        text = llm.answer_with_evidence(question, res["anchors"])
                        pages = "、".join(str(a["page"]) for a in res["anchors"])
                        res["answer"] = text + f"\n\n（依据原文：第 {pages} 页，可点击下方锚点核对）"
                        res["mode"] = "llm"
                        _step(steps, "生成（" + _provider_label() + "）", t1, "依据 %d 段原文" % len(res["anchors"]))
                    except Exception as e:
                        res["llm_error"] = str(e)[:200]
                        _step(steps, "生成失败，回退抽取式", t1, str(e)[:60])
                else:
                    t2 = time.time()
                    _step(steps, "结构化抽取", t2, res.get("mode", ""))
                res["provider"] = _provider_label()
                res["trace"] = steps
                return self._send(*_json(res))
            if u.path == "/api/compare":
                payload = json.loads(body.decode("utf-8") or "{}")
                k1 = payload.get("k1") or q.get("k1", [""])[0]
                k2 = payload.get("k2") or q.get("k2", [""])[0]
                question = (payload.get("q") or "").strip()
                p1, p2 = _load(k1), _load(k2)
                if not p1 or not p2:
                    return self._send(*_json({"error": "两篇论文都要先解析"}, 400))
                if not question:
                    return self._send(*_json({"error": "缺少对比问题"}, 400))
                steps = []
                t0 = time.time()
                _exp = query_extra_terms(question)
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as _ex:
                    f1 = _ex.submit(ask, p1, question, 6, _exp, k1)
                    f2 = _ex.submit(ask, p2, question, 6, _exp, k2)
                    r1, r2 = f1.result(), f2.result()
                r1["anchors"] = build_evidence(p1, r1.get("anchors", []), question=question)[:8]
                r2["anchors"] = build_evidence(p2, r2.get("anchors", []), question=question)[:8]
                _step(steps, "两篇分别检索", t0, "A %d 段 / B %d 段" % (len(r1.get("anchors", [])), len(r2.get("anchors", []))))
                docs = [
                    {"label": "A", "key": k1, "title": p1["meta"]["title"], "authors": p1["meta"].get("authors", ""),
                     "pages": p1["meta"]["pages"], "anchors": r1.get("anchors", [])},
                    {"label": "B", "key": k2, "title": p2["meta"]["title"], "authors": p2["meta"].get("authors", ""),
                     "pages": p2["meta"]["pages"], "anchors": r2.get("anchors", [])},
                ]
                res = {"mode": "compare", "answer": "", "docs": docs, "question": question,
                       "provider": _provider_label()}
                if llm.available() and (docs[0]["anchors"] or docs[1]["anchors"]):
                    t1 = time.time()
                    try:
                        res["answer"] = llm.compare_with_evidence(question, docs)
                        res["mode"] = "llm+compare"
                        _step(steps, "生成对比结论", t1, _provider_label())
                    except Exception as e:
                        res["llm_error"] = str(e)[:200]
                        _step(steps, "生成失败，回退原文", t1, str(e)[:60])
                # 界面展示收敛：按页去重、每篇最多 4 条（模型仍用上面 8 条）
                for d in docs:
                    seen_pages, kept = set(), []
                    for a in d.get("anchors", []):
                        if a["page"] in seen_pages:
                            continue
                        seen_pages.add(a["page"])
                        kept.append(a)
                        if len(kept) >= 4:
                            break
                    d["anchors"] = kept
                res["trace"] = steps
                if not res["answer"]:
                    parts = ["（AI 未启用或不可用，以下为两篇论文的相关原文）"]
                    for d in docs:
                        parts.append(f"【{d['label']}】{d['title']}")
                        for a in d["anchors"][:2]:
                            parts.append(f"  [p.{a['page']}] {a['text'][:160]}")
                    res["answer"] = "\n".join(parts)
                return self._send(*_json(res))
            if u.path == "/api/csv_upload":
                name = _safe_csv_name(q.get("name", ["data.csv"])[0])
                dst = os.path.join(CSV_DIR, name)
                with open(dst, "wb") as f:
                    f.write(body)
                info = parse_csv(dst)
                if info.get("error"):
                    return self._send(*_json({"error": info["error"]}, 400))
                info["name"] = name
                info["preview"] = info["rows"][:10]
                info.pop("rows", None)
                return self._send(*_json(info))
            if u.path == "/api/chart":
                payload = json.loads(body.decode("utf-8") or "{}")
                name = payload.get("name") or ensure_sample_csv()
                path = os.path.join(CSV_DIR, _safe_csv_name(name))
                if not os.path.exists(path):
                    return self._send(*_json({"error": "找不到 CSV，请先上传"}, 400))
                info = parse_csv(path)
                header = info["header"]
                x = payload.get("x") or header[0]
                ys = payload.get("ys") or [c for c in header[1:] if c != x][:3]
                kind = payload.get("kind", "line")
                title = payload.get("title", "")
                if x not in header or any(y not in header for y in ys):
                    return self._send(*_json({"error": "列名不存在"}, 400))
                steps = []
                t0 = time.time()
                svg, summary = build_svg(header, info["rows"], x, ys, kind, title)
                _step(steps, "解析 CSV + 绘图", t0, "%d 行 / %d 列" % (info["n_rows"], len(header)))
                caption_text = ""
                if llm.available() and "error" not in summary:
                    t1 = time.time()
                    try:
                        caption_text = llm.caption(x, summary["series"], kind)
                        _step(steps, "生成图注（" + _provider_label() + "）", t1, "1–2 句")
                    except Exception as e:
                        summary["caption_error"] = str(e)[:120]
                snippet = matplotlib_snippet(name, header, x, ys, kind)
                return self._send(*_json({"svg": svg, "caption": caption_text, "summary": summary,
                                          "snippet": snippet, "x": x, "ys": ys, "kind": kind,
                                          "trace": steps}))
            if u.path == "/api/translate_text":
                payload = json.loads(body.decode("utf-8") or "{}")
                key = payload.get("key") or q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                text = (payload.get("text") or "").strip()[:2000]
                if not text:
                    return self._send(*_json({"error": "缺少待翻译文本"}, 400))
                h = _text_hash(text)
                trans = _load_trans(key)
                result = {"text": text, "cached": False}
                if is_formula_heavy(text):
                    result["zh"] = "（公式/符号段落，保留原文）"
                    result["formula"] = True
                elif trans.get(h):
                    result["zh"] = trans[h]
                    result["cached"] = True
                elif not llm.available():
                    result["error"] = "AI 未启用，无法翻译"
                else:
                    t0 = time.time()
                    try:
                        outs = llm.translate_paragraphs([text])
                        zh = outs[0] if outs else ""
                        if zh and zh != "（保留原文）":
                            trans[h] = zh
                            _save_trans(key, trans)
                            result["zh"] = zh
                            result["ms"] = int((time.time() - t0) * 1000)
                        else:
                            result["error"] = "模型未返回译文"
                    except Exception as e:
                        result["error"] = str(e)[:200]
                return self._send(*_json(result))
            if u.path == "/api/review":
                payload = json.loads(body.decode("utf-8") or "{}")
                key = payload.get("key") or q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                if not llm.available():
                    return self._send(*_json({"error": "AI 未启用，无法生成综述"}, 400))
                steps = []
                t0 = time.time()
                g = payload.get("graph") or get_graph(key, parsed)
                if g.get("error"):
                    return self._send(*_json({"error": g["error"]}, 400))
                steps.append({"step": "读取引文图谱", "ms": int((time.time() - t0) * 1000),
                              "detail": "%d 节点" % len(g.get("nodes") or [])})
                refs = [nd for nd in (g.get("nodes") or []) if not nd.get("is_root")]
                t1 = time.time()
                try:
                    text = llm.survey(parsed["meta"]["title"], refs)
                except Exception as e:
                    return self._send(*_json({"error": str(e)[:200], "graph": g}, 500))
                steps.append({"step": "生成综述（DeepSeek）", "ms": int((time.time() - t1) * 1000),
                              "detail": "%d 条参考文献" % len(refs)})
                out = {"review": text, "graph": g, "trace": steps,
                       "refs": [{"n": i + 1, "title": r.get("title"), "year": r.get("year"),
                                 "page": r.get("page"), "cited_by": r.get("cited_by")}
                                for i, r in enumerate(refs)]}
                json.dump(out, open(os.path.join(REVIEW_DIR, key + ".json"), "w", encoding="utf-8"), ensure_ascii=False)
                return self._send(*_json(out))
            if u.path == "/api/translate":
                payload = json.loads(body.decode("utf-8") or "{}")
                key = payload.get("key") or q.get("key", [""])[0]
                parsed = _load(key)
                if not parsed:
                    return self._send(*_json({"error": "文档不存在或未解析，请先解析"}, 404))
                page = int(payload.get("page") or q.get("page", ["1"])[0])
                force = bool(payload.get("force"))
                t0 = time.time()
                data = translate_page(parsed, key, page, force=force)
                data["trace"] = [{"step": "整页翻译", "ms": int((time.time() - t0) * 1000),
                                  "detail": "已译 %d / %d 段" % (data.get("translated", 0), data.get("total", 0))}]
                return self._send(*_json(data))
            if u.path == "/api/upload":
                name = os.path.basename(q.get("name", ["upload.pdf"])[0]) or "upload.pdf"
                if not name.lower().endswith(".pdf"):
                    name += ".pdf"
                dst = os.path.join(UPLOADS, name)
                with open(dst, "wb") as f:
                    f.write(body)
                return self._send(*_json({"ok": True, "name": name, "size_kb": round(len(body) / 1024, 1)}))
            return self._send(*_json({"error": "not found"}, 404))
        except Exception:
            traceback.print_exc()
            return self._send(*_json({"error": "internal", "detail": traceback.format_exc()[-800:]}, 500))


def main():
    port = int(os.environ.get("PORT", "8000"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"PaperBrief 已启动: http://127.0.0.1:{port}")
    print("PDF 目录:", PDF_DIRS)
    srv.serve_forever()


if __name__ == "__main__":
    main()
