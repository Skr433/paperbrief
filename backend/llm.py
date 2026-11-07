# -*- coding: utf-8 -*-
"""DeepSeek 大模型接入（生成式问答 / 汇报摘要），带超时与降级。

设计原则（对应命题「抗幻觉」「输出可验证」）：
- 模型只能看到检索出的原文片段，回答必须带 [p.X] 页码；
- 调用失败/超时/无 key 时，上层自动回退到抽取式答案，绝不白屏；
- key 只从环境变量读取，不写进代码、不进仓库。
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


def _cfg() -> dict:
    return {
        "key": (os.environ.get("DEEPSEEK_API_KEY") or "").strip(),
        "base": (os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
        "model": os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
    }


def _ollama_cfg() -> dict:
    return {
        "base": (os.environ.get("OLLAMA_BASE_URL") or "").strip().rstrip("/"),
        "model": os.environ.get("OLLAMA_MODEL") or "qwen2.5:3b",
    }


def provider() -> str:
    if _ollama_cfg()["base"]:
        return "ollama:" + _ollama_cfg()["model"]
    if _cfg()["key"]:
        return "deepseek:" + _cfg()["model"]
    return "none"


def available() -> bool:
    return bool(_ollama_cfg()["base"] or _cfg()["key"])


def chat(messages: list[dict], temperature: float = 0.2, timeout: int = 25, max_tokens: int = 800) -> str:
    oc = _ollama_cfg()
    if oc["base"]:
        return _chat_ollama(messages, oc, temperature, timeout, max_tokens)
    cfg = _cfg()
    if not cfg["key"]:
        raise RuntimeError("DEEPSEEK_API_KEY 未配置")
    payload = json.dumps(
        {"model": cfg["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ).encode("utf-8")
    req = urllib.request.Request(
        cfg["base"] + "/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + cfg["key"]},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:200]
        raise RuntimeError(f"HTTP {e.code}: {detail}")
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {str(e)[:150]}")
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "").strip()


def translate_paragraphs(texts: list[str]) -> list[str]:
    """批量英译中（学术风格）。保留公式/变量/数字/引用编号/图表编号原样。

    返回与输入等长的译文列表；模型漏掉的条目返回空串，由上层决定是否回退原文。
    """
    if not texts:
        return []
    numbered = "\n".join(f"{i + 1}|||{t[:1200]}" for i, t in enumerate(texts))
    messages = [
        {
            "role": "system",
            "content": (
                "你是学术论文翻译。把每段英文翻译成简体中文，要求：\n"
                "1) 公式、变量名、数字、单位、引用编号（如 [11]）、图表编号（如 Figure 3）保持原样；\n"
                "2) 术语准确、书面学术风格，不要口语化；\n"
                "3) 每行输出「序号|||译文」，序号与输入一致，不要解释、不要合并段落；\n"
                "4) 无法翻译的段落输出「序号|||（保留原文）」。"
            ),
        },
        {"role": "user", "content": numbered},
    ]
    out = chat(messages, temperature=0.1, timeout=60, max_tokens=3000)
    result = [""] * len(texts)
    for line in out.splitlines():
        m = re.match(r"^\s*(\d{1,3})\s*\|\|\|\s*(.*)$", line)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(result):
                result[idx] = m.group(2).strip()
    return result


def answer_with_evidence(question: str, anchors: list[dict]) -> str:
    ctx = "\n".join(
        f"[p.{a.get('page')}] {(a.get('text') or '')[:600]}" for a in anchors[:3] if a.get("text")
    )
    if not ctx:
        raise RuntimeError("无可用原文片段")
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研论文阅读助手。只能依据给定的<原文>回答，不得编造或补充原文以外的信息；"
                "每条结论后必须标注来源页码，格式如 [p.7]；原文没有答案时直接回答“原文未给出该信息”。"
                "用中文回答，不超过 120 字。"
            ),
        },
        {"role": "user", "content": f"<原文>\n{ctx}\n</原文>\n\n问题：{question}"},
    ]
    return chat(messages, temperature=0.1, timeout=25, max_tokens=400)


def summarize_brief(brief: dict) -> str:
    """基于汇报包的证据生成中文总结 + 3 条要点，每条带页码。"""
    contrib = brief.get("contributions") or []
    metrics = brief.get("metrics") or []
    lines = [f"摘要：{(brief.get('abstract') or '')[:700]}"]
    for c in contrib[:3]:
        lines.append(f"贡献 [p.{c.get('page')}]：{(c.get('text') or '')[:150]}")
    for m in metrics[:2]:
        lines.append(f"结果 [p.{m.get('page')}]：{(m.get('text') or '')[:150]}")
    material = "\n".join(lines)
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研汇报助手。只能依据给定材料写中文总结与要点，不得添加材料以外的事实；"
                "页码只能引用材料中已给出的 [p.X]，不得自造。输出格式严格如下：\n"
                "总结：<三句话以内>\n要点：\n1) <要点> [p.X]\n2) <要点> [p.X]\n3) <要点> [p.X]"
            ),
        },
        {"role": "user", "content": material},
    ]
    return chat(messages, temperature=0.2, timeout=30, max_tokens=600)


def expand_query(question: str) -> str:
    """把中文问题转成英文检索关键词（用于跨语言检索），只返回关键词。"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是检索助手。把用户的中文科研问题转成用于检索英文论文的关键词，"
                "只输出关键词，空格分隔，最多 10 个，不要编号、不要解释、不要整句。"
            ),
        },
        {"role": "user", "content": question},
    ]
    out = chat(messages, temperature=0.0, timeout=15, max_tokens=80)
    return " ".join(out.replace("\n", " ").split())[:200]


def compare_with_evidence(question: str, docs: list[dict]) -> str:
    """多篇论文对比：每篇给定检索片段，要求模型对比并分别标注 [A p.X] / [B p.Y]。"""
    blocks = []
    for d in docs:
        blocks.append(f"论文{d.get('label')}：{d.get('title', '')}")
        for a in (d.get("anchors") or [])[:12]:
            if a.get("text"):
                blocks.append(f"[{d.get('label')} p.{a.get('page')}] {a['text'][:420]}")
    material = "\n".join(blocks)
    if not material.strip():
        raise RuntimeError("两篇论文均无可用片段")
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研论文对比助手。只能依据给定的两篇论文片段作答，不得编造或引入材料外的知识。"
                "输出中文，用 3–5 条要点对比；每条要点后必须标注来源，格式为 [A p.X] 或 [B p.Y]。"
                "区分两类结论：①材料直接陈述的事实，必须引用原文；"
                "②由两篇材料对比可推出的结论（例如 A 只包含编码器、B 同时包含编码器与解码器），可以给出推断，"
                "但必须写明“据 [A p.X] 与 [B p.Y] 推断”。"
                "遇到具体数字（参数量、上下文长度等）材料未给出时，明确写“原文材料不足”，不要猜测数字。"
            ),
        },
        {"role": "user", "content": f"<材料>\n{material}\n</材料>\n\n对比问题：{question}"},
    ]
    return chat(messages, temperature=0.2, timeout=35, max_tokens=800)



def survey(title: str, refs: list[dict]) -> str:
    """基于参考文献标题写中文综述（技术脉络 / 主题分组 / 未来工作提示）。"""
    lines = []
    for i, r in enumerate(refs[:20], 1):
        lines.append(f"[{i}] {r.get('title', '')}（{r.get('year') or '年份不详'}，被引 {r.get('cited_by', 0)}）")
    material = "论文标题：" + title + "\n参考文献：\n" + "\n".join(lines)
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研综述助手。只能依据给定的参考文献标题与元数据写一段中文综述（250-350 字），"
                "内容包含：①该论文的技术脉络（受哪些前作影响）；②把参考文献按主题分成 2-3 组并各举 1-2 个 [n]；"
                "③基于这些引用推测的后续研究方向。不得编造未给出的论文内容，只能从标题合理推断；"
                "引用编号必须来自给定列表，不要自造。"
            ),
        },
        {"role": "user", "content": material},
    ]
    return chat(messages, temperature=0.3, timeout=40, max_tokens=700)


def caption(x: str, series: dict, kind: str) -> str:
    """根据数据摘要写 1-2 句中文图注（学术风格）。"""
    desc = []
    for name, st in series.items():
        trend = "上升" if st["last"] > st["first"] else ("下降" if st["last"] < st["first"] else "持平")
        desc.append(f"{name}：n={st['n']}，范围 {st['min']}~{st['max']}，均值 {st['mean']}，整体{trend}（{st['first']}→{st['last']}）")
    material = f"图表类型：{kind}；横轴：{x}\n" + "\n".join(desc)
    messages = [
        {
            "role": "system",
            "content": (
                "你是科研写作助手。根据给定的数据摘要写 1–2 句中文图注（Figure Caption），"
                "要求：学术书面语、点明横纵轴含义与主要趋势/对比结论、不超过 90 字；"
                "不要出现“我”“我们”“这张图”等口语，不要编造数据摘要以外的数字。"
            ),
        },
        {"role": "user", "content": material},
    ]
    return chat(messages, temperature=0.2, timeout=25, max_tokens=200)


def arxiv_keywords(title: str) -> list[str]:
    """给 arXiv 推荐用 4 个英文检索词（比机械去停用词更准）。"""
    messages = [
        {"role": "system", "content": "你是检索助手。针对给定论文标题，输出 4 个用于检索相关论文的英文关键词，空格分隔，只输出关键词，不要解释。"},
        {"role": "user", "content": title},
    ]
    out = chat(messages, temperature=0.0, timeout=15, max_tokens=60)
    words = [w.strip(" ,;，。").lower() for w in out.replace("\n", " ").split() if w.strip()]
    return [w for w in words if len(w) > 2][:4]


def translate_arxiv(items: list[dict]) -> dict:
    """批量生成中文标题 + 一句中文摘要，返回 {arxiv_id: {t, s}}。"""
    if not items:
        return {}
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}|||{it.get('title','')}|||{(it.get('summary') or '')[:320]}")
    messages = [
        {"role": "system", "content": (
            "你是学术翻译。为每条 arXiv 论文生成：中文标题（保留模型名/数据集等专有名词原文）与一句中文摘要（不超过 60 字）。"
            "输出格式严格为每行：序号|||中文标题|||中文摘要，不要解释、不要编号以外的前缀。"
        )},
        {"role": "user", "content": "\n".join(lines)},
    ]
    out = chat(messages, temperature=0.1, timeout=60, max_tokens=2500)
    result = {}
    for line in out.splitlines():
        parts = line.split("|||")
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0].strip())
        except ValueError:
            continue
        if 1 <= idx <= len(items):
            result[items[idx - 1].get("arxiv_id", str(idx))] = {
                "t": parts[1].strip()[:200], "s": parts[2].strip()[:300]}
    return result


def rank_arxiv(title: str, items: list[dict]) -> list[int]:
    """让模型给出候选论文的相关度排序，返回按相关度从高到低的序号列表（1-based）。"""
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(f"{i}|||{(it.get('title') or '')[:140]}|||{(it.get('summary') or '')[:220]}")
    material = "当前论文标题：" + title + "\n候选论文：\n" + "\n".join(lines)
    messages = [
        {"role": "system", "content": (
            "你是论文检索评估员。按“是否解决同一问题 / 是否同一技术脉络 / 是否只是共享术语”判断，"
            "把候选论文从最相关到最不相关排序。只输出一行：序号用逗号分隔，例如 3,1,5,2,4,6,7,8。"
            "不要解释，不要遗漏序号。"
        )},
        {"role": "user", "content": material},
    ]
    out = chat(messages, temperature=0.0, timeout=40, max_tokens=120)
    nums = [int(x) for x in re.findall(r"\d+", out) if 1 <= int(x) <= len(items)]
    seen, order = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n)
            order.append(n)
    for n in range(1, len(items) + 1):
        if n not in seen:
            order.append(n)
    return order


def _chat_ollama(messages: list[dict], oc: dict, temperature: float, timeout: int, max_tokens: int) -> str:
    """本地 Ollama（/api/chat）。用于完全离线演示，模型由 OLLAMA_MODEL 指定。"""
    payload = json.dumps({
        "model": oc["model"], "messages": messages, "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(oc["base"] + "/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(timeout, 120)) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        raise RuntimeError("Ollama: " + str(e)[:150])
    return ((data.get("message") or {}).get("content") or "").strip()
