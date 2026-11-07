# -*- coding: utf-8 -*-
"""arXiv 追踪与推荐（命题 1.4）。

能力：
- 按当前论文标题抽取关键词，检索 arXiv 最新相关论文（按提交日期倒序）
- 自定义关键词搜索
- 标注「有代码」（摘要含 github / code available / open-source）
- 只依赖标准库（urllib + xml.etree）；SSL 失败时自动降级为不校验证书
"""

from __future__ import annotations

import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

API = "https://export.arxiv.org/api/query"
CROSSREF = "https://api.crossref.org"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

_STOP = set("""the a an of and for to in on with by from at as is are was were be been using based via its
their this that these those we our you your they it can may will would should could not no than then
into over under between among across per each such more most less least very much many few also however
paper papers approach approaches method methods model models result results show shows study studies
new novel proposed propose present presents describe describes need needs towards toward using based
large small better improve improves performs performance first second third one two three state art""".split())

CODE_HINTS = ("github.com", "gitlab.com", "code is available", "code available", "open-source",
              "open source", "we release", "source code")


def _open(url: str, timeout: int = 18):
    """arXiv 从国内机房不稳定：两次尝试 + 明确错误信息。"""
    import time as _time
    req = urllib.request.Request(url, headers={"User-Agent": "paperbrief/0.12 (mailto:dev@example.com)"})
    last = None
    for attempt in range(2):
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except Exception as e:
            msg = str(e)
            reason = getattr(e, "reason", None)
            if "CERTIFICATE_VERIFY_FAILED" in msg or isinstance(reason, ssl.SSLError):
                try:
                    ctx = ssl._create_unverified_context()
                    return urllib.request.urlopen(req, timeout=timeout, context=ctx)
                except Exception as e2:
                    last = e2
            else:
                last = e
            if attempt == 0:
                _time.sleep(1.5)
            else:
                raise RuntimeError("arXiv 网络超时") from last


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


BAD_KEYWORDS = {"paper", "papers", "study", "method", "methods", "approach", "approaches",
                "model", "models", "using", "based", "towards", "toward", "via", "analysis",
                "new", "novel", "results", "result", "work", "works", "deep"}


def keywords_from_title(title: str, limit: int = 4) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", (title or "").lower())
    out, seen = [], set()
    for w in words:
        if w in _STOP or w in BAD_KEYWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= limit:
            break
    return out




_VENUE_RE = re.compile(r"(accepted|to appear|proceedings|conference|workshop|camera[- ]ready|published in|oral|poster)", re.I)


def detect_venue(comment: str, journal_ref: str) -> tuple[str, str]:
    """从 comment / journal_ref 判断出版状态：返回 (出处名称, '期刊'|'会议'|'')。"""
    if journal_ref:
        return journal_ref, "期刊"
    if comment and _VENUE_RE.search(comment):
        return comment, "会议"
    return "", ""


_FIELD_MAP = {"cs.CL": "NLP", "cs.LG": "机器学习", "cs.CV": "视觉", "cs.AI": "AI",
              "cs.IR": "检索", "cs.NE": "神经计算", "stat.ML": "统计学习"}


def field_of(category: str) -> str:
    return _FIELD_MAP.get(category, category.split(".")[0] if category else "")




def _crossref_get(url: str, timeout: int = 12) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "paperbrief/0.10 (mailto:dev@example.com)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return {}


def _venue_from_crossref(title: str) -> tuple[str, str]:
    """用 Crossref 查论文的出版状态：返回 (venue 名称, '期刊'|'会议'|'')。"""
    import difflib
    q = urllib.parse.quote(_clean(title)[:180])
    data = _crossref_get(f"{CROSSREF}/works?query.bibliographic={q}&rows=3"
                         f"&select=DOI,title,type,container-title,event")
    items = data.get("message", {}).get("items", [])
    best, best_ratio = None, 0.0
    want = _clean(title).lower()
    for it in items:
        cand = _clean((it.get("title") or [""])[0]).lower()
        if not cand:
            continue
        ratio = difflib.SequenceMatcher(None, want, cand).ratio()
        if ratio > best_ratio:
            best, best_ratio = it, ratio
    if not best or best_ratio < 0.86:
        return "", ""
    container = _clean((best.get("container-title") or [""])[0])
    etype = (best.get("type") or "").lower()
    if etype == "journal-article" or (container and "journal" in container.lower()):
        return container, "期刊"
    if etype == "proceedings-article" or (container and ("proceedings" in container.lower()
                                                          or "conference" in container.lower())):
        return container, "会议"
    if container:
        return container, "期刊" if etype != "posted-content" else ""
    return "", ""


def enrich_venues(items: list[dict], max_items: int = 8) -> list[dict]:
    """并发用 Crossref 补出版状态（arXiv 的 comment/journal_ref 大多为空）。"""
    import concurrent.futures
    if not items:
        return items
    todo = [it for it in items[:max_items] if not it.get("venue_type")]
    if not todo:
        return items

    def one(it):
        try:
            venue, vtype = _venue_from_crossref(it.get("title", ""))
            if vtype:
                it["venue"] = venue[:80]
                it["venue_type"] = vtype
        except Exception:
            pass
        return it

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(one, todo))
    return items


def _parse(xml_bytes: bytes, max_items: int) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    items = []
    for entry in root.findall(ATOM + "entry")[:max_items]:
        title = _clean(entry.findtext(ATOM + "title") or "")
        summary = _clean(entry.findtext(ATOM + "summary") or "")
        published = (entry.findtext(ATOM + "published") or "")[:10]
        link = ""
        for l in entry.findall(ATOM + "link"):
            if l.get("rel") == "alternate" or l.get("type") == "text/html":
                link = l.get("href") or ""
        if not link:
            link = (entry.findtext(ATOM + "id") or "").strip()
        authors = [a.findtext(ATOM + "name") or "" for a in entry.findall(ATOM + "author")][:3]
        low = summary.lower()
        comment = _clean(entry.findtext(ARXIV_NS + "comment") or "")
        journal_ref = _clean(entry.findtext(ARXIV_NS + "journal_ref") or "")
        cat = entry.find(ARXIV_NS + "primary_category")
        category = cat.get("term") if cat is not None else ""
        venue, venue_type = detect_venue(comment, journal_ref)
        items.append({
            "title": title, "summary": summary[:400], "published": published,
            "authors": authors, "link": link,
            "has_code": any(h in low for h in CODE_HINTS),
            "arxiv_id": (entry.findtext(ATOM + "id") or "").rsplit("/", 1)[-1],
            "comment": comment[:160], "journal_ref": journal_ref[:160],
            "category": category, "field": field_of(category),
            "venue": venue, "venue_type": venue_type,
            "is_survey": bool(re.search(r"\b(survey|review)\b", title, re.I)),
        })
    return items


def search(query: str, limit: int = 8) -> list[dict]:
    """自定义关键词搜索（AND 连接），按相关性。"""
    q = _clean(query)
    if not q:
        return []
    words = [w for w in re.split(r"[\s,，]+", q) if w]
    expr = "+AND+".join(f"all:{urllib.parse.quote(w)}" for w in words[:6])
    url = f"{API}?search_query={expr}&start=0&max_results={limit}&sortBy=relevance"
    with _open(url) as r:
        return _parse(r.read(), limit)


def recommend(title: str, limit: int = 8, extra_keywords: list[str] | None = None) -> list[dict]:
    """基于论文标题关键词（或外部给定关键词），找最近提交的相关论文。"""
    kws = [k for k in (extra_keywords or []) if k and k.lower() not in BAD_KEYWORDS]
    if not kws:
        kws = keywords_from_title(title)
    if not kws:
        return []
    kws = list(dict.fromkeys(kws))[:4]
    if len(kws) <= 2:
        phrase = urllib.parse.quote(" ".join(kws))
        expr = f'all:"{phrase}"'
    else:
        expr = "+AND+".join(f"all:{urllib.parse.quote(w)}" for w in kws)
    url = f"{API}?search_query={expr}&start=0&max_results={limit * 3}&sortBy=submittedDate&sortOrder=descending"
    with _open(url) as r:
        items = _parse(r.read(), limit * 3)
    # 去重 + 只保留近 18 个月
    seen, out = set(), []
    now_year = time.localtime().tm_year
    for it in items:
        key = it["arxiv_id"].split("v")[0]
        if key in seen:
            continue
        year = int(it["published"][:4]) if it["published"][:4].isdigit() else now_year
        if year < now_year - 1:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= limit:
            break
    return out
