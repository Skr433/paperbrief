# -*- coding: utf-8 -*-
"""环境自检：评审或本机启动前先跑这个，给出明确的 PASS / FAIL，而不是让服务闪退。

用法：python scripts/check_env.py
退出码：0 全部通过；1 有阻断项
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ok = True


def line(flag, label, detail=""):
    global ok
    if not flag:
        ok = False
    print(("[PASS] " if flag else "[FAIL] ") + label + ("  " + detail if detail else ""))


print("PaperBrief 环境自检")
print("-" * 46)

# 1. Python 版本
v = sys.version_info
line(v >= (3, 9), "Python 版本", "%d.%d.%d（需要 3.9+）" % (v[0], v[1], v[2]))

# 2. PyMuPDF
try:
    import pymupdf  # noqa: F401

    line(True, "PyMuPDF", getattr(pymupdf, "__doc__", "").splitlines()[0][:40])
except Exception:
    line(False, "PyMuPDF", "未安装，请执行  pip install -r requirements.txt")

# 3. 官方测试集
pdf_dir = os.environ.get("PAPERBRIEF_PDF_DIR", "")
candidates = [pdf_dir] if pdf_dir else []
candidates += [
    os.path.join(ROOT, "..", "ref_kxyzs", "pdf"),
    os.path.join(ROOT, "..", "..", "03_pdf"),
    os.path.join(ROOT, "..", "pdf"),
]
found = ""
for c in candidates:
    if c and os.path.isdir(c) and any(f.lower().endswith(".pdf") for f in os.listdir(c)):
        found = os.path.abspath(c)
        break
if found:
    n = len([f for f in os.listdir(found) if f.lower().endswith(".pdf")])
    line(True, "PDF 测试集", "%d 篇  %s" % (n, found))
else:
    line(False, "PDF 测试集", "未找到，请设置环境变量 PAPERBRIEF_PDF_DIR 指向 pdf 目录")

# 4. 前端文件
line(os.path.isfile(os.path.join(ROOT, "frontend", "index.html")), "前端文件", "frontend/index.html")

# 5. 写权限（缓存与页面渲染）
try:
    d = os.path.join(ROOT, "data", "parsed")
    os.makedirs(d, exist_ok=True)
    probe = os.path.join(d, ".write_test")
    open(probe, "w").close()
    os.remove(probe)
    line(True, "数据目录可写", "data/")
except Exception as e:
    line(False, "数据目录可写", str(e)[:60])

print("-" * 46)
if ok:
    print("全部通过，可以启动：python backend/server.py")
    sys.exit(0)
print("存在阻断项，请按上面提示修复后重试。")
sys.exit(1)
