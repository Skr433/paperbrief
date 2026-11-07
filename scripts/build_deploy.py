# -*- coding: utf-8 -*-
"""打包部署 zip（唯一来源，替代之前的一次性 _build_*.py）。

用法（在项目根目录）：
    python scripts/build_deploy.py
输出：项目同级 paperbrief_deploy.zip + 桌面 paperbrief/paperbrief_deploy.zip

约定：
- `*.sh` 强制 LF，并在 zip 内校验无 CR（防 bash 报 $'\\r'）
- 排除 __pycache__、data/、docs/、测试集 PDF（PDF 单独放 pdf/ 目录）
- 打包前对所有 backend/scripts 做 py_compile
"""

from __future__ import annotations

import os
import py_compile
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(ROOT), "paperbrief_deploy.zip")
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop", "paperbrief", "paperbrief_deploy.zip")
PDF_SRC = os.path.join(os.path.dirname(ROOT), "ref_kxyzs", "pdf")
PDFS = ["1706.03762v7.pdf", "1706.03762v1.pdf", "1706.03762v1_img.pdf", "N19-1423.pdf",
        "2106.09685v2.pdf", "2303.08774v6.pdf", "2005.14165v4.pdf", "language_understanding_paper.pdf"]


def main() -> int:
    for d in ("backend", "scripts"):
        for f in os.listdir(os.path.join(ROOT, d)):
            if f.endswith(".py"):
                py_compile.compile(os.path.join(ROOT, d, f), doraise=True)
    print("syntax OK")

    deploy = os.path.join(ROOT, "deploy.sh")
    raw = open(deploy, "rb").read()
    norm = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if norm != raw:
        open(deploy, "wb").write(norm)
        print("deploy.sh CRLF -> LF")

    files = []
    for d in ("backend", "frontend", "scripts"):
        for r, _, fs in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in r or os.sep + "data" in r:
                continue
            for f in fs:
                full = os.path.join(r, f)
                files.append((full, os.path.relpath(full, ROOT).replace("\\", "/")))
    for f in ("requirements.txt", "README.md", "AGENTS.md", "deploy.sh", "部署说明.md"):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            files.append((p, f))
    for f in PDFS:
        p = os.path.join(PDF_SRC, f)
        if os.path.exists(p):
            files.append((p, "pdf/" + f))

    if os.path.exists(OUT):
        os.remove(OUT)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for full, rel in files:
            arc = "paperbrief_deploy/" + rel
            if rel.endswith(".sh"):
                z.writestr(arc, open(full, "rb").read().replace(b"\r\n", b"\n"))
            else:
                z.write(full, arc)
    with zipfile.ZipFile(OUT) as z:
        assert b"\r" not in z.read("paperbrief_deploy/deploy.sh"), "zip 内 deploy.sh 含 CR"
    print("zip: %.2f MB, %d files -> %s" % (os.path.getsize(OUT) / 1024 / 1024, len(files), OUT))
    if os.path.isdir(os.path.dirname(DESKTOP)):
        shutil.copy2(OUT, DESKTOP)
        print("copied to Desktop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
