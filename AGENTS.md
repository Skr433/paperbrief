# PaperBrief · Agent 指南

论文汇报包助手（第九届传智杯 AI 大模型创新应用挑战赛，赛道 10040）。
核心承诺：**每条结论都能点回原文页码**，回答失败自动降级，不编造。

## 目录

- `backend/`：server.py（HTTP 路由）、pdf_parse.py（解析）、retrieval.py（BM25+锚点）、brief.py（汇报包）、llm.py（DeepSeek/Ollama 双后端）、graph.py（引文图谱）、chart.py（CSV→SVG）、arxiv.py（arXiv 推荐）
- `frontend/index.html`：单文件前端（无构建）
- `scripts/`：smoke_test.py、http_smoke.py、regression.py（9 题）、evaluate.py（解析+检索命中率+延迟）
- `deploy.sh`：Ubuntu 一键部署（systemd + Nginx）；`部署说明.md`
- 资料与日志在桌面项目 `paperbrief/`（04_资料、05_日志），不在本仓库

## 常用命令

```bash
# 本地
python backend/server.py            # 默认 8000，Windows 可用 start.bat
python scripts/smoke_test.py        # 本地冒烟
python scripts/regression.py --base http://1.14.98.248   # 9 题回归
python scripts/evaluate.py --base http://1.14.98.248     # 评测报告
```

部署：本机 `scp paperbrief_deploy.zip ubuntu@<IP>:/home/ubuntu/`，服务器
`sudo rm -rf paperbrief_deploy && unzip -o paperbrief_deploy.zip && cd paperbrief_deploy && sudo bash deploy.sh`

## 硬规则

1. 改完 Python 先 `py_compile`；部署包内 `*.sh` 必须 LF（CRLF 会让 bash 崩）。
2. 版本号只在 `frontend/index.html` 的 `vX.Y.Z`；部署后页面版本要与预期一致。
3. 改功能后跑 `regression.py` + `evaluate.py`，结果写进 `05_日志/`。
4. 谈要求/评分/截止前，先读桌面 `04_资料/比赛要求与评分速查卡.md`；评分注明官网版/需求文档版，不混用。
5. 不新增第三方依赖（已批准的 fastembed spike 除外；服务器 2G 内存，禁止 torch/matplotlib）。
6. 提交被 `.githooks/pre-commit` 拒绝：说明文件 >20KB，按功能拆分后再提交。
7. 写新函数/新规则前，先搜仓库里有没有现成实现（同一条规则只放一处）。

## 已知坑

- `deploy.sh` 出现 `invalid option name`/`$'\r'`：换行被转成 CRLF，重打包强制 LF。
- 服务器 `rm -rf paperbrief_deploy` 报 Permission denied：`__pycache__` 是 root 属主，先 `sudo rm -rf`。
- pgrep 服务挂：看 `journalctl -u paperbrief -n 50 --no-pager`。
- OpenAlex / Semantic Scholar 对机房 IP 常 429；引文数据走 Crossref。
- 官方测试集缺 GPT-2 原论文：GPT 多篇题只能部分回答，属数据限制，文档如实写。

## 材料位置

- 评分速查卡 / 改进清单 / 评测报告 / 回归表 / 计划：桌面 `paperbrief/04_资料/`
- 每日日志：桌面 `paperbrief/05_日志/`
- 线上环境：`/opt/paperbrief`（`.env` 含 DEEPSEEK_API_KEY，勿入库）
