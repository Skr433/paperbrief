# parser 0.2.9 参考文献抽取验证

> 验证对象：`E:\deepseek\对话\paperbrief\backend\pdf_parse.py`（`PARSER_VERSION = "0.2.9"`）。
> 验证快照：2026-11-08 16:43:36，24,946 B（含 `_refs_by_layout`）。
> 运行方式：只读调用 `pdf_parse.parse_pdf()`，Python 3.12 + PyMuPDF 2.0，未改代码、未重启服务。
> 样本：`E:\deepseek\对话\官方测试集_1108\pdf\` 下 8 篇；接口 `references` 最多返回 300 条，`stats.n_references` 为全量计数。

**代码快照备注**：验证期间 16:48:08–16:48:40，仓库里的 layout 代码被外部拆到 `refs_layout.py`（`pdf_parse.py` 变为 20,398 B，`PARSER_VERSION` 仍为 0.2.9）。已逐行比对，拆分前后逻辑一致；复跑 T5 / UDL / 209 / 2406 计数完全一致（133 / 897 / 118 / 133），故本报告对两份快照均成立。但版本号未变、代码在验证中变动，发布前应冻结并升版本。

---

## 一、结果表

| # | 论文 | 条数（stats / 返回） | 前 / 后 2 条判定 | bbox（抽 3 条） | 耗时 | 问题 |
|---|------|--------------------|------------------|-----------------|------|------|
| 1 | 1910.10683v4（T5） | **133 / 133** | 前 2 完整；后 2 完整 | 3/3 通过 | 2.04 s | 18 处断词（"En- hanced"）；[40] 缺作者/年份、[121] 仅 URL |
| 2 | UnderstandingDeepLearning_02_09_26_C（UDL） | **897 / 300（上限）** | 前 2 丢词（"age2StyleGAN"）；后 2 丢词（"with adaptive…"） | 3/3 通过 | **33.74 s** | 远低于 1281；300 条中 43 条合并、84 条丢词；断词/连字未还原 |
| 3 | 1706.03762v7（Attention） | **40 / 40** | 前 2 完整；后 2 完整 | 3/3 通过 | 0.99 s | 编号 1–40 完整；bbox 只覆盖首行 |
| 4 | 2005.14165v4（GPT-3） | **191 / 191** | 前 1 是上一条的出版信息、前 2 只有首行；后 1/后 2 重复 | 3/3 通过 | 10.36 s | 真 [key] 文献 144 条；50 条无编号碎片；缺前 3 条；末尾重复 |
| 5 | 2103.00020v1（CLIP） | **300 / 300（上限）** | 前 2 是正文段落（related work / 结论致谢）；后 1 截断于 "Ruyssen, P.," | 3/3 通过 | 2.62 s | 正文混入；58 条以逗号结尾的断行碎片；实际完整条目约 236 |
| 6 | 2303.08774v6（GPT-4） | **219 / 219** | 前 2 完整；后 2 是附录代码安全建议 | 3/3 通过 | 1.84 s | 28 条附录正文混入；主文 85 + 系统卡 105 + 28 杂项 |
| 7 | 209_Emergent_Abilities_of_Larg | **118 / 118** | 前 2 完整；后 2 是附录图表/正文 | 3/3 通过 | 0.85 s | 15 条来自 p19–28 附录；真文献约 100–103 条 |
| 8 | 2406.14491v2 | **133 / 133** | 前 2 完整；后 2 是附录影评正文 | 3/3 通过 | 0.72 s | 46 条来自 p15–22 附录；真文献约 87 条；[2]/[86] 跨条合并 |

- 8 篇页面均为 **612 × 792**；共 1,434 条返回记录，**0 条 bbox 越界**（x1>x0、y1>y0、均在页面内）。
- 条数与 v0.2.8 对比：T5 133、UDL 897、Attention 40、GPT-3 191、CLIP 300、GPT-4 219 全部持平，**无计数回归**；209 由 5 → 118、2406 由 7 → 133，但增量中含附录正文（见下）。

---

## 二、逐篇样本与判定

### 1. 1910.10683v4（T5）— 达标
- 前 2：`Rami Al-Rfou, …, Character-level language modeling with deeper self-attention. AAAI, 2019.` / `Rohan Anil, …, Memory-efficient adaptive optimization… arXiv:1901.11150, 2019.`
- 后 2：`Chen Zhu, …, Freelb: En-hanced adversarial training… arXiv:1909.11764, 2019.` / `Yukun Zhu, …, Aligning books and movies… IEEE ICCV, 2015.`
- 判定：作者、标题、出处、年份齐全；仅 [40] 是 "Generating sequences with recurrent neural networks. arXiv preprint"（缺作者/年份/编号）、[121] 只有 URL，属于 2 处碎片。
- bbox 例：p58 `[90.0, 113.84, 522.15, 124.75]`、p60 `[90.0, 385.46, 523.72, 396.37]`、p67 `[90.0, 548.59, 522.19, 559.49]`，均在页内。

### 2. UDL — 不达标（数量 + 文本双重问题）
- 前 2：`Abdal, R., Qin, Y., & Wonka, P. (2019). age2StyleGAN: …`（原文是 **Image2StyleGAN**，`Im-` 被丢）/ `… (2020). age2StyleGAN++: …`
- 后 2：`Harvey, W., … Flexible diffusion modeling of long videos. NeurIPS, 35. 370` / `Hasanzadeh, A., … (2020). with adaptive connection sampling. ICML, 4094–4104. 265`（标题首词 "Stochastic…" 被丢）
- bbox 例：p477 `[94.91, 264.73, 259.26, 272.70]`、p492 `[329.53, 390.76, 515.34, 398.73]`，均合法但只覆盖首行（高约 8 pt，文本 200+ 字符）。
- 量化（返回的 300 条）：
  - **43 条含 ≥2 个 `(YYYY).`**，即两条文献被合成一条（如 [22] "Alvarez… 416 Amari, S.-I. (1998)…"、[12] "Agüera y Arcas… 432 Ahmed, N.…"）；
  - **84 条年份后直接小写**，首词/首段被丢（"age2StyleGAN"、"hoodie project" 缺 "The grey"、"ficiently in learning" 缺 "Ef-"）；
  - 2 条可见跨栏 URL 拼接（[7] "Im- Im- The grey Acoustic"、[180] "Distill… Awful AI… B., Mané,"）。
- 根因：`_refs_by_layout` 每页用全页 x0 聚类边距，右对齐的连字符碎片（如 x0≈267.8）会被当成新栏起点；`flush()` 又丢弃 <30 字符的碎片，于是"整词/整条"消失。双栏 + 两端对齐的 UDL 正好踩中。
- 页眉/页脚/索引：`Draft: please send errata`、`Bibliography`、`Xavier initialization, 113` 均未进入 refs（过滤有效）。条目末尾 "302 / 301, 323" 是 UDL 自己的"引用页码"体例，不是页脚混入。

### 3. 1706.03762v7（Attention）— 达标
- 前 2：`[1] Jimmy Lei Ba… Layer normalization. arXiv:1607.06450, 2016.` / `[2] Dzmitry Bahdanau… Neural machine translation… CoRR, 2014.`
- 后 2：`[39] Jie Zhou… Deep recurrent models… CoRR, 2016.` / `[40] Muhua Zhu… Fast and accurate shift-reduce constituent parsing. ACL, 2013.`
- 判定：40 条编号 [1]–[40] 唯一、无重复；前/后样本完整。bbox 合法，但 p10 `[112.98, 615.04, 504.0, 625.12]` 高 10 pt，而文本已 2 行，锚点只覆盖首行。

### 4. 2005.14165v4（GPT-3）— 不达标（碎片 + 缺条 + 重复）
- 前 2：`Proceedings of the 2019 Conference…, 2019.`（这是 [AJF19] 的出版信息续行，不是独立文献）/ `[BBDIW20] Su Lin Blodgett… Language (technology) is power:`（只有首行，后半在 [2]）
- 后 2：`Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2019.`（无编号续行）/ `[ZSW+19b] Daniel M. Ziegler… Fine-tuning… ArXiv, abs/1909.08593, 2019.`（与前一条是同一篇）
- 独立计数：p68–75 的 `[ADG+16]` 风格 key 共 **144 个、全不重复**；parser 只覆盖 141 个，**缺前 3 条**（[ADG+16]、[AI19]、[AJF19] 的正文），另有 **50 条无 key 的续行碎片**，所以 191 = 141 + 50，不是真实文献数。
- 根因：`References` 标题未被主路径识别，走 `_fallback_refs`；其 `started` 由"含 et al./Proceedings"的块触发，从第 4 条中间开始；之后按 block 切分，每个断行块都成为新条目。

### 5. 2103.00020v1（CLIP）— 不达标（前段正文 + 碎片）
- 前 2：`& Bansal, 2019; Chen et al., 2019; … This line of work focuses on richly connecting vision and language…`（正文 Related Work）/ `We have investigated whether it is possible to transfer the success of task-agnostic web-scale pre-training in NLP…`（结论段）。
- 后 2：`Zeiler, M. D. and Fergus, R. Visualizing… Springer, 2014.`（完整）/ `Zhai, X., Puigcerver, J., Kolesnikov, A., Ruyssen, P.,`（截断，缺后半）。
- 判定：300 条是接口上限而非真实条数；首 3 条是 p27 左栏正文，[3] 才是第一条文献；58 条以逗号结尾、202 条含换行（首行被单独成条，如 [3] 只到 "Dean,"、[4] 是 "J., Devin…"）。按续行合并估算约 **236 条**完整文献，说明 300 里约 60+ 是碎片/正文。
- 根因：fallback 在 p27 找到 `References`（右栏 y≈67.8）后，没有按列限制，左栏正文整段进入；`started` 又被正文中的 "et al." 触发。

### 6. 2303.08774v6（GPT-4）— 不达标（附录/系统卡混入）
- 前 2：`[1] Tom Brown… Language models are few-shot learners…` / `[2] Jordan Hoffmann… Training compute-optimal large language models…`（完整）
- 后 2：`4. Error handling: The code does not check for errors…` / `5. Lack of HTTPS: The server is listening on HTTP…`（p97 附录代码安全建议，不是文献）
- 结构：p18–23 主文文献 **85 条**；p71–78 系统卡自己的 "References" **105 条**（编号又从 [1] 开始，与主文重复）；其余 p34/37/47/50/54/61/64/66/67/68/96/97 混入 **28 条**附录正文（如 p47 "1. Advice or encouragement for self harm behaviors"、p61 "2. Model Mitigations"）。85+105+28+1 条跨行碎片 = 219。
- 根因：layout 抽取从第一个 `References`（p18）一直跑到文末，系统卡第二节 `References` 未被视为新文档边界；附录页首又没有 "appendix" 字样可触发 break。

### 7. 209_Emergent_Abilities_of_Larg — 部分恢复，但仍不达标
- 前 2：`Omri Abend… Bootstrapping language acquisition.` / `Michael Ahn… Do as I can, not as I say… arXiv:2204.01691, 2022.`（完整）
- 后 2：`state-of-the-art performance held by a task-specific model…` / `(2022). D: Alayrac et al. (2022). (A) Math word problems…`（p28 附录图表文字）
- 结构：p12–18 是真文献（约 100–103 条，含 [93]/[94] 一条 URL 被拆成两段）；p19–28 是附录 A–E 的正文/表格，**15 条**被当文献（[103] "et al., 2002) as the evaluation metric…"、[109] "Table 2: Parameters…"、[113] 图表坐标文字）。[102] 末尾还混入 "A BIG-Bench analysis A.1 Cross-entro…"。
- 相比原 5 条是大幅恢复，但 118 不是干净计数。

### 8. 2406.14491v2 — 部分恢复，但仍不达标
- 前 2：`Vaibhav Adlakha… 2022. Topiocqa…` / `Shourya Aggarwal… 2021. Explanations for commonsenseqa…`（完整）
- 后 2：`million during its opening 2010. "'Lightyear' had a great deal of potential…"` / `release of a Pixar film since 2020's "Onward."…`（p22 附录影评正文）
- 结构：p9–14 是真文献（约 87 条）；p15–22 是附录 A–B（Instruction Synthesizer 模板、评测表、影评样本），**46 条**被当文献。[2] 把 "Alon Albalak…" 与 "Rich Caruana. 1997…" 合成一条；[86] 把 "Lima: Less is more…" 与附录 "A Data Collection…" 合成一条（784 字符）。
- 相比原 7 条是恢复，但 133 同样不干净。

---

## 三、发现的问题（按严重度）

### P0 — 会直接展示错误内容

1. **参考文献区间终点失控（4 篇）**
   - 209：15/118 条来自附录 p19–28；
   - 2406：46/133 条来自附录 p15–22；
   - GPT-4：28/219 条来自附录正文，且系统卡 p71–78 的 105 条与主文编号重复；
   - CLIP：前 3 条（p27 左栏）是正文，不是文献。
   - 共同原因：`_refs_by_layout` 只在页首 6 行出现 "appendix/附录" 时停止；TMLR/ACL 附录用 "A …" 标题、GPT-4 系统卡有独立 `References`，都绕过了该规则。fallback 则完全没有"标题之后/列边界"的限制。

2. **UDL 文本损坏（丢词 + 合并 + 漏条）**
   - 返回 300 条中 43 条"两条并一条"、84 条"年份后直接小写"（首词被丢）；"Image2StyleGAN" → "age2StyleGAN"、"Stochastic…" → "with adaptive…"。
   - `n_references=897`，比目标 1281 少约 30%；即使 1281 的口径有争议，300 条样本的丢词/合并已足以判定不达标。
   - 双栏边距 + 右对齐连字符碎片是主因；layout 路径未做 `norm_space`，`eﬀicacy` 等连字（9 处）、`Interna- tional` 断词原样输出。

3. **CLIP 前段正文混入 + 严重碎片化**
   - 用户看到的"参考文献"前两条是 Related Work / 结论段落；58 条以逗号结尾的断行碎片；300 是上限，不是真实条数。

### P1 — 计数或锚点不可信

4. **GPT-3 计数虚高、缺条、重复**
   - 真实带 key 文献 144 条，parser 输出 191 条（50 条续行碎片），且缺开头 3 条、末尾 1 组重复（[189]/[190]）。"191 与基准一致"只是同一个旧 bug 的延续，不代表质量达标。

5. **bbox 合法但只覆盖首行**
   - 全量 1,434 条 0 越界；但 layout 路径的 bbox 只取条目首行（高 8–11 pt），而文本常为 2–5 行：
     UDL 300/300、2406 109/133、209 92/118、GPT-4 91/219、T5 71/133、Attention 27/40 的"文本长度 / bbox 面积"超过 1.5 倍。
   - 点回原文时只高亮一行，虽不违反"页内"检查，但削弱"每条结论可定位到完整条目"的体验。

6. **性能**
   - UDL **33.6 s**（541 页，复测 33.56 s），GPT-3 10.4 s；T5 2.0 s、CLIP 2.6 s、GPT-4 1.8 s、其余 ≤1.1 s。
   - `parse_all_v029.json` 记录的 UDL 0.80 s、GPT-3 0.74 s 与实测差 40 倍（v0.2.8 同文件记录 UDL 34.82 s），该计时不可用于容量评估；若服务端同步解析，UDL 会阻塞 30 s+。

7. **代码在验证期间变动**
   - 16:48 拆分 `refs_layout.py`，`PARSER_VERSION` 未变；本报告已确认行为一致并复跑，但发布前应冻结并 bump 版本（建议 0.2.10），否则无法回答"线上到底是哪版"。

### P2 — 轻微/局部问题

8. **断词与碎片**
   - T5：18 处 "En- hanced" / "expla- nations"；[40]、[121] 碎片；8 条无年份。
   - 209：[93]/[94] 同一条 URL 被拆成两条；[102] 末尾混入附录标题。
   - 2406：[2]、[86] 跨条合并。
   - Attention：唯一问题是 bbox 单行（条目本身完整）。

9. **通过项（应保留）**
   - 页眉页脚/索引过滤有效：UDL 300 条中 0 条 `Draft: please send errata`、0 条 `Bibliography`、0 条 `Xavier initialization, 113` 等索引行；209 无 `Published in TMLR` 页眉。
   - 全量 bbox 0 越界；页码与条目所在页一致（抽查 24 条，page 均正确）。

---

## 四、结论

**是否可发布为 v0.14.8：不建议按"参考文献抽取达标"发布。**

- **达标 2/8**：T5（133/133，样本完整）、Attention（40/40，编号完整）。
- **不达标 6/8**：
  - **UDL**：897/1281（-30%），300 条样本 43 条合并、84 条丢词，33.6 s —— 最严重；
  - **CLIP**：前 2 条是正文，58 条断行碎片，300 只是上限；
  - **GPT-3**：191 条中 50 条碎片，真文献 144 条，缺前 3 条且末尾重复；
  - **GPT-4**：219 条含 28 条附录正文 + 105 条系统卡文献（编号重复）；
  - **209**：118 条中 15 条附录正文（真文献约 100–103）；
  - **2406**：133 条中 46 条附录正文（真文献约 87）。
- **无计数回归**：T5、UDL、Attention、GPT-3、CLIP、GPT-4 与 v0.2.8 完全持平；209/2406 从 5/7 提升到 118/133，恢复方向正确，但"恢复"里混入了附录内容，不能算干净通过。

**发布建议**：
1. 若 v0.14.8 还包含其他必须上线的功能，可以作为**过渡版本**发布，但参考文献能力应在发布说明标为 beta，并明确"UDL/CLIP/GPT-3/GPT-4/209/2406 的引用列表可能含附录或碎片"。
2. 若 v0.14.8 的卖点包含"参考文献抽取质量"，则应**阻塞发布**，先修 P0：
   - 以 `References` 标题 + 下一个一级标题（`Appendix` / `A …` / `Index` / 第二个 `References`）为区间边界；
   - layout 路径按"每栏独立边距 + 连字符还原 + 短碎片合并"处理，并对 UDL 做回归（目标 1281）；
   - fallback 从 `References` 标题所在列开始，禁止把正文段落收进来；
   - 统计口径与返回上限分离（`stats.n_references` 全量、`references` 300 上限保持现状即可，但需在 UI 说明"仅展示前 300 条"）。
3. 修复后至少重跑本 8 篇 + `regression.py`，确认 UDL ≥ 1281、CLIP 首条为 `Abadi`、GPT-4 末尾为 `[85]`、209/2406 末尾停在文献区。
