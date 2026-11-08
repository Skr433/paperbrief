# PaperBrief 部署说明（10 分钟上线）

> 适用：腾讯云 / 阿里云轻量应用服务器，**2 核 2G、Ubuntu 22.04**，无需显卡。

## 一、买服务器（10 分钟）

1. 打开腾讯云「轻量应用服务器」或阿里云「轻量应用服务器」。
2. 配置：**2 核 2G、系统镜像 Ubuntu 22.04 LTS、带宽 3–5M**；学生认证更便宜（约 10–15 元/月）。
3. 购买后进入控制台，记下：**公网 IP**、**root 密码**（可重置一次）。
4. **放通防火墙**：轻量服务器 → 防火墙 → 添加规则 → 允许 **TCP 80**（默认常已放通，确认一下）。

## 二、把部署包传到服务器（2 分钟）

在你自己的 Windows 电脑上打开 PowerShell，执行（把 IP 换成你的）：

```powershell
cd "C:\Users\ASUS\Desktop\paperbrief"
scp paperbrief_deploy.zip ubuntu@服务器IP:/home/ubuntu/
```

> 第一次连接会问 `yes/no`，输入 `yes`；然后输入 root 密码（不显示，直接粘贴后回车）。
> 若提示 `scp` 不是命令：Windows 设置 → 应用 → 可选功能 → 安装「OpenSSH 客户端」。

## 三、登录服务器并执行部署（5 分钟）

方法 A（推荐）：腾讯云/阿里云控制台 → 服务器详情 → **登录（网页终端）**，输入 root 密码。

方法 B：本机 PowerShell 执行 `ssh ubuntu@服务器IP`（腾讯云 Ubuntu 镜像默认用户是 ubuntu，不是 root）。

登录后依次粘贴：

```bash
apt-get install -y unzip
cd /home/ubuntu
unzip -o paperbrief_deploy.zip
cd paperbrief_deploy
sudo bash deploy.sh
```

看到最后打印 `部署完成` 和 `{"ok": true...}` 就成功了。

## 四、验证（2 分钟）

浏览器打开：

```
http://服务器IP/
```

依次点：
1. 下拉选 `1706.03762v7.pdf` → 点「解析」→ 左侧出现论文页面、右侧出现统计；
2. 点「生成汇报包」→ 右侧出现摘要/贡献/图表；
3. 点任意 `p.7` → 左侧跳页高亮；
4. 切「问答」→ 问「第 6 作者是谁」→ 出答案 + 锚点。

## 五、常见问题

| 现象 | 原因 / 解决 |
| --- | --- |
| 浏览器打不开 | ① 安全组/防火墙没放通 80；② 服务没起来：`systemctl status paperbrief` 看报错 |
| 页面能开但解析报错 | `journalctl -u paperbrief -n 50 --no-pager` 查看；多为 PDF 目录未配置，检查 `.env` 的 `PAPERBRIEF_PDF_DIR` |
| 上传 PDF 失败 | Nginx 已设 60MB；若仍失败看 `client_max_body_size` |
| 中文乱码 | 已设 `PYTHONIOENCODING=utf-8`，重启服务：`systemctl restart paperbrief` |
| 改完代码不生效 | `systemctl restart paperbrief` |

## 六、后续更新（11/3 接大模型后）

```bash
# 本机
scp paperbrief_deploy.zip ubuntu@服务器IP:/home/ubuntu/
# 服务器
cd /home/ubuntu && unzip -o paperbrief_deploy.zip && bash paperbrief_deploy/deploy.sh
```

只更新代码、保留数据与 `.env`（脚本不会覆盖 `.env`，也不会删除 `data/`）。
