#!/usr/bin/env bash
# PaperBrief 部署脚本 v2（Ubuntu 22.04/24.04，可重复执行）
# 用法：sudo bash deploy.sh
set -uo pipefail

APP_DIR=/opt/paperbrief
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT=8000
STEP="init"

trap 'echo; echo "!!!! 部署失败，出错步骤：$STEP（第 $LINENO 行）"; echo "请把上面 10 行错误信息发出来"; exit 1' ERR

step() { STEP="$1"; echo; echo "==> $1"; }

step "[1/8] 安装系统依赖"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y >/tmp/apt-update.log 2>&1 || echo "[warn] apt-get update 失败，继续（安装阶段可能仍可用）"
apt-get install -y python3 python3-venv python3-pip nginx unzip curl >/tmp/apt-install.log 2>&1 || {
  echo "[error] apt 安装失败，最后 15 行："; tail -n 15 /tmp/apt-install.log; exit 1;
}

step "[2/9] 预检 Python 语法（失败则不停服务、直接中止）"
# 用 /tmp 作为 pycache 前缀：不在解压目录里生成 root 属主的 __pycache__
PYTHONPYCACHEPREFIX=/tmp/pb_pycache python3 -m compileall -q "$SRC_DIR/backend" || {
  echo "[error] backend 存在语法错误，已中止部署（线上服务未受影响）"
  exit 1
}
rm -rf "$SRC_DIR/backend/__pycache__" 2>/dev/null || true

step "[3/9] 停止旧服务并复制程序"
systemctl stop paperbrief >/dev/null 2>&1 || true
mkdir -p "$APP_DIR"
rm -rf "$APP_DIR/backend" "$APP_DIR/frontend" "$APP_DIR/scripts"
cp -r "$SRC_DIR/backend" "$SRC_DIR/frontend" "$SRC_DIR/scripts" "$APP_DIR/"
cp -f "$SRC_DIR/requirements.txt" "$SRC_DIR/README.md" "$APP_DIR/"
if [ -d "$SRC_DIR/pdf" ]; then
  mkdir -p "$APP_DIR/pdf"
  cp -f "$SRC_DIR/pdf/"*.pdf "$APP_DIR/pdf/" 2>/dev/null || true
fi
mkdir -p "$APP_DIR/data/parsed" "$APP_DIR/data/pages" "$APP_DIR/data/uploads"

step "[4/9] 创建虚拟环境"
cd "$APP_DIR"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || { echo "[error] venv 创建失败"; exit 1; }
fi
. .venv/bin/activate

step "[5/9] 安装 Python 依赖"
pip install --upgrade pip -q >/tmp/pip.log 2>&1 || true
pip install -r requirements.txt >>/tmp/pip.log 2>&1 || {
  echo "[error] pip 安装失败，最后 15 行："; tail -n 15 /tmp/pip.log; exit 1;
}
python -c "import pymupdf; print('PyMuPDF OK:', pymupdf.__doc__.splitlines()[0])"

step "[5.5/9] 预下载向量模型（可选，失败不影响启动）"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" EMBED_MODEL="BAAI/bge-small-en-v1.5" \
  "$APP_DIR/.venv/bin/python" -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); print('embedding model ready')" \
  2>/tmp/embed.log || echo "[warn] 向量模型未就绪，检索将退回 BM25（详见 /tmp/embed.log）"

step "[6/9] 写入 .env（已存在则不覆盖）"
if [ ! -f "$APP_DIR/.env" ]; then
  cat > "$APP_DIR/.env" <<'EOF'
PORT=8000
PAPERBRIEF_PDF_DIR=/opt/paperbrief/pdf
DEEPSEEK_API_KEY=
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
EMBED_MODEL=BAAI/bge-small-en-v1.5
HF_ENDPOINT=https://hf-mirror.com
EOF
  echo "    已生成 .env"
else
  echo "    保留已有 .env（不覆盖你的 key）"
fi

step "[7/9] 配置 systemd"
cat > /etc/systemd/system/paperbrief.service <<EOF
[Unit]
Description=PaperBrief web service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$APP_DIR
EnvironmentFile=$APP_DIR/.env
Environment=PYTHONIOENCODING=utf-8
Environment=PYTHONUTF8=1
Environment=HF_ENDPOINT=https://hf-mirror.com
Environment=EMBED_MODEL=BAAI/bge-small-en-v1.5
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/backend/server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable paperbrief >/dev/null 2>&1 || true
systemctl restart paperbrief

step "[8/9] 配置 Nginx"
rm -f /etc/nginx/sites-enabled/default /etc/nginx/conf.d/default.conf 2>/dev/null || true
cat > /etc/nginx/sites-available/paperbrief <<EOF
server {
    listen 80 default_server;
    server_name _;
    server_tokens off;
    client_max_body_size 60m;
    add_header X-Content-Type-Options nosniff always;
    add_header X-Frame-Options SAMEORIGIN always;
    add_header Referrer-Policy no-referrer always;

    location / {
        proxy_pass http://127.0.0.1:$PORT;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_read_timeout 180s;
    }
}
EOF
ln -sf /etc/nginx/sites-available/paperbrief /etc/nginx/sites-enabled/paperbrief
nginx -t >/tmp/nginx-test.log 2>&1 || { echo "[error] nginx 配置检查失败："; cat /tmp/nginx-test.log; exit 1; }
systemctl restart nginx

# 把解压目录属主还给原用户，避免下次 rm -rf 被 root 文件卡住
SRC_OWNER="$(stat -c '%U:%G' "$SRC_DIR" 2>/dev/null || true)"
if [ -n "$SRC_OWNER" ]; then
  chown -R "$SRC_OWNER" "$SRC_DIR" 2>/dev/null || true
fi

step "[9/9] 健康检查"
for i in 1 2 3 4 5; do
  sleep 1
  if curl -sf "http://127.0.0.1:$PORT/api/health" >/tmp/health.json 2>/dev/null; then
    echo "    健康检查通过：$(cat /tmp/health.json)"
    echo
    echo "=============================================="
    echo " 部署完成 ✅"
    echo " 访问地址： http://<服务器公网IP>/"
    echo " 版本：     以页面左上角为准"
    echo " 日志：     journalctl -u paperbrief -n 50 --no-pager"
    echo " 配置：     $APP_DIR/.env"
    echo "=============================================="
    exit 0
  fi
done

echo "[error] 服务未通过健康检查，最后 30 行日志："
journalctl -u paperbrief -n 30 --no-pager || true
exit 1
