FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000 \
    PAPERBRIEF_PDF_DIR=/app/pdf

# 系统依赖：PyMuPDF 运行库 + Tesseract OCR（扫描件兜底）
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY scripts ./scripts
COPY deploy.sh README.md ./

RUN mkdir -p /app/data/parsed /app/data/pages /app/data/uploads /app/pdf \
    && useradd -m -u 10001 paperbrief \
    && chown -R paperbrief:paperbrief /app
USER paperbrief

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)" || exit 1

CMD ["python", "backend/server.py"]
