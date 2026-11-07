@echo off
setlocal
REM PaperBrief 本地模型版：使用本机 Ollama（离线，不依赖 DeepSeek API）
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PORT=8010
set PAPERBRIEF_PDF_DIR=%~dp0pdf
if not exist "%~dp0pdf" set PAPERBRIEF_PDF_DIR=%~dp0..\ref_kxyzs\pdf
set OLLAMA_BASE_URL=http://127.0.0.1:11434
set OLLAMA_MODEL=qwen2.5-3b-local
set DEEPSEEK_API_KEY=
echo Starting PaperBrief with LOCAL model (Ollama) on port %PORT% ...
python "%~dp0backend\server.py"
