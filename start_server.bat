@echo off
echo ========================================
echo 논문 작성 도우미 v16 서버 시작
echo ========================================
echo.
echo [의존성 확인]
pip install uvicorn fastapi pdfplumber PyPDF2 python-multipart requests -q
echo.
echo [Ollama 상태 확인]
ollama list
echo.
echo [서버 시작] http://localhost:8765
python python/server.py
pause
