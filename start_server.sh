#!/bin/bash
echo "========================================"
echo "논문 작성 도우미 v16 서버 시작"
echo "========================================"
pip install uvicorn fastapi pdfplumber PyPDF2 python-multipart requests -q
echo "[서버 시작] http://localhost:8765"
python python/server.py
