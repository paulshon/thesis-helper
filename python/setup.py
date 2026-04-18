"""
setup.py — 논문 도우미 v19 의존성 설치 스크립트
실행: python python/setup.py
  ① Python 패키지를 설치합니다 (완료 시 .setup_done 마커 생성)
  ② 이미 설치된 경우 패키지 설치를 건너뜁니다 (최초 1회만)
  ③ Ollama 4가지 권장 모델 중 하나를 선택해 다운로드합니다
"""

import subprocess, sys, os, json

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MARKER_FILE = os.path.join(SCRIPT_DIR, ".setup_done")

MODELS = [
    {"num":1,"name":"eeve-korean:10.8b","size":"10.8B","stars":"★★★★★★★★★☆",
     "desc":"한국어 최적화 최고, 속도 빠름",     "cmd":"ollama pull eeve-korean:10.8b"},
    {"num":2,"name":"qwen2.5:14b",      "size":"14B",  "stars":"★★★★☆★★★★★",
     "desc":"다국어+추론 최강, 논문 분석 최고",  "cmd":"ollama pull qwen2.5:14b"},
    {"num":3,"name":"qwen2.5:7b",       "size":"7B",   "stars":"★★★★★★★★☆☆",
     "desc":"가장 가볍고 빠름 (메모리 적게 사용)","cmd":"ollama pull qwen2.5:7b"},
    {"num":4,"name":"exaone3.5:7.8b",   "size":"7.8B", "stars":"★★★★★★★★★☆",
     "desc":"LG 한국어 특화 (32b보다 가벼움)",  "cmd":"ollama pull exaone3.5:7.8b"},
]

PACKAGES = [
    "uvicorn","fastapi","pdfplumber","PyPDF2",
    "python-multipart","requests","transformers","torch",
]

def run(cmd, **kw):
    return subprocess.run(cmd, **kw)

def check_ollama():
    try:
        r = run(["ollama","list"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def get_installed_models():
    try:
        r = run(["ollama","list"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")[1:]
        return [ln.split()[0] for ln in lines if ln.strip()]
    except Exception:
        return []

def banner(text):
    print("\n" + "="*55)
    print(f"  {text}")
    print("="*55)

# ── STEP 1: Python 패키지 (최초 1회만) ──────────────────
def install_packages():
    if os.path.exists(MARKER_FILE):
        with open(MARKER_FILE, encoding="utf-8") as f:
            info = json.load(f)
        print(f"\n✅ Python 패키지가 이미 설치되어 있습니다.")
        print(f"   설치 일시 : {info.get('installed_at','알 수 없음')}")
        print(f"   설치 패키지: {', '.join(info.get('packages',[]))}")
        print("   재설치하려면 python/.setup_done 파일을 삭제하세요.")
        return

    banner("STEP 1 — Python 패키지 설치 (최초 1회)")
    failed = []
    for pkg in PACKAGES:
        print(f"\n  설치 중: {pkg} ...")
        r = run([sys.executable, "-m", "pip", "install", pkg], capture_output=True)
        if r.returncode == 0:
            print(f"  ✅ {pkg} 완료")
        else:
            print(f"  ⚠  {pkg} 실패: {r.stderr.decode()[:120]}")
            failed.append(pkg)

    import datetime
    with open(MARKER_FILE, "w", encoding="utf-8") as f:
        json.dump({"installed_at": datetime.datetime.now().isoformat(),
                   "packages": PACKAGES, "failed": failed}, f, ensure_ascii=False, indent=2)
    if failed:
        print(f"\n⚠  일부 실패: {failed}")
    else:
        print(f"\n✅ 모든 패키지 설치 완료 — .setup_done 마커 저장됨")

# ── STEP 2: Ollama 모델 선택 설치 ───────────────────────
def install_model():
    banner("STEP 2 — Ollama LLM 모델 설치")
    if not check_ollama():
        print("\n  ⚠  Ollama가 설치되어 있지 않습니다.")
        print("     https://ollama.com 에서 설치 후 다시 실행하세요.")
        return

    installed = get_installed_models()
    print("\n  ┌────┬────────────────────┬──────┬─────────────────────────────┐")
    print("  │ 번│ 모델명              │ 크기 │ 설명                        │")
    print("  ├────┼────────────────────┼──────┼─────────────────────────────┤")
    for m in MODELS:
        chk = " ✅" if any(m["name"].split(":")[0] in i for i in installed) else ""
        print(f"  │  {m['num']}│ {m['name']:<18} │ {m['size']:<4} │ {m['desc']:<27}│{chk}")
    print("  ├────┴────────────────────┴──────┴─────────────────────────────┤")
    print("  │  0 — 건너뛰기 (나중에 수동 설치)                             │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ✅ = 이미 설치됨   |   여러 모델: 스크립트를 여러 번 실행\n")

    while True:
        try:
            choice = input("  설치할 모델 번호 (0~4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  건너뜁니다."); return
        if choice == "0":
            print("  모델 설치를 건너뜁니다."); return
        if choice in [str(m["num"]) for m in MODELS]:
            model = next(m for m in MODELS if str(m["num"]) == choice); break
        print("  ❌ 0~4 중 하나를 입력하세요.")

    print(f"\n  다운로드: {model['name']}  ({model['size']}) — {model['desc']}")
    print(f"  명령: {model['cmd']}\n")
    pull = run(["ollama", "pull", model["name"]])
    if pull.returncode == 0:
        print(f"\n  ✅ {model['name']} 설치 완료!")
    else:
        print(f"\n  ⚠  실패. 네트워크/디스크 확인 후: {model['cmd']}")

# ── STEP 3: 완료 안내 ───────────────────────────────────
def print_next_steps():
    banner("설치 완료 — 다음 단계")
    print("""
  1. 서버 실행
     Windows : start_server.bat
     Mac/Linux: ./start_server.sh
     직접    : python python/server.py

  2. index.html 열기

  3. UI 메뉴 "AI 엔진 > 모델 선택" 에서 설치된 모델 선택
""")

if __name__ == "__main__":
    banner("논문 도우미 v19 — 환경 설정")
    install_packages()
    install_model()
    print_next_steps()
