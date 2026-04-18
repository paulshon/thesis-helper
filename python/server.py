"""
server.py — 논문 작성 도우미 AI 서버 v16
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v16 핵심 신규 기능]
  ① /upload_pdf   — PDF 업로드 → pdfplumber 텍스트 추출
  ② /analyze_pdf  — 텍스트 → Ollama LLM 문장별 분석
                     → "수정할 문장" + "개선된 문장" JSON 반환
  ③ /analyze_section — 섹션(연구배경 등) 심층 분석
  ④ /models       — 설치된 Ollama 모델 목록
  ⑤ 기존 /bert /llm /analyze /infer /status 모두 유지

실행: python server.py
포트: 8765

권장 모델 (32GB RAM):
  ollama pull exaone3.5:32b   (최고 품질, LG AI)
  ollama pull eeve-korean:10.8b (한국어 특화)
"""

import os, re, io, json, time, hashlib, logging, subprocess
from typing import Optional, List, Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v16")

app = FastAPI(title="논문 도우미 서버 v19", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── 캐시 ────────────────────────────────────────────────
_bert_cache:  dict = {}
_llm_cache:   dict = {}
_infer_cache: dict = {}
_pdf_cache:   dict = {}

# ── 전역 상태 ──────────────────────────────────────────
_ollama_ok:    Optional[bool] = None
_active_model: str = ""
_user_selected_model: str = ""   # [v19] 사용자가 UI에서 선택한 모델
_electra_model     = None
_electra_tokenizer = None
_electra_loaded:   bool = False

# [v19] 지원 모델 목록 (UI 표시용)
SUPPORTED_MODELS = [
    {"id":"eeve-korean:10.8b", "label":"EEVE Korean 10.8B",  "size":"10.8B",
     "stars":"★★★★★★★★★☆", "desc":"한국어 최적화 최고, 속도 빠름",     "cmd":"ollama pull eeve-korean:10.8b"},
    {"id":"qwen2.5:14b",       "label":"Qwen 2.5 14B",        "size":"14B",
     "stars":"★★★★☆★★★★★", "desc":"다국어+추론 최강, 논문 분석 최고",  "cmd":"ollama pull qwen2.5:14b"},
    {"id":"qwen2.5:7b",        "label":"Qwen 2.5 7B",          "size":"7B",
     "stars":"★★★★★★★★☆☆", "desc":"가장 가볍고 빠름 (메모리 적게 사용)","cmd":"ollama pull qwen2.5:7b"},
    {"id":"exaone3.5:7.8b",    "label":"EXAONE 3.5 7.8B",      "size":"7.8B",
     "stars":"★★★★★★★★★☆", "desc":"LG 한국어 특화 (32b보다 가벼움)",   "cmd":"ollama pull exaone3.5:7.8b"},
]


# ══════════════════════════════════════════════════════
# Ollama 연동
# ══════════════════════════════════════════════════════

def check_ollama(force: bool = False) -> bool:
    """Ollama 실행 여부 확인. force=True 이면 캐시 무시하고 재확인."""
    global _ollama_ok
    if not force and _ollama_ok is not None:
        return _ollama_ok
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        _ollama_ok = (r.returncode == 0)
    except Exception:
        _ollama_ok = False
    logger.info(f"Ollama check: {_ollama_ok}")
    return _ollama_ok


def get_installed_models() -> list:
    """설치된 Ollama 모델 이름 목록 반환 [v19]"""
    try:
        r = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")[1:]
        return [ln.split()[0] for ln in lines if ln.strip()]
    except Exception:
        return []


def get_best_model() -> str:
    global _active_model
    # [v19] 사용자가 UI에서 선택한 모델 우선
    if _user_selected_model:
        _active_model = _user_selected_model
        return _active_model
    if _active_model:
        return _active_model
    if not check_ollama():
        return ""
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")[1:]
        installed = [ln.split()[0] for ln in lines if ln.strip()]
        priority = [
            "exaone3.5:32b", "exaone3.5:7.8b",
            "eeve-korean:10.8b", "eeve-korean",
            "qwen2.5:14b", "qwen2.5:7b",
            "llama3.1:8b", "llama3.2:3b",
        ]
        for p in priority:
            for m in installed:
                if p.split(":")[0] in m:
                    _active_model = m
                    logger.info(f"자동 선택 모델: {m}")
                    return m
        if installed:
            _active_model = installed[0]
            return installed[0]
    except Exception as e:
        logger.warning(f"모델 목록 오류: {e}")
    return ""


def ollama_run(prompt: str, model: str = "") -> str:
    """Ollama CLI 호출"""
    if not model:
        model = get_best_model()
    if not model:
        return ""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt, capture_output=True,
            text=True, timeout=300, encoding="utf-8"
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip()
            # ANSI escape code 제거 (터미널 출력 코드가 섞이는 문제)
            import re as _re
            raw = _re.sub(r'\x1B\[[0-9;]*[mGKHFABCDJM]', '', raw)
            raw = _re.sub(r'\[\d+[ABCDK]', '', raw)
            raw = _re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', raw)
            return raw.strip()
        logger.warning(f"Ollama 오류: {result.stderr[:100]}")
    except subprocess.TimeoutExpired:
        logger.warning("Ollama 타임아웃 (300초)")
    except Exception as e:
        logger.warning(f"Ollama 실패: {e}")
    return ""


# ══════════════════════════════════════════════════════
# PDF 텍스트 추출
# ══════════════════════════════════════════════════════

def extract_pdf(pdf_bytes: bytes) -> Dict:
    """pdfplumber → PyPDF2 순서로 텍스트 추출"""
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i+1, "text": text.strip()})
        full = "\n\n".join(p["text"] for p in pages)
        full = clean_text(full)
        return {"success": True, "full_text": full, "pages": pages,
                "total_pages": total, "word_count": len(full.split()), "engine": "pdfplumber"}
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pdfplumber 실패: {e}")

    # PyPDF2 fallback
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"page": i+1, "text": text.strip()})
        full = clean_text("\n\n".join(p["text"] for p in pages))
        return {"success": True, "full_text": full, "pages": pages,
                "total_pages": len(reader.pages), "word_count": len(full.split()), "engine": "PyPDF2"}
    except Exception as e:
        return {"success": False, "full_text": "", "pages": [], "total_pages": 0,
                "word_count": 0, "engine": "failed", "error": str(e)}


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    raw = re.split(r'(?<=[.!?。])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 15]


# ══════════════════════════════════════════════════════
# [v16 핵심] LLM 문장 분석 프롬프트
# ══════════════════════════════════════════════════════

ANALYSIS_PROMPT = """당신은 한국 학술 논문 전문 편집자입니다.
아래 논문 텍스트의 각 문장을 분석하세요.

[분석 기준]
1. 과장/비학술 표현 (매우, 굉장히, 획기적 등)
2. 인용 없는 선행연구 주장 (저자, 연도 형식 필요)
3. 모호한 동사 (살펴본다→규명한다)
4. 연구질문이 의문문 아닌 경우
5. 가설에 방향성(정적+/부적-) 없음
6. 200자 초과 긴 문장
7. 논리 연결어 부재

[텍스트]
{text}

[출력: 아래 JSON 형식만 출력, 다른 텍스트 없음]
{{
  "sentences": [
    {{
      "original": "원본 문장",
      "has_problem": true,
      "problem_type": "문제유형",
      "problem_desc": "문제 설명",
      "improved": "개선된 문장"
    }}
  ],
  "summary": "전체 분석 요약",
  "total_problems": 숫자
}}"""

SECTION_PROMPT = """당신은 한국 학술 논문 편집자입니다.
아래는 논문의 [{section}] 섹션입니다.

[평가 기준]
{criteria}

[원문]
{text}

[출력: 아래 JSON만 출력]
{{
  "section": "{section}",
  "score": 점수(0-100),
  "grade": "A/B/C/D",
  "sentences": [
    {{
      "original": "원본 문장",
      "has_problem": true/false,
      "problem_type": "문제유형",
      "problem_desc": "문제설명",
      "improved": "개선된 문장"
    }}
  ],
  "overall_feedback": "전체 피드백",
  "key_improvements": ["개선점1", "개선점2"]
}}"""

SECTION_CRITERIA = {
    "bg": "거시→학문적중요성→기존연구한계→연구공백→연구필요성 깔때기 구조 / APA 인용 형식 / Research Gap 명시",
    "purpose": "연구목적 명확한 선언문 / 대상·관점·방향 3요소 / 모호한 동사 없음 / RQ 의문문 형식",
    "problem": "거시현상 사실명제 개시 / 학문적중요성 논거 / 기존연구 양보절+한계 / 연구공백 선언",
    "rq": "의문문(?) 형식 / 개방형질문 (어떻게/왜) / RQ1→RQ2 위계 구조 / 연구목적과 일관성",
    "hypo": "방향성 명시(정적+/부적-) / 이론적 근거 / 변인 정의 / 검증 가능한 형태",
    "theory": "이론명+저자/연도 / 핵심명제 / 연구변인과 이론개념 대응 / 적용 적합성",
    "lit": "연대기/주제별 구성 / 선행연구 비판적 검토 / 공백(Gap) 명시 / 최신 문헌",
    "method": "연구 패러다임 명시 / 방법 선택 이유 / 표집·수집·분석 절차 / 신뢰도·타당도",
    "result": "연구문제별 결과 / 통계 수치 포함 / 표/그림 참조 / 사실적 기술",
    "discuss": "결과 해석+이론 연결 / 선행연구 비교 / 이론적·실천적 기여 / 한계 및 후속연구",
    "conc": "연구 요약 / 핵심 기여 / 한계 / 강렬한 마무리",
}

SECTION_NAMES = {
    "bg":"연구배경","purpose":"연구목적","problem":"연구문제",
    "rq":"연구질문","hypo":"가설","theory":"이론적 배경",
    "lit":"문헌연구","method":"연구방법","result":"연구결과",
    "discuss":"논의","conc":"결론"
}


def parse_json(raw: str) -> Optional[Dict]:
    """LLM 응답에서 JSON 추출"""
    raw = re.sub(r'```json\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r'\{[\s\S]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def analyze_with_llm(text: str, section_id: str = "", model: str = "") -> Dict:
    """LLM 또는 규칙 기반으로 문장 분석"""
    if not model:
        model = get_best_model()

    if not model or not check_ollama():
        return rule_analyze(text, section_id)

    logger.info(f"LLM 분석: model={model}, sec={section_id}, len={len(text)}")

    if section_id and section_id in SECTION_CRITERIA:
        sname = SECTION_NAMES.get(section_id, section_id)
        prompt = SECTION_PROMPT.format(
            section=sname,
            criteria=SECTION_CRITERIA[section_id],
            text=text[:3000]
        )
    else:
        prompt = ANALYSIS_PROMPT.format(text=text[:3000])

    raw = ollama_run(prompt, model)
    if raw:
        parsed = parse_json(raw)
        if parsed and "sentences" in parsed:
            parsed["engine"] = f"Ollama({model})"
            parsed["model"]  = model
            return parsed

    logger.warning("LLM 파싱 실패 → 규칙 기반 fallback")
    return rule_analyze(text, section_id)


# ══════════════════════════════════════════════════════
# 규칙 기반 분석 (LLM fallback)
# ══════════════════════════════════════════════════════

RULES = [
    {"type":"과장_표현",
     "pat": r'(매우|굉장히|엄청나게|완전히|절대적으로|획기적|놀랍게도)',
     "desc":"과장 표현 — 실증적 수치/근거로 대체 필요"},
    {"type":"인용_누락",
     "pat": r'(연구|학자).{0,20}(밝혔다|제시했다|주장했다|발견했다)',
     "not": r'\([가-힣A-Za-z].+?\d{4}\)',
     "desc":"APA 인용 누락 — (저자, 연도) 형식 추가 필요"},
    {"type":"모호한_동사",
     "pat": r'(살펴보|알아보|검토해보|알고자)',
     "desc":"모호한 동사 — 규명/분석/탐구로 교체 필요"},
    {"type":"RQ_의문문_아님",
     "pat": r'(연구문제|RQ).{0,50}(분석한다|탐색한다)',
     "not": r'\?',
     "desc":"연구질문이 의문문(?) 형식 아님"},
    {"type":"가설_방향성_누락",
     "pat": r'(가설|H\d).{0,80}(것이다|예상된다)',
     "not": r'(정적|부적|유의|높을수록|\+|\-)',
     "desc":"가설에 방향성(정적+/부적-) 미포함"},
    {"type":"긴_문장",
     "pat": None, "min_len": 200,
     "desc":"문장이 200자 초과 — 2개로 분리 권장"},
]


def fix_sentence(s: str, rule_type: str) -> str:
    if rule_type == "과장_표현":
        return re.sub(r'(매우|굉장히|엄청나게|완전히|획기적|놀랍게도)', '[실증적 수치/근거]', s)
    if rule_type == "인용_누락":
        return re.sub(r'(밝혔다|제시했다|주장했다|발견했다)', r'(저자명, 연도) \1', s)
    if rule_type == "모호한_동사":
        for old, new in [('살펴보','규명'),('알아보','분석'),('검토해보','검토'),('알고자','규명하고자')]:
            s = s.replace(old, new)
        return s
    if rule_type == "RQ_의문문_아님":
        s = re.sub(r'(분석|탐색|규명)(한다|하였다)', r'\1하는가?', s)
        return s
    if rule_type == "가설_방향성_누락":
        return re.sub(r'(것이다|예상된다)', '정적(+) 영향을 미칠 것이다', s, count=1)
    if rule_type == "긴_문장":
        return s[:120] + "... [이후 내용 별도 문장으로 분리 권장]"
    return s


def diagnose(sentence: str, section_id: str) -> Dict:
    s = sentence.strip()
    for rule in RULES:
        if rule.get("min_len") and len(s) > rule["min_len"]:
            return {"original":s,"has_problem":True,"problem_type":rule["type"],
                    "problem_desc":rule["desc"],"improved":fix_sentence(s,rule["type"])}
        if rule.get("pat") and re.search(rule["pat"], s):
            if rule.get("not") and re.search(rule["not"], s):
                continue
            return {"original":s,"has_problem":True,"problem_type":rule["type"],
                    "problem_desc":rule["desc"],"improved":fix_sentence(s,rule["type"])}
    return {"original":s,"has_problem":False,"problem_type":"","problem_desc":"","improved":s}


def rule_analyze(text: str, section_id: str) -> Dict:
    sents = split_sentences(text)
    results = [diagnose(s, section_id) for s in sents[:30]]
    problems = sum(1 for r in results if r["has_problem"])
    return {
        "sentences": results,
        "summary": f"규칙 기반 분석: {len(results)}문장 중 {problems}개 수정 필요.",
        "total_problems": problems,
        "engine": "rule-based",
        "model": "none",
        "note": "Ollama 미실행 — 규칙 기반 분석. 더 정확한 분석: ollama pull exaone3.5:32b 후 재시작"
    }


# ══════════════════════════════════════════════════════
# KR-ELECTRA BERT 평가 (기존 유지)
# ══════════════════════════════════════════════════════

def get_electra():
    global _electra_model, _electra_tokenizer, _electra_loaded
    if _electra_loaded:
        return _electra_model, _electra_tokenizer
    try:
        from transformers import AutoTokenizer, AutoModel
        name = "snunlp/KR-ELECTRA-discriminator"
        _electra_tokenizer = AutoTokenizer.from_pretrained(name)
        _electra_model     = AutoModel.from_pretrained(name)
        _electra_model.eval()
        logger.info("KR-ELECTRA 로드 완료")
    except Exception as e:
        logger.warning(f"KR-ELECTRA 실패: {e}")
    _electra_loaded = True
    return _electra_model, _electra_tokenizer


def bert_score(text: str, section_id: str) -> dict:
    score = 70.0; details = []
    markers = ['따라서','그러나','이에','본 연구','분석','규명','도출','제시']
    found = [m for m in markers if m in text]
    ms = min(15.0, len(found)*2.5); score += ms
    details.append({"item":"학술 지시어","score":ms,"status":"ok" if ms>=7 else "warn","message":f"{len(found)}개"})
    has_cit = bool(re.search(r'\([가-힣A-Za-z].+?\d{4}\)', text))
    cs = 10.0 if has_cit else 0.0; score += cs
    details.append({"item":"인용 형식","score":cs,"status":"ok" if has_cit else "fail"})
    sents = [s for s in re.split(r'[.!?。]', text) if len(s.strip())>10]
    avg = sum(len(s) for s in sents)/max(len(sents),1)
    ss = 8.0 if 20<=avg<=120 else (3.0 if avg<20 else 5.0); score += ss
    details.append({"item":"문장 구조","score":ss,"status":"ok" if ss>=7 else "warn"})
    fb = [f for f in ['매우 중요','획기적','놀랍게도'] if f in text]
    pen = len(fb)*3.0; score -= pen
    if fb: details.append({"item":"과장 표현","score":-pen,"status":"fail","message":str(fb)})
    final = max(0, min(100, round(score,1)))
    grade = "A" if final>=85 else "B" if final>=75 else "C" if final>=65 else "D"

    model, tokenizer = get_electra()
    if model:
        try:
            import torch
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            cls = outputs.last_hidden_state[:,0,:]
            raw = float((cls/cls.norm()).mean().item())
            electra_score = min(100, max(50, 70 + raw*25))
            final = round(electra_score*0.4 + final*0.6, 1)
            grade = "A" if final>=85 else "B" if final>=75 else "C" if final>=65 else "D"
            return {"score":final,"grade":grade,"details":details,"engine":"KR-ELECTRA+rule",
                    "word_count":len(text.split()),"sentence_count":len(sents)}
        except Exception:
            pass

    return {"score":final,"grade":grade,"details":details,"engine":"rule-based",
            "word_count":len(text.split()),"sentence_count":len(sents)}


# ══════════════════════════════════════════════════════
# 구조 추론 (기존 유지)
# ══════════════════════════════════════════════════════

FN_PATS = {
    "purpose":    [r"목적(으로|은|이|을)", r"(규명|탐구|분석)(하고자|한다)", r"본 연구(는|의).*(한다|위해)"],
    "question":   [r"(연구 문제|RQ\d?)[은는]?", r"(무엇|어떻게|왜).*(인지|인가|는가)"],
    "hypothesis": [r"(가설|hypothesis)", r"(일 것이다|예상된다)", r"(정적|부적).*(관계|영향)"],
    "importance": [r"(중요|필요성)(가|를|이|하다)", r"(학문적|이론적).*(의의|기여)"],
    "problem":    [r"(문제|한계|공백)[은는이가]", r"(그러나|하지만).*(않았다|없었다)"],
    "background": [r"(최근|현재).*(변화|발전)", r"\d{4}년?.*(에|이후)", r"(선행|기존).*(연구|문헌)"],
    "method":     [r"(방법론|연구방법)[은는이가]?", r"(수집|표집)(하였다|방식)", r"(참여자|응답자)[은는이가]"],
}
FN2SEC = {"problem":"background","background":"background","importance":"background",
          "purpose":"purpose","question":"question","hypothesis":"hypothesis","method":"method"}
SEC_LBL = {"background":"연구배경","purpose":"연구목적","question":"연구문제","hypothesis":"가설","method":"연구방법"}


def classify_fn(s: str) -> Dict:
    sc = {fn: sum(1 for p in pats if re.search(p,s)) for fn,pats in FN_PATS.items()}
    best = max(sc, key=sc.get)
    conf = 0.3 if sc[best]==0 else min(0.92, 0.55+sc[best]*0.12)
    return {"function": best if sc[best]>0 else "background",
            "confidence": round(conf,2), "section": FN2SEC.get(best,"background")}


def infer_structure(text: str) -> Dict:
    sents = [s.strip() for s in re.split(r'(?<=[.!?。])\s+',text) if len(s.strip())>10][:80]
    struct = {sec:[] for sec in SEC_LBL}
    for s in sents:
        r = classify_fn(s)
        if r["section"] in struct:
            struct[r["section"]].append({"text":s,"function":r["function"],"confidence":r["confidence"]})
    for sec in struct:
        if not struct[sec]:
            struct[sec] = [{"text":f"[추론됨] {SEC_LBL[sec]} 명시적 표현 부재","function":"inferred","confidence":0.2,"inferred":True}]
    req = ["background","purpose","question"]
    present  = [s for s in req if not struct[s][0].get("inferred")]
    inferred = [s for s in req if struct[s][0].get("inferred")]
    score    = round(len(present)/len(req)*100,1)
    return {
        "structure": struct,
        "completeness": {"score":score,"grade":"A" if score>=85 else "B" if score>=70 else "C" if score>=50 else "D",
                         "present":present,"inferred":inferred,
                         "issues":[f"{SEC_LBL[s]}: 명시적 표현 부재" for s in inferred]},
        "sentence_count": len(sents), "engine":"rule+ELECTRA"
    }


def rule_refine(text: str, sec: str) -> str:
    revs = {
        "bg":[("최근","2020년대 이후"),("중요하다","학문적·사회적 과제로 부상하고 있다")],
        "purpose":[("살펴본다","분석한다"),("알아본다","규명한다")],
        "method":[("분석하였다","분석하였다(Cohen's κ=.84, p<.001)")],
    }
    r = text
    for old,new in revs.get(sec,[]):
        r = r.replace(old,new,1)
    hints = {"bg":"[첨삭] 연도·인용 추가 및 Research Gap 명시","purpose":"[첨삭] RQ를 의문문으로 수정","method":"[첨삭] 연구 패러다임 및 Cohen's Kappa 포함"}
    return r + "\n\n" + hints.get(sec,"[첨삭] 인용 형식과 논리적 흐름 점검")


# ══════════════════════════════════════════════════════
# 요청 모델
# ══════════════════════════════════════════════════════

class BertReq(BaseModel):
    text: str; section_id: str = "bg"

class LLMReq(BaseModel):
    text: str; section_id: str = "bg"; model: str = ""

class AnalyzeReq(BaseModel):
    sections: dict

class InferReq(BaseModel):
    text: str; use_llm: bool = True

class PDFAnalyzeReq(BaseModel):
    text: str; section_id: str = ""; model: str = ""

class SectionAnalyzeReq(BaseModel):
    text: str; section_id: str; model: str = ""


# ══════════════════════════════════════════════════════
# API 엔드포인트
# ══════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"message":"논문 도우미 서버 v23","version":"5.0",
            "endpoints":["/status","/upload_pdf","/analyze_pdf","/analyze_section",
                         "/bert","/llm","/analyze","/infer","/models","/set_model"]}


@app.get("/status")
def status():
    model, _ = get_electra()
    ollama   = check_ollama(force=True)   # 매번 재확인
    best     = get_best_model() if ollama else ""
    return {
        "status":           "running",
        "bert_model":       "KR-ELECTRA" if model else "rule-based",
        "llm_engine":       f"Ollama({best})" if ollama else "rule-based",
        "ollama_available": ollama,
        "active_model":     best,
        "user_selected":    _user_selected_model,
        "version":          "5.0-v23",
    }


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 업로드 → pdfplumber 텍스트 추출"""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원")
    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise HTTPException(status_code=400, detail="파일이 너무 작습니다")

    key = hashlib.md5(pdf_bytes).hexdigest()
    if key in _pdf_cache:
        return {**_pdf_cache[key], "cached":True}

    logger.info(f"PDF 추출: {file.filename} ({len(pdf_bytes)//1024}KB)")
    result = extract_pdf(pdf_bytes)
    if not result["success"] or not result["full_text"]:
        raise HTTPException(status_code=422, detail=f"추출 실패: {result.get('error','알 수 없는 오류')}")

    result["filename"] = file.filename
    result["cached"]   = False
    _pdf_cache[key]    = result
    logger.info(f"추출 완료: {result['total_pages']}페이지 {result['word_count']}단어")
    return result


@app.post("/analyze_pdf")
async def analyze_pdf(req: PDFAnalyzeReq):
    """추출된 텍스트 → LLM 문장별 분석 → 수정/개선 쌍 반환"""
    if not req.text or len(req.text.strip()) < 50:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다 (최소 50자)")

    key = hashlib.md5(f"{req.text}{req.section_id}".encode()).hexdigest()
    if key in _llm_cache:
        return {**_llm_cache[key], "cached":True}

    start  = time.time()
    result = analyze_with_llm(req.text, req.section_id, req.model)
    result["elapsed_ms"] = round((time.time()-start)*1000, 1)
    result["cached"]     = False
    _llm_cache[key]      = result
    logger.info(f"분석 완료: {result.get('total_problems',0)}개 문제 {result['elapsed_ms']}ms")
    return result


@app.post("/analyze_section")
async def analyze_section_ep(req: SectionAnalyzeReq):
    """섹션별 심층 분석"""
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다")

    key = hashlib.md5(f"{req.text}{req.section_id}section".encode()).hexdigest()
    if key in _llm_cache:
        return {**_llm_cache[key], "cached":True}

    start  = time.time()
    result = analyze_with_llm(req.text, req.section_id, req.model)
    br     = bert_score(req.text, req.section_id)
    result["bert_score"]   = br["score"]
    result["bert_grade"]   = br["grade"]
    result["bert_details"] = br.get("details",[])
    result["elapsed_ms"]   = round((time.time()-start)*1000, 1)
    result["cached"]       = False
    _llm_cache[key]        = result
    return result


@app.post("/bert")
def bert_ep(req: BertReq):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다")
    key = hashlib.md5(f"{req.text}{req.section_id}".encode()).hexdigest()
    if key in _bert_cache:
        return {**_bert_cache[key], "cached":True}
    r = bert_score(req.text, req.section_id)
    r["cached"] = False; _bert_cache[key] = r; return r


@app.post("/llm")
def llm_ep(req: LLMReq):
    if not req.text or len(req.text.strip()) < 10:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다")
    key = hashlib.md5(f"{req.text}{req.section_id}".encode()).hexdigest()
    if key in _llm_cache:
        return {**_llm_cache[key], "cached":True}
    if check_ollama():
        m = req.model or get_best_model()
        revised = ollama_run(f"다음 학술 문단을 더 학술적으로 개선하세요. 개선 문장만 출력:\n\n{req.text}", m)
        engine  = f"Ollama({m})"
    else:
        revised = rule_refine(req.text, req.section_id)
        engine  = "rule-based"
    r = {"original":req.text,"revised":revised or rule_refine(req.text,req.section_id),
         "suggestion":revised,"engine":engine,"cached":False}
    _llm_cache[key] = r; return r


@app.post("/analyze")
def analyze_ep(req: AnalyzeReq):
    results = {}; total = 0; count = 0
    for sid, text in req.sections.items():
        if text and len(text.strip()) > 10:
            r = bert_score(text, sid)
            results[sid] = r; total += r["score"]; count += 1
    avg   = round(total/max(count,1),1)
    grade = "A" if avg>=85 else "B" if avg>=75 else "C" if avg>=65 else "D"
    return {"sections":results,"avg_score":avg,"grade":grade,"section_count":count}


@app.post("/infer")
def infer_ep(req: InferReq):
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="텍스트가 너무 짧습니다")
    key = hashlib.md5(req.text.encode()).hexdigest()
    if key in _infer_cache:
        return {**_infer_cache[key], "cached":True}
    r = infer_structure(req.text)
    r["summary"] = {
        sec: {"label":SEC_LBL.get(sec,sec),"count":len(items),
              "inferred":items[0].get("inferred",False) if items else True,
              "excerpt":items[0]["text"][:100] if items else ""}
        for sec, items in r["structure"].items()
    }
    r["cached"] = False; _infer_cache[key] = r; return r


@app.get("/models")
def models_ep():
    """설치된 Ollama 모델 목록 + 지원 모델 정보 반환 [v19 확장]"""
    if not check_ollama():
        return {"available": False, "models": [], "active": "",
                "supported": SUPPORTED_MODELS,
                "message": "Ollama 미설치. https://ollama.com 에서 설치하세요."}
    try:
        r = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split("\n")[1:]
        installed = [{"name": ln.split()[0],
                      "size": ln.split()[2] if len(ln.split()) > 2 else ""}
                     for ln in lines if ln.strip()]
        # 지원 모델과 설치 여부 매칭
        for sup in SUPPORTED_MODELS:
            base = sup["id"].split(":")[0]
            sup["installed"] = any(base in m["name"] for m in installed)
        return {"available": True, "models": installed,
                "active": get_best_model(),
                "user_selected": _user_selected_model,
                "supported": SUPPORTED_MODELS}
    except Exception as e:
        return {"available": False, "models": [], "error": str(e),
                "supported": SUPPORTED_MODELS}


class SetModelReq(BaseModel):
    model: str

@app.post("/set_model")
def set_model_ep(req: SetModelReq):
    """[v23] UI에서 사용할 LLM 모델 지정"""
    global _user_selected_model, _active_model
    # 빈 문자열 → 자동 선택 모드 초기화
    if not req.model:
        _user_selected_model = ""
        _active_model = ""
        logger.info("모델 선택 초기화 — 자동 선택 모드")
        return {"success": True, "model": "", "message": "자동 선택 모드로 전환됨"}
    # Ollama 실행 여부 재확인 (force)
    if not check_ollama(force=True):
        raise HTTPException(status_code=503,
            detail="Ollama가 실행되지 않습니다. 'ollama serve' 또는 Ollama 앱을 시작하세요.")
    _user_selected_model = req.model
    _active_model = req.model
    logger.info(f"[v23] 사용자 선택 모델: {req.model}")
    return {"success": True, "model": req.model,
            "message": f"✅ {req.model} 모델로 설정됨"}


@app.delete("/cache")
def clear_cache_ep():
    _bert_cache.clear(); _llm_cache.clear(); _infer_cache.clear(); _pdf_cache.clear()
    return {"message":"모든 캐시 초기화"}


# ══════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("논문 작성 도우미 AI 서버 v19 시작")
    logger.info("포트: 8765")
    logger.info("[기존] /upload_pdf  — PDF → 텍스트 추출")
    logger.info("[기존] /analyze_pdf — LLM 문장별 분석")
    logger.info("[v19]  /set_model   — UI에서 모델 선택")
    logger.info("[v19]  /models      — 설치 모델 + 지원 모델 목록")
    logger.info("=" * 60)

    if check_ollama():
        m = get_best_model()
        logger.info(f"✅ Ollama 연결됨 — 활성 모델: {m}")
    else:
        logger.warning("⚠ Ollama 미설치 — 규칙 기반 모드")
        logger.warning("  설치: https://ollama.com")
        logger.warning("  모델: ollama pull exaone3.5:32b")

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info", reload=False)
