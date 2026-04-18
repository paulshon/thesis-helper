# 논문 작성 도우미 v9 — 맥락 기반 구조 추론 아키텍처

## v9 핵심 업그레이드 (v8 베이스)

### 🧠 [NEW-1] 문장 기능 분류 엔진 (client + server)
표면적 헤더/제목이 없어도 문장의 **기능(Function)**으로 구조 분류:
- `background` — 연구배경, 현상 설명, 선행연구, 인용
- `problem` — 문제 제기, 한계, 공백 선언
- `importance` — 중요성, 의의, 기여
- `purpose` — 연구목적, 목표 선언
- `question` — 연구문제(RQ), 탐구 질문
- `hypothesis` — 가설 선언, 예상 관계
- `method` — 연구방법, 절차, 측정

### 🔄 [NEW-2] detectSecs → inferStructureClient() 교체
기존: `문장 → 키워드 → 섹션 매칭` (헤더 없으면 실패)
v9:  `문장 → 의미(기능) 분류 → 섹션 재구성` (헤더 없어도 작동)

### ✅ [NEW-3] "해당없음" 완전 제거
모든 섹션은 항상 추론된 결과를 반환:
- 명시적 표현 감지 → ✅ 감지됨 (신뢰도 표시)
- 명시적 표현 없음 → 🔍 추론됨 (개선 안내)
- ❌ "미감지/해당없음" → **영구 삭제**

### 🌐 [NEW-4] 서버 /infer API 연동 (ELECTRA 검증 포함)
- 클라이언트 추론 → 오프라인에서도 완전 동작
- 서버(/infer) 연동 → KR-ELECTRA 검증 + Ollama LLM 보정
- 버튼: `🧠 서버 구조 추론`

### 📊 [NEW-5] 구조 추론 결과 시각화
- 섹션별 신뢰도 % 표시
- 기능(function) 분포 뱃지 표시 (background×3, purpose×1...)
- 구조 완성도 점수 (0~100점)

### ⚡ [NEW-6] 개선안 자동 생성
구조 분석 결과를 기반으로 자동 생성:
- 🔴 필수 — 누락된 섹션 선언 문장 템플릿 제공
- 🟡 권장 — 논리 공백(배경→목적 연결 등) 개선 안내
- 내용 보강 — 1개 문장만 감지된 섹션 확장 권고

### 🔧 [NEW-7] KR-ELECTRA 역할 재정의
기존 v8: KR-ELECTRA = 분류기 (단독 사용 → 실패)
v9:      KR-ELECTRA = **검증/보정기** (LLM이 생성 → ELECTRA가 검증)
AI 패널 표기: `ELECTRA(검증기)` / `LLM(구조추론)`

## v9 서버 신규 API
| 엔드포인트 | 기능 |
|---|---|
| `POST /infer` | 맥락 기반 구조 추론 (표면 헤더 없이도 작동) |
| `POST /validate` | ELECTRA 문장-기능 검증 |
| `POST /suggest` | 개선안 자동 생성 |
| `POST /classify` | 문장 기능 일괄 분류 |

## v8에서 유지된 기능
- 5문단 미시구조 분석 (연구배경/목적/문제/질문/가설)
- BERT 점수 평가 (KR-ELECTRA 검증기)
- LLM 첨삭 (Ollama / 규칙 기반 fallback)
- PDF 뷰어 (PDF.js)
- 새 논문 작성 메뉴 (P1~P5 × S1~S5)
- 프로젝트 저장/불러오기

## 실행 방법
- 브라우저: `index.html` 직접 열기 (v9 클라이언트 추론 즉시 동작)
- 서버: `python python/server.py` 또는 `start_server.bat/sh`
- 서버 주소: http://localhost:8765
