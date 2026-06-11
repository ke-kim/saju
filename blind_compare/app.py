"""
사주 LLM 블라인드 비교 데모
AI 엔지니어링 계획서 섹션 4 — 같은 프롬프트를 Claude / Gemini / GPT 에 동시 투입,
팀이 모델명 모른 채 채점 후 결과 공개.
"""

import os
import csv
import random
import concurrent.futures
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import streamlit as st


# ─── 설정 ────────────────────────────────────────────────────────────────────

# Claude 크레딧 충전 후 True 로 변경
CLAUDE_ENABLED = True

MODEL_KEYS = ["claude", "gemini", "gpt"]

TIERS: dict[str, dict] = {
    "저가형": {
        "label": "저가형 — 오늘의 운세 / 무료 챗봇",
        "claude": {"id": "claude-haiku-4-5-20251001", "name": "Claude Haiku 4.5",  "price_in": 1.00, "price_out":  5.00},
        "gemini": {"id": "gemini-3.5-flash",             "name": "Gemini 3.5 Flash",  "price_in": 0.10, "price_out":  0.40},
        "gpt":    {"id": "gpt-4o-mini",                "name": "GPT-4o mini",        "price_in": 0.15, "price_out":  0.60},
    },
    "고가형": {
        "label": "고가형 — 유료 심화 리포트",
        "claude": {"id": "claude-sonnet-4-6",      "name": "Claude Sonnet 4.6", "price_in":  3.00, "price_out": 15.00},
        "gemini": {"id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro",    "price_in":  1.25, "price_out":  5.00},
        "gpt":    {"id": "gpt-4o",                 "name": "GPT-4o",            "price_in":  2.50, "price_out": 10.00},
    },
}

USD_TO_KRW = 1380

CSV_PATH = Path(__file__).parent / "evaluation_results.csv"
CSV_HEADERS = [
    "timestamp", "tier", "category", "topic", "temperature",
    "system_prompt", "user_prompt",
    "A_model", "A_output", "A_용어", "A_톤", "A_포맷", "A_환각없음", "A_총점",
    "B_model", "B_output", "B_용어", "B_톤", "B_포맷", "B_환각없음", "B_총점",
    "C_model", "C_output", "C_용어", "C_톤", "C_포맷", "C_환각없음", "C_총점",
    "winner",
]

# ─── 주제 목록 ────────────────────────────────────────────────────────────────

TOPICS_GENERAL = [
    "사주풀이 도입", "사주 기본 판", "나의 핵심 성향", "오행 밸런스",
    "반복 패턴", "재물운", "일/직업운", "관계/연애운", "올해 운", "종합 처방",
]

TOPICS_JAMKON = [
    "최애와 나의 궁합", "차기작 성적 (배우)", "이번 시즌 성적 (운동선수)",
    "최애 현재 운세", "팬심 케미 분석",
]

# ─── 기본 사주 데이터 ──────────────────────────────────────────────────────────

DEFAULT_SAJU_GENERAL = (
    "사주팔자: 을해년 무인월 갑자일 병인시\n"
    "일간: 갑목(나무 기운이 강한 사람) / 오행 분포: 목3 화2 토1 금1 수1\n"
    "십성: 정관2 식신1 편재1\n"
    "현재 대운: 32~41세 신유 대운(금의 기운이 강한 시기)\n"
    "태양궁: 물고기자리 / 달궁: 전갈자리 / 상승궁(어센던트): 처녀자리"
)

DEFAULT_SAJU_JAMKON = (
    "[나의 사주]\n"
    "사주팔자: 을해년 무인월 갑자일 병인시\n"
    "일간: 갑목 / 오행: 목3 화2 토1 금1 수1\n"
    "태양궁: 물고기자리\n\n"
    "[최애/대상 정보]\n"
    "이름: (입력)\n"
    "생년월일: (입력, 사주 계산 가능 시 아래 추가)\n"
    "사주팔자: (있으면 입력)"
)

# ─── 시스템 프롬프트 ────────────────────────────────────────────────────────────

DEFAULT_SYSTEM_PROMPT = """\
당신은 사주와 서양 점성술을 함께 해석해 주는 따뜻하고 친근한 상담가입니다.
제공된 사주와 점성술 데이터는 이미 정확히 계산된 값입니다. 당신의 역할은 이 데이터를 바탕으로 해석하는 것입니다.

말투와 표현 원칙:
- 어렵고 딱딱한 한자어 대신 누구나 이해할 수 있는 일상적인 언어로 풀어서 설명하세요.
  예) '정관'→'나를 좋아하고 챙겨주는 안정적인 인연', '식신'→'내 매력과 표현력', '편재'→'예상치 못한 횡재나 활발한 돈의 흐름'
- 20~30대 여성이 편하게 읽을 수 있는 친근하고 따뜻한 말투를 사용하세요.
- "~할 수 있어요", "~해보세요", "~인 경향이 있어요" 같이 부드럽게 제안하는 표현을 쓰세요.
- 단정적인 예언("반드시 ~한다", "절대 ~이다")은 피하고 가능성과 경향을 이야기해 주세요.
- 사주 용어를 처음 사용할 때 괄호 안에 쉬운 설명을 꼭 덧붙여 주세요.
- 의료, 법률, 투자, 생사에 관한 단정적 조언은 하지 마세요.
- 데이터에 없는 천간/지지/십성/행성 배치를 만들어내지 마세요."""

# ─── 주제별 유저 템플릿 ────────────────────────────────────────────────────────

def _base(topic: str) -> str:
    return f"""\
## 사주 & 점성술 데이터 (계산 완료)
{{saju_data}}

- 분석 주제: {topic}

---

## 출력 형식 (반드시 이 순서와 구조로 작성)

## [무료 영역]

**지금 내 흐름은 이래요** (2~3문장, 어려운 용어 없이 쉽게)

**한 줄 결론**

---

## [유료 영역]

"""

TOPIC_TEMPLATES: dict[str, dict[str, str]] = {
    "일반": {
        "사주풀이 도입": _base("사주풀이 도입") + """\
**당신은 이런 사람이에요**
- 첫인상 키워드 3가지 (한 줄씩 쉬운 설명 포함)
- 사람들이 당신에게서 느끼는 에너지
- 당신 자신도 몰랐던 숨은 매력 한 가지

**사주가 말해주는 당신의 본질**
- 일간(나를 대표하는 하늘 기운)이 어떤 사람인지 쉽게 풀어서
- 가장 강한 오행(자연 에너지)이 성격에 미치는 영향
- 별자리(태양궁·달궁)와 사주가 같은 방향을 가리키는 부분

**더 알아보고 싶어질 포인트**
- "이게 나였구나" 싶은 공감 포인트 1~2가지
- 앞으로 더 깊게 볼 수 있는 주제 3가지 추천""",

        "사주 기본 판": _base("사주 기본 판") + """\
**나의 사주 8글자 완전 해석**

📅 년주 (태어난 해 — 조상·사회적 첫인상):
- 천간: (글자 + 쉬운 의미)
- 지지: (글자 + 쉬운 의미)
- 이것이 나에게 의미하는 것

📅 월주 (태어난 달 — 부모·청년기 환경):
- 천간/지지 해석
- 이것이 나에게 의미하는 것

📅 일주 (태어난 날 — 나 자신·배우자):
- 일간(나의 본질): 쉬운 해석
- 일지(나의 내면·배우자): 쉬운 해석

📅 시주 (태어난 시 — 자식·노년·숨은 욕구):
- 천간/지지 해석
- 이것이 나에게 의미하는 것

**전체 사주의 조합 에너지**
- 4개 기둥이 합쳐져 만들어내는 나만의 특성
- 가장 강한 기운과 약한 기운의 균형""",

        "나의 핵심 성향": _base("나의 핵심 성향") + """\
**나의 성향 유형과 점수**
- 유형명: (직관적인 표현, 예: "조용한 리더형", "공감 에너지형")
- 핵심 키워드: 3가지

**일간(나의 본질 에너지) 완전 분석**
- 나는 어떤 종류의 에너지를 가진 사람인가
- 이 에너지가 가진 최대 강점 2가지
- 이 에너지가 만들어내는 약점 2가지 (부드럽게)

**겉모습 vs 내면**
- 사람들 눈에 보이는 나 (상승궁·월주 기반)
- 혼자 있을 때의 진짜 나 (달궁·일지 기반)
- 두 모습의 괴리에서 오는 피로감 해소 팁

**관계에서의 나**
- 친한 사람에게 vs 처음 만나는 사람에게 다른 점
- 나와 잘 맞는 에너지 유형
- 갈등 패턴과 해결 방향""",

        "오행 밸런스": _base("오행 밸런스") + """\
**나의 오행 분포 (에너지 지도)**

🌳 목(木, 나무) ■■■□□  (3/8)
🔥 화(火, 불)  ■■□□□  (2/8)
🌍 토(土, 흙)  ■□□□□  (1/8)
⚔️ 금(金, 쇠)  ■□□□□  (1/8)
💧 수(水, 물)  ■□□□□  (1/8)

**가장 강한 오행 — 나의 주력 에너지**
- 이 에너지가 강할 때 나타나는 좋은 점
- 이 에너지가 과할 때 나타나는 신호들

**가장 약한 오행 — 채워줘야 할 에너지**
- 부족할 때 일상에서 느끼는 감각들
- 이 에너지를 보충하는 생활 팁 (색깔, 방향, 음식, 활동)

**오행 균형 맞추기**
- 지금 당장 시도할 수 있는 작은 변화 3가지
- 이번 대운(큰 흐름)이 균형에 미치는 영향""",

        "반복 패턴": _base("반복 패턴") + """\
**내 삶에서 반복되는 패턴 3가지**

패턴 1️⃣ (제목: 예 — "늘 나중에 후회하는 선택들")
- 구체적으로 어떤 상황에서 반복되는지
- 사주에서 이 패턴이 생기는 이유
- 이 패턴을 인식하고 멈추는 방법

패턴 2️⃣
- 동일 구조

패턴 3️⃣
- 동일 구조

**패턴에서 자유로워지는 실천법**
- 이번 대운(현재 큰 흐름) 동안 변화 가능한 가장 쉬운 것
- 7일 안에 할 수 있는 작은 첫걸음""",

        "재물운": _base("재물운") + """\
**나의 재물운 유형과 점수**
- 유형: (예: "꾸준히 모으는 타입", "한방 기회형", "쓰면서 버는 순환형")
- 점수: 00점 / 100점

**지금 재물 흐름은 이래요**
- 현재 대운(큰 흐름)이 돈에 미치는 영향
- 오행 중 재물과 관련된 에너지 상태
- 별자리가 돈을 대하는 방식에 미치는 영향

**시기별 재물 가이드**

🗓️ 이번 주 (7일):
- 특징:
- 주의할 점:
- 이렇게 해보세요:

📅 이번 달 (30일):
- 특징:
- 주의할 점:
- 이렇게 해보세요:

✅ 지금 이건 해보세요 (3가지)
🚫 이건 잠깐 멈춰요 (2~3가지, 이유 포함)""",

        "일/직업운": _base("일/직업운") + """\
**나의 커리어 유형과 점수**
- 유형: (예: "전문가형", "기획·창작형", "대인관계 중심형")
- 점수: 00점 / 100점

**나에게 맞는 일의 방식**
- 사주 기반 적합 직업 유형 (구체적으로)
- 혼자 vs 팀, 루틴 vs 유연, 안정 vs 도전 성향
- 잘 맞는 직장 문화·환경

**지금 커리어 흐름은 이래요**
- 현재 대운이 일에 미치는 영향
- 기회가 열리는 분야

**시기별 일/직업 가이드**

🗓️ 이번 주 (7일):
- 특징 / 주의할 점 / 이렇게 해보세요

📅 이번 달 (30일):
- 특징 / 주의할 점 / 이렇게 해보세요

✅ 추천 행동 (3가지) vs 🚫 피해야 할 행동 (2가지)""",

        "관계/연애운": _base("관계/연애운") + """\
**나의 연애 유형과 점수**
- 유형: (예: "천천히 스며드는 타입", "먼저 표현하는 타입", "기다리는 타입")
- 점수: 00점 / 100점

**나의 연애 패턴**
- 사주 기반 연애 성향 (쉬운 말로)
- 반복되는 연애 패턴 2가지
- 나와 잘 맞는 상대 유형 (오행·에너지 기반)

**지금 애정운 흐름**
- 현재 대운이 연애에 미치는 영향
- 별자리(달궁 에너지)가 알려주는 지금 내 감정 상태

**시기별 관계 가이드**

🗓️ 이번 주 (7일):
- 특징 / 주의할 점 / 이렇게 해보세요

📅 이번 달 (30일):
- 특징 / 주의할 점 / 이렇게 해보세요

✅ 지금 연애에서 해볼 것 vs 🚫 잠깐 멈출 것""",

        "올해 운": _base("올해 운") + """\
**올해 전체 흐름 요약**
- 올해의 핵심 에너지 키워드 3가지
- 대운(큰 흐름)과 올해 세운(연간 흐름)의 조합이 만드는 분위기

**상반기 vs 하반기**

📌 상반기 (1~6월):
- 전체 분위기
- 기회가 오는 시기/분야
- 주의해야 할 시기와 이유

📌 하반기 (7~12월):
- 전체 분위기
- 기회가 오는 시기/분야
- 주의해야 할 시기와 이유

**올해 꼭 기억할 포인트**
- 절대 놓치면 안 되는 기회 시기
- 에너지를 아껴야 할 시기
- 올해 나에게 가장 중요한 행동 1가지""",

        "종합 처방": _base("종합 처방") + """\
**나의 삶 균형 진단표**

🌳 커리어/성취: 00점
❤️ 관계/연애: 00점
💰 재물/돈: 00점
🧘 몸/마음 건강: 00점
🌱 자기 성장: 00점

**지금 당장 바꿔야 할 것 TOP 3**

1️⃣ (제목):
- 왜 지금 이게 중요한지
- 구체적으로 어떻게

2️⃣ (제목): 동일 구조

3️⃣ (제목): 동일 구조

**3개월 실천 로드맵**

Month 1 (지금~한 달): 이것부터
Month 2: 이걸 추가로
Month 3: 이 정도면 변화가 보여요

**한 줄 종합 처방전**
> (이 사람의 사주와 별자리를 모두 보고 내린 핵심 조언 한 문장)""",
    },

    "잼컨": {
        "최애와 나의 궁합": """\
## 사주 & 점성술 데이터 (계산 완료)
{saju_data}

- 분석 주제: 최애와 나의 궁합

---

## 출력 형식

## [무료 영역]

**우리 둘의 에너지, 한 줄로는 이래요** (2~3문장)

**궁합 한 줄 결론**

---

## [유료 영역]

**궁합 점수와 유형**
- 점수: 00점 / 100점
- 궁합 유형: (예: "서로 당기는 자석형", "보완형 파트너", "운명적 끌림형")

**오행 케미 분석**
- 나와 최애의 오행 에너지가 어떻게 만나는지
- 서로 끌리는 이유를 에너지로 설명
- 갈등이 생길 수 있는 지점과 해소 방법

**팬으로서의 인연 에너지**
- 팬과 스타 사이에 흐르는 에너지의 특징
- 이 인연이 나에게 미치는 긍정적 영향
- 덕질이 나의 사주 에너지와 어울리는 이유

**지금 이 시기의 케미**
- 이번 달 팬심 에너지 흐름
- 좋은 소식이 올 가능성이 높은 시기
- 덕질 에너지를 가장 잘 쓸 수 있는 방법""",

        "차기작 성적 (배우)": """\
## 대상 정보 (계산 완료)
{saju_data}

- 분석 주제: 차기작 성적 분석 (배우)

---

## 출력 형식

## [무료 영역]

**지금 이 배우의 운기 흐름** (2~3문장)

**한 줄 결론**

---

## [유료 영역]

**차기작 성공 가능성 점수**
- 점수: 00점 / 100점
- 현재 운기 유형: (예: "상승 직전", "이미 피크", "재충전 시기")

**대운 & 세운 흐름 분석**
- 현재 대운(큰 흐름)이 커리어에 미치는 영향
- 올해 연간 운기의 특징
- 이 시기에 잘 맞는 작품 유형/장르

**시기별 성공 가능성**

🗓️ 단기 (3개월):
- 흥행 에너지가 강한 시기 / 주의해야 할 시기

📅 중기 (1년):
- 커리어의 전환점이 될 수 있는 시기

✅ 응원 포인트 (지금 이 배우에게 좋은 에너지가 오는 이유)
⚠️ 살짝 걱정되는 포인트 (부드럽게)""",

        "이번 시즌 성적 (운동선수)": """\
## 대상 정보 (계산 완료)
{saju_data}

- 분석 주제: 이번 시즌 성적 분석 (운동선수)

---

## 출력 형식

## [무료 영역]

**지금 이 선수의 운기** (2~3문장)

**한 줄 결론**

---

## [유료 영역]

**이번 시즌 성적 가능성 점수**
- 점수: 00점 / 100점
- 현재 운기 유형: (예: "절정기", "성장 중", "회복·재정비 시기")

**시즌 흐름 분석**

📌 시즌 전반부:
- 에너지 특징 / 기대 포인트

📌 시즌 후반부:
- 에너지 특징 / 기대 포인트

**주의 시기와 회복 전략**
- 슬럼프가 올 수 있는 시기와 이유
- 부상·컨디션 관리가 특히 중요한 때
- 이 시기를 버티는 방법

**팬으로서 알아두면 좋은 것**
- 지금 이 선수를 응원하는 에너지가 좋은 이유
- 기대해도 좋은 포인트""",

        "최애 현재 운세": """\
## 대상 정보 (계산 완료)
{saju_data}

- 분석 주제: 최애 현재 운세

---

## 출력 형식

## [무료 영역]

**지금 최애에게 어떤 에너지가 흐르고 있어요** (2~3문장)

**한 줄 결론**

---

## [유료 영역]

**현재 운기 점수**
- 점수: 00점 / 100점
- 지금 어떤 시기인가: (예: "도약 직전", "안정과 충전의 시기", "새로운 챕터 시작")

**이번 달 최애의 에너지**

💼 일·커리어 방면:

🤝 대인관계·주변 환경:

🧘 개인 컨디션·마음 상태:

**좋은 소식이 올 가능성**
- 기대해볼 수 있는 시기와 분야
- 팬들에게 좋은 소식이 올 타이밍

**팬으로서 보내는 응원 에너지**
- 지금 이 시기에 최애에게 필요한 것
- 팬심이 가장 잘 전달될 수 있는 방법""",

        "팬심 케미 분석": """\
## 사주 & 점성술 데이터 (계산 완료)
{saju_data}

- 분석 주제: 팬심 케미 분석

---

## 출력 형식

## [무료 영역]

**이 덕질이 나에게 의미 있는 이유** (2~3문장)

**한 줄 결론**

---

## [유료 영역]

**팬심 케미 점수**
- 점수: 00점 / 100점
- 케미 유형: (예: "영감을 주는 인연", "운명적 끌림형", "에너지 충전형")

**이 덕질이 내 사주와 맞는 이유**
- 나의 오행 에너지와 이 스타의 에너지가 만나는 방식
- 덕질이 내 삶에 미치는 긍정적 에너지
- 이 인연이 나의 성장에 기여하는 방식

**덕질 타이밍 분석**
- 지금 팬심 에너지가 강한 이유
- 더 좋아질 수 있는 시기
- 덕질 에너지를 건강하게 유지하는 방법

**나에게 맞는 덕질 스타일**
- 사주로 본 나의 팬십 유형
- 가장 잘 맞는 응원 방식
- 이 덕질로 얻을 수 있는 것""",
    },
}


# ─── 데이터 클래스 ─────────────────────────────────────────────────────────

@dataclass
class ModelResult:
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: Optional[str] = None
    price_in: float = 0.0
    price_out: float = 0.0

    def cost(self) -> float:
        return (self.input_tokens * self.price_in + self.output_tokens * self.price_out) / 1_000_000

    def cost_label(self) -> str:
        if self.error:
            return ""
        c = self.cost()
        return f"💰 ${c:.4f}  (입력 {self.input_tokens:,} tok + 출력 {self.output_tokens:,} tok)"


# ─── API 호출 ───────────────────────────────────────────────────────────────

def call_claude(model_id: str, system: str, user: str, api_key: str,
                temperature: float, price_in: float, price_out: float) -> ModelResult:
    # 크레딧 충전 후 app.py 상단의 CLAUDE_ENABLED = True 로 변경하면 재활성화
    if not CLAUDE_ENABLED:
        return ModelResult(error="Claude 비활성화 — CLAUDE_ENABLED = True 로 변경하면 재활성화")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_id,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return ModelResult(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            price_in=price_in,
            price_out=price_out,
        )
    except Exception as e:
        return ModelResult(error=str(e))


def call_gemini(model_id: str, system: str, user: str, api_key: str,
                temperature: float, price_in: float, price_out: float) -> ModelResult:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model_id,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=8192,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",       threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                ],
            ),
            contents=user,
        )
        usage = resp.usage_metadata
        return ModelResult(
            text=resp.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=usage.candidates_token_count,
            price_in=price_in,
            price_out=price_out,
        )
    except Exception as e:
        return ModelResult(error=str(e))


def call_gpt(model_id: str, system: str, user: str, api_key: str,
             temperature: float, price_in: float, price_out: float) -> ModelResult:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model_id,
            max_completion_tokens=8192,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return ModelResult(
            text=resp.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            price_in=price_in,
            price_out=price_out,
        )
    except Exception as e:
        return ModelResult(error=str(e))


CALLERS = {"claude": call_claude, "gemini": call_gemini, "gpt": call_gpt}


# ─── CSV 저장 ────────────────────────────────────────────────────────────────

def save_evaluation_to_csv(
    tier_key: str, category: str, topic: str, temperature: float,
    system_prompt: str, user_prompt: str,
    order: list, tier: dict, results: dict, scores: dict,
) -> None:
    labels = ["A", "B", "C"]
    winner_label = max(scores, key=lambda lbl: scores[lbl]["총점"]) if scores else ""
    winner_mk = order[labels.index(winner_label)] if winner_label in labels else ""

    row: dict = {
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tier":          tier_key,
        "category":      category,
        "topic":         topic,
        "temperature":   temperature,
        "system_prompt": system_prompt,
        "user_prompt":   user_prompt,
        "winner":        tier[winner_mk]["name"] if winner_mk else "",
    }
    for label, mk in zip(labels, order):
        res = results.get(mk, ModelResult(error="없음"))
        s = scores.get(label, {})
        row[f"{label}_model"]   = tier[mk]["name"]
        row[f"{label}_output"]  = res.text if not res.error else f"[오류] {res.error}"
        row[f"{label}_용어"]    = s.get("용어", 0)
        row[f"{label}_톤"]      = s.get("톤", 0)
        row[f"{label}_포맷"]    = s.get("포맷", 0)
        row[f"{label}_환각없음"] = s.get("환각 없음", 0)
        row[f"{label}_총점"]    = s.get("총점", 0)

    file_exists = CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─── 유틸 ───────────────────────────────────────────────────────────────────

def split_sections(text: str) -> tuple[str, str]:
    marker = "## [유료 영역]"
    idx = text.find(marker)
    if idx == -1:
        return text.strip(), ""
    return text[:idx].strip(), text[idx:].strip()


def run_all_models(tier: dict, system: str, user: str,
                   keys: dict, temperature: float) -> dict:
    def call(mk: str):
        if mk == "claude" and not CLAUDE_ENABLED:
            return mk, ModelResult(error="Claude 비활성화 — CLAUDE_ENABLED = True 로 변경")
        key = keys.get(mk, "")
        if not key:
            return mk, ModelResult(error="API 키 없음")
        info = tier[mk]
        return mk, CALLERS[mk](
            info["id"], system, user, key, temperature,
            info["price_in"], info["price_out"],
        )

    results: dict[str, ModelResult] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(call, mk) for mk in MODEL_KEYS]
        for future in concurrent.futures.as_completed(futures):
            mk, res = future.result()
            results[mk] = res
    return results


def reset_run():
    for key in ("results", "model_order", "scores", "revealed", "ran",
                "balloons_done", "last_sys_prompt", "last_user_prompt"):
        st.session_state.pop(key, None)
    st.session_state.results        = {}
    st.session_state.model_order    = None
    st.session_state.scores         = {}
    st.session_state.revealed       = False
    st.session_state.ran            = False
    st.session_state.balloons_done  = False
    st.session_state.last_sys_prompt  = ""
    st.session_state.last_user_prompt = ""


# ─── 세션 초기화 ────────────────────────────────────────────────────────────

def init_session():
    defaults: dict = {
        "results":           {},
        "model_order":       None,
        "scores":            {},
        "revealed":          False,
        "ran":               False,
        "balloons_done":     False,
        "last_tier":         None,
        "last_topic_key":    "",
        "last_sys_prompt":   "",
        "last_user_prompt":  "",
        # 편집 가능 프롬프트 — 주제 변경 시 이 값을 갱신한 뒤 text_area가 읽어감
        "sys_prompt_draft":  DEFAULT_SYSTEM_PROMPT,
        "tmpl_draft":        TOPIC_TEMPLATES["일반"]["사주풀이 도입"],
        "tmpl_version":      0,   # 초기화 버튼 클릭 시 증가 → text_area key 변경 → 강제 리렌더
        "sys_version":       0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─── UI 컴포넌트 ────────────────────────────────────────────────────────────

def render_output_col(label: str, mk: str, tier: dict,
                      result: ModelResult, show_model: bool):
    header = tier[mk]["name"] if show_model else f"후보 {label}"
    st.subheader(header)

    if result.error:
        st.error(f"❌ {result.error}")
        return

    free_part, paid_part = split_sections(result.text)
    st.markdown(free_part)
    if paid_part:
        with st.expander("유료 영역 펼치기"):
            st.markdown(paid_part)

    st.caption(result.cost_label())

    with st.expander("전체 텍스트 복사"):
        st.code(result.text, language=None)


def render_eval_form(labels: list[str], tier_key: str, category: str,
                     topic: str, temperature: float, order: list,
                     tier: dict, results: dict):
    st.divider()
    st.subheader("평가 (섹션 4 — 4가지 기준)")
    st.caption("모델명 모른 채 채점 → 제출 시 정체 공개됩니다.")

    cols = st.columns(3)
    for col, label in zip(cols, labels):
        with col:
            st.markdown(f"**후보 {label}**")
            st.slider("사주 용어 자연스러움", 1, 5, 3, key=f"s_term_{label}")
            st.slider("한국어 톤 적합성 (20-30대 여성)", 1, 5, 3, key=f"s_tone_{label}")
            st.slider("포맷 준수도", 1, 5, 3, key=f"s_fmt_{label}")
            st.radio(
                "환각 여부 (데이터에 없는 글자 생성)",
                ["없음 (+1)", "있음 (0)"],
                key=f"s_halluc_{label}",
                horizontal=True,
            )

    if st.button("평가 제출 → 정체 공개", type="primary", use_container_width=True):
        scores: dict = {}
        for label in labels:
            term      = st.session_state[f"s_term_{label}"]
            tone      = st.session_state[f"s_tone_{label}"]
            fmt       = st.session_state[f"s_fmt_{label}"]
            halluc_ok = 1 if st.session_state[f"s_halluc_{label}"].startswith("없음") else 0
            scores[label] = {
                "용어": term, "톤": tone, "포맷": fmt,
                "환각 없음": halluc_ok,
                "총점": term + tone + fmt + halluc_ok,
            }
        st.session_state.scores   = scores
        st.session_state.revealed = True
        st.session_state.balloons_done = False

        try:
            save_evaluation_to_csv(
                tier_key=tier_key, category=category, topic=topic,
                temperature=temperature,
                system_prompt=st.session_state.last_sys_prompt,
                user_prompt=st.session_state.last_user_prompt,
                order=order, tier=tier, results=results, scores=scores,
            )
            st.toast(f"✅ 평가 저장 완료 → {CSV_PATH.name}")
        except Exception as e:
            st.warning(f"⚠️ CSV 저장 실패: {e}")

        st.rerun()


def render_results(order: list[str], tier: dict, results: dict, scores: dict):
    import pandas as pd

    if not st.session_state.balloons_done:
        st.balloons()
        st.session_state.balloons_done = True

    st.divider()
    st.subheader("결과 공개")

    labels = ["A", "B", "C"]
    for label, mk in zip(labels, order):
        st.write(f"**후보 {label}** = {tier[mk]['name']}")

    st.divider()

    rows = []
    for label, mk in zip(labels, order):
        s = scores[label]
        rows.append({
            "모델": tier[mk]["name"],
            "용어": s["용어"], "톤": s["톤"], "포맷": s["포맷"],
            "환각 없음": s["환각 없음"],
            "총점 (16점)": s["총점"],
        })

    df = pd.DataFrame(rows).set_index("모델")
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df["총점 (16점)"])

    max_score = df["총점 (16점)"].max()
    top = df[df["총점 (16점)"] == max_score].index.tolist()
    if len(top) == 1:
        st.success(f"🏆 팀 평가 1위: **{top[0]}**")
    else:
        st.info(f"동점 ({', '.join(top)}) — 단가가 낮은 쪽을 우선 고려하세요.")

    st.divider()
    total_usd = sum(
        results[mk].cost()
        for mk in MODEL_KEYS
        if results.get(mk) and not results[mk].error
    )
    st.metric("3개 모델 총 비용", f"${total_usd:.4f}", f"≈ ₩{total_usd * USD_TO_KRW:.1f}")

    if CSV_PATH.exists():
        with open(CSV_PATH, "rb") as f:
            st.download_button(
                "📥 평가 기록 전체 다운로드 (CSV)",
                data=f.read(),
                file_name="evaluation_results.csv",
                mime="text/csv",
            )


# ─── 메인 ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="사주 LLM 블라인드 비교", layout="wide", page_icon="🔮")
    init_session()

    st.title("🔮 사주 LLM 블라인드 비교 데모")
    st.caption("AI 엔지니어링 계획서 섹션 4 | 동일 프롬프트 → 3개 모델 동시 호출 → 팀이 모델명 모른 채 채점")

    if not CLAUDE_ENABLED:
        st.warning("⚠️ Claude 비활성화 상태입니다. 크레딧 충전 후 `app.py` 상단의 `CLAUDE_ENABLED = True` 로 변경하면 재활성화됩니다.")

    # ── 티어 선택 ─────────────────────────────────────────────────────────
    tier_key = st.radio(
        "비교 티어",
        list(TIERS.keys()),
        format_func=lambda k: TIERS[k]["label"],
        horizontal=True,
    )
    active_tier = TIERS[tier_key]

    if st.session_state.last_tier != tier_key:
        reset_run()
        st.session_state.last_tier = tier_key

    with st.expander("현재 티어 모델 확인"):
        cols = st.columns(3)
        for col, mk in zip(cols, MODEL_KEYS):
            info = active_tier[mk]
            disabled = " (비활성화)" if mk == "claude" and not CLAUDE_ENABLED else ""
            col.metric(
                label=info["name"] + disabled,
                value=f"in ${info['price_in']} / out ${info['price_out']}",
                help="$/MTok",
            )

    st.divider()

    # ── 사이드바 — API 키 & 시스템 프롬프트 편집 ──────────────────────────
    with st.sidebar:
        st.header("API 키 상태")
        claude_key = os.getenv("ANTHROPIC_API_KEY", "")
        gemini_key = os.getenv("GOOGLE_API_KEY", "")
        gpt_key    = os.getenv("OPENAI_API_KEY", "")
        claude_status = ("⏸️ 비활성화" if not CLAUDE_ENABLED
                         else ("✅" if claude_key else "❌ .env 미설정"))
        st.write("Claude", claude_status)
        st.write("Gemini", "✅" if gemini_key else "❌ .env 미설정")
        st.write("GPT",    "✅" if gpt_key    else "❌ .env 미설정")
        st.caption("키는 blind_compare/.env 파일에서 관리합니다.")
        st.divider()
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        st.caption("리포트 권장값 0.7")
        st.divider()

        # 시스템 프롬프트 편집
        st.subheader("🛠 시스템 프롬프트 편집")
        if st.button("기본값으로 초기화", key="reset_sys_btn"):
            st.session_state.sys_prompt_draft = DEFAULT_SYSTEM_PROMPT
            st.session_state.sys_version += 1

        new_sys = st.text_area(
            "시스템 프롬프트",
            value=st.session_state.sys_prompt_draft,
            height=280,
            key=f"sys_area_v{st.session_state.sys_version}",
            label_visibility="collapsed",
        )
        st.session_state.sys_prompt_draft = new_sys

    api_keys = {"claude": claude_key, "gemini": gemini_key, "gpt": gpt_key}

    # ── 카테고리 & 주제 선택 ──────────────────────────────────────────────
    st.subheader("분석 설정")
    category_label = st.radio(
        "카테고리",
        ["일반 사주", "잼컨 사주 ✨"],
        horizontal=True,
    )
    category_key = "일반" if "일반" in category_label else "잼컨"
    topic_list   = TOPICS_GENERAL if category_key == "일반" else TOPICS_JAMKON
    topic        = st.selectbox("분석 주제", topic_list)

    topic_state_key = f"{category_key}::{topic}"
    if st.session_state.last_topic_key != topic_state_key:
        st.session_state.tmpl_draft   = TOPIC_TEMPLATES[category_key][topic]
        st.session_state.last_topic_key = topic_state_key
        st.session_state.tmpl_version += 1
        reset_run()

    # ── 사주 데이터 & 유저 템플릿 편집 ────────────────────────────────────
    default_saju = DEFAULT_SAJU_JAMKON if category_key == "잼컨" else DEFAULT_SAJU_GENERAL

    col_l, col_r = st.columns([1, 1])
    with col_l:
        saju_data = st.text_area(
            "사주 & 점성술 데이터",
            value=default_saju,
            height=200,
            key=f"saju_{category_key}",
        )
    with col_r:
        if st.button("주제 기본 템플릿으로 초기화", key="reset_tmpl_btn"):
            st.session_state.tmpl_draft = TOPIC_TEMPLATES[category_key][topic]
            st.session_state.tmpl_version += 1

        new_tmpl = st.text_area(
            "유저 프롬프트 템플릿 (직접 편집 가능)",
            value=st.session_state.tmpl_draft,
            height=170,
            key=f"tmpl_area_v{st.session_state.tmpl_version}",
        )
        st.session_state.tmpl_draft = new_tmpl

    system_prompt = st.session_state.sys_prompt_draft
    user_prompt   = st.session_state.tmpl_draft.replace("{saju_data}", saju_data)


    if st.button("모델 3개에 동시 요청", type="primary", use_container_width=True):
        reset_run()
        order = MODEL_KEYS.copy()
        random.shuffle(order)
        st.session_state.model_order      = order
        st.session_state.ran              = True
        st.session_state.last_tier        = tier_key
        st.session_state.last_sys_prompt  = system_prompt
        st.session_state.last_user_prompt = user_prompt

        labels = ["A", "B", "C"]
        st.divider()
        cols = st.columns(3)
        spinners: dict   = {}
        containers: dict = {}
        for col, label in zip(cols, labels):
            with col:
                st.subheader(f"후보 {label}")
                spinners[label]   = st.empty()
                spinners[label].info("⏳ 호출 중...")
                containers[label] = st.container()

        def _call(mk: str):
            if mk == "claude" and not CLAUDE_ENABLED:
                return mk, ModelResult(error="Claude 비활성화 — CLAUDE_ENABLED = True 로 변경")
            key = api_keys.get(mk, "")
            if not key:
                return mk, ModelResult(error="API 키 없음")
            info = active_tier[mk]
            return mk, CALLERS[mk](
                info["id"], system_prompt, user_prompt, key, temperature,
                info["price_in"], info["price_out"],
            )

        results: dict[str, ModelResult] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_call, mk): mk for mk in MODEL_KEYS}
            for future in concurrent.futures.as_completed(futures):
                mk, res = future.result()
                results[mk] = res
                label = labels[order.index(mk)]
                spinners[label].empty()
                with containers[label]:
                    if res.error:
                        st.error(f"❌ {res.error}")
                    else:
                        free_part, paid_part = split_sections(res.text)
                        st.markdown(free_part)
                        if paid_part:
                            with st.expander("유료 영역 펼치기"):
                                st.markdown(paid_part)
                        st.caption(res.cost_label())
                        with st.expander("전체 텍스트 복사"):
                            st.code(res.text, language=None)

        st.session_state.results = results
        render_eval_form(labels, tier_key=tier_key, category=category_key, topic=topic,
                         temperature=temperature, order=order, tier=active_tier, results=results)
        st.stop()

    # ── 출력 & 평가/결과 (평가 제출 후 rerun) ────────────────────────────
    if not st.session_state.ran:
        return

    order: list[str] = st.session_state.model_order
    results: dict    = st.session_state.results
    labels = ["A", "B", "C"]

    st.divider()
    cols = st.columns(3)
    for col, label, mk in zip(cols, labels, order):
        with col:
            render_output_col(label, mk, active_tier,
                              results.get(mk, ModelResult(error="호출 실패")),
                              st.session_state.revealed)

    if not st.session_state.revealed:
        render_eval_form(
            labels,
            tier_key=tier_key, category=category_key, topic=topic,
            temperature=temperature, order=order, tier=active_tier,
            results=results,
        )
    else:
        render_results(order, active_tier, results, st.session_state.scores)


if __name__ == "__main__":
    main()
