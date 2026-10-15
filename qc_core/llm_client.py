"""
Gemini API(google-genai) 연동 — 얇은 래퍼.

핵심 CNN 분류/트리아지 로직(wafer_qc_tool.py)은 LLM 없이도 완전히 동작한다.
이 모듈이 제공하는 두 기능은 어디까지나 '부가' 기능이다:
  1. summarize_batch()  — 배치(여러 장) 트리아지 결과를 보고 "오늘 특이사항" 브리핑 작성
  2. answer_kb_question() — 원인 지식베이스(JSON)에 한정해서 자연어 질문에 답하는 RAG형 챗봇

API 키가 없거나 호출이 실패해도 앱 자체는 죽지 않도록, 모든 함수는 예외를 잡아서
사람이 읽을 수 있는 에러 메시지 문자열을 돌려준다 (호출부에서 별도 try/except 불필요).

키 설정 — 두 가지 방식을 자동으로 둘 다 지원한다 (로컬 실행 / Streamlit Cloud 배포 양쪽에서
코드 수정 없이 그대로 동작하게 하기 위함):
  - 로컬 실행: 프로젝트 루트(wafer_qc_app/)에 .env 파일을 만들고
        GEMINI_API_KEY=여기에_발급받은_키
    한 줄만 넣으면 된다. .env는 .gitignore 처리되어 있고, 압축해서 남에게 보내거나 GitHub에
    올릴 때는 반드시 .env를 빼고 보낼 것 — API 키는 비밀번호와 동급이다.
  - Streamlit Community Cloud 배포: 앱 설정(Settings) → Secrets 메뉴에
        GEMINI_API_KEY = "여기에_발급받은_키"
    형식으로 등록하면 된다. 이 경우 .env 파일이 없어도(=저장소에 올라가지 않아도) 정상 동작한다.
"""
import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_NAME = "gemini-3.8-flash"

_dotenv_loaded = False


def _ensure_env_loaded():
    global _dotenv_loaded
    if _dotenv_loaded:
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(APP_DIR / ".env")
    except ImportError:
        pass
    _dotenv_loaded = True


def _get_api_key():
    """.env(로컬) 또는 st.secrets(Streamlit Cloud 배포) 중 어디서든 키를 찾아본다."""
    _ensure_env_loaded()
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:  # noqa: BLE001 — secrets.toml이 아예 없는 로컬 환경 등
        return None


def is_configured():
    """GEMINI_API_KEY가 설정돼 있는지만 빠르게 확인 (실제 호출은 안 함)."""
    return bool(_get_api_key())


def _get_client():
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY가 설정되지 않았습니다. 로컬이면 wafer_qc_app 폴더에 .env 파일을 만들고 "
            "GEMINI_API_KEY=발급받은키 를, Streamlit Cloud 배포라면 앱 Settings → Secrets에 "
            "같은 키를 등록해주세요."
        )
    from google import genai
    return genai.Client(api_key=api_key)


def _call(prompt, max_output_tokens=800):
    """google-genai 호출 1회. 실패 시 예외를 그대로 올림(호출부에서 메시지로 변환)."""
    client = _get_client()
    interaction = client.interactions.create(model=MODEL_NAME, input=prompt)
    return interaction.output_text.strip()


def summarize_batch(action_counts, pattern_counts, process_counts, n_total, thresholds):
    """배치 트리아지 결과(집계 수치)를 받아 '오늘 특이사항' 브리핑 생성.

    원본 웨이퍼 이미지는 전혀 보내지 않는다 — 이미 계산된 집계 숫자만 LLM에 전달.
    과장/추측을 막기 위해 '주어진 숫자 밖의 사실을 지어내지 말라'고 명시적으로 지시.
    """
    try:
        data = {
            "총_웨이퍼_수": n_total,
            "판정_건수": action_counts,
            "예측_패턴_분포": pattern_counts,
            "불량_판정의_관련_공정_빈도": process_counts,
            "통과_임계값": thresholds.get("pass"),
            "불량_임계값": thresholds.get("fail"),
        }
        prompt = (
            "너는 반도체 팹의 웨이퍼 QC 엔지니어를 보조하는 분석가다. "
            "아래는 CNN 앙상블이 웨이퍼맵 묶음 하나를 트리아지(통과/불량/직접확인필요)한 "
            "집계 결과다. 이 숫자만 근거로 엔지니어에게 보여줄 3~5문장짜리 한국어 브리핑을 써라.\n\n"
            "규칙:\n"
            "- 주어진 숫자에 없는 사실(특정 장비, 원인, 날짜 비교 등)은 절대 지어내지 마라.\n"
            "- 특정 패턴/공정의 비중이 눈에 띄게 크면 그 사실만 담백하게 짚어줘라.\n"
            "- '~일 수 있습니다' 식으로 추측이면 추측이라고 명확히 표시해라.\n"
            "- 과장된 어조, 이모지, 마크다운 헤더 없이 평범한 문장으로 써라.\n\n"
            f"집계 데이터(JSON):\n{json.dumps(data, ensure_ascii=False, indent=2)}"
        )
        return _call(prompt)
    except Exception as e:  # noqa: BLE001
        return f"[AI 브리핑 생성 실패] {e}"


def answer_kb_question(question, cause_map, recent_single_check=None,
                        recent_batch_summary=None, history=None):
    """지식베이스(JSON)에 한정한 RAG형 Q&A. 지식베이스에 없는 내용은 모른다고 답하게 유도.

    recent_single_check/recent_batch_summary를 넘기면, 앱이 방금 '단일 검사'/'배치 분석' 탭에서
    실제로 계산한 판정 결과를 사람이 다시 설명하지 않아도 바로 질문에 쓸 수 있다 — 이게 그냥
    범용 Gemini 챗봇을 쓰는 것과의 차이점이다: 데이터를 복사-붙여넣기 할 필요 없이, 앱 안에서
    방금 계산된 값을 자동으로 같이 넘긴다.
    """
    try:
        kb_text = json.dumps(
            {k: v for k, v in cause_map.items() if not k.startswith("_")},
            ensure_ascii=False, indent=2,
        )
        refs_text = json.dumps(cause_map.get("_references", {}), ensure_ascii=False, indent=2)

        context_lines = [
            "너는 반도체 웨이퍼 결함 원인 지식베이스를 기반으로 질문에 답하는 보조 도구다.",
            "아래 '지식베이스'와 '참고문헌' 안의 내용에만 근거해서 답하라.",
            "지식베이스에 없는 내용을 추측해서 답하지 말고, 모르면 '지식베이스에 없는 내용입니다'라고 솔직히 말해라.",
            "evidence가 literature인지 domain_inference인지 구분해서 답에 반영해라 "
            "(domain_inference는 확정된 근거가 아니라 업계 통용 추론임을 알려줘라).",
            "'방금 단일검사 결과'나 '방금 배치 결과'가 아래에 주어져 있으면, 사용자가 '이거 왜 이렇게 "
            "나왔어', '방금 그거' 같이 지칭할 때 그 데이터를 가리키는 것으로 이해하고 구체적인 "
            "수치(판정, 확신도, 패턴 등)를 인용해서 답해라. 주어지지 않았으면 지어내지 마라.",
            "",
            f"[지식베이스]\n{kb_text}",
            "",
            f"[참고문헌]\n{refs_text}",
        ]
        if recent_single_check:
            context_lines += ["", f"[방금 단일검사 결과]\n{json.dumps(recent_single_check, ensure_ascii=False)}"]
        if recent_batch_summary:
            context_lines += ["", f"[방금 배치 결과 요약]\n{json.dumps(recent_batch_summary, ensure_ascii=False)}"]
        if history:
            context_lines += ["", "[이전 대화]"]
            for role, text in history[-6:]:
                context_lines.append(f"{role}: {text}")

        context_lines += ["", f"[질문]\n{question}"]
        prompt = "\n".join(context_lines)
        return _call(prompt)
    except Exception as e:  # noqa: BLE001
        return f"[AI 답변 생성 실패] {e}"
