"""
WM-811K 웨이퍼맵 결함 분류 + 3단계 트리아지 — Streamlit 데모 앱

실행:
    pip install -r requirements.txt
    streamlit run app.py

사제동행세미나 프로젝트 "WM-811K 웨이퍼맵 결함 분류 AI 도구"의 최종 데모 화면.
핵심 로직은 qc_core/wafer_qc_tool.py (base/wide/deep 3모델 앙상블 + 지식베이스 조회) —
이 부분은 LLM 없이 완전히 동작한다. qc_core/llm_client.py(Gemini API)는 어디까지나
부가 기능(배치 브리핑, 원인 Q&A)이며, API 키가 없어도 단일 검사 탭은 그대로 쓸 수 있다.
테마 색상은 .streamlit/config.toml 에서 관리 (화이트 배경, 미니멀 카드 스타일).
"""
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "qc_core"))

from wafer_qc_tool import WaferQCTool, preprocess_wafer_map_stages  # noqa: E402
from cnn_arch import CLASS_ORDER  # noqa: E402
import llm_client  # noqa: E402
import audit_log  # noqa: E402

MODEL_DIR = APP_DIR / "models"
CONFIG_DIR = APP_DIR / "config"
SAMPLE_DIR = APP_DIR / "sample_wafers"
DEMO_BATCH_DIR = SAMPLE_DIR / "demo_batch"

st.set_page_config(page_title="웨이퍼맵 결함 분류 AI", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------- 색상 팔레트 (화이트/미니멀)
TEXT = "#0F172A"
MUTED = "#64748B"
CARD_BG = "#F7F9FB"
BORDER = "rgba(15,23,42,0.08)"
PLOT_BG = "#F7F9FB"
TEAL = "#0D9488"     # 통과 / 양품 다이
CORAL = "#DC2626"    # 불량 / 불량 다이
AMBER = "#D97706"    # 직접확인필요
NO_DIE = "#CBD5E1"   # 다이 없음(배경)

ACTION_STYLE = {
    "통과": {"color": TEAL, "icon": "✅"},
    "불량": {"color": CORAL, "icon": "⛔"},
    "직접확인필요": {"color": AMBER, "icon": "🔍"},
}

# 영문 모델 이름("base"/"wide"/"deep")을 화면에 그대로 노출하면 브라우저 자동번역이
# "기지"/"와이드"/"깊은" 같은 엉뚱한 단어로 바꿔버리는 경우가 있어, 한글 표기로 고정.
MODEL_DISPLAY_NAME = {"base": "베이스", "wide": "와이드", "deep": "딥"}

WAFER_CMAP = LinearSegmentedColormap.from_list("wafer_map", [NO_DIE, TEAL, CORAL], N=256)

CUSTOM_CSS = f"""
<style>
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }}
    h1 {{
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px;
        margin-bottom: 0.15rem !important;
        color: {TEXT};
    }}
    h1::after {{
        content: "";
        display: block;
        width: 52px;
        height: 3px;
        margin-top: 8px;
        background: linear-gradient(90deg, {TEAL}, {AMBER}, {CORAL});
        border-radius: 4px;
    }}
    h2 {{ font-size: 1.1rem !important; font-weight: 700 !important; color: {TEXT}; }}
    h3 {{ font-size: 0.98rem !important; font-weight: 700 !important; color: {TEXT};
          margin-top: 0.4rem !important; }}
    p, li, label, .stMarkdown {{ font-size: 0.88rem; }}
    [data-testid="stCaptionContainer"], .stCaption {{
        font-size: 0.78rem !important; color: {MUTED} !important;
    }}
    [data-testid="stWidgetLabel"] p {{ font-size: 0.84rem !important; }}
    section[data-testid="stSidebar"] {{
        border-right: 1px solid {BORDER};
    }}
    .qc-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }}
    .qc-badge {{
        padding: 11px 16px;
        border-radius: 10px;
        color: white;
        font-size: 16.5px;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.3px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.12);
        margin-bottom: 12px;
    }}
    .qc-stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .qc-stat {{
        flex: 1;
        min-width: 90px;
        background: white;
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 10px 6px;
        text-align: center;
    }}
    .qc-stat .qc-stat-label {{ font-size: 10.5px; color: {MUTED}; margin-bottom: 5px; }}
    .qc-stat .qc-stat-value {{ font-size: 16px; font-weight: 800; color: {TEAL}; }}
    .qc-bar-label {{
        font-size: 12px; color: {TEXT}; margin-bottom: 3px;
        display: flex; justify-content: space-between;
    }}
    .qc-bar-track {{
        background: rgba(15,23,42,0.07); border-radius: 5px; height: 8px;
        overflow: hidden; margin-bottom: 12px;
    }}
    .qc-bar-fill {{ height: 100%; border-radius: 5px; }}
    .qc-cause-item {{
        padding: 7px 10px;
        border-left: 3px solid {AMBER};
        background: white;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 12.5px;
        line-height: 1.4;
        color: {TEXT};
    }}
    .qc-meta-line {{ color: {MUTED}; font-size: 12px; margin-top: 1px; }}
    .qc-ai-note {{
        font-size: 11px; color: {MUTED}; margin-top: 6px; font-style: italic;
    }}
</style>
"""


@st.cache_resource
def load_tool():
    return WaferQCTool(MODEL_DIR, CONFIG_DIR)


@st.cache_data
def load_cause_map():
    with open(CONFIG_DIR / "pattern_cause_map.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_wafer(raw_arr):
    """화이트 테마에 맞춘 웨이퍼맵 시각화: 배경(연회색)/양품(청록)/불량(레드) 3색 그라디언트."""
    fig, ax = plt.subplots(figsize=(3.3, 3.3), dpi=150)
    fig.patch.set_facecolor(PLOT_BG)
    ax.set_facecolor(PLOT_BG)
    ax.imshow(raw_arr, cmap=WAFER_CMAP, vmin=0, vmax=2)
    ax.axis("off")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def action_badge(action):
    style = ACTION_STYLE.get(action, {"color": "gray", "icon": ""})
    st.markdown(
        f"<div class='qc-badge' style='background:{style['color']};'>"
        f"{style['icon']}&nbsp;&nbsp;{action}</div>",
        unsafe_allow_html=True,
    )


def stat_row(items):
    """items: list of (label, value_str)."""
    cells = "".join(
        f"<div class='qc-stat'><div class='qc-stat-label'>{label}</div>"
        f"<div class='qc-stat-value'>{value}</div></div>"
        for label, value in items
    )
    st.markdown(f"<div class='qc-card'><div class='qc-stat-row'>{cells}</div></div>",
                unsafe_allow_html=True)


def bar_rows(items, color=None):
    """items: list of (label, pct_0_to_100)."""
    color = color or AMBER
    rows = []
    for label, pct in items:
        rows.append(
            f"<div class='qc-bar-label'><span>{label}</span><span>{pct:.1f}%</span></div>"
            f"<div class='qc-bar-track'><div class='qc-bar-fill' "
            f"style='width:{min(100.0, pct):.1f}%;background:{color};'></div></div>"
        )
    st.markdown(f"<div class='qc-card'>{''.join(rows)}</div>", unsafe_allow_html=True)


def process_chips(process_list):
    chips = "".join(
        f"<span style='display:inline-block;background:{CORAL}14;color:{CORAL};"
        f"border:1px solid {CORAL}40;border-radius:999px;padding:3px 10px;"
        f"font-size:11.5px;font-weight:700;margin:0 6px 6px 0;'>{p}</span>"
        for p in process_list
    )
    return (f"<div style='margin-bottom:8px;'>"
            f"<div class='qc-meta-line' style='margin-bottom:5px;'>🏭 관련 공정 (우선 점검 대상)</div>"
            f"{chips}</div>")


def candidate_process_card(candidates_with_process):
    """'직접확인필요'일 때, 확률 순 후보 패턴별로 관련 공정을 참고용으로 보여준다 — 확정된
    '불량'의 process_chips(코랄색)와 구분하기 위해 안버(주황)색으로, '확정 아님'을 분명히 밝힌다."""
    rows = []
    for c in candidates_with_process:
        if not c["process"] or c["pattern"] == "none":
            continue
        chips = "".join(
            f"<span style='display:inline-block;background:{AMBER}14;color:{AMBER};"
            f"border:1px solid {AMBER}40;border-radius:999px;padding:2px 9px;"
            f"font-size:11px;font-weight:700;margin:0 5px 4px 0;'>{p}</span>"
            for p in c["process"]
        )
        rows.append(
            f"<div style='margin-bottom:7px;'>"
            f"<span style='font-size:12.5px;font-weight:700;color:{TEXT};'>{c['pattern']}</span>"
            f"<span style='font-size:11px;color:{MUTED};'> ({c['prob']*100:.1f}%)</span><br>{chips}</div>"
        )
    if not rows:
        return
    st.markdown(
        f"<div class='qc-card'>"
        f"<div class='qc-meta-line' style='margin-bottom:8px;'>🏭 후보 패턴별 점검 공정 "
        f"(확정 아님 — 모델이 확신하지 못해 참고용으로만 제공, 확률 높은 순)</div>"
        f"{''.join(rows)}</div>",
        unsafe_allow_html=True,
    )


def cause_card(candidates, meta_line, refs, process_list=None):
    items = "".join(f"<div class='qc-cause-item'>{c}</div>" for c in candidates)
    refs_html = f"<div class='qc-meta-line'>참고문헌 키: {', '.join(refs)}</div>" if refs else ""
    process_html = process_chips(process_list) if process_list else ""
    st.markdown(
        f"<div class='qc-card'>"
        f"{process_html}"
        f"<div class='qc-meta-line' style='margin-bottom:8px;'>{meta_line}</div>"
        f"{items}{refs_html}</div>",
        unsafe_allow_html=True,
    )


def render_result(result):
    action_badge(result["action"])
    st.markdown(f"**모델 예측 패턴:** {result['predicted_pattern']}")

    d = result["detail"]
    if result["action"] == "통과":
        c = d["confidences"]
        st.markdown(
            "🟢 세 모델(베이스/와이드/딥)이 모두 **'정상(none)'** 이라고 확신했습니다 — "
            "다음 공정으로 보내도 됩니다."
        )
        stat_row([(f"{MODEL_DISPLAY_NAME[name]} 모델 확신도", f"{c[name]*100:.1f}%")
                  for name in ("base", "wide", "deep")])

    elif result["action"] == "불량":
        meta = (f"확신도 <b style='color:{CORAL};'>{d['confidence']*100:.1f}%</b> · "
                f"심각도: {d['severity']} · 근거수준: {d['evidence']}")
        if d.get("scratch_linearity_score") is not None:
            meta += f" · 형태 필터(길쭉함 지수): {d['scratch_linearity_score']:.3f} (기준 통과)"
        if d["candidates"]:
            st.markdown("**우선 점검 후보 원인** (확정이 아니라 점검 우선순위 참고용)")
            cause_card(d["candidates"], meta, d.get("refs", []), d.get("process", []))
        else:
            process_html = process_chips(d.get("process", [])) if d.get("process") else ""
            st.markdown(f"<div class='qc-card'>{process_html}{meta}</div>", unsafe_allow_html=True)

    else:
        st.markdown("🟠 모델이 확신하지 못했습니다 — **사람이 직접 확인**해야 합니다.")
        if d.get("downgrade_reason"):
            st.markdown(
                f"<div class='qc-card' style='border-left:3px solid {AMBER};'>"
                f"🩹 {d['downgrade_reason']}</div>", unsafe_allow_html=True,
            )
        st.markdown("**상위 후보 패턴**")
        bar_rows([(label, p * 100) for label, p in d["top_candidates"]])
        if d.get("candidates_with_process"):
            candidate_process_card(d["candidates_with_process"])

    with st.expander("세 모델의 전체 클래스별 확신도 보기"):
        for name in ("base", "wide", "deep"):
            st.markdown(f"**{MODEL_DISPLAY_NAME[name]}**")
            probs = result["confidences_by_model"][name]
            st.bar_chart(probs)


# ============================================================== 사이드바


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 설정")
        pass_th = st.slider("통과(PASS) 확신도 임계값", 0.50, 0.999, 0.83, 0.01,
                             help="세 모델 모두 이 값 이상으로 확신해야 '통과'로 판정합니다. 높일수록 "
                                  "더 보수적(안전)이지만 사람이 검토해야 할 비율이 늘어납니다.")
        fail_th = st.slider("불량(FAIL) 확신도 임계값", 0.50, 0.999, 0.80, 0.01,
                             help="wide 모델이 이 값 이상으로 결함이라 확신하면 '불량'으로 판정합니다.")
        st.caption("기본값(0.83 / 0.80)은 val+test set으로 검증된 추천값입니다.")
        st.divider()

        st.markdown("**🩹 Scratch 형태 필터**")
        use_scratch_filter = st.checkbox("사용", value=True,
            help="CNN이 'Scratch'로 불량 확정하려 할 때, 64×64 중간 단계에서 불량 다이들이 실제로 "
                 "길쭉한 선 모양인지 추가로 확인합니다. 모양이 애매하면 '직접확인필요'로만 낮추고, "
                 "'통과'로는 절대 바꾸지 않습니다 — 안전 우선 설계는 그대로 유지됩니다.")
        if use_scratch_filter:
            scratch_th = st.slider("필터 엄격도 (길쭉함 지수 기준)", 0.05, 0.30, 0.15, 0.01,
                help="낮출수록 엄격 = Scratch 오탐을 더 많이 걸러내지만(정밀도↑), 애매한 진짜 "
                     "Scratch까지 같이 사람 확인으로 넘어갑니다(재현율↓). test set 24,703장 실측: "
                     "필터 없음 정밀도 0.51/재현율 0.18 → 0.15 기준 정밀도 0.64/재현율 0.15, "
                     "0.08 기준 정밀도 0.77/재현율 0.12.")
        else:
            scratch_th = None
        st.divider()
        st.markdown(
            f"<span style='color:{MUTED};font-size:11.5px;'>🟢 양품 다이&nbsp;&nbsp;"
            f"🔴 불량 다이&nbsp;&nbsp;⬜ 다이 없음</span>",
            unsafe_allow_html=True,
        )
        st.divider()
        if llm_client.is_configured():
            st.markdown(f"<span style='color:{TEAL};font-size:12px;'>🤖 AI 부가기능(배치 브리핑·Q&A) 연결됨</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<span style='color:{MUTED};font-size:12px;'>🤖 AI 부가기능 미설정</span>",
                        unsafe_allow_html=True)
            st.caption("wafer_qc_app 폴더에 .env 파일을 만들고 GEMINI_API_KEY=발급받은키 를 넣으면 "
                       "배치 브리핑/Q&A 탭이 활성화됩니다. (핵심 판정 기능은 이것 없이도 정상 동작합니다)")
    return pass_th, fail_th, scratch_th


# ============================================================== 탭 1: 단일 검사


def _clean_single_result(input_name, result):
    """Q&A 탭에 넘길 수 있게 결과에서 JSON 직렬화 안 되는 부분(픽셀 배열)만 뺀 요약."""
    return {
        "입력_웨이퍼": input_name,
        "판정": result["action"],
        "예측_패턴": result["predicted_pattern"],
        "세부정보": result["detail"],
        "모델별_confidence": result["confidences_by_model"],
    }


def render_single_tab(tool, pass_th, fail_th, scratch_th):
    st.subheader("웨이퍼맵 입력")
    src = st.radio("입력 방식", ["샘플에서 선택", "직접 업로드 (.npy, 정수 0/1/2 배열)"], horizontal=True)

    raw_arr = None
    label_hint = ""
    input_name = None
    if src == "샘플에서 선택":
        sample_files = sorted(SAMPLE_DIR.glob("*.npy"))
        names = [f.stem for f in sample_files]
        picked = st.selectbox("샘플 패턴 선택", names)
        if picked:
            raw_arr = np.load(SAMPLE_DIR / f"{picked}.npy")
            label_hint = f"샘플의 실제 라벨: **{picked}**"
            input_name = picked
    else:
        uploaded = st.file_uploader("웨이퍼맵 .npy 파일 업로드", type=["npy"])
        if uploaded is not None:
            raw_arr = np.load(uploaded)
            input_name = uploaded.name

    if raw_arr is None:
        st.info("샘플을 선택하거나 파일을 업로드하면 결과가 표시됩니다.")
        return

    if label_hint:
        st.caption(label_hint)

    # 실제로는 원본 -> (정사각형 패딩 + 64x64, 중간 단계) -> 32x32(최종 모델 입력) 2단계 리사이즈를
    # 거친다. "32x32만 보고 판단한다"고 오해하지 않도록 중간 단계까지 전부 보여준다.
    padded_64, x32 = preprocess_wafer_map_stages(raw_arr)
    img_col1, img_col2, img_col3 = st.columns(3)
    with img_col1:
        st.markdown(f"**① 원본** &nbsp;({raw_arr.shape[0]}×{raw_arr.shape[1]})")
        st.pyplot(plot_wafer(raw_arr))
    with img_col2:
        st.markdown("**② 정사각형 패딩 + 64×64** &nbsp;(중간 단계)")
        st.pyplot(plot_wafer(padded_64))
    with img_col3:
        st.markdown("**③ 32×32** &nbsp;(실제 모델 입력)")
        st.pyplot(plot_wafer(x32 * 2))
    st.caption("모델은 ③만 보고 판단합니다 — ①→③은 사람이 보기 위한 것이 아니라, 실제 추론 전에 "
               "거치는 전처리 파이프라인 그대로입니다 (② nearest 리사이즈로 모양 유지 → ③ area "
               "리사이즈로 다운스케일).")

    st.subheader("판정 결과")
    result = tool.inspect(raw_arr, pass_threshold=pass_th, fail_threshold=fail_th,
                           scratch_linearity_threshold=scratch_th)
    st.session_state["single_check_result"] = _clean_single_result(input_name, result)

    # 탭을 전환하거나 다른 위젯을 건드릴 때마다 스크립트 전체가 다시 실행되기 때문에,
    # 입력/임계값이 그대로인데 매번 감사 로그에 중복으로 쌓이지 않도록 시그니처로 한 번만 기록.
    log_sig = (input_name, pass_th, fail_th, scratch_th, result["action"], result["predicted_pattern"])
    if st.session_state.get("_last_logged_single_sig") != log_sig:
        audit_log.log_judgment("single", input_name, result, pass_th, fail_th, raw_wafer_map=raw_arr)
        st.session_state["_last_logged_single_sig"] = log_sig

    render_result(result)


# ============================================================== 탭 2: 배치 분석


def run_batch(tool, items, pass_th, fail_th, scratch_th):
    """items: list of (name, raw_array) -> list of {name, result}."""
    out = []
    for name, raw in items:
        try:
            result = tool.inspect(raw, pass_threshold=pass_th, fail_threshold=fail_th,
                                   scratch_linearity_threshold=scratch_th)
            out.append({"name": name, "result": result, "raw": raw})
        except Exception as e:  # noqa: BLE001
            out.append({"name": name, "error": str(e)})
    return out


def batch_summary(batch_results):
    action_counts = {"통과": 0, "불량": 0, "직접확인필요": 0}
    pattern_counts = {}
    process_counts = {}
    n_total = 0
    for item in batch_results:
        if "error" in item:
            continue
        n_total += 1
        r = item["result"]
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
        pat = r["predicted_pattern"]
        pattern_counts[pat] = pattern_counts.get(pat, 0) + 1
        if r["action"] == "불량":
            for p in r["detail"].get("process", []):
                process_counts[p] = process_counts.get(p, 0) + 1
    return {
        "n_total": n_total,
        "action_counts": action_counts,
        "pattern_counts": pattern_counts,
        "process_counts": process_counts,
    }


def render_batch_summary_ui(summary):
    n = summary["n_total"]
    ac = summary["action_counts"]
    st.markdown(f"**총 {n}장 처리** — 사람이 실제로 봐야 하는 건 "
                f"**{ac.get('직접확인필요', 0)}장 ({ac.get('직접확인필요', 0)/max(n,1)*100:.1f}%)** 뿐입니다.")
    stat_row([
        ("✅ 통과", f"{ac.get('통과', 0)}장 ({ac.get('통과', 0)/max(n,1)*100:.1f}%)"),
        ("⛔ 불량 확정", f"{ac.get('불량', 0)}장 ({ac.get('불량', 0)/max(n,1)*100:.1f}%)"),
        ("🔍 직접확인필요", f"{ac.get('직접확인필요', 0)}장 ({ac.get('직접확인필요', 0)/max(n,1)*100:.1f}%)"),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**예측 패턴 분포**")
        if summary["pattern_counts"]:
            st.bar_chart(summary["pattern_counts"])
    with col2:
        st.markdown("**불량 판정의 관련 공정 빈도**")
        if summary["process_counts"]:
            st.bar_chart(summary["process_counts"])
        else:
            st.caption("이번 배치에 불량 확정 건이 없습니다.")


def render_batch_tab(tool, pass_th, fail_th, scratch_th):
    st.subheader("여러 장을 한 번에 — 배치 트리아지")
    st.caption("실제 팹에서는 한 장씩 보는 게 아니라 하루 수백~수천 장이 쏟아집니다. "
               "이 탭은 그 중 사람이 실제로 봐야 할 비율이 얼마나 줄어드는지 보여줍니다.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        uploaded = st.file_uploader("웨이퍼맵 .npy 여러 개 업로드", type=["npy"],
                                     accept_multiple_files=True, key="batch_uploader")
    with col_b:
        st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
        demo_clicked = st.button(
            f"📦 데모 배치 불러오기 (실제 test set에서 뽑은 {len(list(DEMO_BATCH_DIR.glob('*.npy')))}장)",
            use_container_width=True,
        )

    items = None
    batch_sig = None
    if demo_clicked:
        files = sorted(DEMO_BATCH_DIR.glob("*.npy"))
        items = [(f.stem, np.load(f)) for f in files]
        batch_sig = ("demo", len(files), pass_th, fail_th, scratch_th)
    elif uploaded:
        items = [(f.name, np.load(f)) for f in uploaded]
        batch_sig = ("upload", tuple(sorted(f.name for f in uploaded)), pass_th, fail_th, scratch_th)

    # 업로드된 파일은 사용자가 지우기 전까지 다음 rerun에도 그대로 남아있어서(탭 전환만 해도
    # 스크립트가 다시 돌기 때문에), 시그니처가 그대로면 재계산/중복 로그를 하지 않는다.
    if items and st.session_state.get("_last_logged_batch_sig") != batch_sig:
        with st.spinner(f"{len(items)}장 처리 중..."):
            results = run_batch(tool, items, pass_th, fail_th, scratch_th)
            summary = batch_summary(results)
        audit_log.log_batch(results, pass_th, fail_th)
        st.session_state["batch_results"] = results
        st.session_state["batch_summary"] = summary
        st.session_state["batch_thresholds"] = {"pass": pass_th, "fail": fail_th}
        st.session_state["_last_logged_batch_sig"] = batch_sig

    summary = st.session_state.get("batch_summary")
    if not summary:
        st.info("왼쪽에서 파일을 올리거나, 오른쪽 '데모 배치 불러오기' 버튼을 눌러보세요.")
        return

    render_batch_summary_ui(summary)

    st.divider()
    st.markdown("**🤖 AI 브리핑**")
    if not llm_client.is_configured():
        st.caption("AI 브리핑을 쓰려면 사이드바 안내대로 .env에 GEMINI_API_KEY를 설정해주세요.")
    else:
        if st.button("이번 배치 브리핑 생성"):
            with st.spinner("Gemini한테 물어보는 중..."):
                briefing = llm_client.summarize_batch(
                    summary["action_counts"], summary["pattern_counts"], summary["process_counts"],
                    summary["n_total"], st.session_state.get("batch_thresholds", {}),
                )
            st.session_state["batch_briefing"] = briefing
        if st.session_state.get("batch_briefing"):
            st.markdown(f"<div class='qc-card'>{st.session_state['batch_briefing']}</div>",
                        unsafe_allow_html=True)
            st.markdown("<div class='qc-ai-note'>※ AI가 위 집계 숫자만 보고 생성한 요약입니다 — "
                        "원본 이미지를 직접 본 게 아니라 숫자 기반 해석이니 참고용으로만 보세요.</div>",
                        unsafe_allow_html=True)

    with st.expander(f"전체 {summary['n_total']}건 상세 보기"):
        for item in st.session_state.get("batch_results", []):
            if "error" in item:
                st.markdown(f"- `{item['name']}` — 처리 실패: {item['error']}")
                continue
            r = item["result"]
            icon = ACTION_STYLE.get(r["action"], {}).get("icon", "")
            st.markdown(f"- `{item['name']}` — {icon} **{r['action']}** (예측: {r['predicted_pattern']})")


# ============================================================== 탭 3: 원인 Q&A


def render_qa_tab(cause_map):
    st.subheader("원인 지식베이스 Q&A")
    st.caption("패턴별 원인 후보/참고문헌에 한정해서 답하는 챗봇입니다. 지식베이스에 없는 내용은 "
               "'모른다'고 답하도록 설계되어 있습니다 — 근거 없는 추측을 지어내지 않습니다.")

    if not llm_client.is_configured():
        st.warning("이 탭을 쓰려면 wafer_qc_app 폴더에 .env 파일을 만들고 "
                   "GEMINI_API_KEY=발급받은키 를 넣어주세요. (사이드바 참고)")
        return

    # 이 챗봇이 그냥 "제미나이 켜놓고 쓰는 것"과 다른 점 — 방금 단일검사/배치분석 탭에서
    # 나온 결과를 사람이 다시 설명해줄 필요 없이 자동으로 같이 넘겨서, "이거 왜 이렇게 나왔어?"
    # 같은 질문이 바로 통하게 함. 뭘 참고 중인지 투명하게 보여준다.
    single_ctx = st.session_state.get("single_check_result")
    batch_ctx = st.session_state.get("batch_summary")
    ctx_bits = []
    if single_ctx:
        ctx_bits.append(f"최근 단일검사 1건 ({single_ctx['입력_웨이퍼']} → {single_ctx['판정']})")
    if batch_ctx:
        ctx_bits.append(f"최근 배치 결과 ({batch_ctx['n_total']}건)")
    if ctx_bits:
        st.caption("📎 지금 같이 참고 중: " + " · ".join(ctx_bits) +
                   " — 데이터를 다시 설명 안 하고 바로 물어보셔도 됩니다.")
    else:
        st.caption("📎 아직 단일검사/배치분석을 안 돌리셨어요 — 지금은 지식베이스만 보고 답합니다. "
                   "먼저 웨이퍼를 하나 검사해보시면 그 결과도 같이 참고해서 답해줍니다.")

    st.session_state.setdefault("chat_history", [])

    for role, text in st.session_state["chat_history"]:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)

    question = st.chat_input("예: 방금 그 결과 왜 이렇게 나온 거야?")
    if question:
        st.session_state["chat_history"].append(("user", question))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("지식베이스 찾아보는 중..."):
                answer = llm_client.answer_kb_question(
                    question, cause_map,
                    recent_single_check=single_ctx,
                    recent_batch_summary=batch_ctx,
                    history=st.session_state["chat_history"],
                )
            st.markdown(answer)
        st.session_state["chat_history"].append(("assistant", answer))


# ============================================================== 탭 4: 모니터링 / 감사 로그


def render_monitoring_tab():
    st.subheader("모니터링 & 감사 로그")
    st.caption("실제 운영에서 필요한 두 가지 — '누가 어떤 판정을 봤고 뭘 바꿨는지(감사 추적)'와 "
               "'모델 성능이 흔들리고 있는지(모니터링)'를 가볍게 보여주는 탭입니다. 통계적으로 "
               "정교한 drift detection은 아니고, 사람이 모델 추천을 얼마나 자주 뒤집는지를 보는 "
               "가장 기본적인 추세 신호입니다.")

    n_total = audit_log.total_count()
    if n_total == 0:
        st.info("아직 기록된 판정이 없습니다. 단일 검사나 배치 분석을 한 번 돌려보세요.")
        return

    # ---- 검토 대기열 (감사 추적: 사람이 뭘 바꿨는지 여기서 기록됨) ----
    st.markdown("**🔍 검토 대기 중인 '직접확인필요' 건**")
    pending = audit_log.fetch_pending_review(limit=20)
    if not pending:
        st.caption("검토 대기 중인 건이 없습니다.")
    else:
        for row in pending:
            with st.form(key=f"review_form_{row['id']}"):
                st.markdown(f"`{row['input_name']}` — 모델 1순위 추천: **{row['predicted_pattern']}** "
                            f"({row['ts']})")
                c1, c2, c3 = st.columns([1, 1, 2])
                with c1:
                    decision = st.selectbox("최종 판단", CLASS_ORDER,
                                             index=CLASS_ORDER.index(row["predicted_pattern"])
                                             if row["predicted_pattern"] in CLASS_ORDER else 0,
                                             key=f"decision_{row['id']}")
                with c2:
                    reviewer = st.text_input("검토자", key=f"reviewer_{row['id']}")
                with c3:
                    note = st.text_input("메모 (선택)", key=f"note_{row['id']}")
                if st.form_submit_button("검토 완료로 기록"):
                    audit_log.record_review(row["id"], reviewer or "익명", decision, note)
                    st.rerun()

    st.divider()

    # ---- 재학습 후보 데이터 (사람이 확정한 건 = 다음 재학습의 재료) ----
    st.markdown("**🔁 재학습 후보 데이터**")
    st.caption("사람이 '직접확인필요' 건을 검토해서 최종 라벨을 확정하면, 그 웨이퍼맵+확정라벨이 "
               "여기 쌓입니다 — 모델이 지금 실제로 틀리거나 애매해하는 사례이기 때문에, 다음 "
               "재학습에서 가장 가치있는 데이터입니다. 쌓인 건들을 다시 열어보고, 잘못 확정된 "
               "건은 재확인하거나 아예 뺄 수 있습니다.")

    show_used = st.checkbox("이미 재학습에 쓴 건도 같이 보기", value=False, key="retrain_show_used",
                             help="매일 자동 재학습(아래 참고)이 끝나면 쓰인 건에 표시가 남습니다 — "
                                  "기본적으로는 아직 한 번도 안 쓰인 '새로 쌓인' 건만 보여줍니다.")
    pool = audit_log.fetch_retrain_pool(include_used=show_used)
    pool_stats = audit_log.retrain_pool_stats(include_used=show_used)

    if pool_stats["n_total"] == 0:
        st.caption("아직 쌓인 재학습 후보가 없습니다 — 위에서 '직접확인필요' 건을 검토하면 쌓입니다.")
    else:
        stat_row([
            ("확정된 건수", f"{pool_stats['n_total']}건"),
            ("모델이 틀렸던 건수", f"{pool_stats['n_model_was_wrong']}건"),
            ("라벨 종류", f"{len(pool_stats['by_label'])}종"),
        ])
        if pool_stats["by_label"]:
            st.caption("확정 라벨별 건수: " +
                       " · ".join(f"{k} {v}건" for k, v in sorted(pool_stats['by_label'].items())))

        with st.expander(f"📋 재학습 후보 목록 보기 / 정리 ({len(pool)}건)"):
            for item in pool:
                mismatch = item["predicted_pattern"] != item["confirmed_label"]
                used_tag = " · ✅ 이미 재학습에 사용됨" if item.get("used_in_training_ts") else ""
                c1, c2, c3 = st.columns([5, 1, 1])
                with c1:
                    st.markdown(
                        f"`{item['input_name']}` — 모델: **{item['predicted_pattern']}** → 확정: "
                        f"**{item['confirmed_label']}**{' 🩹(모델이 틀렸던 건)' if mismatch else ''} "
                        f"&nbsp;·&nbsp;검토자: {item['reviewer_name'] or '익명'} ({item['reviewed_ts']}){used_tag}"
                    )
                    if item.get("reviewer_note"):
                        st.caption(f"메모: {item['reviewer_note']}")
                with c2:
                    if st.button("🔁 재확인", key=f"pool_reset_{item['id']}",
                                 help="검토 대기 목록으로 되돌려서 다시 확인합니다."):
                        audit_log.reset_review(item["id"])
                        st.rerun()
                with c3:
                    if st.button("🗑️ 삭제", key=f"pool_delete_{item['id']}",
                                 help="재학습 후보에서 완전히 뺍니다 (원본 웨이퍼맵 파일도 삭제)."):
                        audit_log.delete_pool_entry(item["id"])
                        st.rerun()

        col_exp1, col_exp2 = st.columns([1, 2])
        with col_exp1:
            if st.button("📦 재학습용 데이터셋 만들기 (.npz)"):
                export_pool = audit_log.fetch_retrain_pool(include_used=False)
                X = np.empty((len(export_pool), 32, 32), dtype=np.float32)
                y = np.empty(len(export_pool), dtype=np.int64)
                skipped = 0
                for i, item in enumerate(export_pool):
                    label = item["confirmed_label"]
                    if label not in CLASS_ORDER:
                        skipped += 1
                        continue
                    raw = np.load(item["wafer_path"])
                    X[i] = preprocess_wafer_map_stages(raw)[1]
                    y[i] = CLASS_ORDER.index(label)
                buf = BytesIO()
                np.savez(buf, X_32=X, y=y, class_order=np.array(CLASS_ORDER))
                st.session_state["retrain_pool_npz"] = buf.getvalue()
                st.session_state["retrain_pool_skipped"] = skipped
        with col_exp2:
            if st.session_state.get("retrain_pool_npz"):
                st.download_button(
                    "⬇️ retrain_pool.npz 다운로드",
                    data=st.session_state["retrain_pool_npz"],
                    file_name="retrain_pool.npz",
                    mime="application/octet-stream",
                )
                if st.session_state.get("retrain_pool_skipped"):
                    st.caption(f"(라벨을 알 수 없는 {st.session_state['retrain_pool_skipped']}건은 제외)")
        st.caption("⚠️ 실제 가중치 재학습은 이 앱 안에서 바로 돌리지 않습니다 — CNN 재학습은 "
                   "1 epoch에도 수 분이 걸려서 Streamlit 요청-응답 방식과는 안 맞습니다. "
                   "**기본으로 권장하는 방법**은 위 버튼으로 내보낸 파일을 "
                   "`scripts/20_retrain_from_review.py`에 수동으로 넘기는 것입니다 — 이 스크립트는 "
                   "검토 건수가 `--min_pool_size`(기본 20건) 미만이면 그냥 건너뜁니다(실측상 "
                   "수십 건 이하로는 돌려봐도 효과가 거의 없거나 트레이드오프만 생겼기 때문 — "
                   "아래 README 참고). `scripts/21_daily_retrain_job.py`로 Windows 작업 "
                   "스케줄러에 등록해서 **매일 자동으로** 돌리는 방법도 있지만, 혼자 검토하는 "
                   "지금 단계에서는 하루에 쌓이는 양이 적어서 권장하지 않습니다 — 여러 명이 "
                   "검토를 나눠 맡는 규모로 커졌을 때를 위한 확장 시나리오로 남겨둔 것입니다. "
                   "'통과' 판정은 base/wide/deep 세 모델이 모두 동의해야 나오므로, 두 경우 모두 "
                   "**세 모델을 같은 데이터로 같이** fine-tune합니다(wide만 재학습하면 시간이 "
                   "지날수록 세 모델의 기준이 벌어지기 때문). `cnn_best.npz`/`cnn_wide_best.npz`/"
                   "`cnn_deep_best.npz`는 건드리지 않고 candidate 파일로만 저장하며, 모델별 "
                   "지표뿐 아니라 세 모델을 합친 **앙상블 미탐지율**(진짜 불량인데 '통과'로 샌 "
                   "비율)이 재학습 후 올라가지 않았는지까지 로그에 남깁니다 — 사람이 그걸 직접 "
                   "확인한 뒤에만, 세 candidate를 묶어서 같이 교체합니다. 자동으로 몰래 모델을 "
                   "바꿔치기하지 않는다는 원칙은 여기도 동일합니다.")

    st.divider()

    # ---- 드리프트 신호 ----
    st.markdown("**📉 드리프트 신호** — 사람 최종판단이 모델 1순위 추천과 달랐던 비율")
    drift = audit_log.drift_signal()
    if not drift:
        st.caption("아직 검토 완료된 건이 없어서 계산할 수 없습니다.")
    else:
        rate = drift["mismatch_rate"]
        stat_row([
            ("검토된 건수", f"{drift['n_reviewed']}건"),
            ("모델과 다르게 판단한 건수", f"{drift['n_mismatch']}건"),
            ("불일치율", f"{rate*100:.1f}%"),
        ])
        if drift["n_reviewed"] >= 5 and rate > 0.3:
            st.warning("최근 검토 결과가 모델 1순위 추천과 자주 다릅니다 — 최근 공정/패턴이 "
                       "바뀌었거나 모델 재학습이 필요할 수 있다는 신호입니다.")
        else:
            st.caption("아직 특별한 이상 신호는 없습니다. (참고용 — 검토 건수가 적으면 신뢰도는 낮습니다)")

    st.divider()

    # ---- 일별 판정 추이 ----
    st.markdown("**📈 일별 판정 추이**")
    daily = audit_log.daily_action_counts()
    if daily:
        days = sorted(set(r["day"] for r in daily))
        actions = ["통과", "불량", "직접확인필요"]
        chart_data = {a: {d: 0 for d in days} for a in actions}
        for r in daily:
            if r["action"] in chart_data:
                chart_data[r["action"]][r["day"]] = r["n"]
        st.bar_chart(chart_data)
    else:
        st.caption("아직 데이터가 없습니다.")

    st.divider()

    # ---- 전체 감사 로그 ----
    with st.expander(f"전체 감사 로그 보기 (최근 {min(n_total, 200)}건 / 총 {n_total}건)"):
        recent = audit_log.fetch_recent(200)
        table = [
            {
                "시각": r["ts"], "출처": r["source"], "입력": r["input_name"],
                "예측패턴": r["predicted_pattern"], "판정": r["action"],
                "확신도": f"{r['top1_confidence']*100:.1f}%" if r["top1_confidence"] is not None else "-",
                "검토자": r["reviewer_name"] or "-", "최종판단": r["reviewer_decision"] or "-",
            }
            for r in recent
        ]
        st.dataframe(table, use_container_width=True)


# ============================================================== main


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🔬 WM-811K 웨이퍼맵 결함 분류 + 트리아지")
    st.caption(
        "규칙 기반 필터 → CNN 확신도 기반 자동 판정 → 애매한 건만 사람이 확인하는 3단계 파이프라인. "
        "base/wide/deep 세 독립 CNN이 전부 동의해야만 '통과'를 내보내는 안전 우선 설계입니다. "
        "(val/test set 기준 검증 완료 — 자세한 내용은 발표자료 참고)"
    )

    tool = load_tool()
    cause_map = load_cause_map()
    pass_th, fail_th, scratch_th = render_sidebar()

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 단일 검사", "📦 배치 분석", "💬 원인 Q&A", "📋 모니터링"])
    with tab1:
        render_single_tab(tool, pass_th, fail_th, scratch_th)
    with tab2:
        render_batch_tab(tool, pass_th, fail_th, scratch_th)
    with tab3:
        render_qa_tab(cause_map)
    with tab4:
        render_monitoring_tab()


if __name__ == "__main__":
    main()
