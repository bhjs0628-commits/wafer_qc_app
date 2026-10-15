"""
WM-811K 웨이퍼맵 결함 분류 + 3단계 트리아지 핵심 로직.

app.py(Streamlit UI)가 이 모듈을 불러 쓴다. 판정 로직은 원 프로젝트의
scripts/14_triage.py에서 val set으로 설계하고 scripts/19_final_test_eval.py로
test set 검증까지 마친 것과 동일하다 (test 기준: 자동통과 58.0%, 미탐지 0.77%).

구성:
  - base/wide/deep 세 독립 CNN이 *모두* 'none'을 예측하고 확신도가 전부
    PASS_THRESHOLD 이상이어야만 "통과" (한 모델이라도 의심하면 "직접확인필요"로)
  - wide 모델이 결함을 예측하고 확신도가 FAIL_THRESHOLD 이상이면 "불량"
    (config/pattern_cause_map.json에서 원인 후보를 붙여 줌 — 확정 아님, 점검 우선순위용)
  - 그 외는 "직접확인필요"
"""
import json
from pathlib import Path

import numpy as np
import cv2

from cnn_arch import (
    build_model, build_model_wide, build_model_deep, load_weights,
    CLASS_ORDER, LABEL_TO_IDX, IMG_SIZE,
)

NONE_IDX = LABEL_TO_IDX["none"]

DEFAULT_PASS_THRESHOLD = 0.83
DEFAULT_FAIL_THRESHOLD = 0.80


def pad_to_square(arr):
    h, w = arr.shape
    side = max(h, w)
    if h == w:
        return arr
    padded = np.zeros((side, side), dtype=arr.dtype)
    top, left = (side - h) // 2, (side - w) // 2
    padded[top:top + h, left:left + w] = arr
    return padded


def preprocess_wafer_map_stages(raw_wafer_map):
    """전처리 중간 단계까지 전부 노출하는 버전. 실제로 모델이 보는 건 32x32(x32)뿐이지만,
    원본 -> 정사각형 패딩 + 64x64(중간 단계) -> 32x32(최종) 2단계 리사이즈를 거치기 때문에,
    화면에 "원본만" 보여주면 실제로 모델에 어떤 전처리가 들어갔는지가 가려진다. 그래서 중간
    단계(64x64)도 같이 반환해서 UI에서 3장을 다 보여줄 수 있게 한다.

    반환: (padded_64, x32)
      - padded_64: uint8, 0/1/2 스케일 그대로 (정사각형 패딩 + nearest 리사이즈까지만 적용)
      - x32: float32, [0,1] 스케일 (최종 모델 입력 — preprocess_wafer_map()의 반환값과 동일)
    """
    arr = np.asarray(raw_wafer_map, dtype=np.uint8)
    arr = pad_to_square(arr)
    padded_64 = cv2.resize(arr, (64, 64), interpolation=cv2.INTER_NEAREST)
    norm_64 = padded_64.astype(np.float32) / 2.0
    x32 = cv2.resize(norm_64, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return padded_64, x32


def preprocess_wafer_map(raw_wafer_map):
    """원본 웨이퍼맵(가변 크기, 정수 0/1/2) -> 모델 입력(32x32, float32 [0,1]).
    중간 단계가 궁금하면 preprocess_wafer_map_stages()를 쓸 것 — 둘은 같은 로직을 공유한다."""
    _, x32 = preprocess_wafer_map_stages(raw_wafer_map)
    return x32


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def scratch_linearity_score(padded_64):
    """64x64 중간 단계(블러 전 해상도)에서, 불량 다이들이 '한 줄로 길게 늘어선 모양(선형)'인지
    점수화한다. 불량 다이를 8-연결 컴포넌트로 묶고 그 중 가장 큰 덩어리만 골라 PCA로 긴 축/짧은
    축 분산 비율을 구함 — 0에 가까우면 직선(진짜 Scratch다운 모양), 1에 가까우면 둥근 덩어리.

    test set 24,703장 전체로 검증: CNN이 Scratch로 예측하고 FAIL_THRESHOLD를 넘겨 '불량'
    판정까지 간 59건 중, 실제로는 'none' 등 다른 패턴인 오탐 29건과 진짜 Scratch 30건을 이
    점수로 비교하면 진짜 Scratch는 뚜렷하게 더 낮은 값(길쭉함)을 보인다 — 완전히 분리되진
    않지만, 임계값으로 걸러내면 오탐을 꽤 줄일 수 있다는 게 실측으로 확인됐다.

    반환: (score, size) — 가장 큰 덩어리가 없거나 너무 작으면 (None, size)."""
    mask = (padded_64 == 2).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if n_labels <= 1:
        return None, 0
    sizes = [int((labels == lbl).sum()) for lbl in range(1, n_labels)]
    best = int(np.argmax(sizes)) + 1
    biggest_size = sizes[best - 1]
    ys, xs = np.nonzero(labels == best)
    if len(xs) < 4:
        return None, biggest_size
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pts -= pts.mean(axis=0)
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    lam_small, lam_big = eigvals[0], eigvals[1]
    if lam_big <= 1e-9:
        return 0.0, biggest_size
    return float(lam_small / lam_big), biggest_size


class WaferQCTool:
    def __init__(self, model_dir, config_dir,
                 pass_threshold=DEFAULT_PASS_THRESHOLD, fail_threshold=DEFAULT_FAIL_THRESHOLD):
        model_dir = Path(model_dir)
        config_dir = Path(config_dir)
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

        self.base = build_model()
        load_weights(self.base, model_dir / "cnn_best.npz")
        self.wide = build_model_wide()
        load_weights(self.wide, model_dir / "cnn_wide_best.npz")
        self.deep = build_model_deep(seed_base=777)
        load_weights(self.deep, model_dir / "cnn_deep_best.npz")

        with open(config_dir / "pattern_cause_map.json", "r", encoding="utf-8") as f:
            self.cause_map = json.load(f)

    def _predict_one(self, x32):
        xb = x32[None, None, :, :].astype(np.float32)
        return {
            "base": _softmax(self.base.forward(xb))[0],
            "wide": _softmax(self.wide.forward(xb))[0],
            "deep": _softmax(self.deep.forward(xb))[0],
        }

    def _candidates_with_process(self, wide_probs, top_k):
        """'직접확인필요'로 빠질 때도, 확률 순 후보 패턴별로 관련 공정 후보를 같이 붙여준다 —
        확정된 '불량'은 아니지만, 그래도 1순위 후보가 뭔지는 이미 알고 있으니 "일단 이 공정부터
        점검해보라"는 참고 정보를 주는 게 사람이 맨땅에서 보는 것보다 낫다는 판단. 'none'(정상
        추정)에는 점검할 공정이 없으므로 process를 빈 리스트로 둔다."""
        top_order = np.argsort(-wide_probs)[:top_k]
        out = []
        for i in top_order:
            label = CLASS_ORDER[i]
            entry = self.cause_map.get(label, {}) if label != "none" else {}
            out.append({
                "pattern": label,
                "prob": float(wide_probs[i]),
                "process": entry.get("process", []),
            })
        return out

    def inspect(self, raw_wafer_map, top_k=3, pass_threshold=None, fail_threshold=None,
                scratch_linearity_threshold=None):
        """원본 웨이퍼맵 1장 -> 판정 결과 dict.

        scratch_linearity_threshold: None이면 비활성화. 값을 주면, CNN이 Scratch로 '불량' 판정
        하려는 순간에 64x64 중간 단계에서 모양이 충분히 길쭉한지(scratch_linearity_score) 추가로
        확인한다. 통과하면 그대로 '불량', 통과 못하면 '직접확인필요'로 낮춘다(절대 '통과'로 바뀌지
        않음 — 안전 우선 설계 그대로 유지). 값이 작을수록(예: 0.05) 더 엄격 = 정밀도↑ 재현율↓.
        """
        pass_th = self.pass_threshold if pass_threshold is None else pass_threshold
        fail_th = self.fail_threshold if fail_threshold is None else fail_threshold

        padded_64, x32 = preprocess_wafer_map_stages(raw_wafer_map)
        probs = self._predict_one(x32)

        wide_probs = probs["wide"]
        pred_idx = int(np.argmax(wide_probs))
        pred_label = CLASS_ORDER[pred_idx]
        wide_conf = float(wide_probs[pred_idx])

        votes_ok = all(
            int(np.argmax(probs[name])) == NONE_IDX and float(probs[name][NONE_IDX]) >= pass_th
            for name in ("base", "wide", "deep")
        )

        result = {
            "predicted_pattern": pred_label,
            "preprocessed_32x32": x32,
            "confidences_by_model": {k: {CLASS_ORDER[i]: float(v[i]) for i in range(len(CLASS_ORDER))}
                                      for k, v in probs.items()},
        }

        if votes_ok:
            result["action"] = "통과"
            result["detail"] = {"confidences": {k: float(probs[k][NONE_IDX]) for k in ("base", "wide", "deep")}}
            return result

        if pred_idx != NONE_IDX and wide_conf >= fail_th:
            scratch_score = None
            if pred_label == "Scratch" and scratch_linearity_threshold is not None:
                scratch_score, _ = scratch_linearity_score(padded_64)
                confirmed = (scratch_score is not None) and (scratch_score <= scratch_linearity_threshold)
            else:
                confirmed = True

            if confirmed:
                entry = self.cause_map.get(pred_label, {})
                result["action"] = "불량"
                result["detail"] = {
                    "confidence": wide_conf,
                    "process": entry.get("process", []),
                    "candidates": entry.get("candidates", []),
                    "severity": entry.get("severity", "medium"),
                    "evidence": entry.get("evidence", "domain_inference"),
                    "refs": entry.get("refs", []),
                    "note": entry.get("note", ""),
                }
                if scratch_score is not None:
                    result["detail"]["scratch_linearity_score"] = scratch_score
                return result

            # Scratch 형태 필터를 통과하지 못함 -> '불량' 확정 대신 사람 확인으로 낮춤 (통과로는 절대 안 바뀜)
            top_order = np.argsort(-wide_probs)[:top_k]
            result["action"] = "직접확인필요"
            result["detail"] = {
                "top_candidates": [(CLASS_ORDER[i], float(wide_probs[i])) for i in top_order],
                "candidates_with_process": self._candidates_with_process(wide_probs, top_k),
                "downgrade_reason": (
                    f"CNN 1순위는 'Scratch'였지만 결함 모양이 충분히 길쭉하지 않아(길쭉함 지수 "
                    f"{scratch_score:.3f} > 기준 {scratch_linearity_threshold:.2f}) 자동 '불량' 확정 "
                    f"대신 사람 확인으로 전환했습니다."
                ) if scratch_score is not None else (
                    "CNN 1순위는 'Scratch'였지만 불량 다이가 너무 적어 모양을 판단할 수 없어 "
                    "자동 '불량' 확정 대신 사람 확인으로 전환했습니다."
                ),
            }
            return result

        top_order = np.argsort(-wide_probs)[:top_k]
        result["action"] = "직접확인필요"
        result["detail"] = {
            "top_candidates": [(CLASS_ORDER[i], float(wide_probs[i])) for i in top_order],
            "candidates_with_process": self._candidates_with_process(wide_probs, top_k),
        }
        return result
