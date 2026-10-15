"""
CNN 아키텍처 정의 — 학습/평가/트리아지 스크립트가 공통으로 사용 (가중치 shape를
일치시키기 위해 모델 구조를 한 곳에서만 정의).

입력 해상도 32x32로 다운스케일한 이유: 이 샌드박스는 GPU가 없고 CPU 2코어뿐이라
PyTorch/TensorFlow도 설치할 수 없었음(네트워크 정책상 pip가 pypi에 접근 불가) —
그래서 cnn_lib.py에 NumPy로 직접 합성곱/역전파를 구현했다. 64x64 입력으로 벤치마크한
결과 1 epoch(16.5만장)에 약 20분이 걸려 비현실적이었고, 32x32로 낮추니 약 5분으로
줄어 실습 가능한 수준이 됐다. (06_preprocess.py가 만든 64x64 데이터 자체는 그대로
보존되어 있으므로, 나중에 GPU 환경에서 PyTorch로 재학습할 때는 64x64 원본을 그대로
쓸 수 있다 — 이번 다운스케일은 "이 샌드박스에서 손으로 짠 CNN을 학습시키기 위한"
임시 조치임을 분명히 해 둔다.)
"""
import cv2
import numpy as np
from cnn_lib import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Sequential

IMG_SIZE = 32  # 32x32로 다운스케일해서 학습 (사유는 위 docstring 참고)

CLASS_ORDER = ["none", "Center", "Donut", "Edge-Loc", "Edge-Ring",
               "Loc", "Random", "Scratch", "Near-full"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}
N_CLASSES = len(CLASS_ORDER)


def downscale_batch(X64):
    """(N,64,64) float32[0,1] -> (N,32,32) float32[0,1]. 범주형 값이 아니라 이미 정규화된
    연속값이므로(0/0.5/1의 평균 등 다양한 값이 섞일 수 있음) area 보간을 사용 —
    여러 다이를 하나의 더 큰 셀로 합칠 때 다운샘플링에 적합한 보간 방식."""
    N = X64.shape[0]
    out = np.empty((N, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(N):
        out[i] = cv2.resize(X64[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return out


def build_model(seed_base=0):
    return Sequential([
        Conv2D(1, 8, 5, 2, seed=seed_base + 1), ReLU(), MaxPool2D(2, 2),    # 32x32 -> 16x16
        Conv2D(8, 16, 3, 1, seed=seed_base + 2), ReLU(), MaxPool2D(2, 2),   # 16x16 -> 8x8
        Conv2D(16, 32, 3, 1, seed=seed_base + 3), ReLU(), MaxPool2D(2, 2),  # 8x8 -> 4x4
        Flatten(),
        Dense(32 * 4 * 4, 64, seed=seed_base + 4), ReLU(),
        Dense(64, N_CLASSES, seed=seed_base + 5),
    ])


def build_model_wide(seed_base=0):
    """베이스라인보다 채널 폭을 2배로 늘린 버전 (8/16/32->16/32/64, dense 64->128).
    미탐지(불량인데 none으로 아주 높은 확신도로 오판) 문제가 모델 용량 부족 때문인지
    확인하기 위한 비교 실험용 — 입력 해상도(32x32)와 레이어 구조(conv3+dense2)는 그대로
    유지하고 '폭'만 키워서, 변수를 하나만 바꾼 공정한 비교가 되도록 함."""
    return Sequential([
        Conv2D(1, 16, 5, 2, seed=seed_base + 1), ReLU(), MaxPool2D(2, 2),    # 32x32 -> 16x16
        Conv2D(16, 32, 3, 1, seed=seed_base + 2), ReLU(), MaxPool2D(2, 2),   # 16x16 -> 8x8
        Conv2D(32, 64, 3, 1, seed=seed_base + 3), ReLU(), MaxPool2D(2, 2),   # 8x8 -> 4x4
        Flatten(),
        Dense(64 * 4 * 4, 128, seed=seed_base + 4), ReLU(),
        Dense(128, N_CLASSES, seed=seed_base + 5),
    ])


def build_model_deep(seed_base=0):
    """3번째 앙상블 멤버 — base/wide와 '폭'이 아니라 '구조'를 다르게 해서 오류 패턴의
    다양성을 더 키우기 위한 버전. 첫 블록에 conv를 하나 더 쌓아(풀링 전에 3x3 두 번)
    receptive field를 더 점진적으로 키우고, 5x5 큰 커널 대신 3x3만 사용. 채널 폭은
    base와 비슷하게 가볍게 유지해 학습 속도를 base 수준으로 맞춤(앙상블 3번째 멤버를
    학습하는 데 시간이 너무 오래 걸리지 않도록)."""
    return Sequential([
        Conv2D(1, 8, 3, 1, seed=seed_base + 1), ReLU(),
        Conv2D(8, 8, 3, 1, seed=seed_base + 2), ReLU(), MaxPool2D(2, 2),   # 32x32 -> 16x16
        Conv2D(8, 16, 3, 1, seed=seed_base + 3), ReLU(), MaxPool2D(2, 2),  # 16x16 -> 8x8
        Conv2D(16, 32, 3, 1, seed=seed_base + 4), ReLU(), MaxPool2D(2, 2), # 8x8 -> 4x4
        Flatten(),
        Dense(32 * 4 * 4, 64, seed=seed_base + 5), ReLU(),
        Dense(64, N_CLASSES, seed=seed_base + 6),
    ])


def save_weights(model, path):
    arrays = {}
    for name, p, _ in model.all_params():
        arrays[name] = p
    np.savez(path, **arrays)


def load_weights(model, path):
    data = np.load(path)
    for name, p, _ in model.all_params():
        p[...] = data[name]
