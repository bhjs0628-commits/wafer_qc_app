"""
순수 NumPy로 구현한 경량 CNN 라이브러리.

이 샌드박스에는 네트워크 정책상 PyTorch/TensorFlow를 설치할 수 없어서(pip가 pypi에 접근
못 함), 합성곱/풀링/완전연결 계층과 역전파를 NumPy로 직접 구현했다. im2col을 쓰는 대신
"커널 오프셋(i,j)마다 한 번씩 이동시킨 슬라이스에 대해 einsum"을 kh*kw번 반복하는 방식을
썼다 — 구현이 단순해서 버그 위험이 적고, 커널이 작아서(3x3, 5x5) 반복 횟수도 적다.

레이어마다 forward(x)/backward(dout)를 가지며, backward는 그 직전 forward에서 저장해 둔
캐시를 사용한다 (PyTorch의 autograd와 비슷한 구조를 손으로 만든 것).
"""
import numpy as np


class Conv2D:
    def __init__(self, in_ch, out_ch, k, pad, stride=1, seed=None):
        rng = np.random.default_rng(seed)
        # He 초기화 (ReLU와 궁합이 좋음)
        fan_in = in_ch * k * k
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), size=(out_ch, in_ch, k, k)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k, self.pad, self.stride = k, pad, stride
        self.dW = None
        self.db = None
        self._cache = None

    def forward(self, x):
        N, C, H, W = x.shape
        k, p, s = self.k, self.pad, self.stride
        x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode="constant") if p else x
        out_h = (H + 2 * p - k) // s + 1
        out_w = (W + 2 * p - k) // s + 1
        out = np.zeros((N, self.W.shape[0], out_h, out_w), dtype=np.result_type(x, self.W))
        for i in range(k):
            for j in range(k):
                sl = x_pad[:, :, i:i + s * out_h:s, j:j + s * out_w:s]  # (N,C,out_h,out_w)
                out += np.einsum("nchw,oc->nohw", sl, self.W[:, :, i, j], optimize=True)
        out += self.b[None, :, None, None]
        self._cache = (x_pad, out_h, out_w, x.shape)
        return out

    def backward(self, dout):
        x_pad, out_h, out_w, x_shape = self._cache
        N, C, H, W = x_shape
        k, p, s = self.k, self.pad, self.stride
        self.dW = np.zeros_like(self.W)
        self.db = dout.sum(axis=(0, 2, 3))
        dx_pad = np.zeros_like(x_pad)
        for i in range(k):
            for j in range(k):
                sl = x_pad[:, :, i:i + s * out_h:s, j:j + s * out_w:s]
                self.dW[:, :, i, j] = np.einsum("nohw,nchw->oc", dout, sl, optimize=True)
                dx_pad[:, :, i:i + s * out_h:s, j:j + s * out_w:s] += np.einsum(
                    "nohw,oc->nchw", dout, self.W[:, :, i, j], optimize=True
                )
        dx = dx_pad[:, :, p:p + H, p:p + W] if p else dx_pad
        return dx

    def params(self):
        return {"W": (self.W, lambda: self.dW), "b": (self.b, lambda: self.db)}


class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout):
        return dout * self._mask

    def params(self):
        return {}


class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size, self.stride = size, stride

    def forward(self, x):
        N, C, H, W = x.shape
        s = self.size
        out_h, out_w = H // s, W // s
        x_crop = x[:, :, :out_h * s, :out_w * s]
        # (N,C,Ho,s,Wo,s) -> (N,C,Ho,Wo,s,s): 풀링 창 안쪽 두 축을 마지막으로 모아서
        # 윈도우별 argmax를 한 번에 계산할 수 있게 한다.
        reshaped = x_crop.reshape(N, C, out_h, s, out_w, s).transpose(0, 1, 2, 4, 3, 5)
        flat = reshaped.reshape(N, C, out_h, out_w, s * s)
        out = flat.max(axis=-1)
        # 동률(ties, 특히 ReLU 이후 0이 여러 개 겹치는 경우)이 있어도 argmax는 "첫 번째" 위치
        # 하나만 고르므로, 역전파 때 그라디언트가 여러 위치로 중복되지 않는다.
        argmax_idx = flat.argmax(axis=-1)
        self._cache = (argmax_idx, x.shape, out_h, out_w, s)
        return out

    def backward(self, dout):
        argmax_idx, x_shape, out_h, out_w, s = self._cache
        N, C, H, W = x_shape
        dflat = np.zeros((N, C, out_h, out_w, s * s), dtype=dout.dtype)
        np.put_along_axis(dflat, argmax_idx[..., None], dout[..., None], axis=-1)
        dreshaped = dflat.reshape(N, C, out_h, out_w, s, s).transpose(0, 1, 2, 4, 3, 5)
        dx_crop = dreshaped.reshape(N, C, out_h * s, out_w * s)
        dx = np.zeros(x_shape, dtype=dout.dtype)
        dx[:, :, :out_h * s, :out_w * s] = dx_crop
        return dx

    def params(self):
        return {}


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._shape)

    def params(self):
        return {}


class Dense:
    def __init__(self, in_dim, out_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, np.sqrt(2.0 / in_dim), size=(in_dim, out_dim)).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        self.dW = None
        self.db = None

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout):
        self.dW = self._x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self):
        return {"W": (self.W, lambda: self.dW), "b": (self.b, lambda: self.db)}


def softmax_xent_loss(logits, y, sample_weight=None):
    """logits: (N, n_classes), y: (N,) int. sample_weight: (N,) 또는 None.
    반환: (loss_scalar, dlogits)"""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    N = logits.shape[0]
    log_likelihood = -np.log(np.clip(probs[np.arange(N), y], 1e-12, None))
    if sample_weight is None:
        loss = log_likelihood.mean()
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1
        dlogits /= N
    else:
        w = sample_weight
        loss = (log_likelihood * w).sum() / w.sum()
        dlogits = probs.copy()
        dlogits[np.arange(N), y] -= 1
        dlogits *= w[:, None]
        dlogits /= w.sum()
    return loss, dlogits, probs


class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def all_params(self):
        """레이어별로 (param_array, grad_getter) 쌍을 모두 모아서 반환."""
        out = []
        for li, layer in enumerate(self.layers):
            for name, (p, g) in layer.params().items():
                out.append((f"{li}.{name}", p, g))
        return out


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def step(self, named_params):
        self.t += 1
        for name, p, grad_fn in named_params:
            g = grad_fn()
            if g is None:
                continue
            if name not in self.m:
                self.m[name] = np.zeros_like(p)
                self.v[name] = np.zeros_like(p)
            self.m[name] = self.b1 * self.m[name] + (1 - self.b1) * g
            self.v[name] = self.b2 * self.v[name] + (1 - self.b2) * (g * g)
            m_hat = self.m[name] / (1 - self.b1 ** self.t)
            v_hat = self.v[name] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
