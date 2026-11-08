"""数字滤波器工具 (PAM4 模型)。

与 NRZ 模型相同的实现: 有 scipy 用其 C 实现, 否则回退纯 numpy DF2T。
"""
from __future__ import annotations

import numpy as np

try:  # 可选加速
    from scipy.signal import lfilter as _scipy_lfilter  # type: ignore
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _scipy_lfilter = None
    _HAS_SCIPY = False


def bilinear_biquad(b_analog, a_analog, fs):
    """双线性变换: 把 2 阶模拟系统映射为数字 IIR, 返回 (bz, az)。"""
    b0, b1 = b_analog
    a0, a1, a2 = a_analog
    c = 2.0 * fs
    bz = [b0 * c + b1, 2.0 * b1, b1 - b0 * c]
    az = [a0 * c * c + a1 * c + a2,
          -2.0 * a0 * c * c + 2.0 * a2,
          a0 * c * c - a1 * c + a2]
    return bz, az


def lfilter(b, a, x):
    """IIR 滤波 (scipy 优先, numpy 回退)。"""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    if _HAS_SCIPY:
        return _scipy_lfilter(b, a, np.asarray(x, dtype=float))
    return _lfilter_numpy(b, a, x)


def _lfilter_numpy(b, a, x):
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    N = max(len(b), len(a))
    bb = np.zeros(N)
    bb[:len(b)] = b
    aa = np.zeros(N)
    aa[:len(a)] = a
    if N == 1:
        return bb[0] * np.asarray(x, dtype=float)
    s = np.zeros(N - 1)
    y = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        xi = x[i]
        yi = bb[0] * xi + s[0]
        y[i] = yi
        for j in range(N - 1):
            s[j] = bb[j + 1] * xi - aa[j + 1] * yi + (s[j + 1] if j + 1 < N - 1 else 0.0)
    return y
