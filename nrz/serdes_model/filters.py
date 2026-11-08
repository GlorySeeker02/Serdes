"""数字滤波器工具: 双线性变换与 IIR 滤波。

核心模型只依赖 numpy。若环境中安装了 scipy, ``lfilter`` 会自动使用其
C 实现加速 (结果与纯 numpy 版本一致); 否则回退到纯 numpy 的直接 II 型
转置结构。因此本模块在 "只有 numpy" 和 "numpy+scipy" 两种环境下都能工作。
"""
from __future__ import annotations

import numpy as np

try:  # 可选加速, 非必需依赖
    from scipy.signal import lfilter as _scipy_lfilter  # type: ignore
    _HAS_SCIPY = True
except Exception:  # noqa: BLE001
    _scipy_lfilter = None
    _HAS_SCIPY = False


def bilinear_biquad(b_analog, a_analog, fs):
    """双线性变换: s -> 2*fs*(1-z^-1)/(1+z^-1)。

    把 2 阶模拟系统
        H(s) = (b0*s + b1) / (a0*s^2 + a1*s + a2)
    映射为数字 IIR。返回 (bz, az), 均为按 z^-1 降幂排列的三系数列表。

    推导: 令 w = z^-1, s = c(1-w)/(1+w), c = 2*fs, 代入并通分即可得到
    下述系数; 数字滤波器的直流增益 sum(bz)/sum(az) = b1/a2 保持不变。
    """
    b0, b1 = b_analog
    a0, a1, a2 = a_analog
    c = 2.0 * fs
    bz = [b0 * c + b1, 2.0 * b1, b1 - b0 * c]
    az = [a0 * c * c + a1 * c + a2,
          -2.0 * a0 * c * c + 2.0 * a2,
          a0 * c * c - a1 * c + a2]
    return bz, az


def lfilter(b, a, x):
    """IIR 滤波, 返回与 x 等长的 float64 数组。

    有 scipy 时调用其 C 实现 (快), 否则回退纯 numpy。两者都先把分母
    首系数归一化为 1, 结果一致。
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]
    if _HAS_SCIPY:
        return _scipy_lfilter(b, a, np.asarray(x, dtype=float))
    return _lfilter_numpy(b, a, x)


def _lfilter_numpy(b, a, x):
    """纯 numpy 回退: 直接 II 型转置结构 (DF2T), 支持任意小阶数。"""
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    b = b / a[0]
    a = a / a[0]
    N = max(len(b), len(a))
    bb = np.zeros(N)
    bb[:len(b)] = b
    aa = np.zeros(N)
    aa[:len(a)] = a
    if N == 1:                       # 0 阶 (纯增益), 无状态
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
