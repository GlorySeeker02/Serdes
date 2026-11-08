"""分析工具: 眼图、误码率、收敛轨迹。

只依赖 numpy; 绘图交给示例脚本 (优先 matplotlib, 否则 Pillow)。
"""
from __future__ import annotations

import numpy as np


def compute_ber(decisions, bits, skip=200, max_lag=40):
    """计算误码率 (BER)。

    跳过前 skip 个 bit (CTLE/CDR/DFE 收敛预热), 并在 [-max_lag, max_lag]
    范围内搜索最优整数延迟对齐。

    返回
    ----
    (BER, latency) : BER 为误码率, latency 为最优整数延迟。
    """
    dec = np.where(decisions > 0, 1, 0).astype(np.int8)
    ref = np.asarray(bits, dtype=np.int8)
    best = None
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = dec[skip:]
            b = ref[skip + lag:]
        else:
            a = dec[skip - lag:]
            b = ref[skip:]
        L = min(len(a), len(b))
        if L <= 0:
            continue
        errs = int(np.sum(a[:L] != b[:L]))
        if best is None or errs < best[0]:
            best = (errs, L, lag)
    errs, L, lag = best
    return errs / L, lag


def eye_segments(wave, samples_per_ui):
    """把波形按 UI 折叠, 返回 (时间轴[UI], 二维片段矩阵)。

    矩阵形状 (rows, samples_per_ui), 每行是一个 UI 的波形片段;
    时间轴以 UI 中心为 0, 范围 [-0.5, 0.5)。
    """
    per = int(samples_per_ui)
    rows = len(wave) // per
    seg = wave[:rows * per].reshape(rows, per)
    t_ui = (np.arange(per) - per / 2) / per
    return t_ui, seg


def eye_hist2d(wave, samples_per_ui, nbins=64):
    """眼图 2D 直方图 (眼图密度)。

    返回
    ----
    (hist[nbins, samples_per_ui], amp_min, amp_max)
    """
    per = int(samples_per_ui)
    rows = len(wave) // per
    seg = wave[:rows * per].reshape(rows, per)
    amp_min, amp_max = float(seg.min()), float(seg.max())
    span = (amp_max - amp_min) or 1.0
    idx = np.clip(((seg - amp_min) / span * (nbins - 1)).astype(int), 0, nbins - 1)
    hist = np.zeros((nbins, per))
    for i in range(per):
        hist[:, i] = np.bincount(idx[:, i], minlength=nbins)
    return hist, amp_min, amp_max
