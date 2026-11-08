"""分析工具 (PAM4): 符号误码率 SER、比特误码率 BER、眼图直方图。"""
from __future__ import annotations

import numpy as np

from .signals import pam4_to_bits


def _align_count(dec, ref, skip, max_lag):
    """在 [skip:] 上搜索最优整数 lag, 返回 (误码数, 比较长度, lag)。"""
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
    return best


def compute_ser(decisions, tx_levels, skip=200, max_lag=40):
    """符号误码率 SER。返回 (ser, lag)。"""
    dec = np.asarray(decisions, dtype=float)
    ref = np.asarray(tx_levels, dtype=float)
    errs, L, lag = _align_count(dec, ref, skip, max_lag)
    return errs / L, lag


def compute_ber(decisions, tx_levels, skip=200, max_lag=40):
    """比特误码率 BER (Gray 解码后比较)。返回 (ber, lag, ser)。"""
    ser, lag = compute_ser(decisions, tx_levels, skip, max_lag)
    db = pam4_to_bits(decisions)
    tb = pam4_to_bits(tx_levels)
    dskip = 2 * skip
    if lag >= 0:
        a = db[dskip:]
        b = tb[dskip + 2 * lag:]
    else:
        a = db[dskip - 2 * lag:]
        b = tb[dskip:]
    L = min(len(a), len(b))
    errs = int(np.sum(a[:L] != b[:L]))
    return errs / L, lag, ser


def eye_hist2d(wave, samples_per_ui, nbins=64):
    """按 UI 折叠的 2D 密度直方图, 返回 (hist[nbins, per], lo, hi)。"""
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


def eye_openings(wave, samples_per_ui, levels=(-3.0, -1.0, 1.0, 3.0)):
    """在眼图中心处估计 3 个眼开口 (相邻电平电平差, 用采样点均值估计)。

    返回 (时间轴[UI], 各电平中心均值)。仅作示意, 采样在 UI 中心。
    """
    per = int(samples_per_ui)
    rows = len(wave) // per
    seg = wave[:rows * per].reshape(rows, per)
    center = seg[:, per // 2]
    means = [float(np.mean(center[center > (levels[i] + levels[i + 1]) / 2]))
             if np.any(center > (levels[i] + levels[i + 1]) / 2) else float('nan')
             for i in range(len(levels))]
    return means
