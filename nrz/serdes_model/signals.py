"""信号源模块。

提供 PRBS 伪随机序列生成、UI 速率符号序列到 NRZ 波形的整形,
以及加性高斯噪声注入。这是整个链路的激励来源。

原理:
    PRBS 由线性反馈移位寄存器 (LFSR) 产生, 特征多项式为
    x^order + ... + 1, 周期 2^order - 1。PRBS 频谱近似白噪声,
    是 SerDes 链路验证的标准测试码型 (PRBS7/PRBS31)。
"""
from __future__ import annotations

import numpy as np

# PRBS 反馈抽头 (0 基索引), 对应本原多项式
_POLY_TAPS = {
    7: (6, 5),      # x^7 + x^6 + 1
    9: (8, 4),      # x^9 + x^5 + 1
    11: (10, 8),    # x^11 + x^9 + 1
    15: (14, 13),   # x^15 + x^14 + 1
    23: (22, 17),   # x^23 + x^18 + 1
    31: (30, 27),   # x^31 + x^28 + 1
}


def prbs_bits(n: int, order: int = 7) -> np.ndarray:
    """生成 n 位 PRBS 序列 (LFSR)。

    参数
    ----
    n : 序列长度 [bit]
    order : 阶数, 决定周期 2^order - 1

    返回
    ----
    0/1 整数数组, 长度 n。
    """
    if order not in _POLY_TAPS:
        raise ValueError(f"不支持的 PRBS 阶数: {order}, 可选 {sorted(_POLY_TAPS)}")
    t1, t2 = _POLY_TAPS[order]
    state = [1] * order
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        out[i] = state[-1]
        new = state[t1] ^ state[t2]
        state = [new] + state[:-1]
    return out


def symbols_to_nrz(symbols, samples_per_ui, rise_ui=0.2,
                   rj_sigma_ui=0.0, rng=None):
    """把 UI 速率的符号序列 (任意电平) 整形为 NRZ 连续波形。

    每个符号持续 samples_per_ui 个采样点; 相邻符号电平不同时用
    升余弦沿过渡 (rise_ui 为过渡区间占 UI 的比例)。可选在跳变沿
    相位上叠加随机抖动 rj_sigma_ui (单位 UI)。

    返回
    ----
    波形数组, 长度 = len(symbols) * samples_per_ui。
    """
    symbols = np.asarray(symbols, dtype=float)
    spui = int(samples_per_ui)
    wave = np.repeat(symbols, spui).astype(float)
    rise = max(1, int(round(rise_ui * spui)))
    if rng is None:
        rng = np.random.default_rng(0)

    for k in range(1, len(symbols)):
        a, b = symbols[k - 1], symbols[k]
        if a != b:
            center = k * spui + rng.normal(0.0, rj_sigma_ui * spui)
            lo = int(max(0, np.floor(center - rise)))
            hi = int(min(len(wave), np.ceil(center + rise)))
            for idx in range(lo, hi):
                t = (idx - (center - rise)) / (2 * rise)
                t = min(max(t, 0.0), 1.0)
                shape = 0.5 * (1.0 - np.cos(np.pi * t))
                wave[idx] = a + (b - a) * shape
    return wave


def bits_to_nrz(bits, samples_per_ui, rise_ui=0.2, swing=1.0,
                rj_sigma_ui=0.0, rng=None):
    """把 0/1 bit 序列转为 NRZ 波形 (1 -> +swing, 0 -> -swing)。"""
    levels = np.where(np.asarray(bits) > 0, swing, -swing).astype(float)
    return symbols_to_nrz(levels, samples_per_ui, rise_ui, rj_sigma_ui, rng)


def add_awgn(wave, sigma, rng=None):
    """叠加零均值高斯白噪声。"""
    if rng is None:
        rng = np.random.default_rng(0)
    return wave + rng.normal(0.0, sigma, size=wave.shape)
