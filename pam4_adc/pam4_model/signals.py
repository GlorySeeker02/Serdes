"""信号源 (PAM4): PRBS、比特<->PAM4 符号 (Gray)、多电平 NRZ 整形。

PAM4 每个符号携带 2 比特, 采用 Gray 映射使相邻电平只差 1 比特:
    00 -> -3,  01 -> -1,  11 -> +1,  10 -> +3
"""
from __future__ import annotations

import numpy as np

from .params import PAM4_LEVELS, LEVEL_BITS

_POLY_TAPS = {
    7: (6, 5), 9: (8, 4), 11: (10, 8), 15: (14, 13), 23: (22, 17), 31: (30, 27),
}


def prbs_bits(n: int, order: int = 7) -> np.ndarray:
    """生成 n 位 PRBS 序列 (LFSR)。"""
    if order not in _POLY_TAPS:
        raise ValueError(f"不支持的 PRBS 阶数: {order}")
    t1, t2 = _POLY_TAPS[order]
    state = [1] * order
    out = np.empty(n, dtype=np.int8)
    for i in range(n):
        out[i] = state[-1]
        new = state[t1] ^ state[t2]
        state = [new] + state[:-1]
    return out


def bits_to_pam4(bits) -> np.ndarray:
    """比特流 -> PAM4 符号 (Gray), 长度 = len(bits)//2。

    Gray 映射: 00->-3, 01->-1, 11->+1, 10->+3。
    以两位二进制值 a*2+b 索引查表: [00,01,10,11] -> [-3,-1,+3,+1]。
    """
    bits = np.asarray(bits, dtype=np.int8)
    n = (len(bits) // 2) * 2
    pairs = bits[:n].reshape(-1, 2)
    table = np.array([-3.0, -1.0, 3.0, 1.0])   # 索引 = a*2+b
    return table[pairs[:, 0] * 2 + pairs[:, 1]]


def pam4_to_bits(levels) -> np.ndarray:
    """PAM4 符号 (可含判决电平) -> 比特流 (Gray), 长度 = 2×len(levels)。"""
    levels = np.asarray(levels, dtype=float)
    lv = np.asarray(PAM4_LEVELS)
    idx = np.argmin(np.abs(levels[:, None] - lv[None, :]), axis=1)
    bits = np.empty(len(levels) * 2, dtype=np.int8)
    for i, j in enumerate(idx):
        bits[2 * i], bits[2 * i + 1] = LEVEL_BITS[j]
    return bits


def symbols_to_nrz(symbols, samples_per_ui, rise_ui=0.2,
                   rj_sigma_ui=0.0, rng=None):
    """把 UI 速率的符号序列 (任意电平) 整形为 NRZ 波形 (升余弦沿)。"""
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
                wave[idx] = a + (b - a) * 0.5 * (1.0 - np.cos(np.pi * t))
    return wave


def bits_to_nrz(bits, samples_per_ui, rise_ui=0.2, swing=1.0,
                rj_sigma_ui=0.0, rng=None):
    """0/1 bit -> ±swing NRZ 波形 (用于串扰等辅助信号)。"""
    levels = np.where(np.asarray(bits) > 0, swing, -swing).astype(float)
    return symbols_to_nrz(levels, samples_per_ui, rise_ui, rj_sigma_ui, rng)


def add_awgn(wave, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    return wave + rng.normal(0.0, sigma, size=np.shape(wave))
