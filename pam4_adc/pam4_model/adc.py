"""ADC 模型: 采样后量化 + 限幅。

ADC-based 接收机的核心: 把模拟波形数字化, 之后所有均衡在数字域完成。
理想 N-bit 量化器的量化步长 Δ = 2·FS / 2^N, 量化噪声近似均匀分布,
标准差 σ_q = Δ / sqrt(12)。
"""
from __future__ import annotations

import numpy as np


def quantize(x, nbits, full_scale):
    """把 x 量化为 nbits 位、满量程 ±full_scale 的离散值。"""
    step = 2.0 * full_scale / (2 ** nbits)
    q = np.clip(np.asarray(x, float), -full_scale, full_scale)
    return np.round(q / step) * step


def quantize_sample(v, nbits, full_scale):
    """单个样本量化。"""
    step = 2.0 * full_scale / (2 ** nbits)
    v = min(max(v, -full_scale), full_scale)
    return round(v / step) * step


def quantization_sigma(nbits, full_scale):
    """量化噪声标准差。"""
    step = 2.0 * full_scale / (2 ** nbits)
    return step / np.sqrt(12.0)
