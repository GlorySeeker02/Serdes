"""信道模型。

用频域传递函数描述有损传输线。损耗分两部分:
    - 趋肤效应: 高频电流挤向导体表面, 损耗 ∝ sqrt(f)
    - 介质损耗: 绝缘体极化损耗, 损耗 ∝ f
在奈奎斯特频点给定额定插入损耗, 并按比例分配二者。

简化假设: 相位取零相位 (纯幅度衰减), 即忽略群时延, 便于
后续延迟对齐。真实信道会引入与频率相关的群时延。
"""
from __future__ import annotations

import numpy as np

from .signals import bits_to_nrz, prbs_bits


def channel_transfer(freqs, loss_db_at_nyquist, nyquist_hz, skin_ratio=0.5):
    """计算频域传递函数 H(f)。

    损耗模型 (dB):
        L(f) = L_nyq * [ skin_ratio * (f/f_nyq)^0.5
                        + (1 - skin_ratio) * (f/f_nyq) ]
    """
    f = np.asarray(freqs, dtype=float)
    r = np.zeros_like(f)
    nz = f > 0
    r[nz] = f[nz] / nyquist_hz
    loss_db = loss_db_at_nyquist * (skin_ratio * np.sqrt(r) + (1.0 - skin_ratio) * r)
    mag = 10.0 ** (-loss_db / 20.0)
    return mag.astype(complex)   # 零相位


def apply_channel(wave, dt, params, rng=None):
    """将波形通过信道, 返回滤波后波形 (含串扰与噪声)。"""
    n = len(wave)
    if params.channel_type == "ideal":
        out = wave.astype(float).copy()
    else:
        freqs = np.fft.rfftfreq(n, d=dt)
        H = channel_transfer(freqs, params.channel_loss_db_at_nyquist,
                             params.nyquist_ghz * 1e9, params.channel_skin_db_ratio)
        out = np.fft.irfft(np.fft.rfft(wave) * H, n=n)

    # 串扰: 用一个独立 PRBS 流模拟相邻通道耦合
    # 注意: Transmitter.generate_bits() 也用 prbs_bits(nbits, 同阶数), LFSR 初态
    # 相同 -> 两条序列会完全相同。若直接相加, "串扰"就变成了主信号的相干副本
    # (反而把眼图抬高), 完全失去串扰的物理意义。所以这里把攻击源序列在自身
    # 周期内旋转一段, 保证它与本通道数据不相关 (同谱、不同码型)。
    if params.channel_xtalk_amp > 0:
        nbits = int(np.ceil(n / params.samples_per_ui))
        cbits = prbs_bits(nbits, params.prbs_order)
        shift = max(1, (2 ** params.prbs_order) // 3)
        cbits = np.roll(cbits, shift)
        xtalk = bits_to_nrz(cbits, params.samples_per_ui,
                            rise_ui=params.nrz_rise_ui,
                            swing=params.channel_xtalk_amp)
        out = out + xtalk[:n]

    # 加性高斯白噪声
    if params.channel_noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        out = out + rng.normal(0.0, params.channel_noise_sigma, n)
    return out
