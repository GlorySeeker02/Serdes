"""信道模型 (PAM4 模型)。

与 NRZ 模型相同的频域有损传输线:
    - 趋肤效应损耗 ∝ sqrt(f)
    - 介质损耗 ∝ f
零相位近似 (纯幅度衰减), 便于延迟对齐。
"""
from __future__ import annotations

import numpy as np

from .params import PAM4_LEVELS
from .signals import bits_to_pam4, prbs_bits, symbols_to_nrz


def channel_transfer(freqs, loss_db_at_nyquist, nyquist_hz, skin_ratio=0.5):
    """频域传递函数 H(f) (零相位), 幅度按趋肤+介质损耗模型。"""
    f = np.asarray(freqs, dtype=float)
    r = np.zeros_like(f)
    nz = f > 0
    r[nz] = f[nz] / nyquist_hz
    loss_db = loss_db_at_nyquist * (skin_ratio * np.sqrt(r) + (1.0 - skin_ratio) * r)
    mag = 10.0 ** (-loss_db / 20.0)
    return mag.astype(complex)


def apply_channel(wave, dt, params, rng=None):
    """波形过信道, 返回含串扰与噪声的输出。"""
    n = len(wave)
    if params.channel_type == "ideal":
        out = wave.astype(float).copy()
    else:
        freqs = np.fft.rfftfreq(n, d=dt)
        H = channel_transfer(freqs, params.channel_loss_db_at_nyquist,
                             params.nyquist_ghz * 1e9, params.channel_skin_db_ratio)
        out = np.fft.irfft(np.fft.rfft(wave) * H, n=n)

    # 串扰: 用一路独立的 PAM4 邻道信号模拟耦合 (峰值 = channel_xtalk_amp)
    # 注意: 主数据也是 prbs_bits(n_symbols*2, 同阶数), LFSR 初态相同 -> 若攻击源
    # 直接按 2 比特/符号做 Gray 映射, 两条序列会逐符号完全相同, 相加只是把主信号
    # 放大 (眼图反而变好), 完全失去串扰的物理意义。所以在自身周期内旋转偶数个
    # 比特 (= 整数个符号), 保证与主数据不相关 (同谱、不同码型)。
    if params.channel_xtalk_amp > 0:
        nsym = int(np.ceil(n / params.samples_per_ui))
        cbits = prbs_bits(nsym * 2, params.prbs_order)
        shift = max(2, ((2 ** params.prbs_order) // 3) // 2 * 2)
        csyms = bits_to_pam4(np.roll(cbits, shift))
        csyms = csyms * (params.channel_xtalk_amp / PAM4_LEVELS[-1])
        xtalk = symbols_to_nrz(csyms, params.samples_per_ui,
                               rise_ui=params.nrz_rise_ui)
        out = out + xtalk[:n]

    if params.channel_noise_sigma > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        out = out + rng.normal(0.0, params.channel_noise_sigma, n)
    return out
