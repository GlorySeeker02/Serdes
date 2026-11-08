"""发送端 (TX): 串行化器、TX FFE 预加重、驱动整形。

预加重 (pre-emphasis / de-emphasis) 原理:
    信道高频损耗会衰减上升/下降沿, 造成码间干扰 (ISI)。TX FFE
    通过 FIR 滤波在发送端预先放大跳变分量 (相对减小低频分量),
    使到达接收端的信号高频与低频平衡。
"""
from __future__ import annotations

import numpy as np

from .signals import prbs_bits, symbols_to_nrz


class Transmitter:
    """发送端行为级模型。

    流程: PRBS 比特 -> 符号映射 (+1/-1) -> TX FFE -> NRZ 整形。
    """

    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(0)

    def generate_bits(self):
        return prbs_bits(self.params.nbits, self.params.prbs_order)

    def apply_ffe(self, bits):
        """TX 前馈均衡 (预加重/去加重)。

        对 UI 速率符号序列做 FIR 滤波: taps = [pre, main, post]。
        main 为主抽头 (≈1), pre/post 用于预加重高频跳变分量。
        """
        taps = np.asarray(self.params.tx_taps, dtype=float)
        syms = np.where(np.asarray(bits) > 0, 1.0, -1.0)
        return np.convolve(syms, taps, mode="same")

    def to_waveform(self, symbols):
        """符号序列 -> NRZ 波形 (含沿整形与 TX 抖动)。"""
        return symbols_to_nrz(
            symbols, self.params.samples_per_ui,
            rise_ui=self.params.nrz_rise_ui,
            rj_sigma_ui=self.params.tx_rj_sigma_ui,
            rng=self.rng,
        )

    def transmit(self, bits):
        """完整发送: 比特 -> FFE -> 波形。"""
        return self.to_waveform(self.apply_ffe(bits))
