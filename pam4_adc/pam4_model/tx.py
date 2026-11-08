"""发送端 (PAM4): 比特 -> Gray 符号 -> TX FFE -> 多电平 NRZ 波形。"""
from __future__ import annotations

import numpy as np

from .signals import prbs_bits, bits_to_pam4, symbols_to_nrz


class Transmitter:
    """PAM4 发送端。"""

    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(0)

    def generate_bits(self):
        return prbs_bits(self.params.n_symbols * 2, self.params.prbs_order)

    def bits_to_symbols(self, bits):
        return bits_to_pam4(bits)

    def apply_ffe(self, symbols):
        """TX FFE (预加重), 对符号序列做 FIR。"""
        taps = np.asarray(self.params.tx_ffe_taps, dtype=float)
        return np.convolve(np.asarray(symbols, float), taps, mode="same")

    def to_waveform(self, symbols):
        return symbols_to_nrz(
            symbols, self.params.samples_per_ui,
            rise_ui=self.params.nrz_rise_ui,
            rj_sigma_ui=self.params.tx_rj_sigma_ui,
            rng=self.rng,
        )

    def transmit(self, bits):
        """完整发送: 比特 -> 符号 -> FFE -> 波形 (返回 波形, 符号)。"""
        syms = self.bits_to_symbols(bits)
        wave = self.to_waveform(self.apply_ffe(syms))
        return wave, syms
