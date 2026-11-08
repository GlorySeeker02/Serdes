"""PAM4 ADC-based SerDes 行为级模型。

架构 (数字域均衡):
    TX: PRBS -> Gray 映射为 PAM4 符号 -> TX FFE -> 多电平 NRZ 整形
    信道: 趋肤+介质损耗 (零相位近似) + 串扰 + 噪声
    RX: VGA/AGC -> ADC (采样+量化) -> 数字 FFE -> 数字 DFE -> PAM4 判决 -> CDR

核心只依赖 numpy; 绘图可选 matplotlib。
"""
from .params import PAM4Params, PAM4_LEVELS, PAM4_THRESHOLDS
from .signals import prbs_bits, bits_to_pam4, pam4_to_bits, symbols_to_nrz, add_awgn
from .filters import bilinear_biquad, lfilter
from .tx import Transmitter
from .channel import channel_transfer, apply_channel
from .adc import quantize, quantize_sample, quantization_sigma
from .rx import Receiver, pam4_slice, design_ffe
from .analysis import compute_ser, compute_ber, eye_hist2d, eye_openings

__all__ = [
    "PAM4Params",
    "PAM4_LEVELS",
    "PAM4_THRESHOLDS",
    "prbs_bits",
    "bits_to_pam4",
    "pam4_to_bits",
    "symbols_to_nrz",
    "add_awgn",
    "bilinear_biquad",
    "lfilter",
    "Transmitter",
    "channel_transfer",
    "apply_channel",
    "quantize",
    "quantize_sample",
    "quantization_sigma",
    "Receiver",
    "pam4_slice",
    "design_ffe",
    "compute_ser",
    "compute_ber",
    "eye_hist2d",
    "eye_openings",
]
