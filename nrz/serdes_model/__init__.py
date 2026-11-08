"""SerDes 高速有线收发链路 —— 行为级教学模型。

模块划分 (对应真实收发机结构):

    params   : 全局参数配置 (所有可调参数集中管理)
    signals  : PRBS 信号源、NRZ 整形、噪声注入
    filters  : 双线性变换、IIR 滤波 (scipy 可选加速, numpy 回退)
    tx       : 发送端 (串行化器、TX FFE 预加重、驱动整形)
    channel  : 有损信道 (趋肤+介质损耗、串扰、噪声)
    rx       : 接收端 (VGA、AGC、CTLE、CDR、DFE、判决器、解串器)
    analysis : 眼图、误码率、浴盆曲线、收敛分析
    plotting : matplotlib 可视化 (可选, 核心不依赖)

核心只依赖 numpy, 便于在任何环境运行与移植。
"""
from .params import SerdesParams
from .signals import prbs_bits, bits_to_nrz, symbols_to_nrz, add_awgn
from .filters import bilinear_biquad, lfilter
from .tx import Transmitter
from .channel import channel_transfer, apply_channel
from .rx import Receiver, design_ctle, deserialize
from .analysis import compute_ber, eye_segments, eye_hist2d

__all__ = [
    "SerdesParams",
    "prbs_bits",
    "bits_to_nrz",
    "symbols_to_nrz",
    "add_awgn",
    "bilinear_biquad",
    "lfilter",
    "Transmitter",
    "channel_transfer",
    "apply_channel",
    "Receiver",
    "design_ctle",
    "deserialize",
    "compute_ber",
    "eye_segments",
    "eye_hist2d",
]
