"""PAM4 ADC-based SerDes 行为级模型 —— 参数配置。

架构要点 (与 NRZ 模拟均衡不同):
    模拟前端尽量简单 (VGA/AGC), 随后 ADC 采样量化, 全部均衡 (FFE/DFE)
    在数字域完成。这就是现代高速 SerDes 的 "ADC-based / DSP-based" 架构。
"""
from __future__ import annotations

from dataclasses import dataclass

# PAM4 标准电平 (整数形式, 峰值 ±3; 归一化峰值 ±1 时等价 -1,-1/3,1/3,1)
PAM4_LEVELS = (-3.0, -1.0, 1.0, 3.0)
# 三个判决门限 = 相邻电平中点
PAM4_THRESHOLDS = (-2.0, 0.0, 2.0)
# Gray 映射: 双比特 -> 电平 (MSB 先)
LEVEL_BITS = ((0, 0), (0, 1), (1, 1), (1, 0))  # 对应 -3,-1,+1,+3


@dataclass
class PAM4Params:
    """PAM4 ADC-based 收发器参数。"""

    # ---------- 时间基 ----------
    baud_rate_gbps: float = 28.0        # 符号率 [GBd] (比特率 = 2×)
    samples_per_ui: int = 32            # 每 UI 采样点数
    n_symbols: int = 4096               # 发送符号数

    # ---------- 信号源 ----------
    prbs_order: int = 7
    nrz_rise_ui: float = 0.2

    # ---------- 发送端 ----------
    tx_swing: float = 1.0
    tx_ffe_taps: tuple = (-0.1, 1.0, -0.1)  # [pre, main, post]
    tx_rj_sigma_ui: float = 0.0

    # ---------- 信道 ----------
    channel_type: str = "lossy"         # 'lossy' | 'ideal'
    channel_loss_db_at_nyquist: float = 6.0
    channel_skin_db_ratio: float = 0.5
    channel_noise_sigma: float = 0.01
    channel_xtalk_amp: float = 0.0

    # ---------- RX 模拟前端 ----------
    vga_gain: float = 1.0
    analog_ctle_peaking_db: float = 0.0   # 模拟 CTLE (ADC-based 下通常很小/关闭)

    # ---------- ADC ----------
    adc_bits: int = 6                     # ADC 分辨率
    adc_fs: float = 4.0                   # ADC 满量程 ±FS

    # ---------- 数字 FFE ----------
    ffe_ntaps: int = 5                    # 居中抽头数 (预/主/后)
    ffe_taps: tuple = ()                  # 空 -> 按信道自动设计
    ffe_train: bool = False               # 是否 LMS 自适应
    ffe_mu: float = 2e-4

    # ---------- 数字 DFE ----------
    dfe_ntaps: int = 3
    dfe_mu: float = 0.005
    dfe_train: bool = True

    # ---------- CDR ----------
    cdr_type: str = "bangbang"
    cdr_kp: float = 1.0 / 64.0
    cdr_ki: float = 1.0 / 2048.0
    cdr_initial_phase_ui: float = 0.0

    # ---------- 派生量 ----------
    ui: float = 0.0
    dt: float = 0.0
    fs: float = 0.0
    nyquist_ghz: float = 0.0
    bits_per_symbol: int = 2

    def __post_init__(self):
        self.ui = 1e-9 / self.baud_rate_gbps
        self.dt = self.ui / self.samples_per_ui
        self.fs = 1.0 / self.dt
        self.nyquist_ghz = self.baud_rate_gbps / 2.0
