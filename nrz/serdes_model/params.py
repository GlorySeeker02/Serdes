"""全局参数配置。

所有可调参数集中在此, 便于扫参实验: 只改配置、不改流程。
单位约定: 时间统一用 UI (1 bit 周期), 数据速率用 Gbps。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SerdesParams:
    """SerDes 行为级模型参数。

    默认值为一个 10 Gbps 通用 NRZ 收发器, 可直接运行。
    """

    # ---------- 时间基 ----------
    data_rate_gbps: float = 10.0        # 数据速率 [Gbps]
    samples_per_ui: int = 32            # 每 UI 采样点数 (时间分辨率)
    nbits: int = 4096                   # 发送 bit 总数

    # ---------- 信号源 ----------
    prbs_order: int = 7                 # PRBS 阶数 (7->PRBS7, 31->PRBS31)
    nrz_rise_ui: float = 0.2            # NRZ 上升/下降时间 [UI]

    # ---------- 发送端 TX ----------
    tx_swing: float = 1.0               # 发送摆幅 [V]
    tx_taps: tuple = (-0.15, 0.85, -0.2)  # 3-tap FFE [pre, main, post]
    tx_rj_sigma_ui: float = 0.0         # TX 随机抖动标准差 [UI]

    # ---------- 信道 ----------
    channel_type: str = "lossy"         # 'lossy' | 'ideal'
    channel_loss_db_at_nyquist: float = 20.0   # 奈奎斯特频点插入损耗 [dB]
    channel_skin_db_ratio: float = 0.5         # 损耗中趋肤效应(∝√f)占比
    channel_noise_sigma: float = 0.01   # 加性高斯白噪声标准差 [V]
    channel_xtalk_amp: float = 0.0      # 串扰幅度 [V]

    # ---------- RX 模拟前端 ----------
    vga_gain: float = 1.0               # 可变增益放大器增益 (固定, AGC 见文档)
    ctle_dc_gain: float = 1.0           # CTLE 直流增益
    ctle_peaking_db: float = 6.0        # CTLE 高频 peaking [dB]
    ctle_peak_freq_ghz: float = 5.0     # CTLE peaking 频率 [GHz]

    # ---------- CDR ----------
    cdr_type: str = "bangbang"          # 'bangbang'(量化鉴相) | 'linear'(模拟边沿)
    cdr_kp: float = 1.0 / 64.0          # 比例增益 [UI/单位误差]
    cdr_ki: float = 1.0 / 2048.0        # 积分增益 [UI/单位误差/UI]
    cdr_initial_phase_ui: float = 0.0   # 初始采样相位 [UI]

    # ---------- DFE ----------
    dfe_ntaps: int = 5                  # 反馈抽头数 (h1..hN)
    dfe_mu: float = 0.01                # sign-sign LMS 步长
    dfe_train: bool = True              # 是否开启自适应

    # ---------- 派生量 (由 __post_init__ 计算) ----------
    ui: float = field(init=False)          # 位周期 [s]
    dt: float = field(init=False)          # 采样间隔 [s]
    fs: float = field(init=False)          # 采样率 [Hz]
    nyquist_ghz: float = field(init=False)  # 奈奎斯特频率 [GHz]

    def __post_init__(self):
        self.ui = 1e-9 / self.data_rate_gbps
        self.dt = self.ui / self.samples_per_ui
        self.fs = 1.0 / self.dt
        self.nyquist_ghz = self.data_rate_gbps / 2.0
