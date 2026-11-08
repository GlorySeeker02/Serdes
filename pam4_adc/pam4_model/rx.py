"""PAM4 ADC-based 接收机。

架构 (数字域均衡):
    信道输出 -> VGA/AGC -> [ADC 采样+量化] -> 数字 FFE -> 数字 DFE
             -> PAM4 判决 (3 门限) -> 解串/解码

各模块原理:
    ADC   : 按恢复相位采样, 量化到 adc_bits 位、满量程 ±adc_fs。
    AGC   : 把 99 分位幅度归一化到 PAM4 峰值 3, 使 ADC 满量程匹配。
    FFE   : 居中(非因果)线性均衡器, 按信道幅度反设计, 压前/后驱 ISI。
    DFE   : sign-sign LMS, 用已判决符号抵消后驱 ISI。
    CDR   : PAM4 Alexander 鉴相 (边沿 - 中点), 二阶 PI 环路。
"""
from __future__ import annotations

import numpy as np

from .params import PAM4_LEVELS, PAM4_THRESHOLDS
from .channel import channel_transfer
from .adc import quantize_sample


def pam4_slice(z):
    """PAM4 判决: 3 个门限 (-2, 0, +2) 映射到 4 个电平。"""
    if z < PAM4_THRESHOLDS[0]:
        return -3.0
    if z < PAM4_THRESHOLDS[1]:
        return -1.0
    if z < PAM4_THRESHOLDS[2]:
        return 1.0
    return 3.0


def design_ffe(params, max_boost_db=12.0):
    """按信道幅度响应设计居中线性均衡器 (ZF, 限制最大提升)。

    返回长度 ffe_ntaps 的抽头, 主抽头归一化为 1。
    """
    p = params
    N = 8192
    # FFE 是符号间隔 (1 样本/UI) 的, 其频率轴只到 baud/2 (符号率奈奎斯特),
    # 不能用采样率 fs (fs = spui*baud, 高出 32 倍), 否则会在无信号的高频段盲目提升。
    f_hz = np.fft.rfftfreq(N) * (p.baud_rate_gbps * 1e9)
    M = np.abs(channel_transfer(f_hz, p.channel_loss_db_at_nyquist,
                                p.nyquist_ghz * 1e9, p.channel_skin_db_ratio))
    boost_max = 10.0 ** (max_boost_db / 20.0)
    H_eq = np.minimum(1.0 / np.maximum(M, 1e-9), boost_max)
    h_eq = np.fft.irfft(H_eq, n=N)          # 零相位对称冲激响应
    m = p.ffe_ntaps // 2
    fir = np.concatenate([h_eq[N - m:], h_eq[:m + 1]])   # 负..正, 中心在 m
    if abs(fir[m]) > 1e-12:
        fir = fir / fir[m]
    return fir


class Receiver:
    """PAM4 ADC-based 接收端。"""

    def __init__(self, params):
        self.params = params
        if len(params.ffe_taps) == 0:
            self.ffe_taps = design_ffe(params)
        else:
            self.ffe_taps = np.asarray(params.ffe_taps, dtype=float)

    def front_end(self, x):
        """VGA + 块级 AGC: 把 99 分位幅度归一化到 PAM4 峰值 3。

        该操作与采样相位无关, 复用时可调用一次后传入 run()。
        """
        p = self.params
        x = np.asarray(x, dtype=float) * p.vga_gain
        if len(x):
            level = float(np.percentile(np.abs(x), 99))
            if level > 1e-9:
                x = x * (3.0 / level)
        return x

    def run(self, x, n_symbols, front_end=None):
        """逐符号执行 ADC + FFE + DFE + PAM4 判决 + CDR。

        返回
        ----
        decisions : 判决符号 (±3, ±1), 长度 n_symbols
        phase_hist: CDR 相位轨迹
        dfe_hist  : DFE 抽头轨迹 (n_symbols x dfe_ntaps)
        z_hist    : 判决器输入 (DFE 后, 长度 n_symbols)
        """
        p = self.params
        need = n_symbols * p.samples_per_ui
        if front_end is None:
            front_end = self.front_end(x)
        y = np.asarray(front_end, dtype=float)
        if len(y) < need:
            raise ValueError(f"前端波形长度 {len(y)} 小于 n_symbols*samples_per_ui={need}")
        n = len(y)
        spui = p.samples_per_ui
        adc_bits, adc_fs = p.adc_bits, p.adc_fs
        step = 2.0 * adc_fs / (2 ** adc_bits)

        n_ff = len(self.ffe_taps)
        m = n_ff // 2
        ffe = np.asarray(self.ffe_taps, dtype=float).copy()

        def sample(k, off):
            idx = (k + off + phase) * spui
            i = int(np.floor(idx))
            f = idx - i
            if i < 0 or i + 1 >= n:
                return 0.0
            return y[i] * (1.0 - f) + y[i + 1] * f

        def aq(v):
            return quantize_sample(v, adc_bits, adc_fs)

        phase = p.cdr_initial_phase_ui
        integral = 0.0
        taps = np.zeros(p.dfe_ntaps)
        hist = np.zeros(p.dfe_ntaps)
        decisions = np.empty(n_symbols)
        phase_hist = np.empty(n_symbols)
        dfe_hist = np.empty((n_symbols, p.dfe_ntaps))
        z_hist = np.empty(n_symbols)
        s_prev = 0.0

        for k in range(n_symbols):
            # 1) ADC: 边沿采样 + 数据窗 (居中 FFE 需要 k-m..k+m)
            e = aq(sample(k, 0.0))
            xwin = np.empty(n_ff)
            for t, jj in enumerate(range(-m, m + 1)):
                xwin[t] = aq(sample(k, 0.5 + jj))

            # 2) 数字 FFE
            yf = float(np.dot(ffe, xwin))
            # 3) 数字 DFE
            fb = float(np.dot(taps, hist))
            z = yf - fb
            s = pam4_slice(z)

            # 4) PAM4 Alexander 鉴相
            if s_prev != 0.0 and s != s_prev:
                mid = 0.5 * (s_prev + s)
                err = (e - mid) * (1.0 if s_prev > s else -1.0)
                integral += p.cdr_ki * err
                phase += p.cdr_kp * err + integral

            # 5) 自适应
            if s_prev != 0.0:
                es = z - s
                if p.dfe_train:
                    taps += p.dfe_mu * np.sign(es) * hist
                if p.ffe_train:
                    ffe -= p.ffe_mu * es * xwin

            # 6) 记录
            hist = np.roll(hist, 1)
            hist[0] = s
            s_prev = s
            decisions[k] = s
            phase_hist[k] = phase
            dfe_hist[k] = taps.copy()
            z_hist[k] = z

        return decisions, phase_hist, dfe_hist, z_hist
