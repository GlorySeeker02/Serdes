"""接收端 (RX): VGA、CTLE、CDR、DFE、判决器、解串器。

采用行为级、逐 UI 处理的因果模型, 结构对应真实接收机:

    信道输出 -> VGA(固定增益) -> CTLE(连续时间均衡) -> CDR 采样
              -> DFE(判决反馈均衡) -> 判决器 -> 解串器

各模块原理:
    CTLE : 一零二极连续时间均衡器, 经双线性变换离散化为 IIR,
           补偿信道高频损耗、减小前驱 ISI。
    CDR  : bang-bang (Alexander) 鉴相器 + 二阶环路滤波 (PI),
           从数据边沿恢复采样时钟相位。
    DFE  : 判决反馈均衡, 用已判决 bit 抵消后驱 ISI,
           sign-sign LMS 自适应收敛。

核心只依赖 numpy (双线性变换、IIR、线性插值均为手写实现),
无需 scipy/matplotlib。
"""
from __future__ import annotations

import numpy as np

from .filters import bilinear_biquad, lfilter

# 向后兼容: 旧版以私有函数名暴露这两个工具, 保留别名
_bilinear_biquad = bilinear_biquad
_lfilter = lfilter


def design_ctle(params):
    """设计 CTLE 滤波器 (一零二极, 双线性变换为 IIR)。

    模拟传递函数:
        H(s) = A_dc * (1 + s/wz) / ((1 + s/wp1)(1 + s/wp2))
    其中 wp1 < wz < wp2, 在 f_peak 附近产生约 peaking_db 的高频增益。
    """
    p = params
    a_dc = p.ctle_dc_gain
    pk = 10.0 ** (p.ctle_peaking_db / 20.0)
    f_peak = p.ctle_peak_freq_ghz * 1e9
    wz = 2.0 * np.pi * f_peak / np.sqrt(pk)
    wp1 = 2.0 * np.pi * f_peak * np.sqrt(pk)
    wp2 = 2.0 * np.pi * (p.data_rate_gbps * 1e9 * 1.5)
    b_analog = [a_dc / wz, a_dc]
    a_analog = [1.0 / (wp1 * wp2), 1.0 / wp1 + 1.0 / wp2, 1.0]
    return bilinear_biquad(b_analog, a_analog, p.fs)


class Receiver:
    """接收端行为级模型。"""

    def __init__(self, params):
        self.params = params
        self.bd, self.ad = design_ctle(params)

    def front_end(self, x):
        """VGA + CTLE + 块级 AGC 模拟前端。

        AGC (自动增益控制) 原理: sign-sign LMS 的误差 sign(z - s) 只有在
        判决器输入的主抽头幅度 ≈ ±1 时才反映真实 ISI 方向; 若幅度偏离 ±1,
        误差符号会被判决电平 s 主导而失效。这里用 |x| 的鲁棒中位数估计
        NRZ 主抽头电平 (NRZ 平顶段占多数样本, 中位数 ≈ 主抽头), 归一化到 1。

        该操作与采样相位无关, 多个接收机复用时可调用一次后传入 run()。
        """
        p = self.params
        x = x * p.vga_gain
        x = lfilter(self.bd, self.ad, x)
        level = float(np.median(np.abs(x))) if len(x) else 1.0
        if level > 1e-9:
            x = x / level
        return x

    # 向后兼容旧方法名
    _front_end = front_end

    def run(self, x, nbits, front_end=None):
        """逐 UI 执行 CDR + DFE + 判决。

        参数
        ----
        x        : 信道输出波形 (长度应 >= nbits * samples_per_ui)
        nbits    : 处理比特数
        front_end: 可选, 预计算的前端输出 (VGA+CTLE+AGC 后)。多个接收机
                   复用同一前端 (例如扫相位) 时可避免重复滤波。

        返回
        ----
        decisions : +1/-1 判决序列 (长度 nbits)
        phase_hist: CDR 相位轨迹 (长度 nbits)
        dfe_hist  : DFE 抽头收敛轨迹 (nbits x dfe_ntaps)
        z_hist    : DFE 均衡后的采样 (判决器输入, 长度 nbits)
        """
        p = self.params
        need = nbits * p.samples_per_ui
        if front_end is None:
            front_end = self.front_end(x)
        y = np.asarray(front_end, dtype=float)
        if len(y) < need:
            raise ValueError(f"前端波形长度 {len(y)} 小于 nbits*samples_per_ui={need}")
        n = len(y)
        spui = p.samples_per_ui

        phase = p.cdr_initial_phase_ui
        integral = 0.0
        taps = np.zeros(p.dfe_ntaps)
        decisions = np.empty(nbits)
        phase_hist = np.empty(nbits)
        dfe_hist = np.empty((nbits, p.dfe_ntaps))
        z_hist = np.empty(nbits)
        hist = np.zeros(p.dfe_ntaps)   # 最近 dfe_ntaps 个历史判决

        def sample(k, off):
            """在 (k + off + phase) 个 UI 处对 y 线性插值采样。"""
            idx = (k + off + phase) * spui
            i = int(np.floor(idx))
            frac = idx - i
            if i < 0 or i + 1 >= n:
                return 0.0
            return y[i] * (1.0 - frac) + y[i + 1] * frac

        s_prev = 0.0
        for k in range(nbits):
            # 1) 采样: 边沿 (off=0) 与数据中心 (off=0.5)
            e = sample(k, 0.0)
            d = sample(k, 0.5)

            # 2) DFE 判决反馈
            fb = float(np.dot(taps, hist))
            z = d - fb
            s = 1.0 if z >= 0.0 else -1.0

            # 3) Alexander 鉴相: 相位误差 = 边沿采样 × 跳变方向
            #    bang-bang: 边沿采样量化为 ±1; linear: 保留模拟边沿值
            if p.cdr_type == "bangbang":
                e = 1.0 if e >= 0.0 else -1.0
            err = e * (s_prev - s)
            integral += p.cdr_ki * err
            phase += p.cdr_kp * err + integral

            # 4) sign-sign LMS 自适应 DFE
            if p.dfe_train and s_prev != 0.0:
                e_slice = z - s
                taps += p.dfe_mu * np.sign(e_slice) * hist

            # 5) 更新历史判决与记录
            hist = np.roll(hist, 1)
            if p.dfe_ntaps > 0:        # dfe_ntaps=0 时没有历史寄存器, 跳过
                hist[0] = s
            s_prev = s
            decisions[k] = s
            phase_hist[k] = phase
            dfe_hist[k] = taps.copy()
            z_hist[k] = z

        return decisions, phase_hist, dfe_hist, z_hist


def deserialize(decisions, word_width=8):
    """解串器: 把串行判决按 word_width 分组为并行字 (0/1)。"""
    bits = np.where(decisions > 0, 1, 0).astype(np.int8)
    usable = len(bits) - (len(bits) % word_width)
    return bits[:usable].reshape(-1, word_width)
