"""lab 公共工具: 建链路、量眼图、打印表格。

所有分阶段实验脚本都 import 本模块, 以便:
  * 统一以 ``SerdesParams()`` 默认值作为"基线链路";
  * 统一输出格式, 直接复制进笔记。

本模块不定义任何新原理, 只是把 serdes_model 里的函数组合成
"实验员视角"的工具: 改一个参数 -> 跑一次 -> 量一个数。
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serdes_model import (  # noqa: E402
    SerdesParams, Transmitter, Receiver, apply_channel,
    symbols_to_nrz, channel_transfer,
)
from serdes_model.analysis import compute_ber  # noqa: E402
from serdes_model.filters import lfilter  # noqa: E402

OUT = ROOT / "output" / "lab"
OUT.mkdir(parents=True, exist_ok=True)

#: "没有 TX FFE" 的抽头 —— 研究纯信道行为时用它做基线
NOFFE = (0.0, 1.0, 0.0)

np.set_printoptions(suppress=True, precision=4)


# ======================================================================
# 参数
# ======================================================================
def params(**kw) -> SerdesParams:
    """按默认值建参数, 再覆盖指定字段, 并重算派生量 (ui/dt/fs)。

    改 ``data_rate_gbps`` 或 ``samples_per_ui`` 后必须重算派生量,
    否则 UI 长度、采样率会与波形不匹配 —— 这是新手最常见的坑。
    """
    p = SerdesParams()
    for k, v in kw.items():
        if not hasattr(p, k):
            raise AttributeError(f"SerdesParams 没有字段 {k!r}")
        setattr(p, k, v)
    p.__post_init__()
    return p


def variant(p: SerdesParams, **kw) -> SerdesParams:
    """复制一份参数并覆盖若干字段 (不改原对象), 用于扫参。"""
    q = dataclasses.replace(p, **kw)
    q.__post_init__()
    return q


# ======================================================================
# 链路
# ======================================================================
class Link:
    """一次 TX -> 信道 的波形准备。

    只跟"发送什么、经什么信道"有关, 与 RX 参数无关, 因此可以在
    同一份波形上反复换 RX 参数跑 (扫 CTLE / CDR / DFE 时都复用)。
    """

    def __init__(self, p: SerdesParams, seed: int = 0):
        self.p = p
        self.tx = Transmitter(p)
        self.bits = self.tx.generate_bits()
        self.wave_tx = self.tx.transmit(self.bits)
        self.wave_ch = apply_channel(self.wave_tx, p.dt, p)

    def front_end(self, p: SerdesParams | None = None):
        """VGA + CTLE + AGC。与采样相位无关, 扫相位时只算一次。"""
        return Receiver(p or self.p).front_end(self.wave_ch)

    def run(self, p: SerdesParams | None = None, front_end=None, nbits=None) -> "Result":
        p = p or self.p
        n = p.nbits if nbits is None else nbits
        rx = Receiver(p)
        dec, ph, dfe, z = rx.run(self.wave_ch, n, front_end=front_end)
        return Result(p=p, rx=rx, dec=dec, ph=ph, dfe=dfe, z=z, bits=self.bits, n=n)


@dataclasses.dataclass
class Result:
    """一次 RX 运行的产物 + 常用指标。"""

    p: SerdesParams
    rx: Receiver
    dec: np.ndarray
    ph: np.ndarray
    dfe: np.ndarray
    z: np.ndarray
    bits: np.ndarray
    n: int

    def ber(self, skip: int = 300, max_lag: int = 40):
        return compute_ber(self.dec, self.bits[: self.n], skip=skip, max_lag=max_lag)

    @property
    def phase_final(self) -> float:
        """锁定后的相位 (末段中位数), 比取最后一个点稳健。"""
        return float(np.median(self.ph[-500:]))

    @property
    def taps_final(self) -> np.ndarray:
        return self.dfe[-1]

    def phase_jitter(self, skip: int = 300) -> float:
        """锁定后相位抖动的标准差 [UI]。"""
        return float(np.std(self.ph[skip:]))

    def eye_z(self, skip: int = 300, tail: float = 0.02):
        """判决器输入 z 的 (垂直眼高, 垂直厚度)。

        这才是"DFE 收干净了没有"的直接读数: z 已经是每 UI 一个点、
        且扣掉了反馈量, 所以它的两簇内沿就是剩余的眼开度。
        """
        z = self.z[skip:]
        b = np.asarray(self.bits[: self.n])[skip:] > 0
        hi, lo = z[b], z[~b]
        h = float(np.quantile(hi, tail) - np.quantile(lo, 1.0 - tail))
        s = float(0.5 * (np.quantile(hi, 1 - tail) - np.quantile(hi, tail)
                         + np.quantile(lo, 1 - tail) - np.quantile(lo, tail)))
        return h, s

    def settle_time(self, tol: float = 0.03, win: int = 64) -> int:
        """锁定时间 [bit]: 相位(滑动平均后)**首次**进入稳态值 ±tol 的时刻。

        为什么用"首次进入"而不是"最后一次离开": bang-bang + 积分支路会让相位
        缓慢游走, "最后一次越界"会落在曲线末尾, 读数恒等于总比特数, 毫无意义。
        锁定快不快看这个值, 稳不稳看 phase_jitter()。返回 n 表示一直没进入。
        """
        ph = self.ph
        ref = float(np.median(ph[-500:]))
        if len(ph) < win:
            return len(ph)
        ma = np.convolve(ph, np.ones(win) / win, mode="valid")
        ok = np.flatnonzero(np.abs(ma - ref) <= tol)
        return int(ok[0] + win) if len(ok) else len(ph)


# ======================================================================
# 采样与眼图指标
# ======================================================================
def sample_ui(y, sp: int, n: int, phase: float = 0.0):
    """在 (k + 0.5 + phase) * sp 处线性插值, 取 n 个"数据中心"采样。"""
    y = np.asarray(y, dtype=float)
    idx = (np.arange(n) + 0.5 + phase) * sp
    i = np.clip(np.floor(idx).astype(int), 0, len(y) - 2)
    f = idx - np.floor(idx)
    return y[i] * (1.0 - f) + y[i + 1] * f


def eye_stats(y, sp: int, bits, phase: float = 0.0, tail: float = 0.02):
    """(垂直眼高, 垂直厚度), 单位 V。

    眼高取"内沿": 高电平组的下 ``tail`` 分位 减 低电平组的上 ``tail`` 分位。
    用分位数而不是 min/max, 是为了让极少数最坏码型/噪声尖峰不至于让
    指标抖到没法比较; 取 2% 尾, 相当于"98% 的码型都不会比这个更差"。

    垂直厚度 = 两组各自 96% 跨度的平均, 反映噪声/残余 ISI 把眼沿"抹厚"的程度。
    """
    v = sample_ui(y, sp, len(bits), phase)
    b = np.asarray(bits) > 0
    hi, lo = v[b], v[~b]
    h = float(np.quantile(hi, tail) - np.quantile(lo, 1.0 - tail))
    s = float(0.5 * (np.quantile(hi, 1 - tail) - np.quantile(hi, tail)
                     + np.quantile(lo, 1 - tail) - np.quantile(lo, tail)))
    return h, s


def eye_height(y, sp: int, bits, phase: float = 0.0, tail: float = 0.02) -> float:
    """垂直眼高 [V] (见 eye_stats)。理想 NRZ 为 2.0。"""
    return eye_stats(y, sp, bits, phase, tail)[0]


def eye_sigma(y, sp: int, bits, phase: float = 0.0, tail: float = 0.02) -> float:
    """垂直厚度 [V] (见 eye_stats)。纯理想信道 + 无噪声时接近 0。"""
    return eye_stats(y, sp, bits, phase, tail)[1]


def eye_profile(y, sp: int, bits, tail: float = 0.02):
    """UI 内 32 个相位处的垂直眼高 -> "眼图的数学定义"。

    返回 (phase[UI], opening[V]); 曲线在 UI 中心处最高, 靠近边沿迅速降到 0,
    曲线的宽度就是水平眼宽, 高度就是垂直眼高。
    """
    ph = np.arange(sp) / sp - 0.5
    op = np.array([eye_height(y, sp, bits, phase=p, tail=tail) for p in ph])
    return ph, op


def eye_width(y, sp: int, bits, frac: float = 0.5) -> float:
    """水平眼宽 [UI]: 垂直眼高高于峰值 ``frac`` 倍所占的相位跨度。

    frac=0.5 就是常说的"眼宽 @50% 眼高", 也是时钟抖动容限的粗略估计。
    """
    ph, op = eye_profile(y, sp, bits)
    peak = float(op.max())
    if peak <= 0:
        return 0.0
    ok = op > frac * peak
    if not np.any(ok):
        return 0.0
    return float(np.ptp(ph[ok]) + 1.0 / sp)


def bathtub(link: Link, phases, p: SerdesParams | None = None,
            train: bool = True, skip: int = 300, front_end=None, max_lag: int = 0):
    """关掉 CDR 环路, 扫固定采样相位, 返回每个相位的 BER。

    这就是浴盆曲线的数值版本: BER 随相位变化的"盆底"宽度 = 水平眼宽。

    max_lag 默认 0: 这里相位是**故意**扫的, 不能让 compute_ber 用整数延迟
    去"对齐" —— 否则采样点偏出一整个 UI 时, 它会把判决整体移一位, 把本应
    全错的相位算成 0 误码, 浴盆曲线就失真了。
    """
    p = p or link.p
    y = front_end if front_end is not None else Receiver(p).front_end(link.wave_ch)
    bers = []
    for ph in phases:
        q = variant(p, cdr_kp=0.0, cdr_ki=0.0,
                    cdr_initial_phase_ui=float(ph), dfe_train=train)
        r = link.run(q, front_end=y)
        bers.append(r.ber(skip=skip, max_lag=max_lag)[0])
    return np.asarray(bers, dtype=float)


def width_at(phases, bers, thr: float = 1e-6) -> float:
    """浴盆曲线中 BER < thr 的相位跨度 [UI]。"""
    ok = np.asarray(bers) < thr
    if not np.any(ok):
        return 0.0
    return float(np.ptp(np.asarray(phases)[ok]))


# ======================================================================
# 脉冲响应 (ISI 抽头)
# ======================================================================
def pulse_cursors(p: SerdesParams, loss_db=None, skin=None, xform=None,
                  tx_taps=None, pre: int = 16, ones: int = 40,
                  tail: int = 16, span: int = 8, margin: int = 3,
                  phase: float = 0.0):
    """单比特响应 (SBR) 的 ISI 游标值。

    做法: 送一个理想阶跃 (0...0,1...1,0...0) 进信道, 采样得到阶跃响应 S[k];
    单比特响应就是它的差分 h[k] = S[k] - S[k-1]。

    为什么不是"直接送一个孤立脉冲"? 因为模型用 FFT 做循环卷积, 如果波形
    两端电平不同, 首尾的跳变会通过"环绕"污染整个结果。所以这里两端都补 0,
    把上升沿和下降沿都放在窗口内部, 只在上升沿附近取数。

    tx_taps : TX FFE 抽头; 默认取 ``p.tx_taps``。FFE 是符号域 FIR,
              "对阶跃做 FFE" 等价于 "对游标做卷积", 所以这里直接卷 h,
              不必真的重发一遍波形。
    phase   : 采样相位 [UI]。ISI 系数是**随采样相位变化**的, 要跟 RX 的
              CDR 锁定相位对齐着比较 (否则会看到 h1 差一倍)。

    返回 (offsets, cursors, main):
        offsets  : -span .. +span 的整数 UI 偏移
        cursors  : 各偏移处的 h[k], 已按主游标归一化 (h[0] = 1)
        main     : 归一化前的主游标值 (即"一个比特能爬到阶跃的百分之几")

    物理含义: cursors[k] 就是第 k 个 ISI 系数。DFE 抵消 offset >= +1 的后游标,
    CTLE/TX FFE 主要改善 offset <= -1 的前游标。
    """
    sp = p.samples_per_ui
    n = pre + ones + tail
    syms = np.zeros(n)
    syms[pre:pre + ones] = 1.0
    wave = symbols_to_nrz(syms, sp, rise_ui=p.nrz_rise_ui)

    q = params(
        channel_type="lossy",
        channel_noise_sigma=0.0,
        channel_xtalk_amp=0.0,
        channel_loss_db_at_nyquist=(p.channel_loss_db_at_nyquist if loss_db is None else loss_db),
        channel_skin_db_ratio=(p.channel_skin_db_ratio if skin is None else skin),
    )
    y = apply_channel(wave, p.dt, q)
    if xform is not None:
        y = xform(y)

    s = sample_ui(y, sp, n, phase)             # 阶跃响应 S[k] (在给定采样相位取)
    h = np.diff(np.concatenate(([0.0], s)))    # h[k] = S[k] - S[k-1]

    # 在 ±(span+margin) 范围内取原始游标, 再把 TX FFE 卷进去
    taps = np.asarray(p.tx_taps if tx_taps is None else tx_taps, dtype=float)
    if len(taps) != 3:
        raise ValueError("目前只支持 3-tap FFE [pre, main, post]")
    lo, hi = pre - span - margin, pre + span + margin
    H = h[lo:hi + 1]
    offs = np.arange(-span, span + 1)
    idx = offs + span + margin
    heff = taps[1] * H[idx] + taps[0] * H[idx + 1] + taps[2] * H[idx - 1]

    main = float(heff[span])
    return offs, heff / main, main


def rc_lowpass(x, tau_ui: float, sp: int):
    """一阶 RC 低通 (单极点), 时间常数 tau 以 UI 为单位。

    用来手工造一个"只看带宽、不看复杂 S 参数"的极简信道:
    它同时抹平边沿 (减小眼宽) 和拖出长尾 (腐蚀眼高), 是理解
    "为什么信道会让眼图闭合"的最小模型。
    """
    a = 1.0 - np.exp(-1.0 / max(tau_ui * sp, 1e-9))
    x = np.asarray(x, dtype=float)
    y = np.empty(len(x))
    acc = float(x[0])
    for i in range(len(x)):
        acc += a * (float(x[i]) - acc)
        y[i] = acc
    return y


def ctle_xform(p: SerdesParams, agc: bool = True):
    """返回一个"只做 VGA+CTLE(+AGC)"的波形变换函数, 用于脉冲响应分析。"""
    rx = Receiver(p)

    def f(y):
        z = lfilter(rx.bd, rx.ad, np.asarray(y, dtype=float) * p.vga_gain)
        if agc:
            lvl = float(np.median(np.abs(z)))
            if lvl > 1e-9:
                z = z / lvl
        return z

    return f


def channel_db(freqs_hz, loss_db, nyq_hz, skin):
    """信道幅度响应 [dB], 直接调模型里的传递函数。"""
    return 20.0 * np.log10(np.abs(channel_transfer(freqs_hz, loss_db, nyq_hz, skin)) + 1e-15)


def ctle_db(p: SerdesParams, freqs_hz):
    """CTLE 幅度响应 [dB]: 冲激响应做 FFT。"""
    rx = Receiver(p)
    n = 8192
    imp = np.zeros(n)
    imp[0] = 1.0
    ir = lfilter(rx.bd, rx.ad, imp)
    H = np.fft.rfft(ir, n=n)
    f = np.fft.rfftfreq(n, d=1.0 / p.fs)
    return np.interp(freqs_hz, f, 20.0 * np.log10(np.abs(H) + 1e-15))


# ======================================================================
# 打印
# ======================================================================
def _dw(text) -> int:
    """显示宽度: 中日韩全角字符按 2 列计。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in str(text))


def h1(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def h2(title: str) -> None:
    print(f"\n--- {title} ---")


def ask(q: str) -> None:
    print(f"  [先猜] {q}")


def obs(s) -> None:
    print(f"  [实测] {s}")


def why(s) -> None:
    print(f"  [解释] {s}")


def warn(s) -> None:
    print(f"  [注意] {s}")


def tbl(headers, rows) -> None:
    """打印对齐的等宽表格 (中文按 2 列宽)。"""
    rows = [[("" if v is None else v) for v in r] for r in rows]
    cells = [[str(h) for h in headers]] + [[str(v) for v in r] for r in rows]
    w = [max(_dw(r[i]) for r in cells) for i in range(len(headers))]
    line = "+".join("-" * (x + 2) for x in w)
    print("+" + line + "+")
    print("| " + " | ".join(str(h) + " " * (w[i] - _dw(h)) for i, h in enumerate(headers)) + " |")
    print("+" + line + "+")
    for r in rows:
        print("| " + " | ".join(str(v) + " " * (w[i] - _dw(v)) for i, v in enumerate(r)) + " |")
    print("+" + line + "+")


def banner(no: str, title: str, goal: str) -> None:
    h1(f"实验 {no}  {title}")
    print(f"目标: {goal}")
    print("用法: 先读『先猜』, 自己想一个答案, 再看『实测』和『解释』。")


def note(s: str) -> None:
    print(f"\n  >> 笔记: {s}")


def get_plt():
    """延迟导入 matplotlib (只在要画图时才付出这个代价)。"""
    from serdes_model.plotting import plt, save_fig, eye_density
    return plt, save_fig, eye_density


__all__ = [
    "OUT", "NOFFE", "params", "variant", "Link", "Result",
    "sample_ui", "eye_height", "eye_sigma", "eye_stats", "eye_profile",
    "eye_width", "bathtub", "width_at",
    "pulse_cursors", "ctle_xform", "channel_db", "ctle_db", "rc_lowpass",
    "h1", "h2", "ask", "obs", "why", "warn", "tbl", "banner", "note",
    "get_plt", "np", "Receiver", "SerdesParams",
]


