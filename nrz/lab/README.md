# `serdes_model` 与 `lab/common.py` 使用手册

两个东西的分工，一句话：

> **`serdes_model/` 是被测对象（DUT）**：一条完整链路，只依赖 numpy，不打印、不画图。
> **`lab/common.py` 是实验室**：信号源开关、探头、记录仪。它不定义任何新原理，
> 只负责"改一个参数 → 跑一次 → 量一个数 → 打印成表"。

这样分层的原因很实际：`serdes_model` 要能被当成"链路本身"来信任（无 matplotlib、
无 I/O 依赖），而实验需要表格、图片、中文输出——这些都放在 `common.py` 里。

```
lab01..lab08.py          ← 每个实验 = 一组 common 的调用
      │
      ▼
lab/common.py            ← Link / Result / 眼图指标 / 浴盆 / 游标 / 打印
      │  import
      ▼
serdes_model/            ← params → signals → tx → channel → rx → analysis
   (纯 numpy)                plotting.py 是唯一例外 (可选)
```

---

# 第一部分：`serdes_model` 逐模块

## 1. `params.py` —— 一个 dataclass 管所有旋钮

```python
@dataclass
class SerdesParams:
    data_rate_gbps: float = 10.0
    samples_per_ui: int = 32
    nbits: int = 4096
    ...
    # 派生量 (init=False)
    ui: float; dt: float; fs: float; nyquist_ghz: float

    def __post_init__(self):          # params.py:62
        self.ui = 1e-9 / self.data_rate_gbps      # 100 ps
        self.dt = self.ui / self.samples_per_ui   # 3.125 ps
        self.fs = 1.0 / self.dt                   # 320 GS/s
        self.nyquist_ghz = self.data_rate_gbps/2  # 5 GHz
```

**字段分 7 组**：时间基 / 信号源 / TX / 信道 / RX 模拟前端 / CDR / DFE。

**唯一的坑**：`ui/dt/fs/nyquist_ghz` 是 `__post_init__` 算的，只在构造时算一次。
所以先构造再 `p.data_rate_gbps = 5` **不会**更新 `p.ui`——波形时间基就错了。
`common.params()` 帮你补了这一步（见第二部分）。

```python
from serdes_model import SerdesParams
p = SerdesParams()
p.ui, p.dt, p.fs, p.nyquist_ghz
# (1e-10, 3.125e-12, 320000000000.0, 5.0)   ← UI 100 ps / dt 3.125 ps / 320 GS/s / 5 GHz
```

## 2. `signals.py` —— 激励：PRBS 与 NRZ 波形

### `prbs_bits(n, order=7)`　`signals.py:26`

标准 LFSR：`state[-1]` 输出，`new = state[t1] ^ state[t2]`，右移补进队首。
抽头表 `_POLY_TAPS` 只支持 7/9/11/15/23/31（对应本原多项式）。

```python
from serdes_model import prbs_bits
bits = prbs_bits(4096, 7)      # 周期 127, 够 32 个完整周期
```

**注意**：初态固定为全 1，所以**同一阶数永远生成同一条序列**。
这条特性在 `channel.py` 里曾导致"串扰 = 主数据的副本"（已修，见附录）。

### `symbols_to_nrz(symbols, samples_per_ui, rise_ui=0.2, rj_sigma_ui=0.0)`　`signals.py:50`

把任意电平的符号序列展开成波形：

1. `np.repeat(symbols, spui)` → 阶梯波（每符号 `spui` 点）
2. 对每个 `symbols[k-1] != symbols[k]` 的位置 `k*spui`，用**升余弦**写一段过渡：
   `rise = max(1, round(rise_ui*spui))`，过渡区间 `[center-rise, center+rise)`
   → **实际过渡宽度 = 2×rise_ui 个 UI**
3. `rj_sigma_ui` 给每次跳变的中心加高斯抖动 `N(0, (rj*spui)²)`

**为什么是升余弦**：它是"带限"的平滑边沿，不会引入额外的高频毛刺；
真实驱动器的边沿也接近这种形状。

### `bits_to_nrz(bits, ...)`　`signals.py:83`

`0/1 → ∓swing` 的薄封装，等价于 `symbols_to_nrz(np.where(bits>0, swing, -swing), ...)`。

### `add_awgn(wave, sigma, rng=None)`　`signals.py:90`

独立的高斯白噪声工具。注意 `apply_channel` 内部已经能加噪声，
这个函数是给"手工搭链路"的时候用的。

## 3. `tx.py` —— 发送端：PRBS → FFE → 波形

```python
class Transmitter:
    def __init__(self, params)      # 自带 rng = default_rng(0)
    def generate_bits(self)         # -> prbs_bits(p.nbits, p.prbs_order)
    def apply_ffe(self, bits)       # tx.py:28  符号域 FIR
    def to_waveform(self, symbols)  # tx.py:38  调 symbols_to_nrz
    def transmit(self, bits)        # tx.py:47  = to_waveform(apply_ffe(bits))
```

`apply_ffe` 就是一个 3 抽头卷积：

```python
taps = np.asarray(self.params.tx_taps)      # (-0.15, 0.85, -0.2)
syms = np.where(bits > 0, 1.0, -1.0)
return np.convolve(syms, taps, mode="same")
```

**三个必须记住的性质**：
- 直流增益 = `sum(taps)` = 0.5 → 长连 1 的稳态电平只有 0.5
- 孤立比特摆幅 = `|pre| + main + |post|` = 1.2 → 跳变被抬高
- `mode="same"` 在两端有边界效应（`np.convolve` 会在两端补 0），
  所以 `lab03` 手工补了 2 个符号再卷积，避免教学展示被边界污染

## 4. `channel.py` —— 有损信道

### `channel_transfer(freqs, loss_db_at_nyquist, nyquist_hz, skin_ratio=0.5)`　`channel.py:18`

```python
r = f / nyquist_hz
loss_db = L_nyq * (skin * sqrt(r) + (1-skin) * r)
mag = 10 ** (-loss_db/20)          # 返回复数, 但相位恒为 0
```

- `√f` = 趋肤效应，`f` = 介质损耗，`skin_ratio` 是配比
- **零相位**：只有幅度衰减，没有群时延。所以本模型的脉冲响应是**对称**的
  （`h(-1) ≈ h(+1)`），真实信道不会这样
- 直流 `H(0) = 1`（`loss_db=0`），这就是"长连 1/0 电平基本不变"的原因

### `apply_channel(wave, dt, params, rng=None)`　`channel.py:34`

三步：

```python
1. 频域滤波: y = irfft(rfft(wave) * H(f))      # 循环卷积!
2. 串扰:     y += xtalk_amp * NRZ(另一路 PRBS)
3. 噪声:     y += N(0, sigma)
```

**这里是全模型最需要警惕的一处**：`rfft/irfft` 做的是**循环卷积**。
若波形首尾电平不同，两端的跳变会"绕回来"污染结果。所以：

- 正常码流（4096 bit）几乎无影响
- 但 `pulse_cursors()` 用的窗口很短，必须**两端补 0** 才能算对（见第二部分）

`channel_type="ideal"` 时直接 `copy()` 直通，用于单独调 TX/RX。

## 5. `filters.py` —— 双线性变换 + IIR

### `bilinear_biquad(b_analog, a_analog, fs)`　`filters.py:19`

把二阶模拟系统 `H(s) = (b0·s + b1)/(a0·s² + a1·s + a2)` 离散化：

```
s = c(1 - w)/(1 + w),  w = z^-1,  c = 2·fs
```

返回 `(bz, az)`，均为 3 个系数（按 `z^-1` 降幂）。性质：直流增益
`sum(bz)/sum(az) = b1/a2` 保持不变。

### `lfilter(b, a, x)`　`filters.py:39`

- 有 scipy → 用其 C 实现（快）
- 没有 → `_lfilter_numpy`（直接 II 型转置，DF2T）逐样本循环

两者都会先把 `a[0]` 归一化为 1，结果一致。这是"核心只依赖 numpy"的关键 20 行。

## 6. `rx.py` —— 接收端（本模型的核心）

### `design_ctle(params)`　`rx.py:30`

由「peaking 多少 dB、峰在哪」反推零极点：

```python
pk   = 10 ** (peaking_db/20)
wz   = 2*pi*f_peak / sqrt(pk)      # 零点在峰左边
wp1  = 2*pi*f_peak * sqrt(pk)      # 极点 1 在峰右边
wp2  = 2*pi*1.5*data_rate          # 极点 2 固定 15 GHz (压住高端)
b_analog = [a_dc/wz, a_dc]
a_analog = [1/(wp1*wp2), 1/wp1 + 1/wp2, 1]
return bilinear_biquad(...)        # -> (bd, ad)
```

### `Receiver.front_end(x)`　`rx.py:56`  —— 三行就是全部

```python
x = x * p.vga_gain                 # 1) VGA
x = lfilter(self.bd, self.ad, x)   # 2) CTLE
level = np.median(np.abs(x))       # 3) AGC: 用中位数估主抽头
if level > 1e-9: x = x / level
```

**关键设计**：这个函数**与采样相位无关**，所以扫相位/扫 DFE 参数时可以只算一次，
然后通过 `run(..., front_end=y)` 复用（`lab06`/`lab08` 扫 40 个相位的浴盆只滤一次波）。

### `Receiver.run(x, nbits, front_end=None)`　`rx.py:77`

逐 UI 循环，每轮 5 步（行号对应当前代码）：

```python
e = sample(k, 0.0)            # rx.py:125  边沿采样
d = sample(k, 0.5)            # rx.py:126  中心采样

fb = dot(taps, hist); z = d - fb      # 129-130  DFE 反馈
s  = 1.0 if z >= 0 else -1.0          # 131      判决

e = 1.0 if e >= 0 else -1.0           # 136      bang-bang 量化
err = e * (s_prev - s)                # 137      Alexander 鉴相
integral += ki * err                  # 138      积分支路(频率)
phase += kp * err + integral          # 139      相位更新

taps += mu * sign(z - s) * hist       # 144      sign-sign LMS

hist = roll(hist, 1); hist[0] = s     # 147     历史寄存器(错误传播就在这里)
```

`sample(k, off)` 是唯一的采样入口：

```python
idx = (k + off + phase) * spui        # 相位 = 小数 UI
i, frac = floor(idx), idx - floor(idx)
return y[i]*(1-frac) + y[i+1]*frac    # 线性插值; 越界返回 0
```

返回值：`(decisions, phase_hist, dfe_hist, z_hist)`
- `decisions` 是 ±1（不是 0/1）
- `z_hist` 是**判决器输入**（DFE 之后），量"眼开度"应该看它
- `dfe_hist` 形状 `(nbits, dfe_ntaps)`

**两条硬约束**：
1. `len(x) >= nbits*samples_per_ui`，否则抛 `ValueError`
2. `nbits` 越大越准，但 4096 bit 以下 DFE 常常"看起来收敛了其实还在漂"

### `deserialize(decisions, word_width=8)`　`rx.py:159`

串行 ±1 → 0/1 → 按字宽 reshape。教学用，说明"解串器就是分组"。

## 7. `analysis.py` —— 误码率与眼图

### `compute_ber(decisions, bits, skip=200, max_lag=40)`　`analysis.py:10`

- 跳过前 `skip` 个 bit（收敛预热）
- 在 `[-max_lag, +max_lag]` 内**搜索最优整数延迟**，取误码最少的那个

**这个延迟搜索是个陷阱**：做浴盆曲线时，如果采样点偏出一整个 UI，
判决序列只是整体移了一位，`compute_ber` 会"帮你对齐"→ 本应全错的相位变成 BER 0。
所以做浴盆必须 `max_lag=0`（`common.bathtub` 已经默认这么做）。

### `eye_segments` / `eye_hist2d`　`analysis.py:40` / `analysis.py:53`

按 UI 折成 `(rows, spui)` 矩阵，再对每一列做幅度直方图。
`plotting.eye_density` 就是把 `eye_hist2d` 画成热力图。

## 8. `plotting.py` —— 只给示例/实验用的可视化

- 顶部强制 `MPLCONFIGDIR = output/.mpl`（避免写用户目录被沙箱拦）
- `matplotlib.use("Agg")`（无头）
- 注册 `simhei/msyh` 中文字体，并 `axes.unicode_minus = False`

`eye_density(wave, sp, ax=..., title=...)`、`save_fig(fig, path)`。

## 9. `__init__.py` —— 公开接口

只导出常用函数，**不导出** `SerdesParams` 之外的类细节。
`from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel` 足够搭链路；
更细的工具从子模块取（`serdes_model.filters` / `serdes_model.analysis`）。

---

# 第二部分：`lab/common.py` 逐块

## 1. 顶部：路径、常量、打印选项

```python
ROOT = Path(__file__).resolve().parent.parent   # -> nrz/
sys.path.insert(0, str(ROOT))                   # 让 lab 能 import serdes_model
OUT  = ROOT / "output" / "lab"                  # 图片/日志统一落这里
NOFFE = (0.0, 1.0, 0.0)                         # "没有 FFE" 的基线抽头
np.set_printoptions(suppress=True, precision=4) # 打印好看
```

有了这 4 行，任何 `labXX.py` 只要 `from common import *` 就能跑，不用管 cwd。

## 2. `params(**kw)` 与 `variant(p, **kw)`

```python
def params(**kw):                # 新建一份, 覆盖字段, 重算派生量
    p = SerdesParams()
    for k, v in kw.items(): setattr(p, k, v)
    p.__post_init__()            # ★ 关键: 重算 ui/dt/fs/nyquist
    return p

def variant(p, **kw):            # 复制一份再覆盖, 不改原对象 (扫参用)
    q = dataclasses.replace(p, **kw)
    q.__post_init__()
    return q
```

**为什么必须 `__post_init__()`**：dataclass 的 `__init__` 只在"构造时"跑一次，
之后再 `setattr` 不会更新 `ui/dt/fs`。少了这一步，改 `data_rate_gbps` 后波形长度
和 UI 就对不上，症状是"眼图莫名其妙地糊了"。

**`params` vs `variant` 的用法分工**：

```python
p   = params(channel_loss_db_at_nyquist=20.0)   # 从默认值起, 写实验
q   = variant(p, cdr_kp=0.0, cdr_ki=0.0)        # 在一份配置上派生扫参
```

## 3. `Link` —— 把"发送什么 + 经什么信道"冻结下来

```python
class Link:
    def __init__(self, p):
        self.tx     = Transmitter(p)
        self.bits   = self.tx.generate_bits()
        self.wave_tx = self.tx.transmit(self.bits)
        self.wave_ch = apply_channel(self.wave_tx, p.dt, p)

    def front_end(self, p=None):  # VGA+CTLE+AGC, 与相位无关
    def run(self, p=None, front_end=None, nbits=None) -> Result
```

**设计要点**：`Link` 只依赖 **TX + 信道** 参数。所以扫 CTLE/CDR/DFE 时，
同一份 `wave_ch` 可以反复用——这是所有扫参实验能跑得快的原因。

```python
link = Link(params(channel_loss_db_at_nyquist=20.0))
for pk in (0, 3, 6, 9, 12):
    q = variant(link.p, ctle_peaking_db=pk)     # 只改 RX
    r = link.run(q)                             # 复用同一份 wave_ch
    print(pk, r.ber()[0])
```

**注意**：`bits` 的长度由**构造时**的 `p.nbits` 决定。扫参时不要改 `nbits`，
否则 `Result.eye_z()` 里的 `bits[:n]` 会错位。

## 4. `Result` —— 一次运行的所有观测量

`rx.run()` 的 4 个返回值 + 原始比特，包成一个对象，再挂上常用指标：

| 成员 | 说明 |
|---|---|
| `.dec / .ph / .dfe / .z` | 判决 / 相位轨迹 / 抽头轨迹 / 判决器输入 |
| `.ber(skip=300, max_lag=40)` | → `(ber, lag)` |
| `.phase_final` | 末 500 点中位数（比取最后一点稳健） |
| `.taps_final` | 最后一个时刻的抽头 |
| `.phase_jitter()` | 锁定后相位标准差 |
| `.eye_z()` | 判决器输入 z 的眼高/厚度（"DFE 收干净没"） |
| `.settle_time(tol, win)` | 首次进入稳态 ±tol 的 bit 数 |

**为什么 `eye_z()` 单独存在**：波形域的眼高（`eye_stats`）量的是"进判决器之前"，
而 `z` 是"DFE 之后、判决之前"，才是真正决定 BER 的量。

**`settle_time` 为什么用"首次进入"**：bang-bang + 积分支路会让相位缓慢游走，
若用"最后一次离开"，读数会永远等于总 bit 数（这个坑我在 lab06 里踩过）。

## 5. 采样与眼图指标

```python
sample_ui(y, sp, n, phase)     # 在 (k+0.5+phase)·sp 处线性插值, 取 n 个点
eye_stats(y, sp, bits, phase)  # -> (垂直眼高, 垂直厚度)
eye_height(...) / eye_sigma(...)   # 上面两个的单项
eye_profile(y, sp, bits)       # -> (32 个相位, 每个相位的眼高)
eye_width(y, sp, bits, frac)   # 眼高 > frac*峰值 的相位跨度
```

**眼高的定义（重要）**：

```python
h = quantile(hi, 0.02) - quantile(lo, 0.98)
```

用 **2% 分位"内沿"**，不是 `median(hi)-median(lo)`，也不是 `min/max`：

- 用中位数：多电平/多码型时会被"最常见的电平"带偏（默认 FFE 下会得到 1.7，而真值是 1.0）
- 用 min/max：一个噪声尖峰就把指标毁掉
- 用 2% 分位：等于"98% 的码型都不会比这更差"，稳定且可比

**垂直厚度** = 两组各自 96% 跨度的平均 → 噪声 + 残余 ISI 的"抹厚"程度。

## 6. `bathtub` / `width_at` —— 时间余量

```python
ber = bathtub(link, phases, p, train=True, front_end=y)
w   = width_at(phases, ber, thr=1e-6)
```

内部对每个相位：`cdr_kp=cdr_ki=0` 固定相位 → 跑一次 → 用 `max_lag=0` 算 BER。

**为什么 `max_lag=0`**：见 `analysis.compute_ber` 那条陷阱。
这一条如果不写，浴盆两端会"看起来很好"，整张图失去意义。

## 7. `pulse_cursors` —— 把 ISI 变成一串数字

```python
offs, h, main = pulse_cursors(p, loss_db=20, xform=None, tx_taps=None, phase=0.0)
```

内部四步，每一步都有原因：

1. **送阶跃而不是孤立脉冲**：`0…0,1…1,0…0`
   （两端补 0 避免 FFT 环绕污染；孤立脉冲有直流偏置问题）
2. **采样得到 `S[k]`**，再差分 `h[k] = S[k] - S[k-1]` → 单比特响应
3. **把 TX FFE 卷进去**：`h_eff = pre·h[k+1] + main·h[k] + post·h[k-1]`
   （FFE 是符号域 FIR，对阶跃做 FFE ≡ 对游标做卷积，不用重发波形）
4. **按主游标归一化**，返回 `(offsets, h, main)`

`main` 是归一化**前**的主游标：0.306 表示"一个比特只能爬到阶跃的 31%"。

**`phase` 参数**：ISI 系数随采样相位变化。跟 DFE 抽头对比时，
必须传 `phase=r.phase_final`，否则 h1 能差一倍（我在 lab07 踩过）。

## 8. 其它工具

| 函数 | 用途 |
|---|---|
| `rc_lowpass(x, tau_ui, sp)` | 一阶 RC，手工造"只有带宽、没有 S 参数"的极简信道（lab01 用它证明"拖尾才是杀手"） |
| `ctle_xform(p, agc=True)` | 返回一个"只做 VGA+CTLE(+AGC)"的闭包，喂给 `pulse_cursors(xform=...)` |
| `channel_db(f, L, nyq, skin)` | 信道幅度响应 [dB] |
| `ctle_db(p, f)` | CTLE 幅度响应 [dB]（冲激响应做 FFT 再插值） |
| `get_plt()` | 延迟导入 matplotlib（不画图就不付这个代价） |

## 9. 打印层

```python
h1(title) / h2(title)          # 一级/二级标题
banner(no, title, goal)        # 实验开场
ask(q) / obs(s) / why(s) / warn(s) / note(s)   # 先猜/实测/解释/注意/笔记
tbl(headers, rows)             # 等宽表格, 中文按 2 列宽对齐
```

`tbl` 里有个 `_dw()`：中日韩字符按 2 列计宽，否则中文表格会歪。
副作用：**表格单元格里不要出现 `|`**（会破坏对齐），所以表头写成 `median(abs x)`
而不是 `median(|x|)`。

---

# 第三部分：怎么用（5 个配方）

## 配方 1：最小端到端（不碰 common，纯 serdes_model）

```python
from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel
from serdes_model.analysis import compute_ber

p = SerdesParams()
tx = Transmitter(p)
bits = tx.generate_bits()
wave = apply_channel(tx.transmit(bits), p.dt, p)

rx = Receiver(p)
dec, ph, dfe, z = rx.run(wave, p.nbits)
print("BER", compute_ber(dec, bits))       # (0.0, 0)
print("相位", ph[-1], "抽头", dfe[-1])
```

跑 `nrz/examples/01_basic_link.py` 就是这段。

## 配方 2：改一个参数，量一个数（用 common）

```python
import sys; sys.path.insert(0, "nrz/lab")
from common import params, Link, variant, eye_stats, tbl

rows = []
for pk in (0, 3, 6, 9, 12):
    p = params(ctle_peaking_db=pk, channel_noise_sigma=0.03)
    link = Link(p)                              # CTLE 变了 -> 前端要重算
    q = variant(p, dfe_train=False,             # RX 参数 -> 只需换一份 p
                cdr_kp=0.0, cdr_ki=0.0)         # 关掉 CDR, 固定相位 0
    y = link.front_end(q)                       # VGA + CTLE + AGC
    h, thick = eye_stats(y, q.samples_per_ui, link.bits, phase=0.0)
    rows.append([pk, f"{h:.3f}", f"{thick:.3f}"])
tbl(["peaking [dB]", "眼高 [V]", "厚度 [V]"], rows)
```

输出（和 lab04 的 4.2 一致，同样能看到"眼高先涨后跌"）：

```
+--------------+----------+----------+
| peaking [dB] | 眼高 [V] | 厚度 [V] |
+--------------+----------+----------+
| 0            | 0.720    | 1.263    |
| 6            | 1.076    | 0.970    |
| 12           | 0.768    | 1.329    |
+--------------+----------+----------+
```

## 配方 3：拿 ISI 系数

```python
from common import params, pulse_cursors
p = params(tx_taps=(0, 1, 0))          # 先关掉 FFE, 看纯信道
offs, h, main = pulse_cursors(p, loss_db=20.0)
for o, v in zip(offs, h):
    print(f"{o:+d}  {v:+.3f}")
print("主游标原始值", round(main, 3))
```

## 配方 4：扫相位画浴盆

```python
import numpy as np
from common import params, Link, bathtub, width_at, get_plt, OUT

p = params(); link = Link(p)
phases = np.linspace(-0.5, 0.5, 41)
ber = bathtub(link, phases, p)          # 内部已固定相位 + max_lag=0
print("窗口宽度", width_at(phases, ber), "UI")

plt, save_fig, _ = get_plt()
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogy(phases, np.clip(ber, 1e-12, 1))
save_fig(fig, OUT / "my_bathtub.png")
```

## 配方 5：写一个新的实验脚本（lab09 骨架）

```python
"""实验 9: 我想验证 XXX。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (banner, h1, h2, ask, obs, why, note, tbl,
                    params, variant, Link, eye_stats)

def exp_9_1_xxx():
    h2("9.1 我的问题")
    rows = []
    for v in (1, 2, 4):
        p = params(channel_loss_db_at_nyquist=20.0)
        link = Link(p)
        r = link.run(p)
        rows.append([v, f"{r.ber()[0]:.2e}"])
    tbl(["变量", "BER"], rows)
    ask("为什么 ... ?")
    why("因为 ...")

def main():
    banner("9", "标题", "一句话目标")
    exp_9_1_xxx()
    h1("小结"); print("...")

if __name__ == "__main__":
    main()
```

放进 `nrz/lab/` 之后，`run_all.py` 会自动发现它（按 `lab[0-9][0-9]_*.py` 匹配）。

---

# 第四部分：8 个常见坑

| # | 症状 | 原因 | 解法 |
|---|---|---|---|
| 1 | 改了 `data_rate_gbps` 后一切都不对 | `ui/dt/fs` 是构造时算的 | 用 `common.params(...)`，或在改完后手动 `p.__post_init__()` |
| 2 | 扫参时 BER 莫名其妙一样 | 改的是 `Link` 里的 TX/信道参数，但 `wave_ch` 已缓存 | RX 参数用 `variant` + `link.run(q)`；TX/信道参数要重建 `Link` |
| 3 | `ValueError: 前端波形长度 ... 小于 ...` | `len(wave) < nbits*samples_per_ui` | 检查 `p.nbits` 与波形长度 |
| 4 | 浴盆曲线"太好" | `compute_ber` 的延迟搜索在对齐 | `max_lag=0`（`common.bathtub` 已默认） |
| 5 | 眼高和肉眼看到的对不上 | 用了 min/max 或中位数 | 用 `eye_stats` 的分位数内沿 |
| 6 | DFE 抽头跟理论值差一倍 | 采样相位不同 | `pulse_cursors(..., phase=r.phase_final)` |
| 7 | 阶跃响应出现"第二个台阶" | FFT 循环卷积的环绕 | 两端补 0（`pulse_cursors` 已做） |
| 8 | 关掉 AGC 后 DFE 跑飞 | sign-sign LMS 依赖幅度 = ±1 | 保持 `front_end()` 的归一化，见 lab05 |

---

# 第五部分：两个模块的"该用谁"

| 我想…… | 用 |
|---|---|
| 搭一条新链路 | `SerdesParams` + `Transmitter` + `apply_channel` + `Receiver` |
| 复用同一份波形扫 RX 参数 | `common.Link` + `variant` |
| 量眼高/眼宽 | `common.eye_stats` / `eye_width` |
| 量时间余量 | `common.bathtub` + `width_at` |
| 量 ISI 系数 | `common.pulse_cursors` |
| 看频响 | `common.channel_db` / `ctle_db` + `get_plt()` |
| 只看频域传递函数 | `channel_transfer` / `design_ctle` |
| 自己写一个自适应算法 | 抄 `rx.py:123-153` 那个循环，替换第 4 步 |
| 换成真实 S 参数信道 | 替换 `apply_channel` 里的 `H(f)`，其余不动 |

