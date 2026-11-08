# SerDes 高速有线收发链路 —— 行为级教学模型

一个用 **Python + NumPy** 编写的 10 Gbps 通用 NRZ SerDes 收发器行为级模型，
覆盖完整高速有线收发链路，用于理解各模块的数学原理与协同工作方式。

核心模型 **只依赖 NumPy**（双线性变换、IIR 滤波、线性插值均为手写），
绘图示例依赖 matplotlib。

> **想系统学一遍？** 从 [`LEARNING_PATH.md`](LEARNING_PATH.md) 开始：
> 8 个阶段的学习顺序 + `lab/` 下 8 个可运行实验（每个都打印"先猜 → 实测 → 解释"），
> 覆盖 UI/眼图 → 信道 ISI → TX FFE → CTLE → AGC → CDR → DFE → 系统预算，
> 并附实测基线数据和笔记模板。

---

## 1. 系统架构

```
 TX 侧                                     信道                    RX 侧
┌──────────────────────────┐          ┌──────────┐      ┌──────────────────────────┐
│ PRBS 源 ─► 串行化器 ─►    │          │ 有损信道  │      │ VGA ─► AGC ─► CTLE ─►     │
│ TX FFE 预加重 ─► NRZ 驱动 │ ───────► │ 趋肤+介质 │ ───► │ CDR 采样 ─► DFE ─► 判决  │
│ (可选抖动)                │          │ 损耗/串扰/│      │ ─► 解串器                 │
└──────────────────────────┘          │ 噪声      │      └──────────────────────────┘
                                      └──────────┘
        数据速率 10 Gbps，UI = 100 ps，每 UI 采样 32 点（时间分辨率）
```

数据流：并行 PRBS 比特 → 符号映射 → TX FFE → NRZ 波形 → 有损信道 → 接收端均衡 → 判决 → 恢复比特。

---

## 2. 各模块原理说明

### 2.1 信号源 `signals.py`

- **PRBS**（`prbs_bits`）：线性反馈移位寄存器（LFSR）生成伪随机序列。
  特征多项式例如 PRBS7 = x⁷+x⁶+1，周期 2⁷-1=127。PRBS 频谱近似白噪声，
  是 SerDes 链路验证的标准码型。反馈抽头由本原多项式决定。

- **NRZ 整形**（`symbols_to_nrz` / `bits_to_nrz`）：把 UI 速率的符号序列
  展开成连续波形，每个符号持续 `samples_per_ui` 个采样点，相邻符号电平不同时
  用升余弦沿过渡（`rise_ui` 控制上升时间）。可选在跳变沿相位叠加随机抖动
  `rj_sigma_ui`（单位 UI），模拟发送端时钟抖动。

### 2.2 发送端 `tx.py`

- **串行化器**：并行比特 → 串行符号流（本模型中体现为 bit → ±1 符号映射）。
- **TX FFE / 预加重**（`apply_ffe`）：对符号序列做 FIR 滤波 `[pre, main, post]`。
  **原理**：信道高频损耗衰减上升/下降沿，造成码间干扰（ISI）。预加重在发送端
  预先放大跳变分量（相对压低低频分量），使到达接收端的高低频能量平衡。
  注意：FFE 抽头之和就是其直流增益（例如 `-0.15+0.85-0.2 = 0.5`），
  这会降低发送信号的整体幅度。

### 2.3 信道 `channel.py`

用频域传递函数描述有损传输线（零相位近似，忽略群时延）：

```
L(f) [dB] = L_nyq · [ skin_ratio · (f/f_nyq)^0.5 + (1-skin_ratio) · (f/f_nyq) ]
```

- **趋肤效应**：高频电流挤向导体表面，损耗 ∝ √f
- **介质损耗**：绝缘体极化损耗，损耗 ∝ f

`channel_loss_db_at_nyquist` 指定奈奎斯特频点的总插入损耗，`skin_ratio`
分配二者比例。可选叠加串扰（另一路 PRBS 流）与高斯白噪声。
`channel_type="ideal"` 时信道直通（用于单独调试 TX/RX）。

### 2.4 接收端模拟前端 `rx.py`

- **VGA**（可变增益放大器，`vga_gain`）：固定增益，也可作为 AGC 初值。
- **AGC**（自动增益控制）：把主抽头幅度归一化到 ±1。
  **这是本模型最关键的环节** —— 见第 4 节"踩坑记录"。用 `median(|x|)` 估计
  NRZ 主抽头电平（NRZ 平顶段占多数样本，中位数 ≈ 主抽头）。
- **CTLE**（连续时间线性均衡器，`design_ctle`）：一零二极模拟滤波器

  ```
  H(s) = A_dc · (1 + s/ωz) / ((1 + s/ωp1)(1 + s/ωp2))，  ωp1 < ωz < ωp2
  ```

  在 `f_peak` 附近产生约 `peaking_db` 的高频增益，补偿信道高频损耗。
  通过**双线性变换** `s → 2fs·(1-z⁻¹)/(1+z⁻¹)` 离散化为 IIR（手写实现，
  避免 scipy 依赖），再用直接 II 型转置结构滤波。

### 2.5 CDR `rx.py`

时钟数据恢复，从数据边沿恢复采样时钟相位：

- **Alexander 鉴相器**（bang-bang）：`err = e · (s_prev - s)`，
  其中 `e` 是相邻两判决之间的边沿采样，`s_prev`、`s` 是前后判决。
  只有发生跳变（`s_prev ≠ s`）时 `err ≠ 0`，方向指示时钟偏早还是偏晚。
  `cdr_type="bangbang"` 时边沿采样量化为 ±1（经典 bang-bang），
  `"linear"` 时保留模拟边沿值（线性化）。
- **二阶环路滤波（PI）**：`integral += ki·err`（积分支路跟踪频率），
  `phase += kp·err + integral`（比例+积分调整相位）。
- **采样**：数据采样在 `(k+0.5+phase)·UI`，边沿采样在 `(k+phase)·UI`，
  相位 `phase` 即恢复时钟相对标称位置的偏移。

### 2.6 DFE 与判决 `rx.py`

- **判决反馈均衡**：`z = d - Σ hᵢ·a_{k-i}`，用已判决的过去比特抵消
  后驱 ISI（低通信道中，前一个 +1 比特的拖尾会污染当前比特，`h1 > 0`）。
- **sign-sign LMS 自适应**：`hᵢ += μ · sign(z - a_k) · a_{k-i}`，
  无需乘法、硬件实现简单。
- **判决器**：`s = sign(z)`，映射到 ±1。
- **解串器**（`deserialize`）：串行判决按 `word_width` 分组为并行字。

### 2.7 分析 `analysis.py`

- `compute_ber`：跳过收敛预热段，在整数延迟范围内搜索最优对齐后统计误码率。
- `eye_segments` / `eye_hist2d`：按 UI 折叠波形，生成眼图 2D 密度直方图。

---

## 3. 目录结构

```
Serdes/
├── serdes_model/          # 核心库 (只依赖 numpy)
│   ├── __init__.py
│   ├── params.py          # 全局参数配置 dataclass
│   ├── signals.py         # PRBS / NRZ 整形 / 噪声
│   ├── filters.py         # 双线性变换 + IIR 滤波 (scipy 可选加速)
│   ├── tx.py              # 发送端 (串行化 / FFE 预加重)
│   ├── channel.py         # 有损信道
│   ├── rx.py              # 接收端 (VGA/AGC/CTLE/CDR/DFE/判决/解串)
│   ├── analysis.py        # 眼图 / 误码率
│   └── plotting.py        # matplotlib 可视化
├── examples/
│   ├── 01_basic_link.py   # 端到端链路, 打印 BER / CDR 相位 / DFE 抽头
│   ├── 02_equalization.py # 四阶段眼图 + 频响
│   └── 03_bathtub.py      # 浴盆曲线 (BER vs 采样相位)
├── lab/                   # 分阶段学习实验 (配套 LEARNING_PATH.md)
│   ├── README.md          # ★ serdes_model 与 common.py 的使用手册
│   ├── common.py          # 实验用的链路/眼图/游标/打印工具
│   ├── lab01_waveform.py  # S1 时间基 / PRBS / NRZ / 眼图
│   ├── lab02_channel_isi.py  # S2 信道损耗与 ISI 系数
│   ├── lab03_tx_ffe.py    # S3 TX FFE 预加重
│   ├── lab04_ctle.py      # S4 CTLE
│   ├── lab05_agc.py       # S5 AGC (复现"自适应失效"那个坑)
│   ├── lab06_cdr.py       # S6 CDR 环路
│   ├── lab07_dfe.py       # S7 DFE 与错误传播
│   ├── lab08_budget.py    # S8 系统级余量预算
│   └── run_all.py         # 一键跑完, 日志写 output/lab/labNN.log
├── tests/
│   └── test_core.py       # 单元测试 (pytest 兼容, 也可直接运行)
├── pyproject.toml         # 打包与依赖配置
├── requirements.txt
├── output/                # 生成的图片
├── LEARNING_PATH.md       # ★ 学习顺序 + 每阶段实验与实测数据
└── README.md
```

---

## 4. 安装、运行与测试

**依赖**：核心只依赖 `numpy`；绘图需要 `matplotlib`；测试可选 `pytest`。

```powershell
# 方式 A: 直接运行示例 (无需安装, 示例内部已处理 sys.path)
conda run -n base python examples/01_basic_link.py

# 方式 B: 安装为可导入包 (推荐)
pip install -e .
# 之后可在任意目录 import serdes_model
```

三个示例：

```powershell
conda run -n base python examples/01_basic_link.py     # 端到端链路
conda run -n base python examples/02_equalization.py   # 眼图 + 频响
conda run -n base python examples/03_bathtub.py        # 浴盆曲线
```

> **环境提示**：本机 base 环境是 conda-forge + defaults 混合环境，直接调
> `python.exe` 会因 MKL 运行库 DLL 不在搜索路径而崩溃（`0xC06D007F`）。
> 必须用 `conda run` 激活；终端中文乱码时先执行 `chcp 65001`。

### 测试

```powershell
# 直接运行 (不依赖 pytest)
conda run -n base python tests/test_core.py

# 或用 pytest
pip install -e ".[dev]"
pytest -q
```

测试覆盖：PRBS 周期、NRZ 整形、信道直流/奈奎斯特频点、双线性变换与 IIR、
CTLE 直流增益、解串器、误码率对齐、端到端 BER=0、CDR 锁定、DFE 收敛、
前端复用等价性、输入长度校验。

---

## 5. 关键参数（`SerdesParams`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `data_rate_gbps` | 10.0 | 数据速率 |
| `samples_per_ui` | 32 | 每 UI 采样点数（时间分辨率） |
| `nbits` | 4096 | 仿真比特数 |
| `prbs_order` | 7 | PRBS7 码型 |
| `tx_taps` | (-0.15, 0.85, -0.2) | TX FFE [pre, main, post] |
| `channel_loss_db_at_nyquist` | 20.0 | 奈奎斯特频点插入损耗 |
| `ctle_peaking_db` | 6.0 | CTLE 高频 peaking |
| `cdr_kp` / `cdr_ki` | 1/64, 1/2048 | CDR 比例/积分增益 |
| `dfe_ntaps` | 5 | DFE 反馈抽头数 |
| `dfe_mu` | 0.01 | sign-sign LMS 步长 |

---

## 6. 踩坑记录（建模要点）

1. **幅度归一化是前提**：sign-sign LMS 的误差 `sign(z - s)` 只有在判决器
   输入主抽头 ≈ ±1 时才反映真实 ISI 方向。若幅度偏离 ±1（本例中 TX FFE
   直流增益 0.5 + 信道低频损耗把主抽头压到 ~0.3），`sign(z-s)` 恒等于
   `-s`，与 ISI 无关，DFE 抽头被系统性推成负值 → 判决错误 → 拖垮 CDR 相位。
   **教训**：AGC 归一化必须放在均衡之前，否则所有自适应都失效。

2. **LMS 步长要稳**：sign-sign LMS 步长过大会使抽头越过稳定域、判决翻转、
   环路发散。归一化正确后 `mu=0.01` 可用；不稳时可降到 `0.001`。

3. **统一时间基**：全程以 UI 为单位（`sample(k, off)` 采样 `(k+off+phase)·UI`），
   避免秒/比特/采样点混用。

4. **先关自适应单独验证**：调试时先关 CDR（固定相位）扫相位找眼图最优位置，
   再关 DFE 看 CDR 是否锁定，最后合在一起，能快速定位是哪一环的问题。

---

## 7. 扩展方向

- **PAM4**：判决器改为 4 电平，DFE 误差定义相应调整。
- **真实 S 参数信道**：用 S4P 文件替换解析传递函数。
- **因果 AGC / 自适应 CTLE**：把块级 AGC 换成逐采样慢速 AGC 环路。
- **前向纠错（FEC）**：在判决后加 RS/BCH 编解码。
- **抖动/串扰建模**：TX 确定性抖动、多路串扰、电源噪声。

---

## 8. 性能与优化记录

- **IIR 滤波双路径**：`filters.lfilter` 检测到 scipy 时用其 C 实现，否则回退
  纯 numpy（结果一致）。原来纯 Python 逐样本循环是主要瓶颈。
- **前端复用**：`Receiver.front_end()` 与采样相位无关；扫相位时先算一次，
  再通过 `run(..., front_end=y)` 传给多个接收机，避免浴盆示例 82 次重复滤波。
- **零阶滤波器修复**：纯 numpy IIR 回退此前在 `b=[1], a=[1]`（纯增益）时越界，
  已增加 `N==1` 直返分支。
- 进一步可做：`run` 的逐比特 Python 循环改 C 扩展/批量处理；信道接入真实
  S4P 参数；新增 PAM4 判决。
