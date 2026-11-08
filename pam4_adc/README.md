# PAM4 ADC-based SerDes —— 行为级教学模型

一个用 **Python + NumPy** 编写的 PAM4 **ADC-based（数字域均衡）** 收发器行为级模型。
与经典模拟均衡的 SerDes 不同，这里模拟前端尽量简单，**ADC 采样量化后，全部均衡在数字域完成** —— 这正是现代高速 SerDes（112G/224G PAM4）的主流架构。

核心只依赖 NumPy；绘图示例依赖 matplotlib。

---

## 1. 系统架构

```
 TX 侧                                  信道                    RX 侧 (ADC-based)
┌─────────────────────────┐        ┌──────────┐      ┌──────────────────────────────────┐
│ PRBS ─► Gray映射 PAM4    │        │ 有损信道  │      │ VGA/AGC ─► ADC(采样+量化) ─►      │
│ ─► TX FFE ─► 4电平NRZ整形│ ────► │ 趋肤+介质 │ ───► │ 数字 FFE ─► 数字 DFE ─► PAM4判决  │
└─────────────────────────┘        │ 损耗/噪声 │      │ ─► 解串/解码  (+ 数字 CDR)        │
                                   └──────────┘      └──────────────────────────────────┘
        符号率 28 GBd（比特率 56 Gbps），每 UI 采样 32 点，ADC 6 bit
```

数据流：PRBS 比特 → Gray 映射为 PAM4 符号（2 bit/符号）→ TX FFE → 4 电平 NRZ 波形 → 信道 → VGA/AGC → ADC 量化 → 数字 FFE → 数字 DFE → PAM4 判决 → 解码。

---

## 2. 各模块原理

### 2.1 PAM4 信号 `signals.py`

- **Gray 映射**：每个符号 2 比特，`00→-3, 01→-1, 11→+1, 10→+3`，相邻电平仅差 1 比特（降低误码传播）。
- **4 电平 NRZ 整形**：每个符号持续一个 UI，电平跳变用升余弦沿。

### 2.2 发送端 `tx.py`

- 比特 → Gray PAM4 符号 → **TX FFE**（预加重，3-tap）→ 多电平波形。

### 2.3 信道 `channel.py`

与 NRZ 模型同款频域有损传输线：趋肤损耗 `∝√f` + 介质损耗 `∝f`，零相位近似。

### 2.4 ADC `adc.py`

- **量化**：N-bit、满量程 ±FS，步长 `Δ = 2FS/2^N`，量化噪声 `σ_q = Δ/√12`。
- **AGC**：把 99 分位幅度归一化到 PAM4 峰值 3，使 ADC 满量程匹配。

### 2.5 接收机 `rx.py`（核心）

- **ADC 采样**：按 CDR 恢复相位采样，数据点 + 边沿点都量化。
- **数字 FFE**：**符号间隔、居中抽头**（预/主/后），按信道幅度反设计（ZF，带提升上限），用于压前驱/后驱 ISI。
  - 关键：FFE 的频率轴是**符号率**（0..baud/2），不是采样率 `fs`（= spui×baud）。
- **数字 DFE**：sign-sign LMS，用已判决符号抵消后驱 ISI。
- **PAM4 判决**：3 个门限 `(-2, 0, +2)` 映射到 4 电平。
- **数字 CDR**：PAM4 Alexander 鉴相（边沿采样 − 相邻判决中点），二阶 PI 环路。

### 2.6 分析 `analysis.py`

- `compute_ser` / `compute_ber`：符号/比特误码率（带延迟对齐）。
- `eye_hist2d`：眼图 2D 密度直方图（可见 3 个眼）。

---

## 3. 目录结构

```
pam4_adc/
├── pam4_model/          # 核心库 (只依赖 numpy)
│   ├── __init__.py
│   ├── params.py        # PAM4Params
│   ├── signals.py       # PRBS / Gray PAM4 映射 / 波形
│   ├── filters.py       # 双线性变换 + IIR 滤波
│   ├── tx.py            # 发送端
│   ├── channel.py       # 有损信道
│   ├── adc.py           # ADC 量化
│   ├── rx.py            # ADC-based 接收机 (FFE/DFE/判决/CDR)
│   ├── analysis.py      # SER/BER/眼图
│   └── plotting.py      # matplotlib 可视化
├── examples/
│   ├── 01_basic_link.py # 端到端链路
│   ├── 02_eyes.py       # 各阶段眼图
│   └── 03_ser_sweep.py  # SER vs 信道损耗
├── tests/
│   └── test_core.py     # 单元测试
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 4. 安装、运行与测试

**依赖**：核心只依赖 `numpy`；绘图需要 `matplotlib`；测试可选 `pytest`。

```powershell
# 方式 A: 直接运行示例 (无需安装)
conda run -n base python examples/01_basic_link.py

# 方式 B: 安装为可导入包
pip install -e .
```

三个示例：

```powershell
conda run -n base python examples/01_basic_link.py   # 端到端链路
conda run -n base python examples/02_eyes.py         # 眼图
conda run -n base python examples/03_ser_sweep.py    # SER vs 损耗
```

> **环境提示**：本机 base 环境是 conda-forge + defaults 混合环境，直接调
> `python.exe` 会因 MKL DLL 崩溃（`0xC06D007F`），必须用 `conda run` 激活；
> 中文乱码先 `chcp 65001`。

### 测试

```powershell
conda run -n base python tests/test_core.py     # 不依赖 pytest
# 或
pip install -e ".[dev]" && pytest -q
```

---

## 5. 关键参数（`PAM4Params`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `baud_rate_gbps` | 28.0 | 符号率 [GBd]（比特率=2×） |
| `samples_per_ui` | 32 | 每 UI 采样点数 |
| `n_symbols` | 4096 | 符号数 |
| `tx_ffe_taps` | (-0.1, 1.0, -0.1) | TX FFE |
| `channel_loss_db_at_nyquist` | 6.0 | 信道奈奎斯特损耗 |
| `adc_bits` | 6 | ADC 分辨率 |
| `adc_fs` | 4.0 | ADC 满量程 |
| `ffe_ntaps` | 5 | 数字 FFE 抽头 |
| `dfe_ntaps` | 3 | 数字 DFE 抽头 |
| `cdr_kp` / `cdr_ki` | 1/64, 1/2048 | CDR 环路增益 |

---

## 6. 设计要点与已知边界

1. **ADC-based 的分工**：模拟只做 AGC；均衡全在数字域（FFE/DFE），这正是可编程、可自适应的关键。
2. **FFE 频率轴必须用符号率**：FFE 是符号间隔（1 样本/UI），设计带宽只到 `baud/2`；若误用采样率 `fs`（=32×baud），会在无信号的高频段盲目提升，产生畸变（调试时踩过这个坑）。
3. **简化模型的可靠区间**：5-tap ZF FFE + 3-tap DFE 可稳定处理 **≤6 dB** 信道损耗（SER=0）；8 dB 起出现残余 ISI（SER≈1e-2），更高损耗需要更强 FFE、FEC 或 MLSE。
4. **零相位信道近似**：忽略群时延，前/后驱 ISI 对称，FFE 用居中抽头处理。

---

## 7. 扩展方向

- **MMSE / 自适应 FFE**：替换固定 ZF 设计，改善噪声放大与高损耗性能。
- **FEC**：RS(544,514) 等，支持 20 dB+ 损耗链路。
- **MLSE / 序列检测**：ADC-based 收机的高阶检测。
- **真实 S4P 信道**、**PAM4 抖动/串扰**、**多电平 CDR 改进**。
