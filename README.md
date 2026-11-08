# SerDes 行为级建模集（NRZ + PAM4 ADC-based）

本仓库包含两个**独立、可运行**的教学级 SerDes 行为模型，帮助理解不同接收机架构。

| 目录 | 模型 | 接收机架构 | 关键点 |
|---|---|---|---|
| [`nrz/`](nrz/) | NRZ 10 Gbps | 模拟均衡：VGA/AGC + **CTLE + 模拟 DFE** + bang-bang CDR | 经典模拟 SerDes |
| [`pam4_adc/`](pam4_adc/) | PAM4 28 GBd（56 Gbps） | **ADC-based / 数字 DSP**：AGC + **ADC** + 数字 FFE/DFE + 数字 CDR | 现代 ADC/DSP SerDes |

两个模型结构对称（`params/signals/filters/tx/channel/rx/analysis/plotting`），
核心都只依赖 **NumPy**，绘图用 matplotlib。

**学习入口**：从 NRZ 学起 → [`nrz/LEARNING_PATH.md`](nrz/LEARNING_PATH.md)
（8 个阶段的学习顺序 + `nrz/lab/` 下 8 个可运行的分阶段调试实验）。

---

## 快速开始

```powershell
# NRZ 模型
conda run -n base python nrz/examples/01_basic_link.py
conda run -n base python nrz/examples/02_equalization.py
conda run -n base python nrz/examples/03_bathtub.py

# PAM4 ADC-based 模型
conda run -n base python pam4_adc/examples/01_basic_link.py
conda run -n base python pam4_adc/examples/02_eyes.py
conda run -n base python pam4_adc/examples/03_ser_sweep.py
```

> **环境提示**：本机 Anaconda base 是混合环境，直接调 `python.exe` 会因 MKL DLL
> 崩溃（`0xC06D007F`），需用 `conda run` 激活；中文乱码先 `chcp 65001`。

---

## 两个模型的对比

| 维度 | NRZ（`nrz/`） | PAM4 ADC-based（`pam4_adc/`） |
|---|---|---|
| 每符号比特 | 1 | 2（Gray 映射） |
| 判决电平 | 2 | 4（3 个门限） |
| 均衡位置 | 模拟（CTLE）+ 模拟 DFE | **数字**（FFE + DFE，ADC 之后） |
| 时钟恢复 | 模拟 bang-bang CDR | 数字 CDR（基于 ADC 边沿采样） |
| 典型用途 | 10G/25G | 112G/224G 高速链路 |
| 默认可靠损耗 | ~20 dB（CTLE+DFE） | ~6 dB（5-tap FFE + 3-tap DFE） |

---

## 目录

```
Serdes/
├── nrz/          # NRZ 模型 (原名 serdes_model)
├── pam4_adc/     # PAM4 ADC-based 模型
├── .gitignore
└── README.md
```

每个子目录都有自己的 `pyproject.toml`、`requirements.txt`、`README.md`、
`examples/` 与 `tests/`，可独立安装运行。
