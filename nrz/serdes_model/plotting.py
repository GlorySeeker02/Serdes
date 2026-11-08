"""绘图工具: 眼图密度图、频响、收敛曲线 (依赖 matplotlib)。

核心模型不依赖 matplotlib; 本模块只用于示例的可视化。
本模块统一完成 matplotlib 初始化 (无头后端、工作区字体缓存、中文字体),
示例通过 `from serdes_model.plotting import plt, ...` 复用配置。
"""
from __future__ import annotations

import os
from pathlib import Path

# matplotlib 字体缓存写到工作区, 避免写用户目录被沙箱拦截 (Permission denied)
_WORKSPACE = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_WORKSPACE / "output" / ".mpl"))

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 无头后端, 直接保存 PNG
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册中文字体, 否则中文标签显示为方框
for _fp in (r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc"):
    if os.path.exists(_fp):
        try:
            font_manager.fontManager.addfont(_fp)
        except Exception:  # noqa: BLE001
            pass
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from .analysis import eye_hist2d


def eye_density(wave, samples_per_ui, nbins=64, ax=None, title="眼图", cmap="inferno"):
    """在坐标轴 ax 上绘制眼图 2D 密度图 (时间轴以 UI 中心为 0)。"""
    per = int(samples_per_ui)
    hist, lo, hi = eye_hist2d(wave, samples_per_ui, nbins)
    t = (np.arange(per) - per / 2) / per
    amp = np.linspace(lo, hi, nbins)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.pcolormesh(t, amp, hist, shading="auto", cmap=cmap)
    ax.set_xlabel("时间 [UI]")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    return ax


def save_fig(fig, path, dpi=130):
    """保存并关闭 figure。"""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)
