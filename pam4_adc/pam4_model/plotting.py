"""绘图工具 (PAM4): 眼图密度图 (依赖 matplotlib)。

统一完成 matplotlib 初始化 (无头后端、工作区字体缓存、中文字体)。
"""
from __future__ import annotations

import os
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_WORKSPACE / "output" / ".mpl"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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


def eye_density(wave, samples_per_ui, nbins=64, ax=None, title="PAM4 眼图",
                cmap="inferno"):
    """在 ax 上绘制 PAM4 眼图 2D 密度图 (时间轴以 UI 中心为 0)。"""
    per = int(samples_per_ui)
    hist, lo, hi = eye_hist2d(wave, samples_per_ui, nbins)
    t = (np.arange(per) - per / 2) / per
    amp = np.linspace(lo, hi, nbins)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.pcolormesh(t, amp, hist, shading="auto", cmap=cmap)
    for th in (-2.0, 0.0, 2.0):
        ax.axhline(th, color="cyan", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("时间 [UI]")
    ax.set_ylabel("幅度")
    ax.set_title(title)
    return ax


def save_fig(fig, path, dpi=130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return str(path)
