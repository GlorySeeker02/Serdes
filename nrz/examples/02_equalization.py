"""示例 2: 均衡效果与眼图对比。

展示 TX -> 信道 -> CTLE -> DFE 各阶段眼图, 以及信道/CTLE 频响,
直观理解 CTLE + DFE 如何重新打开被信道损耗关闭的眼图。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel
from serdes_model.analysis import compute_ber
from serdes_model.channel import channel_transfer
from serdes_model.filters import lfilter
from serdes_model.plotting import plt, eye_density, save_fig

OUT = Path(__file__).resolve().parent.parent / "output"


def main():
    p = SerdesParams()
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wave_tx = tx.transmit(bits)
    wave_ch = apply_channel(wave_tx, p.dt, p)
    rx = Receiver(p)
    decisions, phase_hist, dfe_hist, z_hist = rx.run(wave_ch, p.nbits)

    ber, _ = compute_ber(decisions, bits)
    print(f"BER = {ber:.3e}")
    OUT.mkdir(exist_ok=True)

    # ---------- 1) 四阶段眼图 ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    eye_density(wave_tx, p.samples_per_ui, ax=axes[0, 0], title="TX 输出 (预加重后)")
    eye_density(wave_ch, p.samples_per_ui, ax=axes[0, 1], title="信道输出 (20dB@Nyquist, 眼图关闭)")
    eye_density(rx.front_end(wave_ch), p.samples_per_ui, ax=axes[1, 0], title="CTLE + AGC 输出 (眼图重新打开)")
    # DFE 均衡后的判决器输入: 每 UI 一个点, 画直方图展示两个清晰电平
    axes[1, 1].hist(z_hist, bins=80, color="steelblue", alpha=0.8)
    axes[1, 1].set_xlabel("判决器输入幅度")
    axes[1, 1].set_ylabel("计数")
    axes[1, 1].set_title("DFE 均衡后 (两级分离, 残差很小)")
    eye_png = OUT / "02_equalization_eye.png"
    save_fig(fig, eye_png)
    print("已保存:", eye_png)

    # ---------- 2) 频响 ----------
    n = len(wave_ch)
    freqs = np.fft.rfftfreq(n, d=p.dt) / 1e9
    H_ch = channel_transfer(np.fft.rfftfreq(n, d=p.dt), p.channel_loss_db_at_nyquist,
                            p.nyquist_ghz * 1e9, p.channel_skin_db_ratio)
    # CTLE 频响: 冲激响应 -> FFT
    imp = np.zeros(4096)
    imp[0] = 1.0
    ir = lfilter(rx.bd, rx.ad, imp)
    H_ctle = np.fft.rfft(ir, n=n)

    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs, 20 * np.log10(np.abs(H_ch) + 1e-12), label="信道")
    ax.plot(freqs, 20 * np.log10(np.abs(H_ctle) + 1e-12), label="CTLE")
    ax.plot(freqs, 20 * np.log10(np.abs(H_ch) * np.abs(H_ctle) + 1e-12),
            label="信道 × CTLE (级联)", lw=2)
    ax.axvline(p.nyquist_ghz, color="gray", ls="--", lw=1, label=f"Nyquist {p.nyquist_ghz}GHz")
    ax.set_xlabel("频率 [GHz]")
    ax.set_ylabel("增益 [dB]")
    ax.set_title("信道与 CTLE 频响 (CTLE 补偿高频损耗)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, freqs.max())
    ax.set_ylim(-30, 12)
    fr_png = OUT / "02_frequency_response.png"
    save_fig(fig2, fr_png)
    print("已保存:", fr_png)


if __name__ == "__main__":
    main()
