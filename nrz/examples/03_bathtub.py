"""示例 3: 浴盆曲线 (BER vs 采样相位)。

浴盆曲线反映"在哪个采样相位范围内误码率可接受", 即眼图在时间轴上的
睁开宽度。这里固定 CDR (关环路), 扫采样相位, 测量各相位 BER。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel
from serdes_model.analysis import compute_ber
from serdes_model.plotting import plt, save_fig

OUT = Path(__file__).resolve().parent.parent / "output"


def sweep_ber(wave_ch, bits, phases, front_end, use_dfe=True):
    bers = []
    for ph in phases:
        p2 = SerdesParams()
        p2.nbits = len(bits)
        p2.cdr_kp = 0.0          # 关闭 CDR 环路, 固定采样相位
        p2.cdr_ki = 0.0
        p2.cdr_initial_phase_ui = ph
        p2.dfe_train = use_dfe
        rx = Receiver(p2)
        dec, _, _, _ = rx.run(wave_ch, p2.nbits, front_end=front_end)
        ber, _ = compute_ber(dec, bits, skip=300)
        bers.append(ber)
    return np.array(bers)


def main():
    p = SerdesParams()
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wave_ch = apply_channel(tx.transmit(bits), p.dt, p)

    phases = np.linspace(-0.5, 0.5, 41)
    OUT.mkdir(exist_ok=True)

    # 前端 (VGA+CTLE+AGC) 与采样相位无关, 只算一次, 所有相位复用
    y = Receiver(SerdesParams()).front_end(wave_ch)
    ber_off = sweep_ber(wave_ch, bits, phases, y, use_dfe=False)
    ber_on = sweep_ber(wave_ch, bits, phases, y, use_dfe=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    clip = 1e-12
    ax.semilogy(phases, np.clip(ber_off, clip, 1), label="CTLE 无 DFE", marker="o", ms=3)
    ax.semilogy(phases, np.clip(ber_on, clip, 1), label="CTLE + DFE", marker="o", ms=3)
    ax.set_xlabel("采样相位 [UI]")
    ax.set_ylabel("误码率 BER")
    ax.set_title("浴盆曲线 (眼图时间轴睁开宽度)")
    ax.set_ylim(clip, 1)
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    png = OUT / "03_bathtub.png"
    save_fig(fig, png)
    print("已保存:", png)

    # 打印 DFE 带来的眼宽收益
    th = 1e-6
    w_off = float(np.ptp(phases[ber_off < th])) if np.any(ber_off < th) else 0.0
    w_on = float(np.ptp(phases[ber_on < th])) if np.any(ber_on < th) else 0.0
    print(f"BER<1e-6 的相位宽度: 无DFE={w_off:.3f} UI, 有DFE={w_on:.3f} UI")


if __name__ == "__main__":
    main()
