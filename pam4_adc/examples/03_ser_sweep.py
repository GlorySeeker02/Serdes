"""示例 3: SER vs 信道损耗 (PAM4 ADC-based)。

扫信道奈奎斯特损耗, 观察 SER 随损耗恶化 —— 直观理解 PAM4 对 SNR 的高要求,
以及数字均衡能力边界。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pam4_model import PAM4Params, Transmitter, Receiver, apply_channel, compute_ser
from pam4_model.plotting import plt, save_fig

OUT = Path(__file__).resolve().parent.parent / "output"


def main():
    losses = np.arange(2.0, 15.0, 2.0)
    sers = []
    for loss in losses:
        p = PAM4Params()
        p.channel_loss_db_at_nyquist = float(loss)
        tx = Transmitter(p)
        bits = tx.generate_bits()
        wave, syms = tx.transmit(bits)
        wc = apply_channel(wave, p.dt, p)
        rx = Receiver(p)
        dec, _, _, _ = rx.run(wc, p.n_symbols)
        ser, _ = compute_ser(dec, syms, skip=300)
        sers.append(ser)
        print(f"loss={loss:4.0f} dB   SER={ser:.3e}")

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses, np.clip(sers, 1e-7, 1), marker="o")
    ax.set_xlabel("信道损耗 @Nyquist [dB]")
    ax.set_ylabel("符号误码率 SER")
    ax.set_title("PAM4 ADC-based: SER vs 信道损耗")
    ax.grid(which="both", alpha=0.3)
    ax.set_ylim(1e-7, 1)
    png = OUT / "03_pam4_ser_sweep.png"
    save_fig(fig, png)
    print("已保存:", png)


if __name__ == "__main__":
    main()
