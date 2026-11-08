"""示例 2: PAM4 各阶段眼图 (TX -> 信道 -> ADC 输入) 与 DFE 后电平分布。

ADC-based 收机的关键: ADC 输入处 (AGC 后) 是 4 电平、3 个眼; 数字均衡在
ADC 之后把这些眼重新打开。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pam4_model import PAM4Params, Transmitter, Receiver, apply_channel, compute_ser
from pam4_model.plotting import plt, eye_density, save_fig

OUT = Path(__file__).resolve().parent.parent / "output"


def main():
    p = PAM4Params()
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wave, syms = tx.transmit(bits)
    wc = apply_channel(wave, p.dt, p)
    rx = Receiver(p)
    dec, phase_hist, dfe_hist, z_hist = rx.run(wc, p.n_symbols)
    ser, _ = compute_ser(dec, syms, skip=300)
    print(f"SER = {ser:.3e}")
    OUT.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    eye_density(wave, p.samples_per_ui, ax=axes[0, 0], title="TX 输出 (PAM4, 4 电平)")
    eye_density(wc, p.samples_per_ui, ax=axes[0, 1], title="信道输出 (损耗后, 3 个眼变窄)")
    eye_density(rx.front_end(wc), p.samples_per_ui, ax=axes[1, 0], title="ADC 输入 (AGC 后)")
    axes[1, 1].hist(z_hist, bins=80, color="steelblue", alpha=0.85)
    axes[1, 1].set_xlabel("DFE 后判决输入")
    axes[1, 1].set_ylabel("计数")
    axes[1, 1].set_title("DFE 后电平分布 (4 电平清晰分离)")
    png = OUT / "02_pam4_eyes.png"
    save_fig(fig, png)
    print("已保存:", png)


if __name__ == "__main__":
    main()
