"""示例 1: PAM4 ADC-based 端到端链路。

验证 TX -> 信道 -> ADC -> 数字 FFE -> DFE -> PAM4 判决 全链路能正确恢复数据。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pam4_model import (PAM4Params, Transmitter, Receiver, apply_channel,
                        compute_ser, compute_ber)


def main():
    p = PAM4Params()
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wave, syms = tx.transmit(bits)
    wc = apply_channel(wave, p.dt, p)

    rx = Receiver(p)
    dec, phase_hist, dfe_hist, z_hist = rx.run(wc, p.n_symbols)

    ser, lag = compute_ser(dec, syms, skip=300)
    ber, _, _ = compute_ber(dec, syms, skip=300)
    print(f"符号率: {p.baud_rate_gbps} GBd (比特率 {2 * p.baud_rate_gbps:.1f} Gbps), 符号数 {p.n_symbols}")
    print(f"符号误码率 SER: {ser:.3e}   比特误码率 BER: {ber:.3e}   (延迟 {lag})")
    print(f"CDR 最终相位: {phase_hist[-1]:+.4f} UI")
    print(f"数字 FFE 抽头: {[round(v, 4) for v in rx.ffe_taps]}")
    print(f"DFE 最终抽头: {[round(v, 4) for v in dfe_hist[-1]]}")
    return ser


if __name__ == "__main__":
    main()
