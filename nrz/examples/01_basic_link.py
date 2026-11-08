"""示例 1: 端到端基本链路 (TX -> 信道 -> RX)。

验证整个 SerDes 链路能正确恢复数据, 打印误码率、CDR 相位、DFE 抽头。
这是理解各模块协作的最小可运行示例。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from serdes_model import SerdesParams, Transmitter, Receiver, apply_channel
from serdes_model.analysis import compute_ber


def main():
    p = SerdesParams()
    tx = Transmitter(p)

    # TX: 生成 PRBS -> FFE 预加重 -> NRZ 波形
    bits = tx.generate_bits()
    wave_tx = tx.transmit(bits)

    # 信道: 有损传输 + 噪声
    wave_ch = apply_channel(wave_tx, p.dt, p)

    # RX: VGA + CTLE + CDR + DFE + 判决
    rx = Receiver(p)
    decisions, phase_hist, dfe_hist, z_hist = rx.run(wave_ch, p.nbits)

    ber, latency = compute_ber(decisions, bits)
    print(f"数据速率: {p.data_rate_gbps} Gbps, 比特数: {p.nbits}")
    print(f"误码率 BER: {ber:.3e}  (对齐延迟 {latency} UI)")
    print(f"CDR 最终相位: {phase_hist[-1]:.4f} UI")
    print(f"DFE 最终抽头: {[round(v, 4) for v in dfe_hist[-1]]}")
    return ber


if __name__ == "__main__":
    main()
