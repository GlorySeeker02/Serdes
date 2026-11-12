"""PAM4 ADC-based 模型单元测试。

兼容 pytest, 也可直接运行:  python tests/test_core.py
"""
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pam4_model import (
    PAM4Params, PAM4_LEVELS, PAM4_THRESHOLDS,
    Transmitter, Receiver, apply_channel, channel_transfer,
    prbs_bits, bits_to_pam4, pam4_to_bits, symbols_to_nrz,
    quantize, quantization_sigma, pam4_slice, design_ffe,
    compute_ser, compute_ber, eye_hist2d,
)


# ---------------- 信号源 ----------------
def test_prbs7_period():
    out = prbs_bits(127 * 3, 7)
    assert np.array_equal(out[:127], out[127:254])
    assert 0 < int(out[:127].sum()) < 127


def test_gray_roundtrip():
    bits = np.random.default_rng(0).integers(0, 2, size=200).astype(np.int8)
    levels = bits_to_pam4(bits)
    assert set(np.unique(levels)).issubset(set(PAM4_LEVELS))
    back = pam4_to_bits(levels)
    assert np.array_equal(back, bits)


def test_pam4_levels():
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.int8)
    levels = bits_to_pam4(bits)
    # 00->-3, 01->-1, 11->+1, 10->+3
    assert np.allclose(levels, [-3.0, -1.0, 1.0, 3.0])


def test_symbols_to_nrz_length():
    w = symbols_to_nrz([1.0, -1.0, 3.0], 4, rise_ui=0.0)
    assert len(w) == 12


# ---------------- 信道 ----------------
def test_channel_dc_gain():
    H = channel_transfer(np.array([0.0]), 8.0, 5e9, 0.5)
    assert abs(H[0] - 1.0) < 1e-9


def test_channel_nyquist_loss():
    H = channel_transfer(np.array([5e9]), 12.0, 5e9, 0.5)
    assert abs(abs(H[0]) - 10 ** (-12.0 / 20.0)) < 1e-9


def test_channel_xtalk_decorrelated_from_data():
    """串扰必须是独立码型: 与主数据不相关。

    攻击源序列若与主数据同初态、同阶数, LFSR 就会生成完全相同的序列,
    相加只是把主信号放大了一点 (眼图反而变好)。这里从波形层面检查:
    "注入的串扰分量" 与 "干净数据波形" 的相关系数应接近 0。
    """
    p = PAM4Params()
    p.n_symbols = 512
    p.channel_type = "ideal"        # 只观察串扰本身, 排除信道滤波
    p.channel_noise_sigma = 0.0
    tx = Transmitter(p)
    wave, _ = tx.transmit(tx.generate_bits())

    clean = apply_channel(wave, p.dt, replace(p, channel_xtalk_amp=0.0))
    xtalked = apply_channel(wave, p.dt, replace(p, channel_xtalk_amp=0.3))
    xt = xtalked - clean
    assert np.abs(xt).max() > 0.1, "串扰没有真正注入"

    r = float(np.corrcoef(xt, clean)[0, 1])
    assert abs(r) < 0.2, f"串扰与主数据相关系数 {r:.3f}, 它只是主数据的相干副本"


# ---------------- ADC ----------------
def test_quantize_levels():
    x = np.array([-5.0, 0.0, 5.0, 1.03])
    q = quantize(x, 4, 4.0)
    assert q[0] == -4.0 and q[2] == 4.0   # 限幅
    assert abs(q[1]) < 1e-9


def test_quantization_sigma():
    s = quantization_sigma(6, 4.0)
    assert s > 0


# ---------------- 判决 ----------------
def test_pam4_slice_thresholds():
    assert pam4_slice(-2.5) == -3.0
    assert pam4_slice(-1.0) == -1.0
    assert pam4_slice(0.5) == 1.0
    assert pam4_slice(2.5) == 3.0


def test_design_ffe_main_tap():
    p = PAM4Params()
    ffe = design_ffe(p)
    assert len(ffe) == p.ffe_ntaps
    assert abs(ffe[p.ffe_ntaps // 2] - 1.0) < 1e-6


# ---------------- 端到端 ----------------
def _run_link(n_symbols=2048):
    p = PAM4Params()
    p.n_symbols = n_symbols
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wave, syms = tx.transmit(bits)
    wc = apply_channel(wave, p.dt, p)
    rx = Receiver(p)
    return p, syms, wc, rx.run(wc, n_symbols)


def test_end_to_end_ser_zero():
    _p, syms, _wc, (dec, _ph, _dfe, _z) = _run_link(2048)
    ser, _ = compute_ser(dec, syms, skip=300)
    assert ser == 0.0


def test_end_to_end_ber_zero():
    _p, syms, _wc, (dec, _ph, _dfe, _z) = _run_link(2048)
    ber, _, _ = compute_ber(dec, syms, skip=300)
    assert ber == 0.0


def test_cdr_locks():
    _p, _syms, _wc, (_dec, ph, _dfe, _z) = _run_link(2048)
    assert abs(ph[-1]) < 0.2


def test_front_end_reuse_equivalent():
    p, _syms, wc, _ = _run_link(512)
    rx = Receiver(p)
    y = rx.front_end(wc)
    d1, _, _, _ = rx.run(wc, 512)
    d2, _, _, _ = rx.run(wc, 512, front_end=y)
    assert np.array_equal(d1, d2)


def test_run_rejects_short_input():
    p = PAM4Params()
    p.n_symbols = 100
    rx = Receiver(p)
    try:
        rx.run(np.zeros(10), 100)
    except ValueError:
        pass
    else:
        raise AssertionError("过短输入应抛 ValueError")


# ---------------- 分析 ----------------
def test_eye_hist2d_shape():
    w = np.random.default_rng(0).normal(size=32 * 50)
    hist, lo, hi = eye_hist2d(w, 32, nbins=16)
    assert hist.shape == (16, 32)
    assert hi > lo


def test_compute_ser_alignment():
    p = PAM4Params()
    p.n_symbols = 500
    tx = Transmitter(p)
    syms = tx.bits_to_symbols(tx.generate_bits())
    dec = np.roll(syms, 3)
    ser, lag = compute_ser(dec, syms, skip=0, max_lag=10)
    assert ser == 0.0 and abs(lag) == 3


# ---------------- 直接运行 (无 pytest) ----------------
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
