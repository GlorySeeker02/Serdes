"""核心模型单元测试。

兼容 pytest，也可直接运行:

    python tests/test_core.py

不依赖 pytest: 会逐个执行 test_* 函数并打印 PASS/FAIL。
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from serdes_model import (
    SerdesParams, Transmitter, Receiver, apply_channel, channel_transfer,
    prbs_bits, bits_to_nrz, symbols_to_nrz, add_awgn,
    bilinear_biquad, lfilter, design_ctle, deserialize,
    compute_ber, eye_hist2d,
)


# ---------------- 信号源 ----------------
def test_prbs7_period_and_diversity():
    out = prbs_bits(127 * 3, 7)
    assert len(out) == 381
    assert set(np.unique(out)).issubset({0, 1})
    # 周期 127: 相邻两个周期完全相同
    assert np.array_equal(out[:127], out[127:254])
    assert np.array_equal(out[127:254], out[254:381])
    # 非平凡序列 (不是全 0 或全 1)
    assert 0 < int(out[:127].sum()) < 127


def test_prbs31_runs():
    out = prbs_bits(200, 31)
    assert len(out) == 200
    assert set(np.unique(out)).issubset({0, 1})


def test_prbs_invalid_order_raises():
    try:
        prbs_bits(10, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("非法 PRBS 阶数应抛 ValueError")


def test_bits_to_nrz():
    bits = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    w = bits_to_nrz(bits, 8, rise_ui=0.1, swing=2.0)
    assert len(w) == len(bits) * 8
    assert w.min() >= -2.0 - 1e-9 and w.max() <= 2.0 + 1e-9
    assert np.isclose(w[1 * 8 + 4], 2.0)    # bit1 中心, 两侧都是 1
    assert np.isclose(w[5 * 8 + 4], -2.0)   # bit5 中心, 两侧都是 0


def test_symbols_to_nrz_arbitrary_levels():
    syms = np.array([0.5, -0.25, 1.0])
    w = symbols_to_nrz(syms, 4, rise_ui=0.0)
    assert len(w) == 12
    assert np.allclose(w[:4], 0.5)


def test_add_awgn_shape():
    w = np.zeros(100)
    out = add_awgn(w, 0.1, np.random.default_rng(0))
    assert out.shape == w.shape
    assert abs(out.mean()) < 0.05


# ---------------- 信道 ----------------
def test_channel_dc_gain():
    H = channel_transfer(np.array([0.0]), 20.0, 5e9, 0.5)
    assert abs(H[0] - 1.0) < 1e-9


def test_channel_nyquist_loss():
    H = channel_transfer(np.array([5e9]), 20.0, 5e9, 0.5)
    assert abs(abs(H[0]) - 10 ** (-20.0 / 20.0)) < 1e-9


def test_channel_ideal_passthrough():
    p = SerdesParams()
    p.channel_type = "ideal"
    p.channel_noise_sigma = 0.0
    p.channel_xtalk_amp = 0.0
    w = np.random.default_rng(0).normal(size=200)
    assert np.allclose(apply_channel(w, p.dt, p), w)


# ---------------- 滤波器 ----------------
def test_bilinear_dc_gain_preserved():
    # H(s) = 1/(s/1e4 + 1) 的直流增益应为 1
    bz, az = bilinear_biquad([0.0, 1.0], [1.0 / 1e4, 0.0, 1.0], fs=1e9)
    assert abs(np.sum(bz) / np.sum(az) - 1.0) < 1e-6


def test_lfilter_identity():
    x = np.random.default_rng(0).normal(size=300)
    assert np.allclose(lfilter([1.0], [1.0], x), x)


def test_lfilter_first_order():
    b = [1.0, 0.5]
    a = [1.0, -0.5]
    x = np.ones(60)
    y = lfilter(b, a, x)
    ref = np.zeros(60)
    for i in range(60):
        ref[i] = x[i] + (0.5 * x[i - 1] if i > 0 else 0.0) + (0.5 * ref[i - 1] if i > 0 else 0.0)
    assert np.allclose(y, ref)


def test_ctle_dc_gain():
    p = SerdesParams()
    bz, az = design_ctle(p)
    assert abs(np.sum(bz) / np.sum(az) - p.ctle_dc_gain) < 1e-6


# ---------------- 分析 ----------------
def test_deserialize_shape():
    dec = np.array([1.0, -1.0] * 10)
    words = deserialize(dec, 4)
    assert words.shape == (5, 4)
    assert set(np.unique(words)).issubset({0, 1})


def test_compute_ber_perfect():
    bits = prbs_bits(500, 7)
    dec = np.where(bits > 0, 1.0, -1.0)
    ber, lag = compute_ber(dec, bits, skip=0)
    assert ber == 0.0 and lag == 0


def test_compute_ber_shifted():
    bits = prbs_bits(500, 7)
    base = np.where(bits > 0, 1.0, -1.0)
    dec = np.roll(base, 3)
    ber, lag = compute_ber(dec, bits, skip=0, max_lag=10)
    assert ber == 0.0 and abs(lag) == 3


def test_eye_hist2d_shape():
    w = np.random.default_rng(0).normal(size=32 * 50)
    hist, lo, hi = eye_hist2d(w, 32, nbins=16)
    assert hist.shape == (16, 32)
    assert hi > lo


# ---------------- 端到端 ----------------
def _run_link(nbits=2048):
    p = SerdesParams()
    p.nbits = nbits
    tx = Transmitter(p)
    bits = tx.generate_bits()
    wc = apply_channel(tx.transmit(bits), p.dt, p)
    rx = Receiver(p)
    return p, bits, wc, rx.run(wc, nbits)


def test_end_to_end_ber_zero():
    _p, bits, _wc, (dec, _ph, _dfe, _z) = _run_link(2048)
    ber, _ = compute_ber(dec, bits, skip=300)
    assert ber == 0.0


def test_cdr_locks():
    _p, _bits, _wc, (_dec, ph, _dfe, _z) = _run_link(2048)
    assert abs(ph[-1]) < 0.2


def test_dfe_tap_positive():
    _p, _bits, _wc, (_dec, _ph, dfe, _z) = _run_link(2048)
    assert dfe[-1][0] > 0.0


def test_front_end_reuse_equivalent():
    p, _bits, wc, _ = _run_link(512)
    rx = Receiver(p)
    y = rx.front_end(wc)
    d1, _, _, _ = rx.run(wc, 512)
    d2, _, _, _ = rx.run(wc, 512, front_end=y)
    assert np.array_equal(d1, d2)


def test_run_rejects_short_input():
    p = SerdesParams()
    p.nbits = 100
    rx = Receiver(p)
    try:
        rx.run(np.zeros(10), 100)
    except ValueError:
        pass
    else:
        raise AssertionError("过短输入应抛 ValueError")


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
