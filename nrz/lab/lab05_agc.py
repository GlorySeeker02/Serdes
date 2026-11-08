"""实验 5 (Stage 5): AGC —— 幅度归一化。

为什么单独给 AGC 一个 Stage? 因为它是"看不见的开关":
  关掉它, DFE/CDR 的自适应会全部失效, 而表面上看不出任何代码错误。
本实验复现 README 里记录的那个坑。

运行:
    python nrz/lab/lab05_agc.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_stats, sample_ui, np, Receiver,
)
from serdes_model.filters import lfilter  # noqa: E402


def fe_no_agc(p, wave):
    """只做 VGA + CTLE, 不做 AGC 归一化。"""
    rx = Receiver(p)
    return lfilter(rx.bd, rx.ad, np.asarray(wave, dtype=float) * p.vga_gain)


def exp_5_1_with_vs_without():
    h2("5.1 有 AGC / 无 AGC: 只差一个除法, 结果天壤之别")
    p = params(channel_loss_db_at_nyquist=20.0)
    link = Link(p)
    rx = Receiver(p)
    y_agc = rx.front_end(link.wave_ch)
    y_raw = fe_no_agc(p, link.wave_ch)

    rows = []
    for tag, y in (("有 AGC", y_agc), ("无 AGC", y_raw)):
        r = link.run(p, front_end=y)
        ber, _ = r.ber(skip=300)
        lvl = float(np.median(np.abs(y)))
        taps = r.taps_final
        rows.append([tag, f"{lvl:.3f}", f"{np.mean(np.abs(sample_ui(y, p.samples_per_ui, p.nbits))):.3f}",
                     " ".join(f"{t:+.2f}" for t in taps), f"{ber:.2e}"])
    tbl(["配置", "median(abs x)", "采样点 abs(d) 均值", "DFE 抽头 h1..h5", "BER"], rows)

    ask("关掉 AGC 后 DFE 抽头变成了负的, 为什么?")
    why("sign-sign LMS 的误差是 sign(z - s), 而 s = sign(z) = ±1。只有当判决点"
        "幅度已经接近 ±1 时, sign(z-s) 才反映'偏大还是偏小'(即 ISI 方向)。"
        "幅度只有 0.3 时, z-s 的符号恒等于 -s, 与 ISI 完全无关: "
        "抽头被推进到一个'让所有误差同号'的假平衡, 于是集体变负, 判决全错。")
    warn("这是行为级模型里最容易踩的坑: 自适应算法本身没错, 但它默认了"
         "'信号幅度已归一化'这个前提。工程上对应的是: **AGC 必须放在自适应"
         "均衡之前**, 且必须真正锁定, 否则 DFE/CDR 会一起跑飞。")


def exp_5_2_estimator():
    h2("5.2 AGC 怎么估幅度: median(|x|) 为什么够用")
    p = params(channel_loss_db_at_nyquist=20.0)
    link = Link(p)
    y_ctle = fe_no_agc(p, link.wave_ch)
    y_agc = Receiver(p).front_end(link.wave_ch)

    absx = np.abs(y_ctle)
    center = np.abs(sample_ui(y_ctle, p.samples_per_ui, p.nbits))
    rows = [
        ["median(abs x) 全部样本", f"{np.median(absx):.3f}"],
        ["mean(abs x) 全部样本", f"{np.mean(absx):.3f}"],
        ["median(abs d) 仅采样点", f"{np.median(center):.3f}"],
        ["90 分位 abs(x)", f"{np.quantile(absx, 0.90):.3f}"],
    ]
    tbl(["估计量", "数值"], rows)

    obs(f"AGC 用的估计量 = median(|x|) = {np.median(absx):.3f}; "
        f"归一化后 median(|x|) = {np.median(np.abs(y_agc)):.3f}")

    ask("NRZ 波形里, 为什么中位数能代表主抽头电平?")
    why("NRZ 的波形大部分时间停在平顶 (0 或 1), 只有窄窄的边沿在过渡。"
        "所以 |x| 的分布有两个峰 (≈ 主抽头电平), 中间是稀疏的过渡区; "
        "中位数落在峰上, 几乎不受过渡样本影响 —— 比均值 (会被边沿和噪声拉偏) 稳健。")
    warn("如果信道有严重下垂 (droop) 或占空比失真, 平顶电平会随码型漂移, "
         "单个中位数就不够用了; 真实芯片改用 AGC 环路 + 峰值检测 + 长时间平均。")


def exp_5_3_gain_invariance():
    h2("5.3 增益不变性: VGA 增益怎么变, AGC 之后都应该一样")
    rows = []
    base = None
    for g in (0.2, 0.5, 1.0, 2.0, 5.0):
        p = params(channel_loss_db_at_nyquist=20.0, vga_gain=g)
        link = Link(p)
        y = Receiver(p).front_end(link.wave_ch)
        r = link.run(p, front_end=y)
        ber, _ = r.ber(skip=300)
        hgt, _ = eye_stats(y, p.samples_per_ui, link.bits, phase=0.0)
        if base is None:
            base = y
        diff = float(np.max(np.abs(y - base)))
        rows.append([f"{g:.1f}", f"{np.median(np.abs(y)):.3f}", f"{hgt:.3f}",
                     f"{diff:.2e}", f"{ber:.2e}"])
    tbl(["vga_gain", "AGC 后 median(abs x)", "垂直眼高 [V]", "与 gain=0.2 的最大差", "BER"], rows)
    why("AGC 后整个过程与 VGA 增益无关: 归一化把 0.2 倍和 5 倍的输入映射到同一条波形。"
        "这就是'AGC 在前'的第二个好处 —— 后面所有自适应算法的动态范围都被压缩到 ±1 附近。")
    note("注意噪声不会完全不变: 这里噪声在信道里加, 放大 5 倍后再归一化, 相对"
         "信噪比其实会变。本表之所以完全一致, 是因为 vga_gain 在 AGC 之前、"
         "两者是同一个线性缩放。真实链路里 VGA 的增益会改变噪声相对幅度。")


def main():
    banner("5", "AGC 幅度归一化",
           "把判决点的幅度锁定在 ±1, 让后面所有自适应算法有意义")
    exp_5_1_with_vs_without()
    exp_5_2_estimator()
    exp_5_3_gain_invariance()
    h1("Stage 5 小结")
    print("""  1. AGC = 用一个稳健的幅度估计把主抽头拉到 ±1; 本模型用 median(|x|)。
  2. sign-sign LMS 的前提是"误差方向 = ISI 方向", 这要求幅度已归一化。
  3. 幅度不对时 DFE 抽头会集体变负, BER 崩溃, 而代码看起来没错 —— 最难查的坑。
  4. AGC 之后所有模块的动态范围都变成"以 1 为单位", 环路参数才好设计。
  5. 顺序很重要: VGA -> CTLE -> AGC -> CDR/DFE (自适应必须在归一化之后)。

  下一步: Stage 6 用归一化好的波形去恢复时钟 (CDR)。""")


if __name__ == "__main__":
    main()


