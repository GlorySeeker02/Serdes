"""实验 3 (Stage 3): TX FFE / 预加重。

要建立的三个直觉:
  1. FFE 是一个 3 抽头 FIR: 用"压低低频 + 抬高跳变"换高频余量;
  2. 前抽头 (pre) 专门抵消前游标, 后抽头 (post) 专门抵消后游标;
  3. 预加重不是越多越好: 直流增益 (摆幅) 和噪声放大都是代价。

运行:
    python nrz/lab/lab03_tx_ffe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    NOFFE, Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_stats, pulse_cursors, get_plt, OUT, np,
)


def exp_3_1_what_ffe_does():
    h2("3.1 FFE 对符号流做了什么")
    p = params(tx_taps=NOFFE)
    bits = np.array([1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1])
    syms = np.where(bits > 0, 1.0, -1.0)

    print("  比特      :", "  ".join(f"{b:2d}" for b in bits))
    print("  发送符号  :", "  ".join(f"{v:+.0f}" for v in syms))
    for taps in ((-0.15, 0.85, -0.20), (-0.30, 1.00, -0.15)):
        # 两端各补 2 个符号, 避免卷积的边界效应干扰教学展示
        pad = np.concatenate([syms[:1]] * 2 + [syms] + [syms[-1:]] * 2)
        out = np.convolve(pad, np.array(taps), mode="same")[2:-2]
        print(f"  taps={taps} ->", "  ".join(f"{v:+.2f}" for v in out))
    obs("注意长连 1 的稳态只有 +0.50, 而孤立 1 能冲到 +1.20 (默认抽头): "
        "低频被压低、跳变被抬高, 这就是预加重。")

    rows = []
    for taps in (NOFFE, (-0.15, 0.85, -0.20), (-0.30, 1.00, -0.15), (0.0, 1.0, -0.30)):
        s = float(np.sum(taps))
        peak = abs(taps[0]) + taps[1] + abs(taps[2])
        rows.append([str(taps), f"{s:+.2f}", f"{peak:.2f}", f"{20 * np.log10(s):+.1f}"])
    tbl(["tx_taps [pre, main, post]", "直流增益 Σ", "最大摆幅", "低频损失 [dB]"], rows)

    ask("为什么 pre/post 必须是负的? 它们凭什么「补偿」高频?")
    why("抽头之和 (直流增益) 被压到 < 1, 而孤立比特的摆幅 (|pre|+main+|post|) 却 > 1: "
        "等价于'低通被压低、高通被抬高', 即在发送端就做了一个高通整形, "
        "正好抵消信道的低通。所以预加重本质上是**频域预失真**, 不是'放大信号'。")


def exp_3_2_cursor_view():
    h2("3.2 游标视角: pre 抽头打前游标, post 抽头打后游标")
    p = params(tx_taps=NOFFE, channel_noise_sigma=0.0)
    ref = None
    rows = []
    for taps in (NOFFE, (-0.15, 0.85, -0.20), (-0.40, 0.85, 0.0),
                 (0.0, 0.85, -0.30), (-0.25, 0.85, -0.15)):
        offs, cur, main = pulse_cursors(p, loss_db=20.0, tx_taps=taps)
        if ref is None:
            ref = offs
        pick = [list(offs).index(k) for k in (-2, -1, 0, 1, 2)]
        rows.append([str(taps)] + [f"{cur[i]:+.3f}" for i in pick])
    tbl(["tx_taps", "h(-2)", "h(-1)", "h(0)", "h(+1)", "h(+2)"], rows)

    ask("pre 更负, 哪一个游标先变小?")
    why("h(-1) 也就是前游标。因为 FFE 是符号域 FIR, 等效于 h_eff = conv(h_ch, taps): "
        "pre 抽头作用在**前一个**符号上, 只能改 h(-1) 附近; post 抽头只能改 h(+1) 附近。"
        "两者几乎不互相干扰 —— 这就是 TX FFE 能同时对付前、后游标的原因。")
    obs("看 pre=-0.40 那一行: h(-1) 从 +0.404 一路压到 -0.082 —— 压过头了, "
        "符号翻负。均衡过度的第一个征兆就是游标反号, 它会额外制造新的 ISI。")

    note("注意 h(0) 也在变: FFE 的 main 抽头如果不等于 1, 主游标会跟着缩。"
         "实际芯片里 main 通常固定, 用 pre/post 的负值换高频, 代价是直流增益下降。")
    warn("物理限制: 抽头太负会让输出摆幅超出驱动器的线性范围 (本模型没有建模削顶), "
         "真实芯片上过大的预加重会引入非线性, 反而恶化眼图。")


def exp_3_3_sweep():
    h2("3.3 端到端扫描: 20 dB 信道下 pre/post 怎么选")
    rows = []
    for pre in (0.0, -0.10, -0.20, -0.30):
        for post in (0.0, -0.10, -0.20):
            taps = (pre, 1.0, post)
            p = params(tx_taps=taps, channel_loss_db_at_nyquist=20.0,
                       channel_noise_sigma=0.01)
            link = Link(p)
            q = variant(p, dfe_train=False, cdr_kp=0.0, cdr_ki=0.0,
                        cdr_initial_phase_ui=0.0)
            y = link.front_end(q)
            hgt, _ = eye_stats(y, q.samples_per_ui, link.bits, phase=0.0)
            ber, _ = link.run(q, front_end=y).ber(skip=300)
            rows.append([f"{pre:+.2f}", f"{post:+.2f}", f"{hgt:.3f}", f"{ber:.2e}"])
    tbl(["pre", "post", "AGC 后垂直眼高 [V]", "BER (无 DFE, 无 CTLE 峰值)"], rows)

    ask("这张表里预加重越重、眼高越大, 难道没有上限?")
    why("本表只看到收益, 因为 (1) AGC 把直流增益的损失补回来了, (2) 20 dB 信道的 "
        "ISI 还远没压完, (3) 噪声只有 0.01。真实上限来自三处: 驱动器摆幅削顶、"
        "噪声与串扰被 AGC 同比例放大、以及 tap 反号 (见 3.2)。3.4 把噪声加大, "
        "用更重的预加重把代价逼出来。")


def exp_3_4_cost():
    h2("3.4 预加重的代价: 摆幅、噪声、眼高")
    rows = []
    for pre, post in ((0.0, 0.0), (-0.15, -0.10), (-0.30, -0.20), (-0.45, -0.30)):
        p = params(tx_taps=(pre, 1.0, post), channel_loss_db_at_nyquist=6.0,
                   channel_noise_sigma=0.02)
        link = Link(p)
        q = variant(p, dfe_train=False, cdr_kp=0.0, cdr_ki=0.0,
                    cdr_initial_phase_ui=0.0)
        y = link.front_end(q)
        hgt, thick = eye_stats(y, q.samples_per_ui, link.bits, phase=0.0)
        ber, _ = link.run(q, front_end=y).ber(skip=300)
        rows.append([f"{pre:+.2f}/{post:+.2f}", f"{1 + pre + post:+.2f}",
                     f"{hgt:.3f}", f"{thick:.3f}", f"{ber:.2e}"])
    tbl(["pre/post", "直流增益", "AGC 后眼高 [V]", "垂直厚度 [V]", "BER"], rows)

    why("在只损耗 6 dB 的信道上开很重的预加重: AGC 把幅度拉回 ±1 (主游标看起来没变), "
        "但 AGC 的增益 = 1/直流增益, 噪声被同比例放大 4 倍; 于是眼高反而被抹掉, "
        "垂直厚度从 0.41 涨到 1.78。这就是'过度均衡'的典型症状: 眼不变大, 只是更「毛」。")
    note("经验法则: 预加重的量应该略小于信道在该频段的损耗, 均衡量不够时游标反号就是"
         "过头的信号。区分两种病: 均衡不足 -> ISI 主导 (眼高小, 厚度不涨); "
         "均衡过度 -> 噪声/串扰主导 (眼高反而下降, 厚度明显变大)。")


def main():
    banner("3", "TX FFE / 预加重",
           "在发送端用已知数据做频域预失真, 把前/后游标压下去")
    exp_3_1_what_ffe_does()
    exp_3_2_cursor_view()
    exp_3_3_sweep()
    exp_3_4_cost()
    h1("Stage 3 小结")
    print("""  1. 3-tap FFE = 符号域 FIR: y[k] = pre·a[k+1] + main·a[k] + post·a[k-1]。
  2. 它不能"放大信号", 只能重分配频谱: 直流增益 Σtaps < 1, 跳变摆幅 > 1。
  3. pre 抽头 -> 前游标 h(-1); post 抽头 -> 后游标 h(+1); 两者几乎解耦。
  4. 代价是摆幅 (直流增益) 和噪声放大 (AGC 跟着放大输入噪声)。
  5. FFE 是"已知数据"的均衡: 它利用发送端知道未来的比特, 所以能处理前游标。

  下一步: Stage 4 转到接收端, 用连续的模拟滤波器 (CTLE) 做同样的高频补偿。""")


if __name__ == "__main__":
    main()
