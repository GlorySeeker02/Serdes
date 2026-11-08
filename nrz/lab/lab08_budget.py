"""实验 8 (Stage 8): 系统级余量预算。

要建立的三个直觉:
  1. 链路能不能工作, 最终归结为"垂直余量"和"水平余量"够不够;
  2. 抖动吃掉水平余量, 噪声和串扰吃掉垂直余量, 两者用浴盆曲线合起来看;
  3. 均衡器救的是"眼高", 救不了"信噪比" —— AGC 把幅度拉回来不等于把 SNR 拉回来。

运行:
    python nrz/lab/lab08_budget.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_stats, bathtub, width_at, get_plt, OUT, np,
)
from serdes_model.filters import lfilter  # noqa: E402
from serdes_model import Receiver  # noqa: E402


def exp_8_1_bathtub():
    h2("8.1 浴盆曲线: 时间余量")
    p = params()
    link = Link(p)
    phases = np.linspace(-0.5, 0.5, 41)
    y = link.front_end(p)
    ber_dfe = bathtub(link, phases, p, train=True, front_end=y)
    ber_raw = bathtub(link, phases, p, train=False, front_end=y)
    w1 = width_at(phases, ber_dfe)
    w2 = width_at(phases, ber_raw)
    obs(f"BER<1e-6 的相位窗口: 有 DFE {w1:.3f} UI, 无 DFE {w2:.3f} UI")
    obs(f"理想 NRZ (无限带宽) 的理论上限是 1.0 UI; 本链路只发挥了 {w1 * 100:.0f}%")

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    clip = 1e-12
    ax.semilogy(phases, np.clip(ber_dfe, clip, 1), marker="o", ms=3, label="CTLE+DFE")
    ax.semilogy(phases, np.clip(ber_raw, clip, 1), marker="s", ms=3, label="仅 CTLE")
    ok = phases[ber_dfe < 1e-6]
    if len(ok):
        ax.axvspan(float(ok.min()), float(ok.max()), alpha=0.12, color="tab:green",
                   label="BER<1e-6 的相位窗口")
    ax.set_xlabel("采样相位 [UI]")
    ax.set_ylabel("BER")
    ax.set_ylim(clip, 1)
    ax.set_title("浴盆曲线: 水平 (时间) 余量")
    ax.grid(which="both", alpha=0.3)
    ax.legend(fontsize=8)
    png = OUT / "lab08_bathtub.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")

    ask("为什么两端会冲到 0.5 (瞎猜) 而不是慢慢变差?")
    why("因为采样点移出眼图后, 判决值与当前比特几乎无关, 只剩随机的 ±1 —— "
        "BER 就饱和在 0.5。浴盆的「盆壁」之所以很陡, 是因为眼图边沿的斜率有限, "
        "一旦越过边界就迅速失效。")


def exp_8_2_jitter():
    h2("8.2 发送端抖动 (RJ) 如何吃掉水平余量")
    rows = []
    for rj in (0.0, 0.02, 0.05, 0.10, 0.15):
        p = params(tx_rj_sigma_ui=rj)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        hz, _ = r.eye_z()
        rows.append([f"{rj:.2f}", f"{hz:+.3f}", f"{r.phase_final:+.3f}",
                     f"{r.phase_jitter():.4f}", f"{ber:.2e}"])
    tbl(["rj_sigma [UI]", "z 眼高 [V]", "CDR 相位", "CDR 抖动 [UI]", "BER"], rows)

    ask("抖动加在发送端, 为什么 CDR 的相位抖动也跟着变大?")
    why("抖动让每个边沿的位置随机偏移。bang-bang 鉴相器看到的是'边沿偏了', "
        "它无法区分这是时钟偏了还是数据边沿本身就抖 —— 于是把抖动当成相位误差"
        "去追, 相位跟着一起抖。这就是为什么 CDR 需要**低带宽**环路: "
        "让它跟踪慢漂移 (ppm), 而不去追快抖动。")
    note("经验值: 一个健康的链路, 抖动 + 噪声 + ISI 总预算里, 水平余量通常要求"
         "留 0.2~0.3 UI 给 CDR 跟踪误差。")


def exp_8_3_noise():
    h2("8.3 噪声如何吃掉垂直余量")
    rows = []
    for sigma in (0.04, 0.06, 0.08, 0.10, 0.12):
        p = params(channel_noise_sigma=sigma)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        hz, sz = r.eye_z()
        # 眼高/厚度 就是"信号幅度 / (噪声+残余ISI 的散布)", 是垂直余量的直观读数
        rows.append([f"{sigma:.2f}", f"{hz:+.3f}", f"{sz:.3f}",
                     f"{hz / sz:.1f}" if sz > 0 else "-", f"{ber:.2e}"])
    tbl(["噪声 sigma", "z 眼高 [V]", "z 厚度 [V]", "眼高/厚度", "BER"], rows)

    ask("为什么 BER 对噪声的敏感度是指数级的?")
    why("判决错误 = 噪声把采样点推过门限。高斯尾概率 Q(x) ~ exp(-x^2/2), "
        "所以眼高每减小一点, BER 会按指数掉。工程上用「眼高 = 12σ」对应 BER~1e-16 "
        "(Q(6)) 来做预算。")
    warn("本模型没有建模确定性抖动 (DJ)、占空比失真 (DCD) 和电源噪声, "
         "真实链路的噪声「厚度」会明显比这里大。")


def exp_8_4_ladder():
    h2("8.4 眼高阶梯: 每一级损失了多少")
    p = params()
    link = Link(p)
    sp = p.samples_per_ui
    r = link.run(p)

    lvl = []
    lvl.append(("TX 输出 (理想 ±1)", 2.0))
    lvl.append(("TX FFE 后 (Σtaps=0.5)", 1.0))
    lvl.append(("信道输出", eye_stats(link.wave_ch, sp, link.bits)[0]))
    y_ctle = lfilter(Receiver(p).bd, Receiver(p).ad, link.wave_ch * p.vga_gain)
    lvl.append(("CTLE 后 (未归一)", eye_stats(y_ctle, sp, link.bits)[0]))
    lvl.append(("AGC 归一化后", eye_stats(link.front_end(p), sp, link.bits)[0]))
    hz, _ = r.eye_z()
    lvl.append(("DFE 后 (判决器输入)", hz))

    base = lvl[0][1]
    rows = [[name, f"{v:+.3f}", f"{v / base * 100:.0f}%"] for name, v in lvl]
    tbl(["阶段", "垂直眼高 [V]", "占理想值"], rows)

    ask("AGC 把幅度从 0.9 拉回 1.1, 是「赚」了吗?")
    why("不是。AGC 是无差别放大: 信号和噪声同乘一个系数, 眼高和厚度一起变大, "
        "**信噪比不变**。它真正的价值是让后面自适应算法的动态范围固定下来 "
        "(Stage 5), 而不是改善链路质量。所以「眼高」必须和「厚度」一起看。")

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [n for n, _ in lvl][1:]
    vals = [v for _, v in lvl][1:]
    colors = ["tab:blue", "tab:red", "tab:orange", "tab:green"]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" ")[0] for n in names], fontsize=9)
    ax.set_ylabel("垂直眼高 [V]")
    ax.set_title("眼高阶梯: 信道把它打下去, 均衡把它拉回来")
    for i, v in enumerate(vals):
        ax.text(i, v + (0.03 if v >= 0 else -0.09), f"{v:.2f}", ha="center", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    png = OUT / "lab08_ladder.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")


def main():
    banner("8", "系统级余量预算",
           "把抖动、噪声、ISI 换算成「眼高/眼宽」, 看整条链路还剩多少余量")
    exp_8_1_bathtub()
    exp_8_2_jitter()
    exp_8_3_noise()
    exp_8_4_ladder()
    h1("Stage 8 小结")
    print("""  1. 链路能力 = 垂直余量 (眼高/噪声) x 水平余量 (眼宽/抖动), 缺一不可。
  2. 浴盆曲线 = 水平余量的直接读数; BER 在两端饱和到 0.5 是正常现象。
  3. 发送端 RJ 会被 CDR 部分"追"进去, 所以 CDR 必须是低带宽环路。
  4. BER 对噪声是指数的 (Q 函数), 眼高预算常用 "12 sigma" 作为 1e-16 的判据。
  5. AGC 把幅度拉回来不等于改善 SNR: 信号和噪声一起放大。看指标要成对看。

  到这里 NRZ 链路的主干就走完了。下一步可以:
    - 回看 02 的眼图与 08 的浴盆, 把它们和笔记里的"预算表"对起来;
    - 进 pam4_adc/, 看同样的数学在 PAM4 + ADC 架构里怎么变形。""")


if __name__ == "__main__":
    main()



