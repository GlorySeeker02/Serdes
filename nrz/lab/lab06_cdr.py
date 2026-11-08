"""实验 6 (Stage 6): CDR 时钟数据恢复。

要建立的三个直觉:
  1. CDR 在"边沿"和"数据中心"各采一个点, 用边沿值判断时钟是早了还是晚了;
  2. 二阶 (PI) 环路: 比例支路管相位, 积分支路管频率 (跟踪 ppm 偏差);
  3. 环路带宽是"锁定速度 vs 抖动"的取舍。

运行:
    python nrz/lab/lab06_cdr.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    bathtub, width_at, get_plt, OUT, np,
)


def exp_6_1_capture():
    h2("6.1 捕获: 初相位偏离 ±0.35 UI 也能拉回来")
    fig = None
    plt = None
    traj = []
    rows = []
    for init in (-0.35, -0.15, 0.0, 0.15, 0.35):
        p = params(cdr_initial_phase_ui=init)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        rows.append([f"{init:+.2f}", f"{r.phase_final:+.4f}", f"{r.settle_time(0.03, 16)}",
                     f"{r.phase_jitter():.4f}", f"{ber:.2e}"])
        traj.append((init, r.ph))
    tbl(["初相位 [UI]", "最终相位 [UI]", "锁定时间 [bit]", "稳态抖动 [UI]", "BER"], rows)

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for init, ph in traj:
        ax.plot(np.arange(len(ph)), ph, lw=1.0, label=f"init={init:+.2f}")
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("bit 序号")
    ax.set_ylabel("采样相位 [UI]")
    ax.set_title("CDR 相位轨迹: 从任意初相位收敛到最优采样点")
    ax.set_ylim(-0.5, 0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    png = OUT / "lab06_cdr_trajectory.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")

    ask("为什么最终相位略小于 0 (约 -0.03 UI), 而不是正好 0?")
    why("0 只是'标称中心'。真实最优采样点由信道的群时延/边沿不对称决定, CDR 会"
        "锁到使边沿能量对称的位置, 与几何中心差一点点是正常的 (这就是"
        "'相位偏移/静态相位误差')。")


def exp_6_2_loop_gain():
    h2("6.2 环路增益: 锁定速度与抖动的取舍")
    rows = []
    for kp, ki in ((1 / 8, 1 / 128), (1 / 16, 1 / 256), (1 / 32, 1 / 512),
                   (1 / 64, 1 / 2048), (1 / 128, 1 / 4096), (1 / 256, 1 / 8192)):
        p = params(cdr_kp=kp, cdr_ki=ki, cdr_initial_phase_ui=-0.3)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        rows.append([f"1/{round(1 / kp)}", f"1/{round(1 / ki)}",
                     f"{r.settle_time(0.03, 16)}", f"{r.phase_jitter():.4f}",
                     f"{r.phase_final:+.3f}", f"{ber:.2e}"])
    tbl(["kp", "ki", "锁定时间 [bit]", "稳态抖动 [UI]", "最终相位", "BER"], rows)

    ask("kp 越大锁得越快, 为什么不能一直加大?")
    why("kp 是比例增益: 每个 bit 都按鉴相误差直接改相位。增益太大时, 环路对"
        "单个 bit 的噪声/ISI 过度反应 -> 相位围绕最优点来回抖 (稳态抖动变大), "
        "严重时甚至把采样点推到眼图边上, BER 反而变差。"
        "工程上用'环路带宽 ≈ 数据率的 1/1000 ~ 1/2000'作为起点。")
    note("ki 管的是**频率**(积分), 它决定能否跟踪 ppm 级频偏: 只有 kp 时, "
         "残留频偏会变成固定相位误差; 有了 ki 才能把相位误差积到 0。")


def exp_6_3_detector():
    h2("6.3 鉴相器: bang-bang 还是线性?")
    rows = []
    for ctype in ("bangbang", "linear"):
        for noise in (0.01, 0.03):
            p = params(cdr_type=ctype, channel_noise_sigma=noise)
            link = Link(p)
            r = link.run(p)
            ber, _ = r.ber(skip=300)
            rows.append([ctype, f"{noise:.2f}", f"{r.phase_final:+.4f}",
                         f"{r.phase_jitter():.4f}", f"{ber:.2e}"])
    tbl(["cdr_type", "噪声 sigma", "最终相位", "稳态抖动 [UI]", "BER"], rows)

    why("bang-bang (Alexander) 只用边沿采样的符号: err = sign(e)·(s_prev - s), "
        "硬件只要一个判决器 + 异或门, 但输出是 1 bit 的, 环路本质上是「抖动的」; "
        "linear 保留边沿的模拟幅度, 误差更平滑, 抖动更小, 但需要多一个线性"
        "采样支路 (更像 ADC-based 接收机的做法)。")
    warn("bang-bang 的稳态抖动不是噪声, 而是**极限环**: 相位在最优点的两侧"
         "来回走一步 (步长 ∝ kp)。所以它不会随噪声减小而消失, 只能靠减小 kp。")


def exp_6_4_bathtub():
    h2("6.4 把 CDR 关掉, 手动扫相位: 这就是浴盆曲线")
    p = params()
    link = Link(p)
    phases = np.linspace(-0.45, 0.45, 37)
    y = link.front_end(p)
    ber_dfe = bathtub(link, phases, p, train=True, front_end=y)
    ber_raw = bathtub(link, phases, p, train=False, front_end=y)

    rows = []
    for ph, b1, b2 in zip(phases[::4], ber_dfe[::4], ber_raw[::4]):
        rows.append([f"{ph:+.2f}", f"{b1:.2e}", f"{b2:.2e}"])
    tbl(["采样相位 [UI]", "BER (有 DFE)", "BER (无 DFE)"], rows)
    obs(f"BER<1e-6 的相位宽度: 有 DFE = {width_at(phases, ber_dfe):.3f} UI, "
        f"无 DFE = {width_at(phases, ber_raw):.3f} UI")

    ask("盆底宽度代表什么? 打开 DFE 会把它变宽吗?")
    why("盆底 = 时间余量, 即采样时钟能偏多少还不误码。注意 DFE 几乎不改变它 "
        "(实测 0.675 -> 0.700 UI): DFE 消除的是残余 ISI, 相当于把眼图**上下**"
        "拉开, 但它不改变边沿的斜率, 所以救的是垂直余量而不是水平余量。"
        "要把盆底加宽, 得提高带宽 (更强的 CTLE/FFE) 或降低数据率。")
    obs("曲线左右不对称: 负相位一侧更宽。因为采样点靠近 UI 左边界时, 前一个"
        "比特的尾巴还在, 而这个尾巴**可以被 DFE 用过去的判决减掉**; 正相位一侧"
        "依赖的是'未来'比特, DFE 因果上无能为力, 所以右边界只剩真正的眼宽。")
    warn("这也解释了为什么'浴盆宽度'不能单独作为均衡效果的指标 —— 它衡量的"
         "是时间余量, 而均衡器主要在改善电压余量。两个余量要分开记。")

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    clip = 1e-12
    ax.semilogy(phases, np.clip(ber_dfe, clip, 1), marker="o", ms=3, label="有 DFE")
    ax.semilogy(phases, np.clip(ber_raw, clip, 1), marker="s", ms=3, label="无 DFE")
    ax.set_xlabel("采样相位 [UI]")
    ax.set_ylabel("BER")
    ax.set_ylim(clip, 1)
    ax.set_title("浴盆曲线 (横轴 = 采样时钟偏移)")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    png = OUT / "lab06_bathtub.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")


def main():
    banner("6", "CDR 时钟数据恢复",
           "从数据边沿里提取采样相位, 并把'锁定速度 vs 抖动'量化出来")
    exp_6_1_capture()
    exp_6_2_loop_gain()
    exp_6_3_detector()
    exp_6_4_bathtub()
    h1("Stage 6 小结")
    print("""  1. Alexander 鉴相: err = e·(s_prev - s), 只有跳变时才有信息。
  2. PI 环路: 比例支路 (kp) 管相位, 积分支路 (ki) 管频率 (跟踪 ppm)。
  3. kp 大 -> 锁得快但抖动大; kp 小 -> 抖动小但锁得慢。ki 太大会过冲。
  4. bang-bang 的稳态抖动是极限环, 不随噪声消失; linear 鉴相更平滑。
  5. 关掉环路扫相位 = 浴盆曲线, 盆底宽度就是时间余量。

  下一步: Stage 7 用恢复出来的时钟做判决, 再把残余后游标用 DFE 收掉。""")


if __name__ == "__main__":
    main()




