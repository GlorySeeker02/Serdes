"""实验 1 (Stage 1): 时间基、PRBS、NRZ 波形、眼图。

要建立的三个直觉:
  1. 一个 UI 是多久? 32 个采样点是怎么来的?
  2. PRBS 为什么是 SerDes 的标准激励? (游程分布 -> 频谱 -> 最坏 ISI)
  3. 眼图是怎么"折"出来的? 上升时间如何吃掉眼高?

运行:
    python nrz/lab/lab01_waveform.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params,
    eye_height, eye_stats, eye_profile, eye_width, rc_lowpass, get_plt, OUT, np,
)

from serdes_model import bits_to_nrz, prbs_bits  # noqa: E402


def exp_1_1_time_base():
    h2("1.1 时间基: UI / 采样点 / 采样率")
    p = params()

    tbl(["物理量", "数值", "怎么来的"], [
        ["数据率", f"{p.data_rate_gbps} Gbps", "参数"],
        ["UI (1 bit 周期)", f"{p.ui * 1e12:.1f} ps", "1 / 数据率"],
        ["采样间隔 dt", f"{p.dt * 1e12:.3f} ps", "UI / samples_per_ui"],
        ["采样率 fs", f"{p.fs / 1e9:.0f} GS/s", "1 / dt"],
        ["奈奎斯特频率", f"{p.nyquist_ghz} GHz", "数据率 / 2"],
        ["波形总长", f"{p.nbits * p.samples_per_ui} 点", "nbits x samples_per_ui"],
    ])

    ask("为什么 sampling rate 是 320 GS/s, 而不是 10 GS/s?")
    why("10 GS/s 只能在每个 UI 采 1 点, 看不到边沿位置, CDR/眼图都无从谈起。"
        "32 点/UI 是为了让'采样相位'能连续调节 —— CDR 调的正是这个小数相位。")

    ask("10 Gbps 的 NRZ 信号, 频谱主瓣到哪里? 判决真正关心的又是哪个频点?")
    why("理论谱是 sinc(f/10GHz), 主瓣到 10 GHz; 但一个 UI 只承载 1 bit, "
        "能区分 1010 的最快变化就是 5 GHz —— 所以信道指标几乎总在 "
        "Nyquist = 数据率/2 处标定 (本模型 loss_db_at_nyquist 就是它)。"
        "5 GHz 以上的损耗主要影响边沿陡度, 不直接决定眼高。")


def exp_1_2_one_ui():
    h2("1.2 一个 UI 里到底发生了什么")
    p = params()
    bits = np.array([1, 1, 1, 0, 0, 1, 1, 1])
    wave = bits_to_nrz(bits, p.samples_per_ui, rise_ui=p.nrz_rise_ui)

    print("  比特        :", "  ".join(f"{b}" for b in bits))
    print("  对应电平    :", "  ".join("+1" if b else "-1" for b in bits))
    print("\n  每个 UI 抽 8 个点 (每 4 点取 1), 行 = UI, 列 = UI 内相位 0/32..28/32")
    sp = p.samples_per_ui
    for k, b in enumerate(bits):
        seg = wave[k * sp:(k + 1) * sp][::4]
        print(f"  UI{k} (bit={b}): " + "  ".join(f"{v:+.2f}" for v in seg))

    ask("为什么 bit 从 1 变 0 的那一个 UI 里, 数值不是直接跳变?")
    why("真实驱动器的边沿有限, 模型用升余弦过渡模拟, 过渡宽度 = 2 x nrz_rise_ui "
        "个 UI。边沿越缓, 到下一个 UI 中心时越可能还没到终值 —— 这就是 ISI 的源头。")

    ask("'眼图'到底是什么?")
    why("把长波形按每 32 点切成一片 (每片 = 1 UI), 全部叠画在同一张图上。"
        "凡是 0->1 或 1->0 的片都会经过中间区域, 于是中间留出一个空白菱形 = 眼。"
        "空白的高度 = 电压余量, 宽度 = 时间余量 (采样时钟能偏多少还不误码)。")


def exp_1_3_prbs():
    h2("1.3 PRBS7: 为什么不用随机比特, 也不用人造 1010 图案")
    bits = prbs_bits(127 * 3, 7)

    period = np.array_equal(bits[:127], bits[127:254])
    obs(f"周期 127 连续两段完全相同: {period}")

    # 游程长度分布 (连续相同比特的长度), 除以 3 得到"每周期"次数
    runs = []
    cur, cnt = bits[0], 1
    for b in bits[1:]:
        if b == cur:
            cnt += 1
        else:
            runs.append(cnt)
            cur, cnt = b, 1
    runs.append(cnt)
    runs = np.array(runs)
    theory = [32, 16, 8, 4, 2, 1, 1]      # PRBS7 一个周期内的理论游程数
    rows = []
    for L in range(1, 8):
        n = int(np.sum(runs == L)) / 3.0
        rows.append([L, f"{n:.0f}", theory[L - 1], f"{n / 64 * 100:.0f}%"])
    tbl(["游程长度", "每周期次数", "理论值", "占全部游程"], rows)
    obs(f"一个周期共 {len(runs) / 3:.0f} 个游程 (理论 (2^7)/2 = 64)")

    ask("最长的 0 游程有几个比特? 这会激励出什么频率?")
    why("7 个连 0 => 最低频率成分约 1/(7 UI) ≈ 1.4 GHz, 正好压在信道损耗开始"
        "抬头的频段, 所以 PRBS7 能覆盖'最坏低频下垂 + 最坏高频衰减'。"
        "1010 图案只有 5 GHz 一个频点, 测不出低频问题; 真随机序列又不可复现。")

    note("PRBS 阶数越高, 最长游程 2^n-1 越大, 低频覆盖越好 (PRBS31 用于 112G)。"
         "本模型里换 prbs_order=7 -> 15 就能看到这个变化。")


def exp_1_4_rise_time():
    h2("1.4 上升时间吃掉的是什么: 眼宽, 而不是眼高")
    rows = []
    store = {}
    for rise in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
        p = params(nrz_rise_ui=rise, channel_type="ideal",
                   channel_noise_sigma=0.0, channel_xtalk_amp=0.0)
        link = Link(p)
        h, s = eye_stats(link.wave_ch, p.samples_per_ui, link.bits, phase=0.0)
        w = eye_width(link.wave_ch, p.samples_per_ui, link.bits)
        store[rise] = (h, w)
        rows.append([f"{rise:.2f}", f"{rise * 2:.2f}", f"{h:.3f}", f"{s:.3f}", f"{w:.3f}"])
    tbl(["nrz_rise_ui", "过渡宽度 [UI]", "垂直眼高 [V]", "垂直厚度 [V]",
         "水平眼宽 @50% [UI]"], rows)

    note("眼高不是 2.0 而是 1.0: TX FFE 抽头 (-0.15, 0.85, -0.2) 之和 = 0.5, "
         "长连 1 的稳态只有 0.5, 所以理想眼高 = 2 x 0.5 = 1.0。"
         "眼高由**直流增益**决定, 眼宽由**边沿**决定 —— 两个余量来自两个不同的机制。")

    ask("过渡变慢, 为什么眼高几乎不变, 眼宽却明显缩小?")
    why("本模型里过渡宽度 = 2 x rise_ui, 且过渡以 UI 边界为中心。所以半个过渡"
        "落在相邻 UI 的中心: 中心点'刚好'来得及到终值 -> 眼高保住; 但整个"
        "UI 里能采样的安全窗口被过渡占掉了 -> 眼宽缩小。"
        "也就是说, 上升时间首先吃掉的是**时间余量**, 而不是电压余量。")
    warn("所以'边沿慢 -> 眼高小'是个常见误解。只看边沿, 眼高几乎不塌 "
         "(rise_ui 到 0.5 仍是 1.000); 让眼图真正闭合的是信道的**拖尾**, "
         "下面 1.4b 用一阶 RC 做对照。")

    # ---- 1.4b: 真正让眼闭合的是"带宽"(拖尾), 不是"边沿陡度" ----
    h2("1.4b 对照组: 换成有拖尾的一阶 RC 低通, 眼高才真的塌")
    rows = []
    for tau in (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
        p = params(channel_type="ideal", nrz_rise_ui=0.05,
                   channel_noise_sigma=0.0, channel_xtalk_amp=0.0)
        link = Link(p)
        y = rc_lowpass(link.wave_ch, tau, p.samples_per_ui)
        h, s = eye_stats(y, p.samples_per_ui, link.bits, phase=0.0)
        w = eye_width(y, p.samples_per_ui, link.bits)
        f3db = p.data_rate_gbps / (2 * np.pi * tau)      # 一阶 RC 的 -3dB 频率 [GHz]
        rows.append([f"{tau:.2f}", f"{f3db:.2f}", f"{h:.3f}", f"{s:.3f}", f"{w:.3f}"])
    tbl(["tau [UI]", "等效 -3dB 带宽 [GHz]", "垂直眼高 [V]", "垂直厚度 [V]",
         "水平眼宽 [UI]"], rows)

    why("一阶 RC 的冲激响应是指数拖尾, 一直拖到后面好几个 UI: 前一个 bit 的尾部"
        "叠在当前 bit 上, 码型不同则落点不同 -> 眼'沿'被抹厚, 内沿被吃掉。"
        "注意它是**渐进**的: tau 到 0.5 UI (带宽 3.2 GHz, 只有 Nyquist 的 0.64 倍)"
        "眼高才掉一半; tau=1.2 UI 时眼高变负, 说明眼已经完全倒过来了。"
        "结论: 真正决定眼高的是**带宽/记忆(拖尾)**, 不是**坡度**。")
    note("真实信道的趋肤(∝√f)与介质(∝f)损耗等效于一个多极点、带长尾的低通, "
         "20 dB @ 5 GHz 的信道已经比 tau=1 UI 的 RC 狠得多 —— Stage 2 会看到眼完全闭上。")

    plt, save_fig, eye_density = get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, rise in zip(axes, (0.05, 0.2, 0.5)):
        q = params(nrz_rise_ui=rise, channel_type="ideal",
                   channel_noise_sigma=0.0, channel_xtalk_amp=0.0)
        lk = Link(q)
        eye_density(lk.wave_ch, q.samples_per_ui, ax=ax,
                    title=f"rise_ui={rise} (理想信道)")
    png = OUT / "lab01_rise_time_eye.png"
    save_fig(fig, png)
    obs(f"眼图已保存: {png}")

    fig2, axes2 = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, tau in zip(axes2, (0.02, 0.1, 0.35)):
        q = params(channel_type="ideal", nrz_rise_ui=0.05,
                   channel_noise_sigma=0.0, channel_xtalk_amp=0.0)
        lk = Link(q)
        eye_density(rc_lowpass(lk.wave_ch, tau, q.samples_per_ui),
                    q.samples_per_ui, ax=ax, title=f"RC tau={tau} UI")
    png2 = OUT / "lab01_rc_lowpass_eye.png"
    save_fig(fig2, png2)
    obs(f"RC 低通眼图已保存: {png2}")


def exp_1_5_eye_folding():
    h2("1.5 亲手把波形折成眼图 (眼图的数学定义)")
    p = params()
    link = Link(p)
    sp = p.samples_per_ui
    y = link.wave_ch[: 512 * sp]
    bits = link.bits[:512]
    seg = y.reshape(512, sp)

    print("  取 512 个 UI, 每 UI 32 点, 折成 512x32 矩阵。看第 0 列 (UI 边界):")
    col0 = seg[:, 0]
    obs(f"min={col0.min():+.3f} max={col0.max():+.3f}, "
        f"落在 |v|<0.2 的样本 {int(np.sum(np.abs(col0) < 0.2))}/512 -> 正在跳变, 弧线满屏")

    print("\n  看第 16 列 (UI 中心):")
    col16 = seg[:, 16]
    hi, lo = col16[bits > 0], col16[bits <= 0]
    obs(f"高电平组: 中位数 {np.median(hi):+.3f}, 2%~98% 分位 "
        f"[{np.quantile(hi, 0.02):+.3f}, {np.quantile(hi, 0.98):+.3f}]")
    obs(f"低电平组: 中位数 {np.median(lo):+.3f}, 2%~98% 分位 "
        f"[{np.quantile(lo, 0.02):+.3f}, {np.quantile(lo, 0.98):+.3f}]")
    obs(f"两簇之间的空隙 (垂直眼高) = "
        f"{np.quantile(hi, 0.02) - np.quantile(lo, 0.98):+.3f} V")

    ph, op = eye_profile(y, sp, bits)
    print("\n  逐相位算垂直眼高 (左 = 前一个 UI 边界, 中 = UI 中心, 右 = 后一个边界):")
    for i in range(0, sp, 2):
        bar = "#" * max(0, int(round(op[i] * 40)))
        print(f"    phase {ph[i]:+.3f} UI  {op[i]:+.3f} V  {bar}")

    why("眼图 = 每一列 (每个相位) 上样本的直方图。中心列两簇分得最开 -> "
        "眼高最大; 边界列样本散布在全范围 -> 眼高为 0。"
        "'最优采样相位'就是让采样点落在眼高最大的那一列。")

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ph, op, marker="o", ms=3, color="steelblue")
    ax.axvline(0.0, color="gray", ls="--", lw=1, label="UI 中心")
    ax.set_xlabel("采样相位 [UI]")
    ax.set_ylabel("垂直眼高 [V]")
    ax.set_title("眼图的数学定义: 眼高随采样相位变化")
    ax.grid(alpha=0.3)
    ax.legend()
    png = OUT / "lab01_eye_profile.png"
    save_fig(fig, png)
    obs(f"眼高-相位曲线已保存: {png}")

    note("这条曲线的'高度'就是垂直眼高, '宽度'就是水平眼宽。"
         "后面所有均衡手段 (FFE/CTLE/DFE) 都是在把这条曲线抬高、加宽。")


def main():
    banner("1", "时间基 / PRBS / NRZ / 眼图",
           "把'比特 -> 波形 -> 眼图'这条链路走通, 建立 UI 与眼图的直觉")
    exp_1_1_time_base()
    exp_1_2_one_ui()
    exp_1_3_prbs()
    exp_1_4_rise_time()
    exp_1_5_eye_folding()
    h1("Stage 1 小结")
    print("""  1. UI = 1/数据率; 每 UI 32 点是给"相位"留的连续调节空间。
  2. NRZ: 1 bit = 1 symbol, 判决只需要一个 0 门限; Nyquist = 数据率/2。
  3. 眼图 = 按 UI 折叠后的样本二维分布。垂直眼高 = 电压余量, 水平眼宽 = 时间余量。
  4. 两个余量来自两个机制: 眼高 <- 直流增益; 眼宽 <- 边沿/过渡。
  5. 仅靠"边沿慢"不会让眼高塌; 真正的杀手是信道的**拖尾**(带宽/记忆)。
  6. PRBS7 的 7 连 0 保证激励里有低频, 1010 图案做不到。

  下一步: Stage 2 把波形送进有损信道, 用脉冲响应把"拖尾"量化成 ISI 系数。""")


if __name__ == "__main__":
    main()
