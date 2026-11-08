"""实验 7 (Stage 7): DFE 判决反馈均衡。

要建立的三个直觉:
  1. DFE 用"已经判决过的比特"去抵消后游标, 是无噪声放大的均衡 (不放大噪声);
  2. 但它依赖判决正确 —— 判决错了会传播 (错误传播 / burst);
  3. 抽头个数由 h[k] 的拖尾长度决定, 由 sign-sign LMS 自适应得到。

运行:
    python nrz/lab/lab07_dfe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_stats, pulse_cursors, ctle_xform, get_plt, OUT, np,
)


def exp_7_1_taps_converge():
    h2("7.1 DFE 抽头收敛: 收敛到哪里?")
    # 用更小的步长 + 更长的序列, 让抽头真正收敛, 才好跟理论值对齐
    p = params(nbits=30000, dfe_mu=0.0005)
    link = Link(p)
    r = link.run(p)
    ber, _ = r.ber(skip=2000)
    taps = r.dfe[-4000:].mean(axis=0)

    # 独立算一遍"应该收敛到哪里": CTLE 输出信号的 ISI 系数
    offs, cur, main = pulse_cursors(p, loss_db=p.channel_loss_db_at_nyquist,
                                    xform=ctle_xform(p, agc=False),
                                    phase=r.phase_final)

    rows = []
    for i in range(p.dfe_ntaps):
        k = i + 1
        rows.append([f"h{k}", f"{cur[list(offs).index(k)]:+.3f}", f"{taps[i]:+.3f}"])
    tbl(["抽头", "理论值 (CTLE 后 h[k])", "DFE 收敛值 (末段均值)"], rows)
    obs(f"BER = {ber:.2e}")
    traj = [f"{r.dfe[i, 0]:+.3f}" for i in (50, 200, 1000, 10000, 29999)]
    obs("h1 在 bit 50/200/1000/10000/29999 时分别为: " + ", ".join(traj))

    ask("DFE 的抽头值必须等于 ISI 系数吗?")
    why("在理想情况下是的: 抽头 h_i 乘的是过去判决 a_{k-i}, 所以要抵消的就是"
        "同一位置的 ISI 系数。但 sign-sign LMS 只有误差的**符号**, 加上量化噪声, "
        "收敛值会在真值附近抖动; 而且它收敛到的是「让判决错误最少」的点, 不是"
        "严格的零 ISI 点 (MMSE 而非 ZF)。所以表里两列能对上, 但不会完全相等。")
    obs("特别注意 h4/h5: 理论值 0.02~0.03, 收敛值却几乎为 0。不是公式错了 —— "
        "sign-sign LMS 只有 1 bit 的误差信息, 比『判决抖动』还小的抽头它根本看不见, "
        "自然学不到。所以抽头超过 ISI 拖尾长度之后, 多出来的只会随机游走。")
    note("另一个容易踩的坑: ISI 系数是**随采样相位变化**的。本表用的是 CDR 锁定的"
         "相位 (-0.045 UI), 如果你按 UI 中心去算理论值, h1 会差一倍 —— 比较之前"
         "先确认相位一致 (pulse_cursors(..., phase=...))。")
    warn("如果你看到两列差很多, 先别怀疑公式: 多半是**没收敛**(比特数不够或 mu 太小) "
         "或者 mu 太大导致抽头一直在抖。7.3 会把这两个失效模式量化出来。")

    plt, save_fig, _ = get_plt()
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(p.dfe_ntaps):
        ax.plot(np.arange(p.nbits), r.dfe[:, i], lw=0.8, label=f"h{i + 1}")
    ax.set_xlabel("bit 序号")
    ax.set_ylabel("抽头值")
    ax.set_title("DFE 抽头收敛轨迹 (sign-sign LMS)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    png = OUT / "lab07_dfe_taps.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")


def exp_7_2_ntaps():
    h2("7.2 抽头个数: 少了收不干净, 多了反而更差")
    rows = []
    for ntaps in (0, 1, 2, 3, 5, 8):
        p = params(dfe_ntaps=ntaps, channel_loss_db_at_nyquist=26.0,
                   channel_noise_sigma=0.03)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        hz, sz = r.eye_z()
        rows.append([ntaps, f"{hz:+.3f}", f"{sz:.3f}",
                     f"{r.taps_final[0] if ntaps else 0.0:+.3f}",
                     f"{r.phase_final:+.3f}", f"{ber:.2e}"])
    tbl(["dfe_ntaps", "判决器 z 眼高 [V]", "z 垂直厚度 [V]", "h1 终值",
         "CDR 相位", "BER"], rows)

    ask("抽头越来越多, 眼高会一直变大吗?")
    why("不会。前几个抽头把主要后游标收掉, 眼高快速上升 (0 -> 3 个抽头); "
        "再往后收益变小, 甚至变差 —— 因为 sign-sign LMS 的每个抽头都在抖, "
        "抽头越多, 反馈量的抖动越大; 而且判决一旦出错, 更多抽头会把错误放大得更远。")
    obs("所以抽头个数要「够用就好」: 判断依据是 2.3 的游标表 —— h[k] 衰减到 0.03 "
        "以下时, 再加抽头就没意义了。")

    note("工程上这对应「DFE 长度 vs 功耗/噪声」的折中: 112G 链路用到 10+ 个抽头, "
         "是因为它的 h[k] 拖尾本来就长, 不是为了「多加几个更保险」。")


def exp_7_3_step_size():
    h2("7.3 步长 mu: 收敛速度与稳定性")
    rows = []
    for mu in (0.001, 0.005, 0.01, 0.05, 0.1, 0.3):
        p = params(dfe_mu=mu)
        link = Link(p)
        r = link.run(p)
        ber, _ = r.ber(skip=300)
        hz, _ = r.eye_z()
        rows.append([f"{mu:.3f}", f"{r.taps_final[0]:+.3f}", f"{hz:+.3f}",
                     f"{r.phase_final:+.3f}", f"{ber:.2e}"])
    tbl(["dfe_mu", "h1 终值", "z 眼高 [V]", "CDR 相位", "BER"], rows)

    ask("mu 太大会怎样? 太小又会怎样?")
    why("mu 太大: 每个 bit 的抽头修正量过大, 抽头在最优点附近来回跳, 眼高和 BER "
        "都变差 (极端时发散)。mu 太小: 4096 个 bit 内还没收敛完, 眼高偏小。"
        "sign-sign LMS 的稳态抖动幅度与 mu 成正比, 而收敛时间与 1/mu 成正比。")
    warn("本模型里 h1 的真值约 0.04。mu 从 0.001 到 0.1 都能跑, 是因为信号已归一化; "
         "如果关掉 AGC (见 Stage 5), mu=0.01 就会发散。")


def exp_7_4_error_propagation():
    h2("7.4 DFE 的软肋: 错误传播 (以及它什么时候会失控)")
    rows = []
    for loss, noise in ((26.0, 0.03), (26.0, 0.05), (28.0, 0.05)):
        for train in (False, True):
            p = params(channel_loss_db_at_nyquist=loss, channel_noise_sigma=noise,
                       dfe_train=train)
            link = Link(p)
            r = link.run(p)
            err = (r.dec > 0).astype(int) != link.bits[: r.n]
            idx = np.flatnonzero(err)
            # 相邻错误占比: 错误是"散落"还是"成串"
            adj = float(np.mean(np.diff(idx) <= 2)) if len(idx) > 1 else 0.0
            hz, _ = r.eye_z()
            rows.append([f"{loss:.0f}", f"{noise:.2f}", "开" if train else "关",
                         f"{hz:+.3f}", len(idx), f"{adj:.2f}"])
    tbl(["损耗 [dB]", "噪声 sigma", "DFE", "z 眼高 [V]", "错误个数", "相邻错误占比"], rows)

    ask("第一组 DFE 明显有用, 第三组却比不开还差几十倍, 为什么?")
    why("DFE 生效的前提是**判决基本正确**。第一组眼还开着 (z 眼高 0.2+), 抽头收敛到"
        "真值, 错误变少。第三组眼已经闭合, 一开始判决就是错的, 错误判决进入历史"
        "寄存器 -> 反馈量错 -> 后面判得更错, 同时 LMS 还在「学习」这些错误, 抽头被"
        "推离真值 (这里直接跑飞), 于是 BER 从 3e-2 恶化到 3.8e-1。")
    obs("看最后一列: 正常工作时相邻错误占比 ~0, 失控时高达 0.74 —— 错误成串出现, "
        "这正是 error propagation 的指纹, 也是现场调试时判断「DFE 已经崩了」的依据。")

    note("缓解办法: (1) 保证 AGC 与 CDR 先在 DFE 之前锁好 (次序!); (2) 用更保守的 mu, "
         "或 DFE 只在锁定后开启; (3) 预编码 (precoding) 从原理上消除错误传播, "
         "PAM4 链路常用; (4) 加 FEC 把 burst 打散。")


def main():
    banner("7", "DFE 判决反馈均衡",
           "用已判决的比特抵消后游标, 并看清它的代价 (错误传播)")
    exp_7_1_taps_converge()
    exp_7_2_ntaps()
    exp_7_3_step_size()
    exp_7_4_error_propagation()
    h1("Stage 7 小结")
    print("""  1. DFE: z = d - Σ h_i·a_{k-i}, 用过去判决抵消后游标; 判决 s = sign(z)。
  2. 因为反馈的是「判决」而不是「模拟量」, DFE 不放大噪声 (相对 FFE/CTLE 的优势)。
  3. sign-sign LMS: h_i += mu·sign(z-s)·a_{k-i}, 只需符号, 硬件极省。
  4. 抽头个数由 ISI 拖尾长度决定 (看 2.3 的游标表), 本链路 5 个够。
  5. 软肋是错误传播: 错一个会传染一串, 表现为错误聚簇。
  6. 它只能处理后游标 (因果); 前游标仍归 TX FFE / CTLE 管。

  下一步: Stage 8 把整条链路拉到系统级, 看抖动/噪声/余量预算。""")


if __name__ == "__main__":
    main()




