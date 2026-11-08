"""实验 2 (Stage 2): 有损信道与 ISI。

要建立的三个直觉:
  1. 信道损耗随频率上升, 在 Nyquist 处标定 (本模型 20 dB);
  2. 频域损耗 = 时域拖尾 = ISI, 可以用"脉冲响应游标"精确量化;
  3. 眼高随损耗的下降不是线性的, 到某个点会突然全闭 (BER 掉到 0.5)。

运行:
    python nrz/lab/lab02_channel_isi.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    NOFFE,
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_height, eye_stats, eye_width, pulse_cursors, channel_db, sample_ui,
    get_plt, OUT, np,
)


def exp_2_1_freq_response():
    h2("2.1 信道频响: 损耗怎么随频率涨")
    p = params()
    freqs = np.array([0.5, 1.0, 2.5, 5.0, 7.5, 10.0]) * 1e9
    L = -channel_db(freqs, p.channel_loss_db_at_nyquist,
                    p.nyquist_ghz * 1e9, p.channel_skin_db_ratio)
    rows = []
    for f, l in zip(freqs, L):
        rows.append([f"{f / 1e9:.1f}", f"{l:.2f}", f"{10 ** (-l / 20):.4f}",
                     "Nyquist" if f == 5e9 else ""])
    tbl(["频率 [GHz]", "插损 [dB]", "幅度 |H|", "备注"], rows)

    ask("为什么指标总是标在 Nyquist (5 GHz) 而不是 10 GHz?")
    why("1010 图案 = 5 GHz 正弦, 是 NRZ 判决必须能分辨的最快变化; 10 GHz 以上"
        "只影响边沿形状, 对眼高贡献很小 (sinc 谱的能量也已经掉下去了)。")

    ask("模型里为什么把损耗拆成 sqrt(f) 和 f 两部分?")
    why("趋肤效应: 电流挤向导体表面, 有效截面 ∝ 1/sqrt(f) -> 损耗 ∝ sqrt(f); "
        "介质损耗: 极化弛豫 -> 损耗 ∝ f。低频段趋肤占主导, 高频段介质占主导。"
        "skin_ratio 就是这两者的配比, 决定损耗曲线的形状 (2.4 会扫它)。")
    warn("本模型是**零相位**信道 (只有幅度衰减, 没有群时延), 真实信道还会引入"
         "与频率相关的延迟, 使边沿进一步散开。所以这里的眼图会比真实链路乐观一点。")


def exp_2_2_loss_sweep():
    h2("2.2 损耗扫描: 裸判决 (无 VGA/CTLE/AGC/DFE, 固定相位 0)")
    rows = []
    for loss in (0.0, 6.0, 12.0, 16.0, 20.0, 26.0, 32.0):
        p = params(tx_taps=NOFFE, channel_loss_db_at_nyquist=loss,
                   channel_noise_sigma=0.01)
        link = Link(p)
        h, _ = eye_stats(link.wave_ch, p.samples_per_ui, link.bits, phase=0.0)
        w = eye_width(link.wave_ch, p.samples_per_ui, link.bits)
        q = variant(p, cdr_kp=0.0, cdr_ki=0.0,
                    cdr_initial_phase_ui=0.0, dfe_train=False)
        ber, _ = link.run(q, front_end=link.wave_ch).ber(skip=300)
        rows.append([f"{loss:.0f}", f"{h:.3f}", f"{w:.3f}",
                     f"{h / 1.958 * 100:.0f}%", f"{ber:.2e}"])
    tbl(["损耗@Nyq [dB]", "垂直眼高 [V]", "水平眼宽 [UI]", "相对 0dB 眼高", "BER"], rows)

    ask("损耗从 0 dB 加到 20 dB, 眼高是线性下降吗? BER 呢?")
    why("眼高大致按损耗的幅度 (10^(-L/20)) 缩, 不是按 dB 线性。真正危险的是"
        "'最坏码型': 主游标被衰减, 前后游标却几乎不变, 一旦最坏码型的落点越过 0, "
        "判决就全错。所以曲线是'先缓后陡'的悬崖形。")


def exp_2_2b_error_patterns():
    h2("2.2b 误码不是随机的: 错在哪类码型上?")
    p = params(tx_taps=NOFFE, channel_loss_db_at_nyquist=32.0,
               channel_noise_sigma=0.01)
    link = Link(p)
    q = variant(p, cdr_kp=0.0, cdr_ki=0.0,
                cdr_initial_phase_ui=0.0, dfe_train=False)
    r = link.run(q, front_end=link.wave_ch)
    dec = (r.dec > 0).astype(int)
    bits = link.bits
    err = dec != bits

    core = np.arange(1, len(bits) - 1)
    pat = bits[core - 1] * 4 + bits[core] * 2 + bits[core + 1]
    groups = [
        ("111 / 000  长游程内部", (pat == 7) | (pat == 0)),
        ("010 / 101  孤立比特", (pat == 2) | (pat == 5)),
        ("100 / 011 / 110 / 001", ~np.isin(pat, [0, 2, 5, 7])),
    ]
    rows = []
    for name, m in groups:
        rows.append([name, f"{m.mean() * 100:.1f}%", f"{err[core][m].mean():.3f}"])
    tbl(["3 比特邻域模式", "占比", "该类 BER"], rows)
    obs(f"整体 BER = {err[core].mean():.3f}")

    ask("为什么整体 BER 只有百分之十几, 而不是 0.5?")
    why("因为信道直流增益 = 1: 长连 1/0 的稳态电平完全正确, 这类比特怎么判都对。"
        "错的全是'孤立比特'——它们必须靠高频分量才能分辨, 而高频正好被损耗吃掉了。"
        "这也解释了为什么 PRBS7 (有一堆孤立比特) 比 1010 图案更严苛: "
        "1010 虽然频率最高, 但它是纯周期信号, 判决器可以'记住'它; PRBS 的"
        "随机性让任何码型都无法被预测。")

    note("工程含义: 同样的 BER 指标, 不同码型的分布完全不同。测链路必须用 PRBS, "
         "而且要看清'是哪种码型错了'——这直接指向是低频 droop 还是高频衰减的问题。")


def exp_2_3_pulse_response():
    h2("2.3 脉冲响应: 把 ISI 变成一串可读的数字")
    p = params(tx_taps=NOFFE, channel_noise_sigma=0.0)
    offs, c, main = pulse_cursors(p, loss_db=20.0)

    rows = []
    for o, v in zip(offs, c):
        if o == 0:
            tag = "主游标 (main cursor)"
        elif o < 0:
            tag = "前游标 / pre-cursor"
        else:
            tag = "后游标 / post-cursor"
        rows.append([f"{o:+d}", f"{v:+.4f}", tag])
    tbl(["偏移 [UI]", "归一化 h[k]", "含义"], rows)

    isi = float(np.sum(np.abs(c[offs != 0])))
    obs(f"主游标原始值 = {main:.3f} (即: 单个比特只能爬到阶跃的 {main * 100:.0f}%)")
    obs(f"其余游标绝对值之和 = {isi:.3f} (相对主游标) -> 最坏码型可偏离 "
        f"±{isi:.2f} 个主游标")
    why("ISI 总和 > 1 就意味着: 某个码型组合下, 采样点会被推到判决门限的另一侧 "
        "——眼图必然闭合, 判决必然出错。所以'能不能均衡'只看这一串数就够了。")

    ask("h(+1) 为正, 意味着什么? 为什么 h(-1) 和 h(+1) 差不多大?")
    why("正的 h1 表示前一个 +1 比特在当前 UI 中心还残留正电压 (低通拖尾), "
        "DFE 用 -h1·a_{k-1} 把它减掉。h(-1) ≈ h(+1) 是本模型特有的: 信道用的"
        "是**零相位**传递函数 (只有幅度衰减, 没有群时延), 等价于一个非因果的"
        "对称滤波器。真实信道有传播延迟, 前游标会小得多。")

    note("这串数字就是教科书的 ISI 系数 h[k]。均衡的全部目标: 把这串数逼近 "
         "[..., 0, 0, 1, 0, 0, ...]。DFE 只能动 k>0 的部分 (因果), "
         "k<0 的部分必须靠 TX FFE / CTLE。")

    # 不同损耗下的游标形状对比
    h2("2.3b 损耗越大, 游标拖得越远 (需要更多抽头)")
    rows = []
    for loss in (6.0, 12.0, 20.0, 26.0):
        _, cc, mm = pulse_cursors(p, loss_db=loss)
        rows.append([f"{loss:.0f}", f"{mm:.3f}"] + [f"{v:+.3f}" for v in cc[offs >= 1][:5]])
    tbl(["损耗 [dB]", "主游标(原始)", "h1", "h2", "h3", "h4", "h5"], rows)
    why("损耗越大: 主游标越小, 后游标越大且衰减越慢 -> DFE 需要更多抽头才能收干净。"
        "这就是为什么 112G 链路要 10+ 个 DFE 抽头, 而短距 10G 只要 3~5 个。")


def exp_2_3c_cross_check():
    h2("2.3c 交叉验证: 从真实波形里用最小二乘反推 ISI 系数")
    p = params(tx_taps=NOFFE, channel_noise_sigma=0.01)
    link = Link(p)
    sp = p.samples_per_ui
    a = np.where(link.bits > 0, 1.0, -1.0)
    n = 3000
    d = sample_ui(link.wave_ch, sp, n)          # 每个 UI 中心采一个点
    lo, hi = -3, 6
    start, stop = hi, n + lo                    # 保证切片长度一致
    A = np.stack([a[start - k:stop - k] for k in range(lo, hi + 1)], axis=1)
    coef, *_ = np.linalg.lstsq(A, d[start:stop], rcond=None)
    k0 = list(range(lo, hi + 1)).index(0)
    emp = coef / coef[k0]

    _, cur, _ = pulse_cursors(p, loss_db=p.channel_loss_db_at_nyquist)
    ref = np.arange(-8, 9)
    rows = []
    for k in range(lo, hi + 1):
        rows.append([f"{k:+d}", f"{emp[k - lo]:+.4f}",
                     f"{cur[list(ref).index(k)]:+.4f}"])
    tbl(["偏移 [UI]", "回归法 (从数据估)", "脉冲法 (从阶跃估)"], rows)

    note("两种方法完全独立: 一个用最小二乘拟合真实码流的采样点, 一个送单比特/阶跃"
         "量冲激响应。它们给出同一串数, 说明'ISI 系数'这个概念是自洽的。")
    why("这个回归技巧在调试时非常好用: 拿到一段 RX 前端波形 + 已知发送比特, "
        "就能直接量出 h[k], 不用去猜信道 S 参数。后面判断'还差几个 DFE 抽头'"
        "用的就是它。")


def exp_2_4_skin_ratio():
    h2("2.4 同样 20 dB, 损耗形状不同, 眼图也不同")
    rows = []
    p0 = params(tx_taps=NOFFE, channel_noise_sigma=0.01)
    for skin in (0.0, 0.3, 0.5, 0.7, 1.0):
        p = variant(p0, channel_skin_db_ratio=skin)
        link = Link(p)
        h, s = eye_stats(link.front_end(p), p.samples_per_ui, link.bits, phase=0.0)
        rows.append([f"{skin:.1f}", f"{(1 - skin) * 100:.0f}% / {skin * 100:.0f}%",
                     f"{h:.3f}", f"{s:.3f}"])
    tbl(["skin_ratio", "介质(f)/趋肤(√f) 占比", "垂直眼高 [V]", "垂直厚度 [V]"], rows)
    note("这一行量的是 CTLE+AGC 之后的眼图 (否则在 20 dB 下眼本来就闭着, 看不出差别)。")
    why("Nyquist 处的总损耗是固定的, 但曲线形状不同 -> 低频段的衰减量不同 -> "
        "长连 1/0 的稳态电平不同 -> 眼图上下不对称的方式也不同。"
        "本表里 skin 越大 (趋肤占比越高) 眼高反而**略好**: sqrt(f) 曲线在高频端更软"
        "(10 GHz 处 sqrt(f) 只多 8.3 dB, 纯 f 要多 20 dB)。"
        "工程上必须同时看低频损耗 (决定 droop) 和高频损耗 (决定边沿), "
        "只报一个'20 dB'是远远不够的。")


def exp_2_5_eye_pictures():
    h2("2.5 眼图: 0 dB 对比 20 dB")
    plt, save_fig, eye_density = get_plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, loss in zip(axes, (0.0, 12.0, 20.0)):
        p = params(tx_taps=NOFFE, channel_loss_db_at_nyquist=loss,
                   channel_noise_sigma=0.0)
        link = Link(p)
        eye_density(link.wave_ch, p.samples_per_ui, ax=ax,
                    title=f"信道输出 {loss:.0f} dB @Nyq")
    png = OUT / "lab02_loss_eye.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")

    PLT = plt
    fig2, ax2 = PLT.subplots(figsize=(7, 4))
    fs = np.fft.rfftfreq(4096, d=params().dt)
    for loss in (6.0, 20.0):
        for skin in (0.0, 0.5, 1.0):
            ax2.plot(fs / 1e9, channel_db(fs, loss, 5e9, skin), lw=1.2,
                     label=f"{loss:.0f} dB, skin={skin:.1f}")
    ax2.axvline(5.0, color="gray", ls="--", lw=1)
    ax2.set_xlim(0, 12)
    ax2.set_ylim(-40, 2)
    ax2.set_xlabel("频率 [GHz]")
    ax2.set_ylabel("插损 [dB]")
    ax2.set_title("信道频响: 总损耗 vs 损耗形状")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)
    png2 = OUT / "lab02_channel_response.png"
    save_fig(fig2, png2)
    obs(f"已保存: {png2}")


def main():
    banner("2", "有损信道与 ISI",
           "看懂频域损耗如何变成时域拖尾, 再用脉冲响应量化 ISI 系数")
    exp_2_1_freq_response()
    exp_2_2_loss_sweep()
    exp_2_2b_error_patterns()
    exp_2_3_pulse_response()
    exp_2_3c_cross_check()
    exp_2_4_skin_ratio()
    exp_2_5_eye_pictures()
    h1("Stage 2 小结")
    print("""  1. 信道损耗按 sqrt(f) (趋肤) + f (介质) 叠加, 指标标在 Nyquist = 数据率/2。
  2. 频域幅度衰减 <-> 时域脉冲拖尾, 两者是同一个物理事实的两面。
  3. 拖尾 = ISI 系数 h[k]: 当前 bit 的判决点被前后若干 bit 污染。
  4. h[k>0] 是后游标 (可用 DFE 抵消), h[k<0] 是前游标 (只能用 TX FFE / CTLE)。
  5. 20 dB 时主游标只剩 ~38%, ISI 总和 > 1 -> 眼完全闭上 (本模型实测 2.2)。
  6. 误码不是随机的: 长游程 0% 错, 孤立比特 ~43% 错。BER 停在 0.11 而不是 0.5。

  下一步: Stage 3 先在发送端动手 —— 用 TX FFE 打"预防针"。
  笔记要点: 把 2.2 的眼高/BER 表、2.2b 的"错在哪类码型"表、2.3 的游标表
  抄进笔记本, 这是后面所有均衡实验的基线。""")


if __name__ == "__main__":
    main()





