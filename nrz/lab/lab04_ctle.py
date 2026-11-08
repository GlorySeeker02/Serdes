"""实验 4 (Stage 4): CTLE 连续时间线性均衡。

要建立的三个直觉:
  1. CTLE = 一个零点 + 两个极点: 在 f_peak 附近抬起高频 ("peaking");
  2. peaking 不是越大越好: 它同时放大高频噪声和串扰;
  3. CTLE 是"因果"的模拟滤波器, 它能压后游标, 但压不掉信道造成的前游标。

运行:
    python nrz/lab/lab04_ctle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    Link, banner, h1, h2, ask, obs, why, note, warn, tbl, params, variant,
    eye_stats, pulse_cursors, ctle_xform, channel_db, ctle_db, get_plt, OUT, np,
)


def exp_4_1_zero_pole():
    h2("4.1 CTLE 的零极点位置")
    rows = []
    for pk_db in (0.0, 3.0, 6.0, 9.0, 12.0):
        p = params(ctle_peaking_db=pk_db)
        pk = 10 ** (pk_db / 20.0)
        fz = p.ctle_peak_freq_ghz / np.sqrt(pk)
        fp1 = p.ctle_peak_freq_ghz * np.sqrt(pk)
        fp2 = p.data_rate_gbps * 1.5
        rows.append([f"{pk_db:.0f}", f"{pk:.2f}", f"{fz:.2f}", f"{fp1:.2f}",
                     f"{fp2:.1f}", f"{fp1 - fz:.2f}"])
    tbl(["peaking [dB]", "线性倍数", "零点 fz [GHz]", "极点 fp1 [GHz]",
         "极点 fp2 [GHz]", "fz 与 fp1 间距"], rows)

    ask("为什么 peaking 是靠「把极点 1 推远离零点」实现的?")
    why("H(s) = (1+s/wz)/((1+s/wp1)(1+s/wp2)), 零点在前、极点在后的那一段频率上"
        "幅度是上升的。peaking 越大 -> fz 与 fp1 拉得越开 -> 上升段越宽、越高; "
        "而 fp2 固定在 15 GHz, 保证高端不再继续抬 (否则会放大到 GHz 以上的噪声)。")
    obs("peaking 大致落在 fz~fp1 之间 (≈5~8 GHz): 第三个极点 fp2 会把这个峰往右拉一点, "
        "所以 4.3 的频响图上峰顶在 7~8 GHz 而不是正好 5 GHz。")


def exp_4_2_peaking_sweep():
    h2("4.2 扫 peaking: 20 dB 信道, 只开 CTLE (无 DFE)")
    rows = []
    for pk_db in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0):
        p = params(ctle_peaking_db=pk_db, channel_loss_db_at_nyquist=20.0,
                   channel_noise_sigma=0.03)
        link = Link(p)
        q = variant(p, dfe_train=False, cdr_kp=0.0, cdr_ki=0.0,
                    cdr_initial_phase_ui=0.0)
        y = link.front_end(q)
        hgt, thick = eye_stats(y, q.samples_per_ui, link.bits, phase=0.0)
        ber, _ = link.run(q, front_end=y).ber(skip=300)
        gain5 = float(ctle_db(p, np.array([5e9]))[0])
        rows.append([f"{pk_db:.0f}", f"{gain5:+.2f}", f"{hgt:.3f}",
                     f"{thick:.3f}", f"{ber:.2e}"])
    tbl(["peaking [dB]", "CTLE @5GHz [dB]", "垂直眼高 [V]", "垂直厚度 [V]", "BER"], rows)

    ask("眼高随 peaking 一直变大吗? 什么时候开始变差?")
    why("先是单调变好 (补上高频), 到某个点后眼高不再涨、垂直厚度继续涨 —— "
        "因为高频增益把噪声也一起抬起来了。本表里眼高在 6~8 dB 附近最大, "
        "而垂直厚度在 6 dB 处最小: 两个指标的最优点并不完全重合, 工程上按 BER 选。")
    warn("本表没有加串扰。真实链路里 CTLE 会同时放大相邻通道的串扰, 所以"
         "最优 peaking 往往比'只看噪声'时更小。4.4 会验证这一点。")


def exp_4_3_response():
    h2("4.3 频响: 信道 x CTLE 级联")
    plt, save_fig, _ = get_plt()
    f = np.linspace(0.05, 12, 400) * 1e9
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(f / 1e9, channel_db(f, 20.0, 5e9, 0.5), color="k", lw=2, label="信道 20 dB")
    for pk in (0.0, 6.0, 12.0):
        p = params(ctle_peaking_db=pk)
        g = ctle_db(p, f)
        ax.plot(f / 1e9, g, lw=1.2, label=f"CTLE peaking={pk:.0f} dB")
        ax.plot(f / 1e9, g + channel_db(f, 20.0, 5e9, 0.5), lw=1.0, ls="--",
                label=f"级联 peaking={pk:.0f} dB")
    ax.axvline(5.0, color="gray", ls=":", lw=1)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("频率 [GHz]")
    ax.set_ylabel("增益 [dB]")
    ax.set_ylim(-45, 20)
    ax.set_title("CTLE 用高频 peaking 把级联频响拉平")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    png = OUT / "lab04_ctle_response.png"
    save_fig(fig, png)
    obs(f"已保存: {png}")

    obs("读图要点: 级联曲线 (虚线) 越接近 0 dB 水平线, 输出眼图越「方」; "
        "低频段被 CTLE 的直流增益 (=1) 保持不动, 所以长连 1/0 的稳态电平不会被改变。")


def exp_4_4_crosstalk():
    h2("4.4 CTLE 的另一面: 它同时放大串扰")
    rows = []
    for pk_db in (0.0, 6.0, 12.0):
        for xt in (0.0, 0.05, 0.10):
            p = params(ctle_peaking_db=pk_db, channel_loss_db_at_nyquist=20.0,
                       channel_noise_sigma=0.01, channel_xtalk_amp=xt)
            link = Link(p)
            q = variant(p, dfe_train=False, cdr_kp=0.0, cdr_ki=0.0,
                        cdr_initial_phase_ui=0.0)
            y = link.front_end(q)
            hgt, thick = eye_stats(y, q.samples_per_ui, link.bits, phase=0.0)
            ber, _ = link.run(q, front_end=y).ber(skip=300)
            rows.append([f"{pk_db:.0f}", f"{xt:.2f}", f"{hgt:.3f}", f"{thick:.3f}",
                         f"{ber:.2e}"])
    tbl(["peaking [dB]", "串扰幅度 [V]", "垂直眼高 [V]", "垂直厚度 [V]", "BER"], rows)

    why("串扰是加在信道输出上的宽带噪声, 它的高频成分最多。CTLE 抬高频 -> 串扰被"
        "同比例放大 -> 眼高收益被吃掉, 厚度却继续涨。这就是'均衡越强、串扰越毒'。")
    note("所以真实链路必须先评估噪声/串扰环境, 再定 peaking; 并且 CTLE 通常放在"
         "VGA 之后、ADC 之前, 增益分配要保证 ADC 不饱和 (本模型没有建模饱和)。")


def exp_4_5_cursor_view():
    h2("4.5 游标视角: CTLE 到底改了哪几个 h[k]")
    p = params(channel_noise_sigma=0.0)
    rows = []
    for tag, xf in (("无 CTLE", None), ("CTLE 6dB", ctle_xform(p, agc=False)),
                    ("CTLE 12dB", ctle_xform(params(ctle_peaking_db=12.0), agc=False))):
        offs, cur, _ = pulse_cursors(p, loss_db=20.0, tx_taps=(0.0, 1.0, 0.0), xform=xf)
        pick = [list(offs).index(k) for k in (-2, -1, 1, 2, 3)]
        rows.append([tag] + [f"{cur[i]:+.3f}" for i in pick])
    tbl(["配置", "h(-2)", "h(-1)", "h(+1)", "h(+2)", "h(+3)"], rows)

    ask("CTLE 能压掉前游标 h(-1) 吗?")
    why("压不掉, 甚至可能更大。CTLE 是因果的最小相位滤波器, 它只能'提前'响应, "
        "而前游标是信道非因果部分 (零相位模型) 造成的。要动前游标, 只能用"
        "在发送端就知道未来比特的 TX FFE。这就是 3.2 与 4.5 的分工。")
    obs("它确实把 h(+1) 从 0.404 压到 0.291 (6 dB) / 0.119 (12 dB) —— 后游标被显著改善, "
        "剩下的残量交给 DFE (Stage 7) 收尾。")


def main():
    banner("4", "CTLE 连续时间线性均衡",
           "用一个零两点的模拟滤波器把高频抬起来, 看清它的收益和代价")
    exp_4_1_zero_pole()
    exp_4_2_peaking_sweep()
    exp_4_3_response()
    exp_4_4_crosstalk()
    exp_4_5_cursor_view()
    h1("Stage 4 小结")
    print("""  1. CTLE: H(s) = A_dc(1+s/wz)/((1+s/wp1)(1+s/wp2)), wp1 < wz < wp2。
  2. peaking 大小由 (fz, fp1) 的间距决定; 峰值落在 fz~fp1 的几何中点。
  3. 收益: 抬高高频 -> 眼高变大; 代价: 噪声/串扰被同比例放大。
  4. CTLE 直流增益 = 1, 不改变长连 1/0 的稳态电平 (所以 AGC 逻辑不受影响)。
  5. 它压得住后游标, 压不住前游标 —— 前游标只能靠 TX FFE, 后游标最终靠 DFE。

  下一步: Stage 5 先解决一个"看不见但致命"的问题 —— 幅度归一化 (AGC)。""")


if __name__ == "__main__":
    main()


