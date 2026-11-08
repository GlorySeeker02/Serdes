"""一键跑完 8 个分阶段实验。

用法:
    python nrz/lab/run_all.py          # 全部跑一遍
    python nrz/lab/run_all.py 3 5      # 只跑 lab03 和 lab05

输出会同时打印到终端并写入 nrz/output/lab/labNN.log, 方便贴进笔记。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "output" / "lab"
OUT.mkdir(parents=True, exist_ok=True)

LABS = sorted(p for p in HERE.glob("lab[0-9][0-9]_*.py"))


def main(argv):
    want = {int(a) for a in argv if a.isdigit()}
    for script in LABS:
        num = int(script.name[3:5])
        if want and num not in want:
            continue
        print(f"\n########## {script.name} ##########", flush=True)
        proc = subprocess.run([sys.executable, str(script)],
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        text = (proc.stdout or "") + (proc.stderr or "")
        print(text)
        (OUT / f"lab{num:02d}.log").write_text(text, encoding="utf-8")
        if proc.returncode != 0:
            print(f"!!! {script.name} 退出码 {proc.returncode}")
    print(f"\n日志已写入: {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
