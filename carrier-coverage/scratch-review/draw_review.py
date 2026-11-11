"""Render the training review and replayed validation examples."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np


STYLES = {
    "control": ("50:50 对照", "#235898", "o", "--"),
    "ofea": ("50:50 + OFEA", "#b84f28", "D", "-"),
}
INK = "#182738"
MUTED = "#506276"
GRID = "#dbe2e9"


def plot(axes, x, y, arm, markers=True):
    label, color, marker, line = STYLES[arm]
    return axes.plot(x, y, label=label, color=color, ls=line, marker=marker if markers else None,
                     lw=2.1, ms=5.2, markeredgecolor="white", markeredgewidth=.65, clip_on=False)[0]


def draw_curves(report, output):
    fig, axes = plt.subplots(2, 3, figsize=(13.1, 7.7))
    fig.subplots_adjust(left=.084, right=.965, top=.775, bottom=.138, wspace=.355, hspace=.54)
    fig.text(.084, .945, "N416 从零训练｜收敛检查", fontsize=22, weight="bold")
    step = report["comparison_step"]
    through = report["train_through"]
    fig.text(.084, .904, f"M 双头编码器  /  Oldpan 64  /  单种子  /  共同验证节点 {step:,} 步",
             fontsize=11, color=MUTED)
    titles = ("新图 · 直接解码", "新图 · 换脸后整串恢复", "新图 · 换脸后误码",
              "训练 · 换脸 BCE", "训练 · Oldpan 特征差", "验证 · 身份阻断与联合成功")
    labels = ("整串恢复率 (%)", "整串恢复率 (%)", "BER (%)",
              "每千步均值", "原始 OFEA（未乘权重）", "符合条件的配对数")
    handles = []
    for arm in STYLES:
        ev = report["evaluations"][arm]
        x = [r["step"] for r in ev]
        new = [r["panels"]["new"] for r in ev]
        handles.append(plot(axes[0, 0], x, [100 * r["direct"]["all"]["exact"] / 556 for r in new], arm))
        plot(axes[0, 1], x, [100 * r["swap"]["all"]["exact"] / 556 for r in new], arm)
        plot(axes[0, 2], x, [100 * r["swap"]["all"]["ber"] for r in new], arm)
        train = report["train"][arm]
        t = [r["step"] for r in train]
        plot(axes[1, 0], t, [r["swap_bce"] for r in train], arm, markers=False)
        plot(axes[1, 1], t, [r["ofea"] for r in train], arm, markers=False)
        identity = [r for r in ev if "identity" in r]
        ix = [r["step"] for r in identity]
        block = [sum(r["identity"]["all"]["cells"][k] for k in ("10", "11")) for r in identity]
        plot(axes[1, 2], ix, block, arm)
    joint = [r["identity"]["all"]["cells"]["11"] for r in identity]
    assert all(r["identity"]["all"]["cells"]["11"] == 0
               for records in report["evaluations"].values() for r in records if "identity" in r)
    joint_handle = axes[1, 2].scatter(ix, joint, s=27, marker="x", c=INK,
                                      linewidths=1.25, zorder=4, label="两组联合成功")
    for k, ax in enumerate(axes.flat):
        ax.set_title(titles[k], loc="left", pad=10, weight="bold", fontsize=12)
        ax.set_ylabel(labels[k])
        end = through if k in (3, 4) else step
        ax.set_xlim(-.025 * end, 1.035 * end)
        ax.set_xticks(np.arange(0, end + 1, 20000))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0"))
        ax.set_xlabel("优化器更新步")
        ax.grid(axis="y", color=GRID, lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#a5b3c1")
        ax.set_axisbelow(True)
    axes[0, 0].set_ylim(-4, 103)
    axes[0, 1].set_ylim(-2, 59)
    axes[0, 2].set_ylim(-1, 53)
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_ylim(bottom=0)
    ymax = max(sum(r["identity"]["all"]["cells"][k] for k in ("10", "11"))
               for records in report["evaluations"].values() for r in records if "identity" in r)
    axes[1, 2].set_ylim(-.25, max(2, ymax + .4))
    axes[1, 2].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1, 2].legend(handles=[joint_handle], frameon=False, loc="upper right", fontsize=9)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.078, .878), ncol=2, frameon=False)
    fig.text(.084, .059, f"消息验证为 139 张未见载体 × 4 条未见消息。身份分母为 480 对。训练曲线截至共同记录 {through:,} 步。",
             color=MUTED, fontsize=10)
    fig.savefig(output / "convergence.png", dpi=170, facecolor="white")
    plt.close(fig)


def draw_examples(report, output):
    with np.load(output / "examples.npz") as data:
        images = data["images"]
    examples = report["examples"]
    assert images.shape == (len(examples), 6, 160, 160, 3)
    names = {"median": "联合组误码中位", "ofea_block": "联合组身份阻断",
             "control_exact": "仅对照组全对", "ofea_exact": "仅联合组全对"}
    fig, axes = plt.subplots(len(examples), 6, figsize=(12.4, 2.0 * len(examples) + 1.34))
    fig.subplots_adjust(left=.175, right=.977, top=.86, bottom=.077, wspace=.1, hspace=.32)
    fig.text(.175, .957, f"第 {report['comparison_step']:,} 步｜同一验证配对的换脸结果",
             fontsize=19, weight="bold")
    fig.text(.175, .919, "每行两组共用一张原图、一条 15 位消息和同一 Oldpan 换脸方向。",
             fontsize=10.5, color=MUTED)
    titles = ("原始载体", "无水印换脸", "对照 · 水印图", "对照 · 换脸", "OFEA · 水印图", "OFEA · 换脸")
    for ax, title in zip(axes[0], titles, strict=True):
        ax.set_title(title, fontsize=10.5, weight="bold", pad=9)
    for row, (sample, frames) in enumerate(zip(examples, images, strict=True)):
        for ax, frame in zip(axes[row], frames, strict=True):
            ax.imshow(frame, interpolation="nearest")
            ax.set_axis_off()
        box = axes[row, 0].get_position()
        y = box.y0 + box.height / 2
        target = sample["target"].capitalize()
        source = sample["source"].capitalize()
        fig.text(.159, y + .017, f"{target} → {source}", ha="right", va="center", weight="bold")
        fig.text(.159, y - .012, names[sample["kind"]], ha="right", va="center", color=MUTED, fontsize=9.7)
        for col, arm in ((2, "control"), (4, "ofea")):
            r = sample[arm]
            axes[row, col].text(.5, -.085, f'{r["psnr"]:.2f} dB', transform=axes[row, col].transAxes,
                                ha="center", va="top", fontsize=9.2, color=MUTED)
            status = (" · 阻断" if r["blocked"] and r["eligible"] else
                      " · 检测失败" if r["detection"] == "failed" else "")
            errors = r["swap_errors"]
            color = "#087b61" if errors == 0 else STYLES[arm][1] if status else MUTED
            axes[row, col + 1].text(.5, -.085, f"错 {errors}/15 位{status}",
                                    transform=axes[row, col + 1].transAxes,
                                    ha="center", va="top", fontsize=9.2, color=color, weight="bold" if errors == 0 else "normal")
    fig.text(.175, .036, "图例按误码中位、两组恢复差异和联合组的阻断个案选取。总体指标见收敛曲线。",
             color=MUTED, fontsize=9.4)
    fig.savefig(output / "face-swap-examples.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/msyh.ttc"))
    args = parser.parse_args()
    font_manager.fontManager.addfont(args.font)
    family = font_manager.FontProperties(fname=args.font).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
                         "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    report = json.loads((args.output / "review.json").read_text(encoding="utf-8"))
    draw_curves(report, args.output)
    draw_examples(report, args.output)
