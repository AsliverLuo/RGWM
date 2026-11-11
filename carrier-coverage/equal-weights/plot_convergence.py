"""Plot the verified validation checkpoints and training BCE of both recipes."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_audit(folder, recipe, size):
    paths = list(folder.glob(f"audit-{recipe}-n{size}-*.json"))
    if recipe == 75 and size == 64:
        paths.append(folder.parent / "exposure-matched/export/n64-seed0/audit-24000.json")
    audits = [load(path) for path in paths]
    return max(audits, key=lambda audit: audit["through"])


def series(audit, paired_nodes, recipe):
    points = {node["step"]: node["panels"]["new"]["swap"]["all"] for node in audit["trajectory"]}
    for node in paired_nodes:
        stats = node[recipe]["swap"]["all"]
        assert stats["n"] == 556 and points.get(node["step"], stats) == stats
        points[node["step"]] = stats
    return sorted(points.items())


def plot(folder, font_path):
    comparison = folder / "comparison-final.json"
    if not comparison.exists():
        comparison = folder / "comparison-78000.json"
    compared = load(comparison)
    font_manager.fontManager.addfont(font_path)
    font = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#a6adb6", "text.color": "#162333",
                         "axes.labelcolor": "#27374a", "xtick.color": "#374657",
                         "ytick.color": "#374657"})
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    fig.subplots_adjust(left=.10, right=.965, top=.80, bottom=.13, wspace=.28, hspace=.32)
    fig.text(.10,.95,"换脸后水印恢复随训练步数的变化",fontsize=21,weight="bold")
    fig.text(.10,.908,"新图指标基于同一组 556 个载体与消息配对，使用真实 Oldpan 换脸。",
             color="#516173",fontsize=11)
    styles = [("75:25",75,"weighted","#315d82","o","--"),
              ("50:50",50,"equal","#9a316a","s","-")]
    handles = []
    coverage = {}
    for col,size in enumerate((64,416)):
        nodes = compared[f"n{size}-seed0"]
        checkpoints = [nodes["coverage"][-1], *nodes["exposure"]]
        axes[0,col].set_title(f"N{size} · {size} 张训练载体",loc="left",weight="bold",pad=9)
        for label, recipe, key, color, marker, line in styles:
            audit = read_audit(folder,recipe,size)
            points = series(audit,checkpoints,key)
            steps = [step/1000 for step,_ in points]
            exact = [stats["exact"] for _,stats in points]
            ber = [100*stats["ber"] for _,stats in points]
            style = dict(color=color,marker=marker,linestyle=line,linewidth=2,markersize=5)
            plotted, = axes[0,col].plot(steps,exact,label=label,**style)
            axes[1,col].plot(steps,ber,**style)
            windows = audit["training_500_step_means"]
            axes[2,col].plot([w["step"]/1000 for w in windows],[w["swap_bce"] for w in windows],
                             color=color,linestyle=line,linewidth=1.45)
            if col == 0:
                handles.append(plotted)
            if size == 416:
                coverage[recipe] = (points[-1][0],windows[-1]["step"])
        for ax in axes[:,col]:
            ax.set_xlim(4, 26 if size == 64 else 163)
            ax.xaxis.set_major_locator(MultipleLocator(6 if size == 64 else 39))
            ax.grid(axis="y",color="#dce2e8",linewidth=.65)
            ax.set_axisbelow(True)
        axes[0,col].set_ylim(bottom=0)
        axes[0,col].yaxis.set_major_locator(MaxNLocator(integer=True,nbins=5))
        axes[1,col].set_ylim(bottom=0)
        axes[2,col].set_yscale("log")
        axes[2,col].set_xlabel("累计优化器步数（千步）")
    for ax,label in zip(axes[:,0],("新图整串恢复（条）","新图 BER（%）","训练换脸 BCE（对数轴）"),strict=True):
        ax.set_ylabel(label)
    fig.legend(handles=handles,loc="upper left",bbox_to_anchor=(.095,.878),frameon=False,ncol=2)
    eval_step, loss_step = coverage[50]
    footer = "训练 BCE 按连续 500 步取均值。"
    if eval_step != coverage[75][0] or loss_step != coverage[75][1]:
        footer += f" N416 的 50:50 已核验验证点至 {eval_step:,} 步，已核验训练损失至 {loss_step:,} 步。"
    fig.text(.10,.035,footer,fontsize=9,color="#516173")
    fig.savefig(folder/"convergence.png",dpi=170,facecolor="white")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font",type=Path,required=True)
    args = parser.parse_args()
    plot(Path(__file__).resolve().parent,args.font)
