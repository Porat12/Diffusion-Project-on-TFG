"""The figures of the report, drawn from the result files: figures/results_table.csv,
figures/guide_vs_judge/guide_vs_judge.csv, results/diagnostics/diagnose_label8.json and the sample grids in results/.
Colors: teal = the network's estimate (DPS, TFG), orange = the OU estimate, slate = mean guidance only,
gray = no guidance.

Run from the repository root:  python report/make_figures.py   (writes report/figures/*.png)
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

from style import GRAY, INK, LIGHT, MUTED, ORANGE, TEAL, setup_matplotlib

ROOT = Path(__file__).resolve().parent.parent           # the repository root
OUT = Path(__file__).parent / "figures"
TINT = "#E7F3F1"
N = 100                                                   # images per run


def load_runs():
    with open(ROOT / "figures" / "results_table.csv", newline="") as f:
        return list(csv.DictReader(f))


def find(rows, experiment, guided=100, k=0, st=False):
    """The run of `experiment` guided in the last `guided` steps, with the OU guess (or nothing) in the last k."""
    for r in rows:
        if (r["experiment"] == experiment and int(r["guided_steps"]) == guided and int(r["ou_steps"]) == k
                and (r["straight_through"] == "True") == st):
            return r
    raise KeyError((experiment, guided, k, st))


def acc(r):
    return 100 * float(r["accuracy"])


def se(r):
    p = float(r["accuracy"])
    return 100 * np.sqrt(p * (1 - p) / N)


def plain(ax, keep=("left", "bottom")):
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(s in keep)
        ax.spines[s].set_color(GRAY)
    ax.tick_params(colors=MUTED, labelsize=14, length=0)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


POINTS = [  # label, experiment, color, marker, label offset (points)
    ("no guidance", "unconditional", GRAY, "o", (10, -4)),
    ("mean guidance only", "mean-only", MUTED, "^", (10, -4)),
    ("DPS", "DPS", TEAL, "o", (-14, 6)),
    ("TFG", "TFG", TEAL, "D", (-12, 10)),
    ("OU-DPS", "OU-DPS", ORANGE, "o", (10, 4)),
    ("OU-TFG", "OU-TFG", ORANGE, "D", (10, -4)),
]


def accuracy_time(rows):
    """Every method guided at all 100 steps: accuracy (judge) against seconds per image."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.add_patch(Rectangle((0.18, 70), 0.3, 28, facecolor=TINT, edgecolor="none", zorder=0))
    ax.text(0.33, 84, "cheap and accurate:\nwhat we wanted", ha="center", va="center", fontsize=14, color=TEAL)
    for label, exp, color, marker, off in POINTS:
        r = find(rows, exp)
        x, y = float(r["seconds_per_image"]), acc(r)
        ax.errorbar(x, y, yerr=se(r), fmt=marker, color=color, ms=11, capsize=3, lw=1.4, zorder=3)
        ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=15, color=color,
                    ha="right" if off[0] < 0 else "left", va="bottom" if off[1] > 0 else "top",
                    weight="bold" if exp.startswith("OU") else "normal")
    ax.set_xlim(0.15, 1.12)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("seconds per image", fontsize=15, color=MUTED)
    ax.set_ylabel("accuracy", fontsize=15, color=MUTED)
    ax.grid(color=LIGHT, lw=1)
    plain(ax)
    save(fig, "accuracy_time.png")


def where_useful(rows):
    """Left: does the guide already see the final class in each guess? Right: guiding only the last t* steps."""
    diag = json.loads((ROOT / "results" / "diagnostics" / "diagnose_label8.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 3.3), gridspec_kw={"wspace": 0.3})
    x = np.arange(1, len(diag) + 1)
    a.axhline(10, color=GRAY, lw=1.4, ls=":")
    a.plot(x, [100 * d["network_agreement"] for d in diag], color=TEAL, lw=2.6)
    a.plot(x, [100 * d["ou_agreement"] for d in diag], color=ORANGE, lw=2.6)
    a.text(52, 50, "network's guess", fontsize=15, color=TEAL, ha="right")
    a.text(55, 20, "OU guess", fontsize=15, color=ORANGE, weight="bold")
    a.text(2, 3, "chance", fontsize=13, color=GRAY)
    a.set_title("how often the guide already sees the final class", fontsize=15, color=MUTED, loc="left")
    a.set_xticks([1, 100], ["noise", "image"])
    a.tick_params(axis="x", pad=8)
    a.set_xlim(1, 100)
    a.set_ylim(0, 105)
    a.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    a.grid(axis="y", color=LIGHT, lw=1)
    plain(a)

    ts = [5, 10, 20, 30, 40, 60, 100]
    b.axhline(acc(find(rows, "TFG")), color=TEAL, lw=1.4, ls=":")
    b.axhline(acc(find(rows, "unconditional")), color=GRAY, lw=1.4, ls=":")
    b.text(101, acc(find(rows, "TFG")) + 2, "TFG", fontsize=14, color=TEAL, ha="right", va="bottom")
    b.text(101, acc(find(rows, "unconditional")) - 2, "no guidance", fontsize=14, color=GRAY, ha="right", va="top")
    for exp, marker, ls in [("OU-DPS", "o", "-"), ("OU-TFG", "D", "--")]:
        rs = [find(rows, exp, guided=t) for t in ts]
        b.errorbar(ts, [acc(r) for r in rs], yerr=[se(r) for r in rs], fmt=marker, ls=ls, color=ORANGE, ms=7,
                   capsize=3, lw=2)
    b.text(72, 31, "OU-DPS, OU-TFG", fontsize=15, color=ORANGE, weight="bold", ha="center")
    b.set_title("accuracy when guiding only the last t* steps", fontsize=15, color=MUTED, loc="left")
    b.set_xlabel("t*", fontsize=15, color=MUTED, labelpad=2, style="italic")
    b.set_xticks(ts[1:], [str(t) for t in ts[1:]])
    b.set_xlim(0, 103)
    b.set_ylim(0, 100)
    b.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
    b.grid(axis="y", color=LIGHT, lw=1)
    plain(b)
    save(fig, "where_useful.png")


def guide_judge():
    """The same 64 images scored by the guide (it steered the sampling) and by the judge (independent)."""
    with open(ROOT / "figures" / "guide_vs_judge" / "guide_vs_judge.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if "t*" not in r["settings"] and "k=" not in r["settings"]
                and not r["experiment"].endswith("-ST")]
    show = [("no guidance", "unconditional", GRAY), ("DPS", "DPS", TEAL), ("TFG", "TFG", TEAL),
            ("mean guidance only", "mean-only", MUTED), ("OU-DPS", "OU-DPS", ORANGE), ("OU-TFG", "OU-TFG", ORANGE)]
    by_exp = {r["experiment"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    plain(ax, keep=("left",))
    y = np.arange(len(show))[::-1] * 1.0
    h = 0.38
    for yi, (label, exp, _) in zip(y, show):
        g, j = 100 * float(by_exp[exp]["guide"]), 100 * float(by_exp[exp]["judge"])
        ax.barh(yi + h / 2, g, height=h, color=GRAY)
        ax.barh(yi - h / 2, j, height=h, color=INK)
        ax.text(g + 1.5, yi + h / 2, f"{g:.0f}%", va="center", fontsize=14, color=MUTED)
        ax.text(j + 1.5, yi - h / 2, f"{j:.0f}%", va="center", fontsize=14, color=INK, weight="bold")
    ax.set_yticks(y, [label for label, *_ in show], fontsize=16)
    for lab, (_, exp, color) in zip(ax.get_yticklabels(), show):
        lab.set_color(color)
        if exp.startswith("OU"):
            lab.set_fontweight("bold")
    ax.set_xlim(0, 118)
    ax.set_xticks([])
    ax.text(0, y[0] + 0.85, "guide", fontsize=14, color=MUTED, va="center")
    ax.text(22, y[0] + 0.85, "judge", fontsize=14, color=INK, va="center", weight="bold")
    ax.add_patch(Rectangle((14, y[0] + 0.72), 5, 0.26, color=GRAY, clip_on=False))
    ax.add_patch(Rectangle((35, y[0] + 0.72), 5, 0.26, color=INK, clip_on=False))
    save(fig, "guide_judge.png")


def samples(cols=range(6)):
    """The first images of three runs; each column starts from the same noise."""
    runs = [("no guidance", "none_label8_rho0.0_guided100_mu0.0_iter0_gamma0.0_n100_seed0.png", GRAY),
            ("TFG", "network_label8_rho0.5_guided100_mu0.25_iter4_gamma0.001_n100_seed0.png", TEAL),
            ("OU-DPS", "ou_label8_rho0.25_guided100_mu0.0_iter0_gamma0.0_n100_seed0.png", ORANGE)]
    fig, axes = plt.subplots(len(runs), len(cols), figsize=(5.9, 3.1), gridspec_kw={"wspace": 0.06, "hspace": 0.1})
    for i, (label, file, color) in enumerate(runs):
        grid = Image.open(ROOT / "results" / file).convert("RGB")       # 8 x 8 images of 32 px, 2 px apart
        for j, c in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(grid.crop((2 + 34 * c, 2, 34 + 34 * c, 34)), interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].set_ylabel(label, fontsize=15, color=color, rotation=0, ha="right", va="center", labelpad=8,
                              weight="bold" if label.startswith("OU") else "normal")
    save(fig, "samples.png")


def tfg_then_ou(rows):
    """TFG for the first 100 - k steps, then k steps with the OU guess, or with no guidance at all (the control)."""
    tfg = find(rows, "TFG")
    ctrl = [(0, tfg)] + [(k, find(rows, "TFG-then-unguided", k=k)) for k in (5, 10, 20, 30, 40, 60)]
    ou = [(0, tfg)] + [(k, find(rows, "TFG-then-OU", k=k)) for k in (10, 20, 30, 40, 60)]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.axhline(acc(tfg), color=TEAL, lw=1.4, ls=":")
    ax.text(61, acc(tfg) + 1.5, "TFG", fontsize=14, color=TEAL, ha="right", va="bottom")
    for pts, color, marker, ls, label, dy in [(ctrl, GRAY, "s", "--", "then no\nguidance", -7),
                                              (ou, ORANGE, "o", "-", "then OU\nguess", 7)]:
        ks, rs = zip(*pts)
        ax.errorbar(ks, [acc(r) for r in rs], yerr=[se(r) for r in rs], fmt=marker, ls=ls, color=color, ms=8,
                    capsize=3, lw=2.2)
        ax.text(62.5, acc(rs[-1]) + dy, label, fontsize=15, color=color, va="center",
                weight="bold" if color == ORANGE else "normal")
    ax.set_xticks([0, 10, 20, 30, 40, 60])
    ax.set_xlim(-2, 75)
    ax.set_ylim(15, 100)
    ax.set_yticks([25, 50, 75, 100], ["25%", "50%", "75%", "100%"])
    ax.set_xlabel("$k$ = last steps without a backward pass", fontsize=15, color=MUTED)
    ax.grid(axis="y", color=LIGHT, lw=1)
    plain(ax)
    save(fig, "tfg_then_ou.png")


def noise_numbers():
    """The noise of the OU estimate, sqrt((1 - abar_t) / abar_t), over the sampler's 100 steps (Section 3)."""
    abar = np.cumprod(1 - np.linspace(1e-4, 0.02, 1000))          # the CIFAR-10 model's noise schedule
    steps = [round(k * 999 / 99) for k in range(100)][::-1]        # 999, 989, ..., 0, as in sampler.py
    std = np.sqrt((1 - abar[steps]) / abar[steps])
    return std[0], int((std < 1).sum())


if __name__ == "__main__":
    setup_matplotlib()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    accuracy_time(rows)
    where_useful(rows)
    guide_judge()
    samples()
    tfg_then_ou(rows)
    start, late = noise_numbers()
    print(f"OU estimate: noise std {start:.0f} at the first step, below 1 in the last {late} steps")
