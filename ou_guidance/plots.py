"""Figures and tables from the result files.

make_plots(results_dir, out_dir) writes, per label L:
  results_table.csv                every run
  README.md                        what each figure shows and which runs it uses
  label<L>_1_main_comparison.png   all methods (guided at all steps): accuracy, FID, time
  label<L>_2_ou_guided_steps.png   OU-DPS / OU-TFG vs t*
  label<L>_3_tfg_then_ou.png       TFG-then-OU and TFG-then-unguided vs k
  label<L>_4_accuracy_vs_time.png  every run
  label<L>_5_accuracy_vs_fid.png   every run
  label<L>_6_samples.png           8 images per method
Only the most common n of each label is used (FID depends on n).
"""
import csv, glob, json, os
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from helpers import kind

ORDER = ["unconditional", "DPS", "TFG", "mean-only", "OU-DPS", "OU-TFG", "OU-DPS-ST", "OU-TFG-ST", "TFG-then-OU",
         "TFG-then-unguided"]
# one colour per method (-ST: the same colour, hatched / dashed / hollow)
COLOR = {"unconditional": "#8a8985", "DPS": "#2a78d6", "TFG": "#eb6834", "mean-only": "#1baf7a", "OU-DPS": "#6250d6",
         "OU-TFG": "#e34948", "OU-DPS-ST": "#6250d6", "OU-TFG-ST": "#e34948", "TFG-then-OU": "#008300",
         "TFG-then-unguided": "#eda100"}
MARKER = {"unconditional": "o", "DPS": "s", "TFG": "D", "mean-only": "^", "OU-DPS": "v", "OU-TFG": "P",
          "OU-DPS-ST": "v", "OU-TFG-ST": "P", "TFG-then-OU": "X", "TFG-then-unguided": "h"}
ABLATION = {"OU-DPS-ST", "OU-TFG-ST"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
STYLE = {"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
         "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 10, "figure.dpi": 130, "savefig.bbox": "tight"}


def error(r):
    """Standard error of an accuracy."""
    return (r["accuracy"] * (1 - r["accuracy"]) / r["n"]) ** 0.5


def describe(r):
    """Settings in one line, e.g. 'TFG: rho=0.5, mu=0.25, gamma=0.001'."""
    if r["method"] == "none":
        return "unconditional: no guidance"
    text = f"{kind(r)}: rho={r['rho']}"
    if r["n_iter"]:
        text += f", mu={r['mu']}, gamma={r['gamma']}"
    if r["guided_steps"] < 100:
        text += f", t*={r['guided_steps']}"
    if r.get("ou_steps"):
        text += f", k={r['ou_steps']}, ou_rho={r['ou_rho']}, ou_mu={r['ou_mu']}"
    return text


def make_plots(results_dir, out_dir):
    rows = []
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        r = json.load(open(f))
        r["file"] = os.path.basename(f)
        rows.append(r)
    os.makedirs(out_dir, exist_ok=True)
    save_table(rows, os.path.join(out_dir, "results_table.csv"))
    readme = ["# Figures\n", f"From the results in `{results_dir}`. All runs start from the same noise (seed 0). "
              "Accuracy: the judge (ConvNeXt), +- standard error. FID: against real images of the label (compare equal n "
              "only). Time: seconds per image on Colab.\n"]
    with plt.rc_context(STYLE):
        for label in sorted({r["label"] for r in rows}):
            n = Counter(r["n"] for r in rows if r["label"] == label).most_common(1)[0][0]
            same = [r for r in rows if r["label"] == label and r["n"] == n]
            readme += plots_for_label(same, label, n, results_dir, out_dir)
    open(os.path.join(out_dir, "README.md"), "w").write("\n".join(readme))
    print("saved to", out_dir)


def plots_for_label(rows, label, n, results_dir, out_dir):
    """All figures of one label; returns their README lines."""
    name = lambda i, what: os.path.join(out_dir, f"label{label}_{i}_{what}.png")
    about = f"label {label}, n = {n} images per run"
    full = {}                                                         # one run per method, guided at all 100 steps
    for r in sorted(rows, key=lambda r: r["file"]):
        if r["guided_steps"] == 100 and not r.get("ou_steps"):
            full[kind(r)] = r
    full = [full[k] for k in ORDER if k in full]
    lines = [f"\n## Label {label} (n = {n})\n"]

    # 1. main comparison: one bar per method, one panel per measure
    fig, axes = plt.subplots(1, 3, figsize=(14, 0.45 * len(full) + 1.6), sharey=True)
    y = list(range(len(full)))
    for axis, (key, title, fmt) in zip(axes, [("accuracy", "accuracy (higher is better)", "{:.2f}"),
                                               ("fid", "FID (lower is better)", "{:.0f}"),
                                               ("seconds_per_image", "seconds per image", "{:.2f}")]):
        values = [r.get(key, float("nan")) for r in full]
        axis.barh(y, values, height=0.6, color=[COLOR[kind(r)] for r in full],
                  hatch=["//" if kind(r) in ABLATION else "" for r in full], edgecolor="white",
                  xerr=[error(r) for r in full] if key == "accuracy" else None, ecolor=INK, capsize=3)
        for yi, v, r in zip(y, values, full):                      # the value, after the bar
            axis.text(v + (error(r) if key == "accuracy" else 0), yi, "  " + fmt.format(v), va="center", fontsize=9)
        axis.set_title(title, loc="left", fontsize=10)
        axis.grid(axis="y", visible=False)
        axis.margins(x=0.25)
    axes[0].set_yticks(y, [f"{kind(r)}  ({r['unet_backward']} U-Net backward)" for r in full])
    axes[0].invert_yaxis()
    fig.suptitle(f"Every method with guidance at all 100 steps ({about})", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(name(1, "main_comparison")); plt.close(fig)
    lines += [f"**label{label}_1_main_comparison.png**: all methods, guided at all 100 steps. Next to each name: "
              "U-Net backward passes. Hatched: straight-through. Runs:"]
    lines += [f"- {describe(r)}  (`{r['file']}`)" for r in full]

    # 2. OU guess: accuracy and FID vs t* (the number of guided last steps)
    sweep = {k: sorted([r for r in rows if kind(r) == k], key=lambda r: r["guided_steps"])
             for k in ["OU-DPS", "OU-TFG", "OU-DPS-ST", "OU-TFG-ST"]}
    sweep = {k: v for k, v in sweep.items() if v}
    if sweep:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for axis, key, title in [(axes[0], "accuracy", "accuracy (higher is better)"), (axes[1], "fid", "FID (lower is better)")]:
            for k, points in sweep.items():
                axis.errorbar([r["guided_steps"] for r in points], [r.get(key, float("nan")) for r in points],
                              [error(r) for r in points] if key == "accuracy" else None, color=COLOR[k], marker=MARKER[k],
                              markersize=7, linewidth=2, linestyle="--" if k in ABLATION else "-",
                              markerfacecolor="white" if k in ABLATION else COLOR[k], capsize=3, label=k)
            reference_lines(axis, full, key, ["unconditional", "DPS", "TFG", "mean-only"])
            axis.set_xlabel("t* = number of last steps with guidance (out of 100)")
            axis.set_title(title, loc="left", fontsize=10)
        axes[0].legend(fontsize=8, frameon=False)
        fig.suptitle(f"OU guess: how many of the last steps should be guided? ({about})", x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(name(2, "ou_guided_steps")); plt.close(fig)
        lines += [f"\n**label{label}_2_ou_guided_steps.png**: OU-DPS / OU-TFG guided only in the last t* steps "
                  "(strengths tuned at t* = 100). Dashed: straight-through. Dotted: the other methods. Runs:"]
        lines += [f"- {describe(r)}  (`{r['file']}`)" for points in sweep.values() for r in points]

    # 3. TFG-then-OU and its control TFG-then-unguided vs k (k = 0 is TFG)
    tfg = [r for r in full if kind(r) == "TFG"]
    runs_k = {k: sorted([r for r in rows if kind(r) == k], key=lambda r: r["ou_steps"])
              for k in ["TFG-then-OU", "TFG-then-unguided"]}
    runs_k = {k: v for k, v in runs_k.items() if v}
    lines_k = {k: tfg + v for k, v in runs_k.items()}                 # both lines start at TFG
    if lines_k:
        ks = sorted({r.get("ou_steps", 0) for points in lines_k.values() for r in points})
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for axis, key, title in [(axes[0], "accuracy", "accuracy (higher is better)"), (axes[1], "fid", "FID (lower is better)"),
                                 (axes[2], "seconds_per_image", "seconds per image")]:
            for k, points in lines_k.items():
                axis.errorbar([r.get("ou_steps", 0) for r in points], [r.get(key, float("nan")) for r in points],
                              [error(r) for r in points] if key == "accuracy" else None, color=COLOR[k], marker=MARKER[k],
                              markersize=8, linewidth=2, capsize=3, label=k)
            if key != "seconds_per_image":
                reference_lines(axis, full, key, ["unconditional", "TFG"])
            axis.set_xticks(ks)
            axis.set_xlabel("k = last steps without U-Net backprop\n(OU guess, or no guidance at all)")
            axis.set_title(title, loc="left", fontsize=10)
        axes[0].legend(fontsize=8, frameon=False, loc="center left")
        fig.suptitle(f"TFG early, then the OU guess or no guidance in the last k steps (k = 0 is TFG; {about})", x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(name(3, "tfg_then_ou")); plt.close(fig)
        lines += [f"\n**label{label}_3_tfg_then_ou.png**: TFG, then in the last k steps the OU guess (TFG-then-OU, "
                  "strengths tuned per k) or no guidance (TFG-then-unguided, the control). Equal lines: the OU guess adds "
                  "nothing. Where the search chose ou_rho = ou_mu = 0, the run is on the control line only. Runs:"]
        lines += [f"- {describe(r)}  (`{r['file']}`)" for r in tfg + [r for v in runs_k.values() for r in v]]

    # 4. and 5. every run as a point: accuracy vs time and vs FID
    for i, key, xlabel, what in [(4, "seconds_per_image", "seconds per image", "accuracy_vs_time"),
                                 (5, "fid", "FID (lower is better)", "accuracy_vs_fid")]:
        fig, axis = plt.subplots(figsize=(9, 6))
        for k in ORDER:
            points = [r for r in rows if kind(r) == k]
            if not points:
                continue
            axis.scatter([r.get(key, float("nan")) for r in points], [r["accuracy"] for r in points], s=55, label=k,
                         marker=MARKER[k], color="white" if k in ABLATION else COLOR[k], edgecolors=COLOR[k], linewidths=1.5)
            for r in points:                                      # tag t* or k
                tag = f"t*={r['guided_steps']}" if r["guided_steps"] < 100 else (f"k={r['ou_steps']}" if r.get("ou_steps") else "")
                if tag:
                    axis.annotate(tag, (r.get(key, float("nan")), r["accuracy"]), xytext=(5, 3),
                                  textcoords="offset points", fontsize=7, color=MUTED)
        if key == "fid":
            axis.invert_xaxis()                                   # better = to the right
        axis.set_xlabel(xlabel); axis.set_ylabel("accuracy")
        axis.legend(fontsize=8, frameon=False, loc="best")
        axis.set_title(f"Every run: {'cost' if i == 4 else 'image quality'} vs accuracy ({about}).\n"
                       "Points without a tag have guidance at all 100 steps.", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(name(i, what)); plt.close(fig)
    lines += [f"\n**label{label}_4_accuracy_vs_time.png**, **label{label}_5_accuracy_vs_fid.png**: every run as a point "
              "(hollow: straight-through). t* = guided last steps; k = last steps with the OU guess or no guidance. "
              "Best: up-left (time), up-right (FID, axis reversed)."]

    # 6. samples: the first row (8 images) of each run's grid
    pngs = [(r, os.path.join(results_dir, r["file"].replace(".json", ".png"))) for r in full]
    pngs = [(r, p) for r, p in pngs if os.path.exists(p)]
    if pngs:
        fig, axes = plt.subplots(len(pngs), 1, figsize=(7, 0.95 * len(pngs) + 0.6), squeeze=False)
        for axis, (r, p) in zip(axes[:, 0], pngs):
            axis.imshow(plt.imread(p)[:36])                       # first row of the grid
            axis.set_xticks([]); axis.set_yticks([]); axis.grid(False)
            axis.set_ylabel(kind(r), rotation=0, ha="right", va="center", fontsize=9)
        fig.suptitle(f"Samples ({about}; each column starts from the same noise)", fontsize=10)
        fig.tight_layout(); fig.savefig(name(6, "samples")); plt.close(fig)
        lines += [f"\n**label{label}_6_samples.png**: the first 8 images of each run of figure 1. A column = the same "
                  "starting noise."]

    # the full table
    lines += ["\n| experiment | settings | n | accuracy | FID | s/img | U-Net backward | file |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (ORDER.index(kind(r)), -r["guided_steps"], r.get("ou_steps", 0))):
        lines.append(f"| {kind(r)} | {describe(r).split(': ', 1)[1]} | {r['n']} | {r['accuracy']:.2f} +- {error(r):.2f} | "
                     f"{r.get('fid', float('nan')):.1f} | {r['seconds_per_image']:.2f} | {r['unet_backward']} | `{r['file']}` |")
    return lines


def reference_lines(axis, full, key, kinds):
    """A dotted line, labelled on the right, for each method in `kinds`."""
    edge = transforms.blended_transform_factory(axis.transAxes, axis.transData)
    names = {}                                                    # value -> methods (equal values share a label)
    for r in full:
        if kind(r) in kinds and key in r:
            axis.axhline(r[key], color=COLOR[kind(r)], linestyle=":", linewidth=1.5)
            names.setdefault(round(r[key], 2 if key == "accuracy" else 0), []).append(kind(r))
    for value, same in names.items():
        axis.text(1.01, value, " / ".join(same), transform=edge, va="center", fontsize=8, color=MUTED)


def save_table(rows, path):
    """Every run in one CSV file."""
    columns = ["experiment", "label", "method", "rho", "mu", "n_iter", "gamma", "guided_steps", "ou_steps", "ou_rho", "ou_mu",
               "straight_through", "n", "seed", "accuracy", "fid", "seconds_per_image", "unet_forward", "unet_backward",
               "classifier_calls", "peak_memory_gb", "file"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["label"], ORDER.index(kind(r)), -r["guided_steps"], r.get("ou_steps", 0))):
            writer.writerow(dict(r, experiment=kind(r)))
