"""Does the guidance fool its own classifier?

Cut every run's saved 8 x 8 grid back into its images and ask both classifiers what they are:
the guide (ResNet, steered the sampling) and the judge (ConvNeXt, independent).
  both say "label"        -> the object was drawn
  only the guide says it  -> the guide was fooled (adversarial patterns)
Writes to out_dir: guide_vs_judge.csv, predictions.json (per image) and label<L>_guide_vs_judge.png.
"""
import csv, glob, json, os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid

from helpers import kind
from models import load_guidance_classifier, load_eval_classifier
from plots import COLOR, ORDER, STYLE, describe


def cut_grid(path, n, nrow=8, size=32, pad=2):
    """The first n images of a grid saved by run.py, in [0, 1] (the png rounds pixels by at most 0.5 / 255)."""
    grid = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).float() / 255
    images = []
    for i in range(n):
        row, col = divmod(i, nrow)
        y, x = pad + row * (size + pad), pad + col * (size + pad)
        images.append(grid[:, y:y + size, x:x + size])
    assert torch.equal(make_grid(images, nrow=nrow, padding=pad), grid), "the pieces do not give back the saved grid"
    return torch.stack(images)


@torch.no_grad()
def classify(images, resnet, processor, convnext, device):
    """The class the guide and the judge give to every image."""
    guide = resnet(images.to(device)).argmax(1).cpu()
    inputs = processor(list(images), return_tensors="pt", do_rescale=False).to(device)   # as in evaluate.py
    judge = convnext(**inputs).logits.argmax(1).cpu()
    return guide, judge


def guide_vs_judge(results_dir, out_dir, classifier):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    resnet = load_guidance_classifier(classifier, device)
    processor, convnext = load_eval_classifier(device)
    rows, predictions = [], {}
    for f in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = json.load(open(f))
        png = f[:-len(".json")] + ".png"
        if not os.path.exists(png):
            continue
        images = cut_grid(png, min(64, r["n"]))
        guide, judge = classify(images, resnet, processor, convnext, device)
        says = lambda c: (c == r["label"]).float().mean().item()
        rows.append(dict(r, experiment=kind(r), settings=describe(r).split(": ", 1)[1], images=len(images),
                         guide=says(guide), judge=says(judge), judge_all=r["accuracy"],
                         fooled=((guide == r["label"]) & (judge != r["label"])).float().mean().item(),
                         file=os.path.basename(f)))
        predictions[os.path.basename(f)] = {"guide": guide.tolist(), "judge": judge.tolist()}
    rows.sort(key=lambda r: (r["label"], ORDER.index(r["experiment"]), -r["guided_steps"], r.get("ou_steps", 0)))

    os.makedirs(out_dir, exist_ok=True)
    columns = ["experiment", "settings", "label", "images", "guide", "judge", "fooled", "judge_all", "file"]
    with open(os.path.join(out_dir, "guide_vs_judge.csv"), "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json.dump(predictions, open(os.path.join(out_dir, "predictions.json"), "w"))
    print_table(rows)
    for label in sorted({r["label"] for r in rows}):
        plot(rows, label, os.path.join(out_dir, f"label{label}_guide_vs_judge.png"))
    print("saved to", out_dir)
    return rows


def print_table(rows):
    """guide, judge: fraction called `label`; fooled: guide yes, judge no; judge_all: the run's accuracy (all images)."""
    print(f"{'experiment':18s} {'settings':50s} label images | guide judge fooled | judge_all")
    for r in rows:
        print(f"{r['experiment']:18s} {r['settings'][:50]:50s} {r['label']:5d} {r['images']:6d} | "
              f"{r['guide']:5.2f} {r['judge']:5.2f} {r['fooled']:6.2f} | {r['judge_all']:9.2f}")


def plot(rows, label, path):
    """Methods guided at all 100 steps: guide (solid) vs judge (light)."""
    full = {}
    for r in rows:
        if r["label"] == label and r["guided_steps"] == 100 and not r.get("ou_steps"):
            full[r["experiment"]] = r
    full = [full[k] for k in ORDER if k in full]
    if not full:
        return
    with plt.rc_context(STYLE):
        fig, axis = plt.subplots(figsize=(8, 0.6 * len(full) + 1.4))
        for i, r in enumerate(full):
            color = COLOR[r["experiment"]]
            for offset, key, alpha in [(-0.19, "guide", 1.0), (0.19, "judge", 0.35)]:
                axis.barh(i + offset, r[key], height=0.36, color=color, alpha=alpha, edgecolor=color)
                axis.text(r[key], i + offset, f"  {r[key]:.2f}", va="center", fontsize=8)
        axis.set_yticks(range(len(full)), [r["experiment"] for r in full])
        axis.invert_yaxis()
        axis.set_xlim(0, 1.12)
        axis.grid(axis="y", visible=False)
        axis.set_xlabel(f"fraction of the images called label {label}")
        axis.bar(0, 0, color="#52514e", label="guide (ResNet, steered the sampling)")
        axis.bar(0, 0, color="#52514e", alpha=0.35, label="judge (ConvNeXt, independent)")
        axis.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
        axis.set_title(f"Does the guidance fool its own classifier? (label {label}, the same {full[0]['images']} images "
                       "for both classifiers)", loc="left", fontsize=10)
        fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    import sys
    guide_vs_judge(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "checkpoints/resnet_cifar10.pt")
